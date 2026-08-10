#!/usr/bin/env python3
"""
run.py - Interactive & CLI Text Generation Engine for TinyGrad Transformer Checkpoints.

Usage Examples:
  # Generate text from prompt using the latest checkpoint in checkpoints/
  uv run python run.py --prompt "Once upon a time" --max-tokens 100 --temperature 0.8

  # Specify explicit checkpoint file
  uv run python run.py --checkpoint checkpoints/model_125m_step_30518.safetensors --prompt "In a small village"

  # Enter interactive text generation loop
  uv run python run.py --interactive
"""

import argparse
import json
import os
import re
import sys

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time

import numpy as np
import tiktoken
from tinygrad import Tensor, TinyJit, Variable, dtypes
from tinygrad.device import Device
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load

from src.model import GPT


def load_vocab_map(dataset_name: str = "tinystories", vocab_map_path: str | None = None):
    """Load vocabulary map for trimming/restoring original GPT-2 token IDs."""
    if not vocab_map_path:
        if dataset_name.lower() == "fineweb":
            vocab_map_path = "data/FineWeb/vocab_map.json"
        else:
            vocab_map_path = "data/TinyStories/vocab_map.json"

    if not os.path.exists(vocab_map_path):
        vocab_map_path = os.path.join(os.path.dirname(__file__), vocab_map_path)

    if os.path.exists(vocab_map_path):
        with open(vocab_map_path) as f:
            vmap = json.load(f)
        orig_to_new = {int(k): int(v) for k, v in vmap.get("orig_to_new", {}).items()}
        new_to_orig = vmap.get("new_to_orig", [])
        return orig_to_new, new_to_orig
    print(f"⚠️ Vocab map file '{vocab_map_path}' not found. Using identity token mapping.", flush=True)
    return None, None


def find_latest_checkpoint(checkpoint_dir: str, model_size: str) -> str | None:
    """Scan checkpoint directory for highest step count matching model scale."""
    if not os.path.exists(checkpoint_dir):
        return None
    pattern = re.compile(rf"model_{model_size.lower()}_step_(\d+)\.safetensors$")
    max_step = -1
    best_ckpt = None
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            s = int(match.group(1))
            if s > max_step:
                max_step = s
                best_ckpt = os.path.join(checkpoint_dir, filename)
    return best_ckpt


def sample_next_token(logits_np: np.ndarray, temperature: float = 0.8, top_k: int = 40) -> int:
    """Sample next token index using temperature scaling and top-k filtering."""
    if temperature <= 0.0:
        return int(np.argmax(logits_np))

    logits = logits_np / temperature

    if top_k > 0 and top_k < len(logits):
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        top_k_logits = logits[top_k_indices]
        top_k_probs = np.exp(top_k_logits - np.max(top_k_logits))
        top_k_probs /= np.sum(top_k_probs)
        return int(np.random.choice(top_k_indices, p=top_k_probs))
    else:
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        return int(np.random.choice(len(probs), p=probs))


def generate_text(
    model: GPT,
    tokenizer,
    orig_to_new: dict | None,
    new_to_orig: list | None,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    seq_len: int = 256,
    use_jit: bool = True,
    profile: bool = False,
) -> str:
    """Autoregressively generate text given a starting prompt using KV-caching and @TinyJit acceleration."""
    model.reset_cache()

    orig_prompt_ids = tokenizer.encode(prompt)
    if not orig_prompt_ids:
        orig_prompt_ids = [50256]  # Fallback to EOS

    if orig_to_new:
        trimmed_prompt_ids = [orig_to_new.get(tid, 0) for tid in orig_prompt_ids]
    else:
        trimmed_prompt_ids = list(orig_prompt_ids)

    current_ids = list(trimmed_prompt_ids)
    raw_vocab_size = model.raw_vocab_size

    def step_fn(x_in: Tensor, start_pos: Variable) -> Tensor:
        return model.forward(x_in, start_pos=start_pos)

    if use_jit:
        step_jit = TinyJit(step_fn)
        # JIT compilation warmup pass so kernel compilation finishes before starting timing
        x_warm = Tensor([[0]], dtype=dtypes.int32).realize()
        v_warm = Variable("start_pos", 0, 511).bind(0)
        _ = step_jit(x_warm, v_warm).realize()
        Device[Device.DEFAULT].synchronize()
        model.reset_cache()
    else:
        step_jit = step_fn

    print(f'\n📝 Prompt: "{prompt}"', flush=True)
    print("-------------------------------------------------------", flush=True)
    sys.stdout.write(prompt)
    sys.stdout.flush()

    token_times_ms = []
    t_start = time.time()

    # 1. Prompt Phase (Fill initial KV cache for prompt tokens)
    t0 = time.time()
    prompt_len = len(trimmed_prompt_ids)
    x_prompt = Tensor([trimmed_prompt_ids], dtype=dtypes.int32).realize()
    v_start_pos_0 = Variable("start_pos", 0, 511).bind(0)
    logits_prompt = model.forward(x_prompt, start_pos=v_start_pos_0).realize()
    logits_np = logits_prompt[0, prompt_len - 1, :raw_vocab_size].cast(dtypes.float32).numpy()

    next_trimmed_id = sample_next_token(logits_np, temperature=temperature, top_k=top_k)
    current_ids.append(next_trimmed_id)

    t1 = time.time()
    ttft_ms = (t1 - t0) * 1000.0
    token_times_ms.append(ttft_ms)

    # Stream first generated token
    if new_to_orig and next_trimmed_id < len(new_to_orig):
        orig_id = new_to_orig[next_trimmed_id]
    else:
        orig_id = next_trimmed_id
    sys.stdout.write(tokenizer.decode([orig_id]))
    sys.stdout.flush()

    # 2. Single-token Autoregressive Generation Phase
    start_pos = prompt_len
    for step_idx in range(1, max_new_tokens):
        t0 = time.time()

        x_in = Tensor([[next_trimmed_id]], dtype=dtypes.int32).realize()
        v_start_pos = Variable("start_pos", 0, 511).bind(start_pos)
        logits = step_jit(x_in, v_start_pos).realize()
        logits_np = logits[0, 0, :raw_vocab_size].cast(dtypes.float32).numpy()

        next_trimmed_id = sample_next_token(logits_np, temperature=temperature, top_k=top_k)
        current_ids.append(next_trimmed_id)
        start_pos += 1

        t1 = time.time()
        step_ms = (t1 - t0) * 1000.0
        token_times_ms.append(step_ms)

        # Stream decoded token to stdout
        if new_to_orig and next_trimmed_id < len(new_to_orig):
            orig_id = new_to_orig[next_trimmed_id]
        else:
            orig_id = next_trimmed_id

        decoded_chunk = tokenizer.decode([orig_id])
        sys.stdout.write(decoded_chunk)
        sys.stdout.flush()

    sys.stdout.write("\n-------------------------------------------------------\n")
    sys.stdout.flush()

    t_end = time.time()
    total_gen_sec = t_end - t_start
    gen_tok_per_sec = max_new_tokens / total_gen_sec if total_gen_sec > 0 else 0.0
    avg_per_token_ms = float(np.mean(token_times_ms[1:])) if len(token_times_ms) > 1 else (float(np.mean(token_times_ms)) if token_times_ms else 0.0)

    if profile:
        print("\n=======================================================", flush=True)
        print("⚡ INFERENCE PROFILING TELEMETRY METRICS", flush=True)
        print("=======================================================", flush=True)
        print(f"Time To First Token (TTFT): {ttft_ms:.2f} ms", flush=True)
        print(f"Average Generation Speed:  {gen_tok_per_sec:.1f} tokens/sec", flush=True)
        print(f"Average Per-Token Latency: {avg_per_token_ms:.2f} ms/token", flush=True)
        print(f"Total Generation Duration: {total_gen_sec:.2f} s ({max_new_tokens} tokens)", flush=True)
        print("=======================================================\n", flush=True)

    if new_to_orig:
        full_orig_ids = [new_to_orig[tid] for tid in current_ids if tid < len(new_to_orig)]
    else:
        full_orig_ids = current_ids

    return tokenizer.decode(full_orig_ids)


def main():
    parser = argparse.ArgumentParser(description="TinyGrad Transformer Inference & Generation Engine")
    parser.add_argument("--dataset", type=str, choices=["tinystories", "fineweb"], default="tinystories", help="Target dataset (tinystories or fineweb)")
    parser.add_argument("--model-size", choices=["15M", "125M"], default="125M", help="Target model scale")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to .safetensors checkpoint file")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory containing model checkpoints")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Initial generation prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering threshold")
    parser.add_argument("--no-jit", action="store_true", default=False, help="Disable @TinyJit compilation")
    parser.add_argument("--profile", action="store_true", default=False, help="Enable detailed inference profiling telemetry")
    parser.add_argument("--interactive", action="store_true", default=False, help="Run interactive prompt loop")
    args = parser.parse_args()

    # Locate checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir, args.model_size)

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No valid checkpoint found for model scale '{args.model_size}' in directory '{args.checkpoint_dir}'.")

    print("\n=======================================================", flush=True)
    print(f"🚀 TINYGRAD INFERENCE ENGINE ({args.model_size} | Dataset: {args.dataset})", flush=True)
    print(f"Checkpoint: {ckpt_path}", flush=True)
    print(f"JIT Acceleration: {not args.no_jit} | Profiling: {args.profile}", flush=True)
    print("=======================================================\n", flush=True)

    # Initialize Tokenizer & Vocab Map
    tokenizer = tiktoken.get_encoding("gpt2")
    orig_to_new, new_to_orig = load_vocab_map(dataset_name=args.dataset)

    # Model Preset Architecture Parameters
    if args.model_size == "125M":
        vocab_size = 13970
        d_model = 768
        n_layers = 12
        n_heads = 12
        d_ff = 3072
    else:
        vocab_size = 13970
        d_model = 288
        n_layers = 6
        n_heads = 6
        d_ff = 1152

    Tensor.training = False
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=512,
        use_swiglu=True,
        use_rope=True,
        pad_vocab_multiple=128,
        pad_vocab_power_of_2=True,
    )

    # Load parameters from safetensors checkpoint
    print(f"📦 Loading weights from '{ckpt_path}'...", flush=True)
    t_load_start = time.time()
    state = safe_load(ckpt_path)
    load_state_dict(model, state, strict=False)
    Tensor.realize(*get_parameters(model))
    t_load_ms = (time.time() - t_load_start) * 1000.0
    print(f"✅ Model weights loaded successfully in {t_load_ms:.2f} ms.\n", flush=True)

    if args.interactive:
        print("💡 Entering Interactive Generation Mode. Press Ctrl+C or type 'exit' to quit.\n", flush=True)
        while True:
            try:
                prompt_input = input("Enter prompt > ").strip()
                if prompt_input.lower() in ["exit", "quit"]:
                    break
                if not prompt_input:
                    prompt_input = "Once upon a time"
                generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    orig_to_new=orig_to_new,
                    new_to_orig=new_to_orig,
                    prompt=prompt_input,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    use_jit=not args.no_jit,
                    profile=args.profile,
                )
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive mode.")
                break
    else:
        generate_text(
            model=model,
            tokenizer=tokenizer,
            orig_to_new=orig_to_new,
            new_to_orig=new_to_orig,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            use_jit=not args.no_jit,
            profile=args.profile,
        )


if __name__ == "__main__":
    main()
