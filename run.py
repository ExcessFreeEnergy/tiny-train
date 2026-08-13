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
import time

# Ensure sys.path includes src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import tiktoken
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes
from tinygrad.helpers import GlobalCounters, Profiling, Timing
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load

from src.chat_engine import apply_repetition_penalty
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
    """Scan checkpoint directory for fine-tuned or highest step count model matching model scale."""
    search_dirs = []
    if os.path.exists(checkpoint_dir):
        search_dirs.append(checkpoint_dir)

    finetuned_dir = os.path.join(os.path.dirname(__file__), "checkpoints_finetuned")
    if os.path.exists(finetuned_dir) and finetuned_dir not in search_dirs:
        search_dirs.append(finetuned_dir)

    if not search_dirs:
        return None

    # Prioritize fused fine-tuned checkpoint if present
    for sdir in search_dirs:
        fused_ckpt = os.path.join(sdir, f"model_{model_size.lower()}_finetuned.safetensors")
        if os.path.exists(fused_ckpt):
            return fused_ckpt

    pattern = re.compile(rf"model_{model_size.lower()}_step_(\d+)\.safetensors$")
    max_step = -1
    best_ckpt = None
    for sdir in search_dirs:
        for filename in os.listdir(sdir):
            match = pattern.match(filename)
            if match:
                s = int(match.group(1))
                if s > max_step:
                    max_step = s
                    best_ckpt = os.path.join(sdir, filename)
    return best_ckpt


def sample(logits: Tensor, temp: float = 0.8, top_k: int = 40) -> Tensor:
    """Sample next token index on device using temperature scaling and top-k / multinomial sampling."""
    assert logits.ndim == 1, "sample expects 1D logits tensor"
    if temp < 1e-6:
        return logits.argmax().cast(dtypes.int32)

    logits = logits.to(Device.DEFAULT)
    logits = (logits != logits).where(-float("inf"), logits)
    t = (logits / temp).softmax()

    if top_k > 0 and top_k < logits.numel():
        counter = Tensor.arange(t.numel(), device=t.device).contiguous()
        counter2 = Tensor.arange(t.numel() - 1, -1, -1, device=t.device).contiguous()
        output = Tensor.zeros(top_k, device=t.device).contiguous()
        output_indices = Tensor.zeros(top_k, device=t.device, dtype=dtypes.int32).contiguous()
        for i in range(top_k):
            t_argmax = (t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1).cast(dtypes.default_int)
            output = output + t_max.unsqueeze(0).pad(((i, top_k - i - 1),))
            output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, top_k - i - 1),))
            t = (counter == t_argmax).where(0, t)
        output_idx = output.multinomial()
        output_token = output_indices[output_idx]
    else:
        output_token = t.multinomial().cast(dtypes.int32)

    return output_token


class GPTEngine:
    """TinyGrad Inference Engine encapsulating JIT execution graph & KV caching."""

    def __init__(self, model: GPT, max_context: int = 1024, use_jit: bool = True):
        self.model = model
        self.max_context = max_context
        self.raw_vocab_size = model.raw_vocab_size
        self.step_jit = TinyJit(self.step) if use_jit else None

    def step(
        self,
        tokens: Tensor,
        start_pos: Variable | int,
        temperature: float,
        top_k: int,
        repetition_penalty: float = 1.15,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        logits = self.model.forward(tokens, start_pos=start_pos)
        last_logits = logits[0, -1, : self.raw_vocab_size]
        if repetition_penalty != 1.0 and context_mask is not None:
            last_logits = apply_repetition_penalty(last_logits, penalty=repetition_penalty, context_mask=context_mask)
        return sample(last_logits, temp=temperature, top_k=top_k)

    def __call__(
        self,
        tokens: Tensor,
        start_pos: int,
        temperature: float = 0.8,
        top_k: int = 40,
        repetition_penalty: float = 1.15,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        if context_mask is None:
            context_mask = Tensor.zeros(self.raw_vocab_size, dtype=dtypes.bool, device=Device.DEFAULT).clone().realize()
        if tokens.shape == (1, 1) and self.step_jit is not None and start_pos != 0:
            v_start_pos = Variable("start_pos", 1, self.max_context - 1).bind(start_pos)
            return self.step_jit(tokens, v_start_pos, temperature, top_k, repetition_penalty, context_mask)
        return self.step(tokens, start_pos, temperature, top_k, repetition_penalty, context_mask)


def generate_text(
    engine: GPTEngine,
    tokenizer,
    orig_to_new: dict | None,
    new_to_orig: list | None,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.15,
    profile: bool = False,
    timing: bool = False,
) -> str:
    """Autoregressively generate text given a starting prompt using KV-caching and @TinyJit acceleration."""
    engine.model.reset_cache()

    orig_prompt_ids = tokenizer.encode(prompt)
    if not orig_prompt_ids:
        orig_prompt_ids = [50256]  # Fallback to EOS

    if orig_to_new:
        trimmed_prompt_ids = [orig_to_new.get(tid, 0) for tid in orig_prompt_ids]
    else:
        trimmed_prompt_ids = list(orig_prompt_ids)

    current_ids = list(trimmed_prompt_ids)
    prompt_len = len(trimmed_prompt_ids)

    mask_np = np.zeros(engine.raw_vocab_size, dtype=bool)
    valid_ids = [tid for tid in trimmed_prompt_ids if 0 <= tid < engine.raw_vocab_size]
    if valid_ids:
        mask_np[valid_ids] = True
    context_mask = Tensor(mask_np, device=Device.DEFAULT).clone().realize()

    print(f'\n📝 Prompt: "{prompt}"', flush=True)
    print("-------------------------------------------------------", flush=True)
    sys.stdout.write(prompt)
    sys.stdout.flush()

    param_bytes = sum(p.nbytes() for p in get_parameters(engine.model))

    with Profiling(enabled=profile):
        # 1. Prompt Phase (Fill initial KV cache for prompt tokens)
        t0 = time.perf_counter()
        x_prompt = Tensor([trimmed_prompt_ids], dtype=dtypes.int32).realize()
        tok_tensor = engine(
            x_prompt,
            start_pos=0,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            context_mask=context_mask,
        ).realize()
        next_trimmed_id = tok_tensor.item()
        current_ids.append(next_trimmed_id)
        if 0 <= next_trimmed_id < engine.raw_vocab_size:
            mask_np[next_trimmed_id] = True
            context_mask = Tensor(mask_np, device=Device.DEFAULT).clone().realize()
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        # Stream first generated token
        orig_id = new_to_orig[next_trimmed_id] if new_to_orig and next_trimmed_id < len(new_to_orig) else next_trimmed_id
        sys.stdout.write(tokenizer.decode([orig_id]))
        sys.stdout.flush()

        # 2. Autoregressive single-token generation phase
        start_pos = prompt_len
        token_times_ms = [ttft_ms]

        t_gen_start = time.perf_counter()
        for _ in range(1, max_new_tokens):
            GlobalCounters.reset()
            t_step_0 = time.perf_counter()

            with Timing(
                "total ",
                enabled=timing,
                on_exit=lambda x: f", {1e9 / x:.2f} tok/s, {GlobalCounters.global_mem / x:.2f} GB/s, param {param_bytes / x:.2f} GB/s",
            ):
                x_in = tok_tensor.reshape(1, 1)
                tok_tensor = engine(
                    x_in,
                    start_pos=start_pos,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    context_mask=context_mask,
                ).realize()
                next_trimmed_id = tok_tensor.item()

            current_ids.append(next_trimmed_id)
            if 0 <= next_trimmed_id < engine.raw_vocab_size:
                mask_np[next_trimmed_id] = True
                context_mask = Tensor(mask_np, device=Device.DEFAULT).clone().realize()

            start_pos += 1
            step_ms = (time.perf_counter() - t_step_0) * 1000.0
            token_times_ms.append(step_ms)

            orig_id = new_to_orig[next_trimmed_id] if new_to_orig and next_trimmed_id < len(new_to_orig) else next_trimmed_id
            sys.stdout.write(tokenizer.decode([orig_id]))
            sys.stdout.flush()

    sys.stdout.write("\n-------------------------------------------------------\n")
    sys.stdout.flush()

    total_gen_sec = time.perf_counter() - t_gen_start
    gen_tok_per_sec = (max_new_tokens - 1) / total_gen_sec if total_gen_sec > 0 else 0.0
    avg_per_token_ms = float(sum(token_times_ms[1:]) / len(token_times_ms[1:])) if len(token_times_ms) > 1 else ttft_ms

    print("\n=======================================================", flush=True)
    print("⚡ INFERENCE TELEMETRY METRICS", flush=True)
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
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty scalar (default 1.15)")
    parser.add_argument("--no-jit", action="store_true", default=False, help="Disable @TinyJit compilation")
    parser.add_argument("--timing", action="store_true", default=False, help="Print per-token timing and memory bandwidth")
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
    print(f"JIT Acceleration: {not args.no_jit} | Profiling: {args.profile} | Timing: {args.timing}", flush=True)
    print("=======================================================\n", flush=True)

    # Initialize Tokenizer & Vocab Map
    tokenizer = tiktoken.get_encoding("gpt2")
    orig_to_new, new_to_orig = load_vocab_map(dataset_name=args.dataset)

    # Load parameters from safetensors checkpoint first to infer exact wte shape
    print(f"📦 Loading weights from '{ckpt_path}'...", flush=True)
    t_load_start = time.perf_counter()
    state = safe_load(ckpt_path)

    wte_shape = state["wte"].shape
    ckpt_padded_vocab_size = wte_shape[0]

    # Auto-detect dataset if checkpoint vocab size > 30000 (FineWeb)
    if ckpt_padded_vocab_size > 30000 and args.dataset.lower() == "tinystories":
        args.dataset = "fineweb"
        orig_to_new, new_to_orig = load_vocab_map(dataset_name=args.dataset)

    # Model Preset Architecture Parameters
    if args.model_size == "125M":
        d_model = 768
        n_layers = 12
        n_heads = 12
        d_ff = 3072
    else:
        d_model = 288
        n_layers = 6
        n_heads = 6
        d_ff = 1152

    if new_to_orig:
        raw_vocab_size = len(new_to_orig)
    else:
        raw_vocab_size = 13970 if args.dataset != "fineweb" else 49685

    # Match vocab padding method based on checkpoint wte shape
    is_power_of_2 = (ckpt_padded_vocab_size & (ckpt_padded_vocab_size - 1) == 0) and (ckpt_padded_vocab_size >= raw_vocab_size)

    Tensor.training = False
    model = GPT(
        vocab_size=raw_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=1024,
        use_swiglu=True,
        use_rope=True,
        pad_vocab_multiple=128,
        pad_vocab_power_of_2=is_power_of_2,
    )

    if "freqs_cis" in state:
        state.pop("freqs_cis")

    load_state_dict(model, state, strict=False)
    Tensor.realize(*get_parameters(model))
    t_load_ms = (time.perf_counter() - t_load_start) * 1000.0
    print(f"✅ Model weights loaded successfully in {t_load_ms:.2f} ms.\n", flush=True)

    engine = GPTEngine(model, max_context=1024, use_jit=not args.no_jit)

    if args.interactive:
        print("💡 Launching Textual Interactive TUI Chat Application...\n", flush=True)
        from chat import TinyChatApp

        app = TinyChatApp(
            dataset=args.dataset,
            model_size=args.model_size,
            checkpoint_path=ckpt_path,
            checkpoint_dir=args.checkpoint_dir,
            use_jit=not args.no_jit,
        )
        app.run()
    else:
        generate_text(
            engine=engine,
            tokenizer=tokenizer,
            orig_to_new=orig_to_new,
            new_to_orig=new_to_orig,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            profile=args.profile,
            timing=args.timing,
        )


if __name__ == "__main__":
    main()
