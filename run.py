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

import numpy as np
import tiktoken
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load

from src.model import GPT


def load_vocab_map(vocab_map_path: str = "data/TinyStories/vocab_map.json"):
    """Load vocabulary map for trimming/restoring original GPT-2 token IDs."""
    if not os.path.exists(vocab_map_path):
        vocab_map_path = os.path.join(os.path.dirname(__file__), "data/TinyStories/vocab_map.json")
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
) -> str:
    """Autoregressively generate text given a starting prompt."""
    orig_prompt_ids = tokenizer.encode(prompt)
    if not orig_prompt_ids:
        orig_prompt_ids = [50256]  # Fallback to EOS

    if orig_to_new:
        trimmed_prompt_ids = [orig_to_new.get(tid, 0) for tid in orig_prompt_ids]
    else:
        trimmed_prompt_ids = list(orig_prompt_ids)

    current_ids = list(trimmed_prompt_ids)
    raw_vocab_size = model.raw_vocab_size

    print(f'\n📝 Prompt: "{prompt}"', flush=True)
    print("-------------------------------------------------------", flush=True)
    sys.stdout.write(prompt)
    sys.stdout.flush()

    for _ in range(max_new_tokens):
        # Truncate context to max seq_len
        ctx_ids = current_ids[-seq_len:]
        ctx_tensor = Tensor([ctx_ids], dtype=dtypes.int32)

        # Forward pass
        logits = model.forward(ctx_tensor)
        logits_np = logits[0, -1, :raw_vocab_size].cast(dtypes.float32).numpy()

        next_trimmed_id = sample_next_token(logits_np, temperature=temperature, top_k=top_k)
        current_ids.append(next_trimmed_id)

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

    # Full decode
    if new_to_orig:
        full_orig_ids = [new_to_orig[tid] for tid in current_ids if tid < len(new_to_orig)]
    else:
        full_orig_ids = current_ids

    return tokenizer.decode(full_orig_ids)


def main():
    parser = argparse.ArgumentParser(description="TinyGrad Transformer Inference & Generation Engine")
    parser.add_argument("--model-size", choices=["15M", "125M"], default="125M", help="Target model scale")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to .safetensors checkpoint file")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory containing model checkpoints")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Initial generation prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering threshold")
    parser.add_argument("--interactive", action="store_true", default=False, help="Run interactive prompt loop")
    args = parser.parse_args()

    # Locate checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint(args.checkpoint_dir, args.model_size)

    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No valid checkpoint found for model scale '{args.model_size}' in directory '{args.checkpoint_dir}'.")

    print("\n=======================================================", flush=True)
    print(f"🚀 TINYGRAD INFERENCE ENGINE ({args.model_size})", flush=True)
    print(f"Checkpoint: {ckpt_path}", flush=True)
    print("=======================================================\n", flush=True)

    # Initialize Tokenizer & Vocab Map
    tokenizer = tiktoken.get_encoding("gpt2")
    orig_to_new, new_to_orig = load_vocab_map()

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
    state = safe_load(ckpt_path)
    load_state_dict(model, state)
    Tensor.realize(*get_parameters(model))
    print("✅ Model weights loaded successfully.\n", flush=True)

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
        )


if __name__ == "__main__":
    main()
