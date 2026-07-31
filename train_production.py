#!/usr/bin/env python3
"""
train_production.py - Production Training Engine for 15M & 125M Parameter Transformer Models.
Features Single Micro-Batch @TinyJit Scoping, Python Gradient Accumulation, and live MFU % telemetry.
"""

import argparse
import json
import math
import os
import time

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.device import Device
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save

from model import GPT


def load_best_config(config_path: str = "best_config.json") -> dict:
    if not os.path.exists(config_path):
        config_path = "config.json"
    with open(config_path) as f:
        return json.load(f)


def get_lr_schedule(it: int, max_iters: int, warmup_iters: int, max_lr: float, min_lr: float) -> float:
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    parser = argparse.ArgumentParser(description="Main Production Transformer Trainer")
    parser.add_argument("--model-size", choices=["15M", "125M"], default="125M", help="Target model parameter scale")
    parser.add_argument("--total-steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--eval-interval", type=int, default=50, help="Steps between validation evaluations")
    args = parser.parse_args()

    config = load_best_config()

    default_float_str = str(config.get("DEFAULT_FLOAT", "BFLOAT16")).upper()
    if default_float_str == "HALF":
        dtypes.default_float = dtypes.half
        loss_scale = 1.0
    elif default_float_str == "BFLOAT16":
        dtypes.default_float = dtypes.bfloat16
        loss_scale = 1.0
    else:
        dtypes.default_float = dtypes.float
        loss_scale = 1.0

    micro_batch_size = int(config.get("MICRO_BATCH_SIZE", 16 if args.model_size == "125M" else 64))
    grad_accum_steps = int(config.get("GRAD_ACCUMULATION_STEPS", 16 if args.model_size == "125M" else 4))
    effective_batch_size = micro_batch_size * grad_accum_steps
    seq_len = int(config.get("SEQUENCE_LENGTH", 256))
    max_lr = float(config.get("LEARNING_RATE", 1e-3))
    min_lr = max_lr * 0.1
    warmup_iters = max(10, int(args.total_steps * 0.05))

    # Model Presets
    if args.model_size == "125M":
        vocab_size = int(config.get("VOCAB_SIZE", 29362))
        d_model = 768
        n_layers = 12
        n_heads = 12
        d_ff = 3072
    else:
        vocab_size = int(config.get("VOCAB_SIZE", 29362))
        d_model = int(config.get("D_MODEL", 288))
        n_layers = int(config.get("N_LAYERS", 6))
        n_heads = int(config.get("N_HEADS", 6))
        d_ff = int(config.get("D_FF", 1152))

    use_swiglu = bool(config.get("USE_SWIGLU", 1))
    use_rope = bool(config.get("USE_ROPE", 1))
    pad_vocab_mult = int(config.get("PAD_VOCAB_MULTIPLE", 128))
    use_jit = bool(config.get("JIT", 1))

    train_bin = "data/TinyStories/train_trimmed.bin"
    valid_bin = "data/TinyStories/valid_trimmed.bin"
    if not os.path.exists(train_bin):
        train_bin = "data/TinyStories/train.bin"

    if os.path.exists(train_bin):
        print(f"📦 Loading training dataset: '{train_bin}'")
        train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    else:
        print("⚠️ Training binary dataset not found. Using synthetic random buffer...")
        train_data = np.random.randint(0, vocab_size, size=(10000000,), dtype=np.uint16)

    if os.path.exists(valid_bin):
        valid_data = np.memmap(valid_bin, dtype=np.uint16, mode="r")
    else:
        valid_data = train_data

    Tensor.training = True
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max(seq_len, 512),
        use_swiglu=use_swiglu,
        use_rope=use_rope,
        pad_vocab_multiple=pad_vocab_mult,
    )
    param_count = model.num_params()
    print("\n=======================================================")
    print(f"🚀 MAIN PRODUCTION TRAINER INITIALIZED ({args.model_size})")
    print(f"Parameters: {param_count:,} | Padded Vocab: {model.vocab_size}")
    print(f"Micro-Batch: {micro_batch_size} | Grad Accum: {grad_accum_steps} | Eff Batch: {effective_batch_size}")
    print(f"Precision: {default_float_str} | RoPE: {use_rope} | SwiGLU: {use_swiglu}")
    print("=======================================================\n")

    params = get_parameters(model)
    optimizer = AdamW(params, lr=max_lr)

    flops_per_step = 6.0 * param_count * effective_batch_size * seq_len

    # Single Micro-Batch @TinyJit Function returning Tensors
    def raw_micro_step(x: Tensor, y: Tensor) -> tuple:
        logits = model.forward(x)
        loss = logits.sparse_categorical_crossentropy(y)
        scaled_loss = (loss / grad_accum_steps) * loss_scale
        scaled_loss.backward()
        grads = [p.grad.realize() if p.grad is not None else Tensor.zeros(*p.shape).realize() for p in params]
        return (loss.realize(), *grads)

    if use_jit:
        micro_step_fn = TinyJit(raw_micro_step)
    else:
        micro_step_fn = raw_micro_step

    def get_micro_batch(data_source: np.ndarray, step_idx: int, accum_idx: int):
        d_len = len(data_source)
        offset = ((step_idx * grad_accum_steps + accum_idx) * micro_batch_size * seq_len) % (d_len - micro_batch_size * seq_len - 1)
        chunk = data_source[offset : offset + micro_batch_size * seq_len + 1].astype(np.int32)
        x_np = chunk[:-1].reshape(micro_batch_size, seq_len)
        y_np = chunk[1:].reshape(micro_batch_size, seq_len)
        return Tensor(x_np).realize(), Tensor(y_np).realize()

    print("⏳ Warming up single micro-batch @TinyJit compilation graph...")
    w_start = time.time()
    for w in range(2):
        xw, yw = get_micro_batch(train_data, 100, w)
        _ = micro_step_fn(xw, yw)
        Device[Device.DEFAULT].synchronize()
    print(f"✅ Single micro-batch @TinyJit Warmup complete in {time.time() - w_start:.2f}s\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    step_times = []

    # Production Training Loop
    for step in range(1, args.total_steps + 1):
        cur_lr = get_lr_schedule(step, args.total_steps, warmup_iters, max_lr, min_lr)
        optimizer.lr = cur_lr

        t0 = time.time()
        optimizer.zero_grad()
        accum_losses = []

        for acc in range(grad_accum_steps):
            x_m, y_m = get_micro_batch(train_data, step, acc)
            res = micro_step_fn(x_m, y_m)
            m_loss = res[0]
            m_grads = res[1:]

            if step == 1 or step == args.total_steps or step % 10 == 0:
                accum_losses.append(float(m_loss.item()))

            for p, g in zip(params, m_grads):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad = (p.grad + g).realize()

        optimizer.step()
        Device[Device.DEFAULT].synchronize()
        t1 = time.time()

        step_ms = (t1 - t0) * 1000.0
        step_times.append(step_ms)

        if step == 1 or step == args.total_steps or step % 10 == 0:
            loss_val = float(np.mean(accum_losses)) if accum_losses else 0.0
            throughput = (effective_batch_size / (step_ms / 1000.0)) if step_ms > 0 else 0.0
            gflops = (flops_per_step / (step_ms / 1000.0)) / 1e9 if step_ms > 0 else 0.0
            mfu_pct = (gflops / 330000.0) * 100.0

            print(
                f"[STEP {step:04d}/{args.total_steps}] loss={loss_val:.4f} | lr={cur_lr:.2e} | "
                f"time={step_ms:.2f}ms | tput={throughput:.1f} smp/s | GFLOPS={gflops:.1f} | MFU={mfu_pct:.2f}%"
            )

        # Checkpointing & Validation Eval
        if step % args.eval_interval == 0 or step == args.total_steps:
            x_v, y_v = get_micro_batch(valid_data, step + 999, 0)
            val_res = micro_step_fn(x_v, y_v)
            val_loss = float(val_res[0].item())
            print(f"📊 Validation Loss at step {step}: {val_loss:.4f}")

            ckpt_path = os.path.join(args.checkpoint_dir, f"model_{args.model_size.lower()}_step_{step}.safetensors")
            state_dict = get_state_dict(model)
            safe_save(state_dict, ckpt_path)
            print(f"💾 Checkpoint saved to '{ckpt_path}'")

    avg_step_ms = float(np.mean(step_times[5:])) if len(step_times) > 5 else float(np.mean(step_times))
    avg_tput = effective_batch_size / (avg_step_ms / 1000.0)
    avg_gflops = (flops_per_step / (avg_step_ms / 1000.0)) / 1e9
    avg_mfu = (avg_gflops / 330000.0) * 100.0

    print("\n=======================================================")
    print(f"🏆 {args.model_size} PRODUCTION TRAINING COMPLETE!")
    print("=======================================================")
    print(f"Average Step Time: {avg_step_ms:.2f} ms")
    print(f"Average Throughput: {avg_tput:.1f} samples/sec")
    print(f"Average Compute: {avg_gflops:.1f} GFLOPS ({avg_gflops / 1000.0:.2f} TFLOPS)")
    print(f"Average MFU: {avg_mfu:.2f}% (Target: ~60%)")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
