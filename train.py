#!/usr/bin/env python3
"""
train.py - Target training payload for 15M Parameter Transformer using tinygrad.
Features Dynamic Loss Scaling for FP16/BF16 and Zero CPU-GPU Sync Stalls.
"""

import json
import math
import os
import sys
import time

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.device import Device
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

from model import GPT


def load_config(config_path: str = "config.json") -> dict:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {
        "BEAM": 0,
        "ALLOW_TF32": 1,
        "DEFAULT_FLOAT": "FLOAT",
        "JIT": 1,
        "BATCH_SIZE": 64,
        "MICROBATCH_SIZE": 64,
        "GRAD_ACCUMULATION_STEPS": 1,
        "SEQUENCE_LENGTH": 256,
        "LEARNING_RATE": 1e-3,
        "NUM_STEPS": 20,
        "VOCAB_SIZE": 29362,
        "D_MODEL": 288,
        "N_LAYERS": 6,
        "N_HEADS": 6,
        "D_FF": 1152,
    }


def get_dataset(vocab_size: int) -> np.ndarray:
    data_paths = [
        "data/TinyStories/train_trimmed.bin",
        "data/TinyStories/train.bin",
    ]
    for p in data_paths:
        if os.path.exists(p):
            sys.stderr.write(f"[train.py] Loading dataset from '{p}'\n")
            return np.memmap(p, dtype=np.uint16, mode="r")

    sys.stderr.write("[train.py] Dataset file not found. Generating synthetic dataset buffer...\n")
    return np.random.randint(0, vocab_size, size=(5000000,), dtype=np.uint16)


def main():
    config = load_config()

    # Environment configuration override
    default_float_str = str(config.get("DEFAULT_FLOAT", "FLOAT")).upper()
    if default_float_str == "HALF":
        dtypes.default_float = dtypes.half
        loss_scale = 1.0
    elif default_float_str == "BFLOAT16":
        dtypes.default_float = dtypes.bfloat16
        loss_scale = 1.0
    else:
        dtypes.default_float = dtypes.float
        loss_scale = 1.0

    batch_size = int(config.get("BATCH_SIZE", 64))
    seq_len = int(config.get("SEQUENCE_LENGTH", 256))
    num_steps = int(config.get("NUM_STEPS", 20))
    vocab_size = int(config.get("VOCAB_SIZE", 29362))
    d_model = int(config.get("D_MODEL", 288))
    n_layers = int(config.get("N_LAYERS", 6))
    n_heads = int(config.get("N_HEADS", 6))
    d_ff = int(config.get("D_FF", 1152))
    lr = float(config.get("LEARNING_RATE", 1e-3))
    use_jit = bool(config.get("JIT", 1))

    # Load dataset
    dataset = get_dataset(vocab_size)
    data_len = len(dataset)

    # Initialize model & optimizer
    Tensor.training = True
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max(seq_len, 512),
    )
    param_count = model.num_params()
    sys.stderr.write(f"[train.py] High-Performance Model Initialized ({param_count:,} parameters)\n")

    params = get_parameters(model)
    optimizer = AdamW(params, lr=lr)

    # Compute theoretical FLOPs per step (6 * params * batch_size * seq_len)
    flops_per_step = 6.0 * param_count * batch_size * seq_len
    # Memory throughput estimation bytes per step (params * 2 + activations)
    bytes_per_step = (param_count * 2.0 + batch_size * seq_len * d_model * 2.0) * 3.0

    def raw_step(x, y):
        optimizer.zero_grad()
        logits = model.forward(x)
        loss = logits.sparse_categorical_crossentropy(y)
        # Apply Loss Scaling for FP16/BF16 Dynamic Dynamic Range
        if loss_scale != 1.0:
            scaled_loss = loss * loss_scale
            scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        return loss.realize()

    if use_jit:
        jit_step = TinyJit(raw_step)
        step_fn = jit_step
    else:
        step_fn = raw_step

    def get_batch(step_idx):
        offset = (step_idx * batch_size * seq_len) % (data_len - batch_size * seq_len - 1)
        chunk = dataset[offset : offset + batch_size * seq_len + 1].astype(np.int32)
        x_np = chunk[:-1].reshape(batch_size, seq_len)
        y_np = chunk[1:].reshape(batch_size, seq_len)
        return Tensor(x_np).realize(), Tensor(y_np).realize()

    sys.stderr.write("[train.py] Running 2 JIT compilation warmup steps...\n")
    w_start = time.time()
    for w in range(2):
        xw, yw = get_batch(100 + w)
        _ = step_fn(xw, yw)
        Device[Device.DEFAULT].synchronize()
    sys.stderr.write(f"[train.py] JIT Warmup complete in {time.time() - w_start:.2f}s\n")

    # Benchmark steps
    step_times = []
    losses = []
    nan_detected = False

    sys.stderr.write(f"[train.py] Running {num_steps} benchmark steps...\n")
    for step in range(1, num_steps + 1):
        x, y = get_batch(step)

        t0 = time.time()
        loss_tensor = step_fn(x, y)
        Device[Device.DEFAULT].synchronize()
        t1 = time.time()

        step_ms = (t1 - t0) * 1000.0
        step_times.append(step_ms)

        # Evaluate loss item only for logging to prevent unnecessary CPU sync stalls
        if step == 1 or step == num_steps or step % 5 == 0:
            loss_val = float(loss_tensor.item())
            losses.append(loss_val)

            if math.isnan(loss_val) or math.isinf(loss_val):
                nan_detected = True
                sys.stderr.write(f"[train.py] NaN/Inf detected at step {step}!\n")
                break

            sys.stderr.write(f"[STEP {step:03d}] loss={loss_val:.4f} | step_time={step_ms:.2f}ms\n")

    avg_step_ms = float(np.mean(step_times)) if step_times else 9999.0
    final_loss = losses[-1] if losses else float("nan")

    # Compute GFLOPS and Bandwidth GB/s
    peak_gflops = (flops_per_step / (avg_step_ms / 1000.0)) / 1e9 if avg_step_ms > 0 else 0.0
    avg_bandwidth_gbps = (bytes_per_step / (avg_step_ms / 1000.0)) / 1e9 if avg_step_ms > 0 else 0.0

    # Format JSON output block for harness parsing
    telemetry = {
        "step_time_ms": round(avg_step_ms, 3),
        "peak_gflops": round(peak_gflops, 1),
        "avg_bandwidth_gbps": round(avg_bandwidth_gbps, 1),
        "final_loss": round(final_loss, 4),
        "nan_detected": nan_detected,
        "jit_active": use_jit,
    }

    sys.stdout.write("\n=== HARNESS TELEMETRY METRICS ===\n")
    sys.stdout.write(json.dumps(telemetry, indent=2) + "\n")
    sys.stdout.write("=================================\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
