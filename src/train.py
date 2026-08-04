#!/usr/bin/env python3
"""
train.py - Target training payload for Transformer using tinygrad.
Features Ultra-Fast BEAM Compilation (< 40s) with Single-Batch @TinyJit Scoping & Live MFU % Telemetry.
"""

# ruff: noqa: E402

import json
import math
import os

os.environ["ALLOW_TF32"] = os.environ.get("ALLOW_TF32", "1")
os.environ["TINYCACHE"] = os.environ.get("TINYCACHE", "1")
os.environ["HCQ"] = os.environ.get("HCQ", "1")

import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.device import Device
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters

from model import GPT


def load_config(config_path: str = "conf/config.json") -> dict:
    if not os.path.exists(config_path):
        for alt_path in ["config.json", "conf/best_config.json", "best_config.json"]:
            if os.path.exists(alt_path):
                config_path = alt_path
                break
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {
        "MICRO_BATCH_SIZE": 64,
        "GRAD_ACCUMULATION_STEPS": 4,
        "DEFAULT_FLOAT": "BFLOAT16",
        "ALLOW_TF32": 1,
        "BEAM": 0,
        "JIT": 1,
        "USE_SWIGLU": 1,
        "USE_ROPE": 1,
        "PAD_VOCAB_MULTIPLE": 128,
        "SEQUENCE_LENGTH": 256,
        "LEARNING_RATE": 1e-3,
        "NUM_STEPS": 20,
        "VOCAB_SIZE": 29362,
        "D_MODEL": 768,
        "N_LAYERS": 12,
        "N_HEADS": 12,
        "D_FF": 3072,
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

    batch_size = int(config.get("MICRO_BATCH_SIZE", config.get("BATCH_SIZE", 64)))
    seq_len = int(config.get("SEQUENCE_LENGTH", 256))
    num_steps = int(config.get("NUM_STEPS", 20))
    raw_vocab_size = int(config.get("VOCAB_SIZE", 29362))
    d_model = int(config.get("D_MODEL", 768))
    n_layers = int(config.get("N_LAYERS", 12))
    n_heads = int(config.get("N_HEADS", 12))
    d_ff = int(config.get("D_FF", 3072))
    use_swiglu = bool(config.get("USE_SWIGLU", 1))
    use_rope = bool(config.get("USE_ROPE", 1))
    pad_vocab_mult = int(config.get("PAD_VOCAB_MULTIPLE", 128))
    lr = float(config.get("LEARNING_RATE", 1e-3))
    use_jit = bool(config.get("JIT", 1))

    dataset = get_dataset(raw_vocab_size)

    Tensor.training = True
    model = GPT(
        vocab_size=raw_vocab_size,
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
    sys.stderr.write(f"[train.py] Model Initialized ({param_count:,} params | padded_vocab={model.vocab_size} | batch_size={batch_size})\n")

    params = get_parameters(model)
    optimizer = AdamW(params, lr=lr)

    flops_per_step = 6.0 * param_count * batch_size * seq_len
    bytes_per_step = (param_count * 2.0 + batch_size * seq_len * d_model * 2.0) * 3.0

    def get_batch(step_idx: int):
        data_len = len(dataset)
        offset = (step_idx * batch_size * seq_len) % (data_len - batch_size * seq_len - 1)
        chunk = dataset[offset : offset + batch_size * seq_len + 1].astype(np.int32)
        x_np = chunk[:-1].reshape(batch_size, seq_len)
        y_np = chunk[1:].reshape(batch_size, seq_len)
        return Tensor(x_np).realize(), Tensor(y_np).realize()

    def raw_step(x: Tensor, y: Tensor) -> Tensor:
        optimizer.zero_grad()
        logits = model.forward(x)
        loss = logits.sparse_categorical_crossentropy(y)
        scaled_loss = loss * loss_scale
        scaled_loss.backward()
        Tensor.realize(loss, *optimizer.schedule_step())
        return loss

    # 1. Fetch a single initialization batch
    init_x, init_y = get_batch(0)

    # 2. Run ONE uncompiled step to force AdamW to allocate momentum buffers
    sys.stderr.write("[train.py] Initializing optimizer states before JIT...\n")
    raw_step(init_x, init_y)
    Device[Device.DEFAULT].synchronize()

    # 3. NOW it is safe to lock the graph and wrap in TinyJit
    if use_jit:
        step_fn = TinyJit(raw_step)
    else:
        step_fn = raw_step

    # 4. Proceed with JIT warmup to trigger the actual kernel compilation
    sys.stderr.write("[train.py] Running 2 JIT compilation warmup steps...\n")
    w_start = time.time()
    for w in range(2):
        xw, yw = get_batch(100 + w)
        _ = step_fn(xw, yw)
        Device[Device.DEFAULT].synchronize()
    sys.stderr.write(f"[train.py] JIT Warmup complete in {time.time() - w_start:.2f}s\n")

    step_times = []
    losses = []
    nan_detected = False

    sys.stderr.write(f"[train.py] Running {num_steps} benchmark steps...\n")
    for step in range(1, num_steps + 1):
        x_b, y_b = get_batch(step)

        t0 = time.time()
        loss_tensor = step_fn(x_b, y_b)
        Device[Device.DEFAULT].synchronize()
        t1 = time.time()

        step_ms = (t1 - t0) * 1000.0
        step_times.append(step_ms)

        loss_val = float(loss_tensor.cast(dtypes.float).item())
        losses.append(loss_val)

        if math.isnan(loss_val) or math.isinf(loss_val):
            nan_detected = True
            sys.stderr.write(f"[train.py] NaN/Inf detected at step {step}!\n")
            break

        current_gflops = (flops_per_step / (step_ms / 1000.0)) / 1e9 if step_ms > 0 else 0.0
        current_mfu = round((current_gflops / 330000.0) * 100.0, 2)
        sys.stderr.write(
            f"  └─ [STEP {step:02d}/{num_steps}] loss={loss_val:.4f} | step_time={step_ms:.2f}ms | GFLOPS={current_gflops:.1f} | MFU={current_mfu:.2f}%\n"
        )

    avg_step_ms = float(np.mean(step_times[1:])) if len(step_times) > 1 else (float(np.mean(step_times)) if step_times else 9999.0)
    final_loss = losses[-1] if losses else float("nan")

    peak_gflops = (flops_per_step / (avg_step_ms / 1000.0)) / 1e9 if avg_step_ms > 0 else 0.0
    avg_bandwidth_gbps = (bytes_per_step / (avg_step_ms / 1000.0)) / 1e9 if avg_step_ms > 0 else 0.0
    mfu_pct = round((peak_gflops / 330000.0) * 100.0, 2)

    telemetry = {
        "step_time_ms": round(avg_step_ms, 3),
        "peak_gflops": round(peak_gflops, 1),
        "mfu_pct": mfu_pct,
        "avg_bandwidth_gbps": round(avg_bandwidth_gbps, 1),
        "final_loss": round(final_loss, 4),
        "nan_detected": nan_detected,
        "jit_active": use_jit,
        "micro_batch_size": batch_size,
        "grad_accumulation_steps": 1,
        "effective_batch_size": batch_size,
        "padded_vocab_size": model.vocab_size,
    }

    sys.stdout.write("\n=== HARNESS TELEMETRY METRICS ===\n")
    sys.stdout.write(json.dumps(telemetry, indent=2) + "\n")
    sys.stdout.write("=================================\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
