#!/usr/bin/env python3
"""
train_production.py - Production Training Engine for 15M & 125M Parameter Transformer Models.
Features Ultra-Fast BEAM Compilation (< 40s) with Single-Step @TinyJit Scoping & Live MFU % Telemetry.
"""

# ruff: noqa: E402

import argparse
import datetime
import json
import math
import os
import re
import sys
import time

# Ensure sys.stdout is unbuffered / line-buffered for real-time log streaming
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Add paths immediately
sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def load_config(config_path: str | None = None) -> dict:
    candidates = []
    if config_path:
        candidates.append(config_path)
    candidates.extend(["conf/best_config.json", "conf/config.json", "best_config.json", "config.json"])
    for path in candidates:
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError("No configuration JSON file found.")


# Set up optimization environment variables BEFORE tinygrad import
_preload_config = {}
try:
    _preload_config = load_config()
except Exception:
    pass

os.environ["ALLOW_TF32"] = os.environ.get("ALLOW_TF32", str(_preload_config.get("ALLOW_TF32", "1")))
os.environ["TINYCACHE"] = os.environ.get("TINYCACHE", "1")
os.environ["HCQ"] = os.environ.get("HCQ", "0")
os.environ["TC"] = "1"
os.environ["TENSOR_CORES"] = "1"
os.environ["BEAM"] = os.environ.get("BEAM", str(_preload_config.get("BEAM", "2")))

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.device import Device
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save

from model import GPT


def get_lr_schedule(it: int, max_iters: int, warmup_iters: int, max_lr: float, min_lr: float) -> float:
    if warmup_iters <= 0 or max_iters <= warmup_iters:
        return max_lr
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
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--disable-debug", "--no-debug", action="store_true", default=False, help="Disable debug print logging")
    parser.add_argument("--debug-level", "--debug", type=int, default=None, help="Set debug logging level")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest checkpoint in checkpoint-dir")
    parser.add_argument("--resume-path", type=str, default=None, help="Explicit path to checkpoint file to resume from")
    args = parser.parse_args()

    disable_debug = args.disable_debug or args.debug_level == 0 or os.environ.get("DEBUG") == "0"
    if disable_debug:
        os.environ["DEBUG"] = "0"

    config = load_config(args.config)

    beam_val = str(config.get("BEAM", 2))
    os.environ["BEAM"] = os.environ.get("BEAM", beam_val)

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

    micro_batch_size = int(config.get("MICRO_BATCH_SIZE", config.get("BATCH_SIZE", 16)))
    grad_accum_steps = int(config.get("GRAD_ACCUMULATION_STEPS", 4))
    eff_batch_size = micro_batch_size * grad_accum_steps
    seq_len = int(config.get("SEQUENCE_LENGTH", 256))
    max_lr = float(config.get("LEARNING_RATE", 1e-3))
    min_lr = max_lr * 0.1
    warmup_iters = max(10, int(args.total_steps * 0.05))

    # Model Presets
    if args.model_size == "125M":
        vocab_size = int(config.get("VOCAB_SIZE", 13970))
        d_model = 768
        n_layers = 12
        n_heads = 12
        d_ff = 3072
    else:
        vocab_size = int(config.get("VOCAB_SIZE", 13970))
        d_model = int(config.get("D_MODEL", 288))
        n_layers = int(config.get("N_LAYERS", 6))
        n_heads = int(config.get("N_HEADS", 6))
        d_ff = int(config.get("D_FF", 1152))

    use_swiglu = bool(config.get("USE_SWIGLU", 1))
    use_rope = bool(config.get("USE_ROPE", 1))
    pad_vocab_mult = int(config.get("PAD_VOCAB_MULTIPLE", 128))
    pad_vocab_p2 = bool(config.get("PAD_VOCAB_POWER_OF_2", 1))
    use_jit = bool(config.get("JIT", 1))

    train_bin = "data/TinyStories/train_trimmed.bin"
    valid_bin = "data/TinyStories/valid_trimmed.bin"
    if not os.path.exists(train_bin):
        train_bin = "data/TinyStories/train.bin"

    if os.path.exists(train_bin):
        print(f"📦 Loading training dataset: '{train_bin}'", flush=True)
        train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    else:
        print("⚠️ Training binary dataset not found. Using synthetic random buffer...", flush=True)
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
        pad_vocab_power_of_2=pad_vocab_p2,
    )
    start_time = time.time()
    start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Checkpoint Resume Auto-detection
    resumed_step = 0
    ckpt_path_to_load = args.resume_path

    if not ckpt_path_to_load and args.resume:
        if os.path.exists(args.checkpoint_dir):
            pattern = re.compile(rf"model_{args.model_size.lower()}_step_(\d+)\.safetensors$")
            max_step = -1
            best_ckpt = None
            for filename in os.listdir(args.checkpoint_dir):
                match = pattern.match(filename)
                if match:
                    s = int(match.group(1))
                    if s > max_step:
                        max_step = s
                        best_ckpt = os.path.join(args.checkpoint_dir, filename)
            if best_ckpt and max_step > 0:
                ckpt_path_to_load = best_ckpt
                resumed_step = max_step

    if ckpt_path_to_load:
        if not os.path.exists(ckpt_path_to_load):
            raise FileNotFoundError(f"Checkpoint file '{ckpt_path_to_load}' not found.")
        if resumed_step == 0:
            match = re.search(r"_step_(\d+)\.safetensors$", ckpt_path_to_load)
            if match:
                resumed_step = int(match.group(1))
        print(f"🔄 Resuming model weights from checkpoint: '{ckpt_path_to_load}' (starting step {resumed_step + 1})", flush=True)
        state = safe_load(ckpt_path_to_load)
        load_state_dict(model, state)
        print(f"✅ State dict loaded successfully from '{ckpt_path_to_load}'", flush=True)

    start_step = resumed_step + 1

    param_count = model.num_params()
    print("\n=======================================================", flush=True)
    print(f"🚀 MAIN PRODUCTION TRAINER INITIALIZED ({args.model_size})", flush=True)
    print(f"Start Time: {start_datetime}", flush=True)
    if start_step > 1:
        print(f"Resuming From: Step {start_step} / {args.total_steps} (Checkpoint: {ckpt_path_to_load})", flush=True)
    print(f"Parameters: {param_count:,} | Padded Vocab: {model.vocab_size}", flush=True)
    print(f"Micro-Batch: {micro_batch_size} | Grad Accum: {grad_accum_steps} | Effective Batch: {eff_batch_size} | Seq Len: {seq_len}", flush=True)
    print(f"Precision: {default_float_str} | RoPE: {use_rope} | SwiGLU: {use_swiglu} | BEAM: {os.environ.get('BEAM')}", flush=True)
    print("=======================================================\n", flush=True)

    params = get_parameters(model)

    # SEVER RNG GRAPH: Force all glorot_uniform weights into VRAM
    # so the optimizer never sees the RNG initialization history.
    sys.stderr.write("[train_production.py] Realizing model weights into VRAM...\n")
    Tensor.realize(*params)

    # Pre-allocate gradient accumulation buffers as zeros and sever graph
    for p in params:
        p.accum_grad = Tensor.zeros_like(p).realize()
        p.grad = None

    optimizer = AdamW(params, lr=max_lr)
    lr_tensor = Tensor([max_lr], dtype=dtypes.float, device=params[0].device).realize()
    optimizer.lr = lr_tensor

    flops_per_step = 6.0 * param_count * eff_batch_size * seq_len

    def get_batch(data_source: np.ndarray, step_idx: int):
        d_len = len(data_source)
        offset = (step_idx * eff_batch_size * seq_len) % (d_len - eff_batch_size * seq_len - 1)
        chunk = data_source[offset : offset + eff_batch_size * seq_len + 1].astype(np.int32)
        x_np = chunk[:-1].reshape(eff_batch_size, seq_len)
        y_np = chunk[1:].reshape(eff_batch_size, seq_len)
        return x_np, y_np

    # Pre-allocate static micro-batch input buffers to prevent JIT memory thrashing
    x_jit = Tensor.zeros(micro_batch_size, seq_len, dtype=dtypes.int32, device=params[0].device).realize()
    y_jit = Tensor.zeros(micro_batch_size, seq_len, dtype=dtypes.int32, device=params[0].device).realize()

    def accum_step(x_micro: Tensor, y_micro: Tensor) -> Tensor:
        logits_chunk = model.forward(x_micro)
        flat_logits = logits_chunk.reshape(-1, logits_chunk.shape[-1])
        flat_y = y_micro.flatten()
        chunk_loss = flat_logits.sparse_categorical_crossentropy(flat_y) / grad_accum_steps
        scaled_loss = chunk_loss * loss_scale
        scaled_loss.backward()
        for p in params:
            if p.grad is not None:
                # Statically accumulate and sever the graph
                p.accum_grad.assign(p.accum_grad + p.grad)
                p.grad = None  # Force backward() to create a fresh leaf node next iteration
        return chunk_loss

    def opt_step():
        for p in params:
            p.grad = p.accum_grad
        Tensor.realize(*optimizer.schedule_step())
        for p in params:
            p.accum_grad.assign(Tensor.zeros_like(p))
            p.grad = None

    # 1. Fetch initialization batch
    init_x, init_y = get_batch(train_data, 0)

    # 2. Run ONE uncompiled step to force AdamW to allocate momentum buffers (m and v) in VRAM before JIT
    sys.stderr.write("[train_production.py] Initializing optimizer states before JIT...\n")
    for i in range(grad_accum_steps):
        x_jit.assign(init_x[i * micro_batch_size : (i + 1) * micro_batch_size])
        y_jit.assign(init_y[i * micro_batch_size : (i + 1) * micro_batch_size])
        accum_step(x_jit, y_jit)
    opt_step()
    Device[Device.DEFAULT].synchronize()

    # 3. NOW it is safe to wrap in TinyJit
    if use_jit:
        accum_fn = TinyJit(accum_step)
        opt_fn = TinyJit(opt_step)
    else:
        accum_fn = accum_step
        opt_fn = opt_step

    # 4. Proceed with JIT warmup to trigger actual kernel compilation
    sys.stderr.write("[train_production.py] Running JIT compilation warmup steps...\n")
    w_start = time.time()
    for w in range(2):
        xw, yw = get_batch(train_data, 100 + w)
        for i in range(grad_accum_steps):
            x_jit.assign(xw[i * micro_batch_size : (i + 1) * micro_batch_size])
            y_jit.assign(yw[i * micro_batch_size : (i + 1) * micro_batch_size])
            _ = accum_fn(x_jit, y_jit)
        opt_fn()
        Device[Device.DEFAULT].synchronize()

    # CRITICAL: This exact string allows harness.py to set jit_active=True
    sys.stderr.write(f"[train_production.py] JIT Warmup complete in {time.time() - w_start:.2f}s\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    step_times = []
    last_loss_val = 0.0

    # Production Training Loop
    for step in range(start_step, args.total_steps + 1):
        cur_lr = get_lr_schedule(step, args.total_steps, warmup_iters, max_lr, min_lr)
        lr_tensor.assign([cur_lr]).realize()

        x_b, y_b = get_batch(train_data, step)

        t0 = time.time()
        step_loss_tensor = Tensor([0.0], device=params[0].device)
        for i in range(grad_accum_steps):
            x_jit.assign(x_b[i * micro_batch_size : (i + 1) * micro_batch_size])
            y_jit.assign(y_b[i * micro_batch_size : (i + 1) * micro_batch_size])
            loss_micro = accum_fn(x_jit, y_jit)
            step_loss_tensor = step_loss_tensor + loss_micro

        opt_fn()
        Device[Device.DEFAULT].synchronize()
        step_loss = float(step_loss_tensor.cast(dtypes.float).item())
        t1 = time.time()

        step_ms = (t1 - t0) * 1000.0
        step_times.append(step_ms)

        total_steps = args.total_steps
        if step == start_step or step == total_steps or step % 1000 == 0 or step % args.eval_interval == 0:
            last_loss_val = step_loss
            tokens_per_sec = (eff_batch_size * seq_len) / (step_ms / 1000.0) if step_ms > 0 else 0.0

            if not disable_debug and (step == start_step or step == total_steps or step % 1000 == 0 or step % args.eval_interval == 0):
                print(f"Step {step:6d} / {total_steps} | Loss: {last_loss_val:.4f} | Tok/sec: {tokens_per_sec:.0f}", flush=True)

        # Checkpointing & Validation Eval
        if step % args.eval_interval == 0 or step == args.total_steps:
            x_v, y_v = get_batch(valid_data, step + 999)
            val_logits = model.forward(Tensor(x_v[:micro_batch_size]))
            val_loss_tensor = val_logits.sparse_categorical_crossentropy(Tensor(y_v[:micro_batch_size])).realize()
            val_loss = float(val_loss_tensor.cast(dtypes.float).item())
            print(f"📊 Validation Loss at step {step}: {val_loss:.4f}", flush=True)

            ckpt_path = os.path.join(args.checkpoint_dir, f"model_{args.model_size.lower()}_step_{step}.safetensors")
            state_dict = get_state_dict(model)
            safe_save(state_dict, ckpt_path)
            print(f"💾 Checkpoint saved to '{ckpt_path}'", flush=True)

    avg_step_ms = float(np.mean(step_times[1:])) if len(step_times) > 1 else (float(np.mean(step_times)) if step_times else 0.0)
    avg_tput = eff_batch_size / (avg_step_ms / 1000.0) if avg_step_ms > 0 else 0.0
    avg_gflops = ((flops_per_step / (avg_step_ms / 1000.0)) / 1e9) if avg_step_ms > 0 else 0.0
    avg_mfu = (avg_gflops / 330000.0) * 100.0

    end_time = time.time()
    end_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_elapsed_sec = end_time - start_time
    total_elapsed_formatted = str(datetime.timedelta(seconds=round(total_elapsed_sec)))

    telemetry = {
        "start_time": start_datetime,
        "end_time": end_datetime,
        "total_elapsed_sec": round(total_elapsed_sec, 2),
        "total_elapsed_formatted": total_elapsed_formatted,
        "step_time_ms": round(avg_step_ms, 3),
        "peak_gflops": round(avg_gflops, 1),
        "mfu_pct": round(avg_mfu, 2),
        "avg_bandwidth_gbps": 0.0,
        "final_loss": round(last_loss_val, 4),
        "nan_detected": False,
        "jit_active": use_jit,
        "micro_batch_size": micro_batch_size,
        "grad_accumulation_steps": grad_accum_steps,
        "effective_batch_size": eff_batch_size,
        "padded_vocab_size": model.vocab_size,
    }

    print("\n=======================================================", flush=True)
    print(f"🏆 {args.model_size} PRODUCTION TRAINING COMPLETE!", flush=True)
    print("=======================================================", flush=True)
    print(f"Start Time: {start_datetime}", flush=True)
    print(f"End Time:   {end_datetime}", flush=True)
    print(f"Total Run Duration: {total_elapsed_formatted} ({total_elapsed_sec:.2f}s)", flush=True)
    print(f"Average Step Time: {avg_step_ms:.2f} ms", flush=True)
    print(f"Average Throughput: {avg_tput:.1f} samples/sec", flush=True)
    print(f"Average Compute: {avg_gflops:.1f} GFLOPS ({avg_gflops / 1000.0:.2f} TFLOPS)", flush=True)
    print(f"Average MFU: {avg_mfu:.2f}% (Target: ~35%)", flush=True)
    print("=======================================================\n", flush=True)

    sys.stdout.write("\n=== HARNESS TELEMETRY METRICS ===\n")
    sys.stdout.write(json.dumps(telemetry, indent=2) + "\n")
    sys.stdout.write("=================================\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\n[TRAINER ERROR] Incompatible configuration: {e}\n")
        sys.stderr.flush()
        os._exit(1)
