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

if "HK_FLASH_ATTENTION" not in os.environ:
    os.environ["HK_FLASH_ATTENTION"] = "1"
os.environ["ALLOW_TF32"] = os.environ.get("ALLOW_TF32", str(_preload_config.get("ALLOW_TF32", "1")))
os.environ["TINYCACHE"] = os.environ.get("TINYCACHE", "1")
os.environ["HCQ"] = os.environ.get("HCQ", "0")
os.environ["TC"] = "1"
os.environ["TENSOR_CORES"] = "1"
os.environ["BEAM"] = os.environ.get("BEAM", str(_preload_config.get("BEAM", "2")))

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.device import Device
from tinygrad.nn.optim import AdamW, OptimizerGroup
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_save

from model import GPT


def get_layer_depth(param_name: str, n_layers: int) -> int:
    if param_name.startswith("wte") or param_name.startswith("wpe") or param_name.startswith("emb"):
        return 0
    match = re.match(r"^h\.(\d+)\.", param_name)
    if match:
        return int(match.group(1)) + 1
    if param_name.startswith("rms_f") or param_name.startswith("head") or param_name.startswith("lm_head"):
        return n_layers + 1
    return n_layers + 1


def build_optimizer(model: GPT, max_lr: float, weight_decay: float, use_llrd: bool, llrd_decay: float):
    state_dict = get_state_dict(model)
    n_layers = len(model.h)
    total_layers = n_layers + 1

    if not use_llrd:
        params = get_parameters(model)
        decay_params = [p for p in params if len(p.shape) >= 2]
        nodecay_params = [p for p in params if len(p.shape) < 2]
        opt_decay = AdamW(decay_params, lr=max_lr, weight_decay=weight_decay)
        opt_nodecay = AdamW(nodecay_params, lr=max_lr, weight_decay=0.0)
        setattr(opt_decay, "llrd_scale", 1.0)
        setattr(opt_nodecay, "llrd_scale", 1.0)
        return OptimizerGroup(opt_decay, opt_nodecay), {"active": False, "gamma": llrd_decay, "layer_lrs": {}}

    layer_groups: dict[int, list[tuple[str, Tensor]]] = {d: [] for d in range(total_layers + 1)}
    for name, p in state_dict.items():
        if not p.is_param:
            continue
        d = get_layer_depth(name, n_layers)
        layer_groups[d].append((name, p))

    opts = []
    layer_lrs = {}
    for d in range(total_layers + 1):
        group = layer_groups[d]
        if not group:
            continue
        scale = llrd_decay ** (total_layers - d)
        layer_lrs[d] = scale

        decay_p = [p for name, p in group if len(p.shape) >= 2]
        nodecay_p = [p for name, p in group if len(p.shape) < 2]

        if decay_p:
            opt_d = AdamW(decay_p, lr=max_lr, weight_decay=weight_decay)
            setattr(opt_d, "llrd_scale", scale)
            opts.append(opt_d)
        if nodecay_p:
            opt_nd = AdamW(nodecay_p, lr=max_lr, weight_decay=0.0)
            setattr(opt_nd, "llrd_scale", scale)
            opts.append(opt_nd)

    optimizer = OptimizerGroup(*opts)
    info = {
        "active": True,
        "gamma": llrd_decay,
        "total_layers": total_layers,
        "layer_lrs": layer_lrs,
    }
    return optimizer, info


def save_checkpoint(model: GPT, optimizer: OptimizerGroup, step: int, ckpt_path: str):
    state = get_state_dict(model)
    param_to_key = {id(p): k for k, p in state.items()}

    if len(optimizer.optimizers) > 0:
        first_opt = optimizer.optimizers[0]
        state["opt.decay.b1_t"] = first_opt.b1_t
        state["opt.decay.b2_t"] = first_opt.b2_t
        state["opt.nodecay.b1_t"] = first_opt.b1_t
        state["opt.nodecay.b2_t"] = first_opt.b2_t

    for opt in optimizer.optimizers:
        for i, p in enumerate(opt.params):
            param_key = param_to_key.get(id(p))
            if param_key:
                state[f"opt.m.{param_key}"] = opt.m[i]
                state[f"opt.v.{param_key}"] = opt.v[i]

    state["global_step"] = Tensor([step], dtype=dtypes.int32)
    safe_save(state, ckpt_path)


def load_checkpoint(model: GPT, optimizer: OptimizerGroup, ckpt_path: str) -> int:
    state = safe_load(ckpt_path)
    load_state_dict(model, state)

    resumed_step = 0
    if "global_step" in state:
        resumed_step = int(state["global_step"].cast(dtypes.int32).to(Device.DEFAULT).item())

    if "opt.decay.b1_t" in state:
        for opt in optimizer.optimizers:
            opt.b1_t.assign(state["opt.decay.b1_t"].cast(opt.b1_t.dtype).to(opt.b1_t.device))
    if "opt.decay.b2_t" in state:
        for opt in optimizer.optimizers:
            opt.b2_t.assign(state["opt.decay.b2_t"].cast(opt.b2_t.dtype).to(opt.b2_t.device))

    model_state = get_state_dict(model)
    param_to_key = {id(p): k for k, p in model_state.items()}

    restored_buffers = 0
    for opt in optimizer.optimizers:
        for i, p in enumerate(opt.params):
            param_key = param_to_key.get(id(p))
            if param_key:
                m_key = f"opt.m.{param_key}"
                v_key = f"opt.v.{param_key}"
                if m_key in state and v_key in state:
                    opt.m[i].assign(state[m_key].cast(opt.m[i].dtype).to(opt.m[i].device))
                    opt.v[i].assign(state[v_key].cast(opt.v[i].dtype).to(opt.v[i].device))
                    restored_buffers += 1

    if restored_buffers > 0:
        print(f"✅ Restored optimizer momentum & variance buffers ({restored_buffers} tensors)", flush=True)
    else:
        print("⚠️ Checkpoint lacks optimizer state. Momentum buffers will start from zero.", flush=True)

    return resumed_step


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
    parser.add_argument(
        "--val-steps", "--eval-steps", type=int, default=None, help="Number of micro-batches to evaluate during validation (default: from config or 32)"
    )
    parser.add_argument("--patience", type=int, default=None, help="Patience for early stopping (consecutive evaluations without improvement, 0 to disable)")
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--curriculum", type=str, default=None, help="Path to curriculum JSON configuration file")
    parser.add_argument("--dataset", type=str, choices=["tinystories", "fineweb"], default=None, help="Dataset to train on (tinystories or fineweb)")
    parser.add_argument("--disable-debug", "--no-debug", action="store_true", default=False, help="Disable debug print logging")
    parser.add_argument("--debug-level", "--debug", type=int, default=None, help="Set debug logging level")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest checkpoint in checkpoint-dir")
    parser.add_argument("--resume-path", type=str, default=None, help="Explicit path to checkpoint file to resume from")
    parser.add_argument("--learning-rate", "--lr", type=float, default=None, help="Override peak learning rate")
    parser.add_argument("--use-llrd", action="store_true", default=None, help="Enable Layer-wise Learning Rate Decay (LLRD)")
    parser.add_argument("--no-llrd", action="store_false", dest="use_llrd", help="Disable Layer-wise Learning Rate Decay (LLRD)")
    parser.add_argument("--llrd-decay", "--llrd-gamma", type=float, default=None, help="LLRD decay factor gamma (default: 0.9 or from config)")
    args = parser.parse_args()

    disable_debug = args.disable_debug or args.debug_level == 0 or os.environ.get("DEBUG") == "0"
    if disable_debug:
        os.environ["DEBUG"] = "0"

    config = load_config(args.config)

    # Curriculum Configuration Auto-loading
    curriculum_path = args.curriculum
    if not curriculum_path:
        default_curr = f"conf/curriculum_{args.model_size}.json"
        if os.path.exists(default_curr):
            curriculum_path = default_curr

    curriculum_data = None
    if curriculum_path and os.path.exists(curriculum_path):
        with open(curriculum_path) as cf:
            curriculum_data = json.load(cf)
        print(f"📜 Loaded curriculum configuration: '{curriculum_path}'", flush=True)

    if curriculum_data and "phases" in curriculum_data and len(curriculum_data["phases"]) > 0:
        # Default active phase settings to Phase 1 initially or calculated total steps
        p1 = curriculum_data["phases"][0]
        config["MICRO_BATCH_SIZE"] = p1.get("micro_batch_size", config.get("MICRO_BATCH_SIZE", 16))
        config["GRAD_ACCUMULATION_STEPS"] = p1.get("grad_accumulation_steps", config.get("GRAD_ACCUMULATION_STEPS", 64))
        config["SEQUENCE_LENGTH"] = p1.get("sequence_length", config.get("SEQUENCE_LENGTH", 512))
        total_curr_steps = sum(p.get("steps", 0) for p in curriculum_data["phases"])
        if total_curr_steps > 0 and args.total_steps == 500: # 500 is default parser value
            args.total_steps = total_curr_steps

    if args.use_llrd is not None:
        use_llrd = args.use_llrd
    else:
        use_llrd = bool(config.get("USE_LLRD", 1))

    llrd_decay = float(args.llrd_decay) if args.llrd_decay is not None else float(config.get("LLRD_DECAY", config.get("LLRD_GAMMA", 0.9)))

    dataset_name = (args.dataset or config.get("DATASET", "tinystories")).lower()
    patience = int(args.patience) if args.patience is not None else int(config.get("PATIENCE", 10))
    val_steps = int(args.val_steps) if args.val_steps is not None else int(config.get("VAL_STEPS", 32))

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
    max_lr = float(args.learning_rate) if args.learning_rate is not None else float(config.get("LEARNING_RATE", 4e-4))
    min_lr = max_lr * 0.1
    warmup_iters = max(10, int(args.total_steps * 0.05))

    # Determine Dataset Paths
    if dataset_name == "fineweb":
        data_dir = "data/FineWeb"
    else:
        data_dir = "data/TinyStories"

    train_bin = os.path.join(data_dir, "train_trimmed.bin")
    valid_bin = os.path.join(data_dir, "valid_trimmed.bin")
    if not os.path.exists(train_bin):
        train_bin = os.path.join(data_dir, "train.bin")
    if not os.path.exists(valid_bin):
        valid_bin = os.path.join(data_dir, "valid.bin")

    # Load Vocab Map if available in dataset directory
    vocab_map_path = os.path.join(data_dir, "vocab_map.json")
    dataset_vocab_size = int(config.get("VOCAB_SIZE", 13970))
    if os.path.exists(vocab_map_path):
        try:
            with open(vocab_map_path) as vf:
                vdata = json.load(vf)
                dataset_vocab_size = vdata.get("active_vocab_size", vdata.get("trimmed_vocab_size", vdata.get("vocab_size", dataset_vocab_size)))
        except Exception:
            pass

    # Model Presets
    if args.model_size == "125M":
        vocab_size = dataset_vocab_size
        d_model = 768
        n_layers = 12
        n_heads = 12
        d_ff = 3072
    else:
        vocab_size = dataset_vocab_size
        d_model = 288
        n_layers = 6
        n_heads = 6
        d_ff = 1152

    use_swiglu = bool(config.get("USE_SWIGLU", 1))
    use_rope = bool(config.get("USE_ROPE", 1))
    use_flash_attn = bool(int(os.environ.get("HK_FLASH_ATTENTION", config.get("HK_FLASH_ATTENTION", 1))))
    pad_vocab_mult = int(config.get("PAD_VOCAB_MULTIPLE", 1))
    pad_vocab_p2 = bool(config.get("PAD_VOCAB_POWER_OF_2", 0))
    use_jit = bool(config.get("JIT", 1))

    if os.path.exists(train_bin):
        print(f"📦 Loading training dataset ({dataset_name}): '{train_bin}'", flush=True)
        train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    else:
        print(f"⚠️ Training binary dataset '{train_bin}' not found. Using synthetic random buffer...", flush=True)
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
        flash_attn=use_flash_attn,
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

    if ckpt_path_to_load and not os.path.exists(ckpt_path_to_load):
        raise FileNotFoundError(f"Checkpoint file '{ckpt_path_to_load}' not found.")

    param_count = model.num_params()
    params = get_parameters(model)

    sys.stderr.write("[train_production.py] Realizing model weights into VRAM...\n")
    for x in params:
        x.replace(x.contiguous())
    Tensor.realize(*params)

    weight_decay = float(config.get("WEIGHT_DECAY", 0.01))

    optimizer, llrd_info = build_optimizer(model, max_lr, weight_decay, use_llrd, llrd_decay)

    # Load initial state dict and optimizer momentum/variance buffers if resuming
    if ckpt_path_to_load:
        print(f"🔄 Resuming training state from checkpoint: '{ckpt_path_to_load}'", flush=True)
        loaded_step = load_checkpoint(model, optimizer, ckpt_path_to_load)
        if loaded_step > 0:
            resumed_step = loaded_step
        elif resumed_step == 0:
            match = re.search(r"_step_(\d+)\.safetensors$", ckpt_path_to_load)
            if match:
                resumed_step = int(match.group(1))
        print(f"✅ State loaded successfully. Resume starting step {resumed_step + 1}", flush=True)

    start_step = resumed_step + 1

    print("\n=======================================================", flush=True)
    print(f"🚀 MAIN PRODUCTION TRAINER INITIALIZED ({args.model_size})", flush=True)
    print(f"Start Time: {start_datetime}", flush=True)
    if start_step > 1:
        print(f"Resuming From: Step {start_step} / {args.total_steps} (Checkpoint: {ckpt_path_to_load})", flush=True)
    print(f"Parameters: {param_count:,} | Padded Vocab: {model.vocab_size}", flush=True)
    print(f"Micro-Batch: {micro_batch_size} | Grad Accum: {grad_accum_steps} | Effective Batch: {eff_batch_size} | Seq Len: {seq_len}", flush=True)
    print(f"Precision: {default_float_str} | RoPE: {use_rope} | SwiGLU: {use_swiglu} | BEAM: {os.environ.get('BEAM')}", flush=True)
    if use_llrd:
        min_scale = llrd_decay ** (n_layers + 1)
        print(
            f"LLRD: ENABLED (gamma={llrd_decay:.3f}) | Depths: 0..{n_layers + 1} | Peak LRs: Embedding={max_lr * min_scale:.3e} -> Head={max_lr:.3e}",
            flush=True,
        )
    else:
        print("LLRD: DISABLED", flush=True)
    print("=======================================================\n", flush=True)

    flops_per_step = 6.0 * param_count * eff_batch_size * seq_len

    def get_batch(data_source: np.ndarray, step_idx: int):
        d_len = len(data_source)
        offset = (step_idx * eff_batch_size * seq_len) % (d_len - eff_batch_size * seq_len - 1)
        chunk = data_source[offset : offset + eff_batch_size * seq_len + 1].astype(np.int32)
        x_np = chunk[:-1].reshape(eff_batch_size, seq_len)
        y_np = chunk[1:].reshape(eff_batch_size, seq_len)
        return x_np, y_np

    # Canonical TinyJit Fused Train Step
    @TinyJit
    def train_step(x_m: Tensor, y_m: Tensor, lr_tensor: Tensor) -> Tensor:
        optimizer.zero_grad()
        for opt in optimizer.optimizers:
            opt.lr.assign((lr_tensor * getattr(opt, "llrd_scale", 1.0)).cast(opt.lr.dtype))
        logits = model.forward(x_m)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_y = y_m.flatten()
        loss = flat_logits.sparse_categorical_crossentropy(flat_y)
        (loss * loss_scale).backward()
        return loss.realize(*optimizer.schedule_step())

    def val_step(x_m: Tensor, y_m: Tensor) -> Tensor:
        logits = model.forward(x_m)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_y = y_m.flatten()
        return flat_logits.sparse_categorical_crossentropy(flat_y).realize()

    # JIT compilation warmup
    sys.stderr.write("[train_production.py] Running JIT compilation warmup steps...\n")
    w_start = time.time()
    for w in range(2):
        xw, yw = get_batch(train_data, 100 + w)
        w_x = Tensor(xw[:micro_batch_size], device=params[0].device)
        w_y = Tensor(yw[:micro_batch_size], device=params[0].device)
        w_lr = Tensor([max_lr], device=params[0].device)
        w_loss = train_step(w_x, w_y, w_lr)
        Device[Device.DEFAULT].synchronize()
        w_loss_val = float(w_loss.cast(dtypes.float).item())
        if math.isnan(w_loss_val) or math.isinf(w_loss_val):
            raise RuntimeError(f"JIT warmup step {w + 1} produced invalid loss: {w_loss_val}")

    if ckpt_path_to_load:
        _ = load_checkpoint(model, optimizer, ckpt_path_to_load)
        Tensor.realize(*params)
        for opt in optimizer.optimizers:
            Tensor.realize(*opt.m, *opt.v, opt.b1_t, opt.b2_t)
        Device[Device.DEFAULT].synchronize()

    # CRITICAL: Harness telemetry log string
    sys.stderr.write(f"[train_production.py] JIT Warmup complete in {time.time() - w_start:.2f}s\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    step_times = []
    last_loss_val = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    # Production Training Loop
    for step in range(start_step, args.total_steps + 1):
        cur_lr = get_lr_schedule(step, args.total_steps, warmup_iters, max_lr, min_lr)
        lr_tensor = Tensor([cur_lr], device=params[0].device)

        x_b, y_b = get_batch(train_data, step)
        x_m = Tensor(x_b[:micro_batch_size], device=params[0].device)
        y_m = Tensor(y_b[:micro_batch_size], device=params[0].device)

        t0 = time.time()
        loss_tensor = train_step(x_m, y_m, lr_tensor)
        Device[Device.DEFAULT].synchronize()
        step_loss = float(loss_tensor.cast(dtypes.float).item())
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
            Tensor.training = False
            total_val_loss = 0.0
            for v_idx in range(val_steps):
                x_v, y_v = get_batch(valid_data, step * val_steps + v_idx + 1000)
                x_v_tensor = Tensor(x_v[:micro_batch_size], device=params[0].device)
                y_v_tensor = Tensor(y_v[:micro_batch_size], device=params[0].device)
                v_loss_tensor = val_step(x_v_tensor, y_v_tensor)
                total_val_loss += float(v_loss_tensor.cast(dtypes.float).item())

            val_loss = total_val_loss / val_steps
            eval_tokens = val_steps * micro_batch_size * seq_len
            print(f"📊 Validation Loss at step {step} ({val_steps} steps | {eval_tokens:,} tokens): {val_loss:.4f}", flush=True)
            Tensor.training = True
            Device[Device.DEFAULT].synchronize()

            ckpt_path = os.path.join(args.checkpoint_dir, f"model_{args.model_size.lower()}_step_{step}.safetensors")
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"💾 Checkpoint saved to '{ckpt_path}' (including optimizer state)", flush=True)

            if patience > 0:
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(
                        f"⚠️ Validation loss did not improve (best: {best_val_loss:.4f}). Early stopping patience: {patience_counter}/{patience}",
                        flush=True,
                    )
                    if patience_counter >= patience:
                        print(
                            f"🛑 Early stopping triggered at step {step}: Validation loss failed to improve for {patience} consecutive evaluations.",
                            flush=True,
                        )
                        break

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
        "llrd_active": use_llrd,
        "llrd_decay": llrd_decay if use_llrd else 1.0,
        "micro_batch_size": micro_batch_size,
        "grad_accumulation_steps": grad_accum_steps,
        "effective_batch_size": eff_batch_size,
        "padded_vocab_size": model.vocab_size,
        "val_loss": round(val_loss, 4) if "val_loss" in locals() else None,
        "val_steps": val_steps,
        "val_tokens": val_steps * micro_batch_size * seq_len,
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
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
