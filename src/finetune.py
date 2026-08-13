#!/usr/bin/env python3
"""
finetune.py - LoRA Fine-Tuning Engine for 125M Transformer Model using tinygrad & uv.

Features:
  - Start from pre-trained 125M base checkpoint (e.g. checkpoints/model_125m_step_5500.safetensors).
  - Low-Rank Adaptation (LoRA): Freezes base parameters and trains low-rank adapters (c_attn, c_proj, w13, w2).
  - Disables KV caching during training (Tensor.training = True) for sequence logits forward pass.
  - Streams Open-Platypus finetuning dataset via np.memmap.
  - @TinyJit compiled micro-batch accumulation & optimizer step.
  - Safetensors checkpointing matching src/train_production.py format.
  - Automatic post-training weight fusion: exports fused .safetensors model compatible with run.py & chat.py.
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
    return {}


_preload_config = load_config()

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

from src.model import GPT, apply_rope


class LoRALinear:
    """Low-Rank Adaptation (LoRA) Linear Layer Wrapper."""

    def __init__(self, base_weight: Tensor, rank: int = 8, alpha: float = 16.0):
        self.base_weight = base_weight.is_param_(False)
        self.in_dim = base_weight.shape[0]
        self.out_dim = base_weight.shape[1]
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Trainable low-rank adapter matrices
        self.lora_A = Tensor.glorot_uniform(self.in_dim, rank)
        self.lora_B = Tensor.zeros(rank, self.out_dim)

    def __call__(self, x: Tensor) -> Tensor:
        base_out = x @ self.base_weight
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    def get_fused_weight(self) -> Tensor:
        return self.base_weight + ((self.lora_A @ self.lora_B) * self.scaling)


def apply_lora(model: GPT, rank: int = 8, alpha: float = 16.0, target_modules: list[str] | None = None) -> list[Tensor]:
    """Freeze base weights and inject LoRA adapters into target linear projections."""
    if target_modules is None:
        target_modules = ["c_attn", "c_proj"]

    # 1. Freeze all base parameters in the model
    for p in get_parameters(model):
        p.is_param_(False)

    # 2. Inject LoRA adapters into target layers across all transformer blocks
    for block in model.h:
        attn = block.attn
        if "c_attn" in target_modules:
            c_attn_lora = LoRALinear(attn.c_attn, rank=rank, alpha=alpha)
            attn.c_attn_lora = c_attn_lora

        if "c_proj" in target_modules:
            c_proj_lora = LoRALinear(attn.c_proj, rank=rank, alpha=alpha)
            attn.c_proj_lora = c_proj_lora

        # Rebind attention __call__ logic
        def make_new_attn(attn_obj):
            has_c_attn = hasattr(attn_obj, "c_attn_lora")
            has_c_proj = hasattr(attn_obj, "c_proj_lora")

            def new_attn_call(x: Tensor, freqs_cis: Tensor | None = None, start_pos: int | None = None) -> Tensor:
                b, t, c = x.shape
                c_attn_out = attn_obj.c_attn_lora(x) if has_c_attn else x @ attn_obj.c_attn
                qkv = c_attn_out.reshape(b, t, 3, attn_obj.n_heads, attn_obj.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                if start_pos is None:
                    if attn_obj.use_rope and freqs_cis is not None:
                        q = apply_rope(q, freqs_cis[:t])
                        k = apply_rope(k, freqs_cis[:t])
                    y = Tensor.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    if attn_obj.use_rope and freqs_cis is not None:
                        freqs = freqs_cis[start_pos : start_pos + t]
                        q = apply_rope(q, freqs)
                        k = apply_rope(k, freqs)

                    if not hasattr(attn_obj, "cache_kv"):
                        max_context = freqs_cis.shape[0] if freqs_cis is not None else 512
                        attn_obj.cache_kv = Tensor.zeros(2, b, attn_obj.n_heads, max_context, attn_obj.head_dim, dtype=k.dtype).contiguous().realize()

                    kv_stacked = Tensor.stack(k, v).cast(attn_obj.cache_kv.dtype)
                    attn_obj.cache_kv[:, :, :, start_pos : start_pos + t, :].assign(kv_stacked).realize()
                    keys = attn_obj.cache_kv[0][:, :, : start_pos + t, :]
                    values = attn_obj.cache_kv[1][:, :, : start_pos + t, :]
                    mask = Tensor.full((1, 1, t, start_pos + t), float("-inf"), dtype=x.dtype).triu(start_pos + 1) if t > 1 else None
                    y = Tensor.scaled_dot_product_attention(q, keys, values, attn_mask=mask)

                y = y.transpose(1, 2).reshape(b, t, c)
                return attn_obj.c_proj_lora(y) if has_c_proj else y @ attn_obj.c_proj

            return new_attn_call

        attn.__call__ = make_new_attn(attn)

        # Inject LoRA into MLP if requested
        mlp = block.mlp
        has_swiglu = hasattr(mlp, "w13")
        if has_swiglu:
            if "w13" in target_modules:
                mlp.w13_lora = LoRALinear(mlp.w13, rank=rank, alpha=alpha)
            if "w2" in target_modules:
                mlp.w2_lora = LoRALinear(mlp.w2, rank=rank, alpha=alpha)

            def make_new_swiglu_mlp(mlp_obj):
                has_w13_lora = hasattr(mlp_obj, "w13_lora")
                has_w2_lora = hasattr(mlp_obj, "w2_lora")

                def new_mlp_call(x: Tensor) -> Tensor:
                    w13_out = mlp_obj.w13_lora(x) if has_w13_lora else x @ mlp_obj.w13
                    w1, w3 = w13_out.chunk(2, dim=-1)
                    swiglu_act = w1.silu() * w3
                    return mlp_obj.w2_lora(swiglu_act) if has_w2_lora else swiglu_act @ mlp_obj.w2

                return new_mlp_call

            mlp.__call__ = make_new_swiglu_mlp(mlp)
        else:
            if "c_fc" in target_modules:
                mlp.c_fc_lora = LoRALinear(mlp.c_fc, rank=rank, alpha=alpha)
            if "c_proj" in target_modules:
                mlp.c_proj_lora = LoRALinear(mlp.c_proj, rank=rank, alpha=alpha)

            def make_new_gelu_mlp(mlp_obj):
                has_cfc_lora = hasattr(mlp_obj, "c_fc_lora")
                has_cproj_lora = hasattr(mlp_obj, "c_proj_lora")

                def new_mlp_call(x: Tensor) -> Tensor:
                    h_act = (mlp_obj.c_fc_lora(x) if has_cfc_lora else x @ mlp_obj.c_fc).gelu()
                    return mlp_obj.c_proj_lora(h_act) if has_cproj_lora else h_act @ mlp_obj.c_proj

                return new_mlp_call

            mlp.__call__ = make_new_gelu_mlp(mlp)

    # 3. Extract trainable LoRA parameters
    trainable_params = [p for p in get_parameters(model) if p.is_param]
    return trainable_params


def save_lora_checkpoint(model: GPT, optimizer: AdamW, step: int, ckpt_path: str):
    """Save LoRA adapter weights and optimizer state to safetensors."""
    state = get_state_dict(model)
    # Extract only LoRA adapter parameters to keep checkpoint lightweight
    lora_state = {k: v for k, v in state.items() if "lora_A" in k or "lora_B" in k}

    param_to_key = {id(p): k for k, p in lora_state.items()}

    for i, p in enumerate(optimizer.params):
        param_key = param_to_key.get(id(p))
        if param_key:
            lora_state[f"opt.m.{param_key}"] = optimizer.m[i]
            lora_state[f"opt.v.{param_key}"] = optimizer.v[i]

    lora_state["opt.b1_t"] = optimizer.b1_t
    lora_state["opt.b2_t"] = optimizer.b2_t
    lora_state["global_step"] = Tensor([step], dtype=dtypes.int32)

    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    safe_save(lora_state, ckpt_path)


def load_lora_checkpoint(model: GPT, optimizer: AdamW, ckpt_path: str) -> int:
    """Load saved LoRA adapter weights and optimizer state."""
    state = safe_load(ckpt_path)
    if "freqs_cis" in state:
        state.pop("freqs_cis")
    load_state_dict(model, state, strict=False)

    resumed_step = 0
    if "global_step" in state:
        resumed_step = int(state["global_step"].cast(dtypes.int32).to(Device.DEFAULT).item())

    if "opt.b1_t" in state:
        optimizer.b1_t.assign(state["opt.b1_t"].cast(optimizer.b1_t.dtype).to(optimizer.b1_t.device))
    if "opt.b2_t" in state:
        optimizer.b2_t.assign(state["opt.b2_t"].cast(optimizer.b2_t.dtype).to(optimizer.b2_t.device))

    state_dict = get_state_dict(model)
    param_to_key = {id(p): k for k, p in state_dict.items()}

    restored_buffers = 0
    for i, p in enumerate(optimizer.params):
        param_key = param_to_key.get(id(p))
        if param_key:
            m_key = f"opt.m.{param_key}"
            v_key = f"opt.v.{param_key}"
            if m_key in state and v_key in state:
                optimizer.m[i].assign(state[m_key].cast(optimizer.m[i].dtype).to(optimizer.m[i].device))
                optimizer.v[i].assign(state[v_key].cast(optimizer.v[i].dtype).to(optimizer.v[i].device))
                restored_buffers += 1

    print(f"✅ Restored {restored_buffers} LoRA optimizer momentum & variance tensors from '{ckpt_path}'", flush=True)
    return resumed_step


def save_fused_checkpoint(model: GPT, output_path: str):
    """Fuse LoRA weights into base model parameters and export clean safetensors checkpoint for inference."""
    for block in model.h:
        attn = block.attn
        if hasattr(attn, "c_attn_lora"):
            attn.c_attn = attn.c_attn_lora.get_fused_weight().realize()
        if hasattr(attn, "c_proj_lora"):
            attn.c_proj = attn.c_proj_lora.get_fused_weight().realize()

        mlp = block.mlp
        if hasattr(mlp, "w13_lora"):
            mlp.w13 = mlp.w13_lora.get_fused_weight().realize()
        if hasattr(mlp, "w2_lora"):
            mlp.w2 = mlp.w2_lora.get_fused_weight().realize()
        if hasattr(mlp, "c_fc_lora"):
            mlp.c_fc = mlp.c_fc_lora.get_fused_weight().realize()
        if hasattr(mlp, "c_proj_lora"):
            mlp.c_proj = mlp.c_proj_lora.get_fused_weight().realize()

    model_state = get_state_dict(model)
    clean_state = {k: v for k, v in model_state.items() if not any(x in k for x in ["c_attn_lora", "c_proj_lora", "w13_lora", "w2_lora", "c_fc_lora"])}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    safe_save(clean_state, output_path)
    print(f"💾 Saved fused fine-tuned model checkpoint to '{output_path}'", flush=True)


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


def get_batch(data_source: np.ndarray, step_idx: int, eff_batch_size: int, seq_len: int):
    d_len = len(data_source)
    offset = (step_idx * eff_batch_size * seq_len) % (d_len - eff_batch_size * seq_len - 1)
    chunk = data_source[offset : offset + eff_batch_size * seq_len + 1].astype(np.int32)
    x_np = chunk[:-1].reshape(eff_batch_size, seq_len)
    y_np = chunk[1:].reshape(eff_batch_size, seq_len)
    return x_np, y_np


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning Engine for 125M Transformer Model")
    parser.add_argument("--base-checkpoint", type=str, default="checkpoints/model_125m_step_5500.safetensors", help="Path to pre-trained base model checkpoint")
    parser.add_argument("--dataset-dir", type=str, default="data/OpenPlatypus", help="Directory for fine-tuning binary dataset (Open-Platypus)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_finetuned", help="Directory to save fine-tuned checkpoints")
    parser.add_argument("--epochs", type=int, default=2, help="Target fine-tuning epochs")
    parser.add_argument("--total-steps", type=int, default=None, help="Explicit total training steps (overrides epochs if specified)")
    parser.add_argument("--eval-interval", type=int, default=50, help="Steps between validation evaluations and adapter checkpoints")
    parser.add_argument("--learning-rate", "--lr", type=float, default=3e-4, help="Peak learning rate for LoRA training")
    parser.add_argument("--lora-rank", "-r", type=int, default=8, help="LoRA adapter rank r")
    parser.add_argument("--lora-alpha", "-a", type=float, default=16.0, help="LoRA scaling factor alpha")
    parser.add_argument("--targets", type=str, default="c_attn,c_proj", help="Comma-separated LoRA target modules (c_attn,c_proj,w13,w2)")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="Micro-batch size per GPU step")
    parser.add_argument("--grad-accum-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length for fine-tuning")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from latest adapter checkpoint in checkpoint-dir")
    parser.add_argument("--resume-path", type=str, default=None, help="Explicit path to adapter checkpoint file to resume from")
    parser.add_argument("--disable-debug", action="store_true", default=False, help="Disable verbose debugging")
    args = parser.parse_args()

    if args.disable_debug:
        os.environ["DEBUG"] = "0"

    dtypes.default_float = dtypes.bfloat16
    loss_scale = 1.0

    micro_batch_size = args.micro_batch_size
    grad_accum_steps = args.grad_accum_steps
    eff_batch_size = micro_batch_size * grad_accum_steps
    seq_len = args.seq_len

    # Load dataset
    data_dir = os.path.abspath(args.dataset_dir)
    train_bin = os.path.join(data_dir, "train_trimmed.bin")
    if not os.path.exists(train_bin):
        train_bin = os.path.join(data_dir, "train.bin")
    valid_bin = os.path.join(data_dir, "valid_trimmed.bin")
    if not os.path.exists(valid_bin):
        valid_bin = os.path.join(data_dir, "valid.bin")

    if not os.path.exists(train_bin):
        raise FileNotFoundError(f"Fine-tuning dataset '{train_bin}' not found. Please run src/prepare_fineweb.py --dataset platypus first.")

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    valid_data = np.memmap(valid_bin, dtype=np.uint16, mode="r") if os.path.exists(valid_bin) else train_data

    total_train_tokens = len(train_data)
    tokens_per_step = eff_batch_size * seq_len
    steps_per_epoch = max(1, total_train_tokens // tokens_per_step)

    if args.total_steps is not None:
        total_steps = args.total_steps
    else:
        total_steps = steps_per_epoch * args.epochs

    max_lr = args.learning_rate
    min_lr = max_lr * 0.1
    warmup_iters = max(10, int(total_steps * 0.05))

    # Inspect vocab_map if present
    vocab_map_path = os.path.join(data_dir, "vocab_map.json")
    fineweb_vocab_map = os.path.join("data/FineWeb", "vocab_map.json")
    dataset_vocab_size = 49685
    if os.path.exists(vocab_map_path):
        with open(vocab_map_path) as vf:
            vdata = json.load(vf)
            dataset_vocab_size = vdata.get("trimmed_vocab_size", vdata.get("active_vocab_size", dataset_vocab_size))
    elif os.path.exists(fineweb_vocab_map):
        with open(fineweb_vocab_map) as vf:
            vdata = json.load(vf)
            dataset_vocab_size = vdata.get("trimmed_vocab_size", dataset_vocab_size)

    # 125M Model Specs
    d_model = 768
    n_layers = 12
    n_heads = 12
    d_ff = 3072
    max_len = 1024
    use_swiglu = True
    use_rope = True

    print("\n=======================================================", flush=True)
    print("🚀 INITIALIZING 125M LORA FINE-TUNING ENGINE", flush=True)
    print(f"Base Checkpoint: '{args.base_checkpoint}'", flush=True)
    print(f"Dataset Directory: '{data_dir}' ({total_train_tokens:,} train tokens)", flush=True)
    print(f"Target Epochs: {args.epochs} ({steps_per_epoch:,} steps/epoch) | Total Steps: {total_steps:,}", flush=True)
    print(f"Micro-Batch: {micro_batch_size} | Grad Accum: {grad_accum_steps} | Effective Batch: {eff_batch_size} | Seq Len: {seq_len}", flush=True)
    print(f"LoRA Rank (r): {args.lora_rank} | LoRA Alpha (a): {args.lora_alpha} | Targets: {args.targets}", flush=True)
    print(f"Peak LR: {max_lr:.3e} | Min LR: {min_lr:.3e} | Warmup Steps: {warmup_iters}", flush=True)
    print("=======================================================\n", flush=True)

    # 1. Instantiate 125M GPT Model
    Tensor.training = True
    model = GPT(
        vocab_size=dataset_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len,
        use_swiglu=use_swiglu,
        use_rope=use_rope,
    )

    # 2. Load pre-trained base weights
    if not os.path.exists(args.base_checkpoint):
        raise FileNotFoundError(f"Base model checkpoint '{args.base_checkpoint}' not found.")

    print(f"📦 Loading base model weights from '{args.base_checkpoint}'...", flush=True)
    ckpt_state = safe_load(args.base_checkpoint)
    base_state = {k: v for k, v in ckpt_state.items() if not k.startswith("opt.") and k != "global_step" and k != "freqs_cis"}
    load_state_dict(model, base_state, strict=False)
    print("✅ Pre-trained base weights loaded successfully!", flush=True)

    # 3. Apply LoRA Adapters
    target_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    lora_params = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, target_modules=target_list)

    # Realize model weights into VRAM
    all_params = get_parameters(model)
    for x in all_params:
        x.replace(x.contiguous())
    Tensor.realize(*all_params)

    # Pre-allocate gradient accumulation buffers for LoRA parameters
    for p in lora_params:
        p.accum_grad = Tensor.zeros_like(p).realize()

    total_model_params = sum(p.numel() for p in all_params)
    total_lora_params = sum(p.numel() for p in lora_params)
    print(
        f"📊 Parameter Summary: Total Model={total_model_params:,} | Trainable LoRA={total_lora_params:,} ({total_lora_params / total_model_params * 100:.2f}%)",
        flush=True,
    )

    optimizer = AdamW(lora_params, lr=max_lr, weight_decay=0.01)

    # Checkpoint Resume logic
    resumed_step = 0
    ckpt_to_load = args.resume_path
    if not ckpt_to_load and args.resume and os.path.exists(args.checkpoint_dir):
        pattern = re.compile(r"lora_125m_step_(\d+)\.safetensors$")
        max_step = -1
        for filename in os.listdir(args.checkpoint_dir):
            match = pattern.match(filename)
            if match:
                s = int(match.group(1))
                if s > max_step:
                    max_step = s
                    ckpt_to_load = os.path.join(args.checkpoint_dir, filename)
        if ckpt_to_load and max_step > 0:
            resumed_step = max_step

    if ckpt_to_load:
        print(f"🔄 Resuming LoRA fine-tuning state from checkpoint: '{ckpt_to_load}'", flush=True)
        resumed_step = load_lora_checkpoint(model, optimizer, ckpt_to_load)

    start_step = resumed_step + 1

    # Define Micro-batch Accumulation Function
    def accum_step(x_micro: Tensor, y_micro: Tensor):
        logits = model.forward(x_micro)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_y = y_micro.flatten()
        chunk_loss = flat_logits.sparse_categorical_crossentropy(flat_y) / grad_accum_steps
        (chunk_loss * loss_scale).backward()
        accum_nodes = []
        for p in lora_params:
            if p.grad is not None:
                accum_nodes.append(p.accum_grad.assign(p.accum_grad + p.grad.cast(p.accum_grad.dtype)))
                p.grad = None
        return chunk_loss, *accum_nodes

    # Define Optimizer Function
    def opt_step():
        for p in lora_params:
            p.grad = p.accum_grad

        # Gradient clipping
        grads = [p.grad for p in lora_params if p.grad is not None]
        if grads:
            total_norm_sq = sum((g.cast(dtypes.float32) ** 2).sum() for g in grads)
            global_norm = total_norm_sq.sqrt()
            clip_coeff = (1.0 / (global_norm + 1e-6)).clip(max_=1.0)
            for p in lora_params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coeff

        opt_nodes = optimizer.schedule_step()
        wipe_nodes = [p.accum_grad.assign(Tensor.zeros_like(p.accum_grad)) for p in lora_params]
        for p in lora_params:
            p.grad = None
        return *opt_nodes, *wipe_nodes

    use_jit = bool(_preload_config.get("JIT", 1))
    if use_jit:
        accum_fn = TinyJit(accum_step)
        opt_fn = TinyJit(opt_step)
    else:
        accum_fn = accum_step
        opt_fn = opt_step

    # JIT compilation warmup
    sys.stderr.write("[finetune.py] Running JIT compilation warmup steps...\n")
    w_start = time.time()
    for w in range(2):
        xw, yw = get_batch(train_data, 10 + w, eff_batch_size, seq_len)
        w_last_loss = None
        for i in range(grad_accum_steps):
            x_m = Tensor(xw[i * micro_batch_size : (i + 1) * micro_batch_size], device=lora_params[0].device)
            y_m = Tensor(yw[i * micro_batch_size : (i + 1) * micro_batch_size], device=lora_params[0].device)
            w_res = accum_fn(x_m, y_m)
            w_last_loss = w_res[0]
        _ = opt_fn()
        Device[Device.DEFAULT].synchronize()
        if w_last_loss is not None:
            w_val = float(w_last_loss.cast(dtypes.float).item())
            if math.isnan(w_val) or math.isinf(w_val):
                raise RuntimeError(f"JIT warmup produced invalid loss: {w_val}")

    if ckpt_to_load:
        _ = load_lora_checkpoint(model, optimizer, ckpt_to_load)
        Tensor.realize(*lora_params)
        Tensor.realize(*optimizer.m, *optimizer.v, optimizer.b1_t, optimizer.b2_t)
        Device[Device.DEFAULT].synchronize()

    sys.stderr.write(f"[finetune.py] JIT Warmup complete in {time.time() - w_start:.2f}s\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_time = time.time()
    start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_times = []
    last_loss_val = 0.0
    val_loss = 0.0

    print("🔥 Starting LoRA Fine-Tuning Execution Loop...\n", flush=True)

    for step in range(start_step, total_steps + 1):
        cur_lr = get_lr_schedule(step, total_steps, warmup_iters, max_lr, min_lr)
        optimizer.lr.assign([cur_lr]).realize()

        x_b, y_b = get_batch(train_data, step, eff_batch_size, seq_len)

        t0 = time.time()
        step_loss_tensor = Tensor.zeros((), dtype=dtypes.float, device=lora_params[0].device)
        for i in range(grad_accum_steps):
            x_m = Tensor(x_b[i * micro_batch_size : (i + 1) * micro_batch_size], device=lora_params[0].device)
            y_m = Tensor(y_b[i * micro_batch_size : (i + 1) * micro_batch_size], device=lora_params[0].device)
            res_micro = accum_fn(x_m, y_m)
            loss_micro = res_micro[0]
            step_loss_tensor = step_loss_tensor + loss_micro

        _ = opt_fn()
        Device[Device.DEFAULT].synchronize()
        step_loss = float(step_loss_tensor.cast(dtypes.float).item())
        t1 = time.time()

        step_ms = (t1 - t0) * 1000.0
        step_times.append(step_ms)

        if step == start_step or step == total_steps or step % 10 == 0 or step % args.eval_interval == 0:
            last_loss_val = step_loss
            tok_per_sec = tokens_per_step / (step_ms / 1000.0) if step_ms > 0 else 0.0
            print(
                f"Step {step:5d} / {total_steps} | Loss: {last_loss_val:.4f} | LR: {cur_lr:.3e} | Tok/sec: {tok_per_sec:.0f} | Step: {step_ms:.1f}ms",
                flush=True,
            )

        # Validation Loss Evaluation & Checkpointing
        if step % args.eval_interval == 0 or step == total_steps:
            Tensor.training = False
            x_v, y_v = get_batch(valid_data, step + 99, eff_batch_size, seq_len)
            val_logits = model.forward(Tensor(x_v[:micro_batch_size], device=lora_params[0].device))
            flat_val_logits = val_logits.reshape(-1, val_logits.shape[-1])
            flat_val_y = Tensor(y_v[:micro_batch_size], device=lora_params[0].device).flatten()
            val_loss_tensor = flat_val_logits.sparse_categorical_crossentropy(flat_val_y).realize()
            val_loss = float(val_loss_tensor.cast(dtypes.float).item())
            print(f"📊 Validation Loss at step {step}: {val_loss:.4f}", flush=True)
            Tensor.training = True

            adapter_ckpt = os.path.join(args.checkpoint_dir, f"lora_125m_step_{step}.safetensors")
            save_lora_checkpoint(model, optimizer, step, adapter_ckpt)
            print(f"💾 LoRA adapter checkpoint saved to '{adapter_ckpt}'", flush=True)

    end_time = time.time()
    end_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_elapsed_sec = end_time - start_time
    total_elapsed_formatted = str(datetime.timedelta(seconds=round(total_elapsed_sec)))
    avg_step_ms = float(np.mean(step_times[1:])) if len(step_times) > 1 else (float(np.mean(step_times)) if step_times else 0.0)
    avg_tput = eff_batch_size / (avg_step_ms / 1000.0) if avg_step_ms > 0 else 0.0

    print("\n=======================================================", flush=True)
    print("🏆 125M LORA FINE-TUNING COMPLETE!", flush=True)
    print("=======================================================", flush=True)
    print(f"Start Time: {start_datetime}", flush=True)
    print(f"End Time:   {end_datetime}", flush=True)
    print(f"Total Duration: {total_elapsed_formatted} ({total_elapsed_sec:.2f}s)", flush=True)
    print(f"Average Step Time: {avg_step_ms:.2f} ms", flush=True)
    print(f"Average Throughput: {avg_tput:.1f} samples/sec", flush=True)
    print(f"Final Loss: {last_loss_val:.4f} | Validation Loss: {val_loss:.4f}", flush=True)
    print("=======================================================\n", flush=True)

    # 4. Fuse weights and export standalone merged safetensors model
    fused_model_path = os.path.join(args.checkpoint_dir, "model_125m_finetuned.safetensors")
    print("🔗 Fusing LoRA adapter weights into base model parameters for standalone inference...", flush=True)
    save_fused_checkpoint(model, fused_model_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
