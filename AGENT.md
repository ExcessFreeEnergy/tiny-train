# AGENT.md: Tinygrad High-Performance Optimization Harness

## 1. Role & Objective
You are an autonomous AI Agent specializing in **hardware-level GPU optimization**, **kernel fusion**, **memory bandwidth saturation**, and **Model FLOPs Utilization (MFU)** optimization for the `tinygrad` deep learning framework.

Your target payload is a Transformer model training locally on an **NVIDIA GeForce RTX 4090** (24 GB VRAM, Ada Lovelace architecture, ~1,008 GB/s peak memory bandwidth, ~330 TFLOPS BF16/FP16 Tensor Core theoretical peak).

Your primary objective is to **target ~60% MFU (~198 TFLOPS)** on the target model architecture, **minimize `step_time_ms`**, **maximize `samples_per_sec` throughput**, and **eliminate memory-bound kernel stalls**, while guaranteeing numerical stability (no NaNs, decreasing training loss).

---

## 2. 2-Stage Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. HARNESS SUITE (Stage 1: ~3 mins)                         │
│    ├── Precision Check (BFLOAT16 + ALLOW_TF32 enabled)      │
│    ├── OOM-Safe Micro-Batch Sweep (Targeting >50 FLOPs/B)   │
│    ├── SwiGLU / GELU Fusion Test (Check for kernel growth)  │
│    └── BEAM Compiler Search (Lock layout for fixed shape)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Writes optimized best_config.json
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. MAIN PRODUCTION TRAINER (Stage 2: train_production.py)  │
│    ├── Loads best_config.json (Zero compiler overhead)      │
│    ├── Streams dataset via np.memmap (TinyStories)          │
│    ├── Cosine LR Decay + Warmup + Safe Checkpointing        │
│    └── Achieves high MFU immediately on Step 1              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Strict 4-Phase Optimization Order of Operations

When instructed to optimize training throughput and overcome memory bandwidth stalls, ALWAYS follow this strict 4-Phase order:

### Phase 1: Precision & Tensor Core Unlock
- Set `DEFAULT_FLOAT="BFLOAT16"` and `ALLOW_TF32=1`.
- Halves VRAM transfer byte volume and lowers the compute-bound threshold, unlocking Ada Lovelace Tensor Cores.

### Phase 2: Micro-Batch Saturation (OOM-Safe Sweep)
- Keep `BEAM=0`. Run the OOM-safe micro-batch sweep in `harness.py` (`python harness.py --sweep-batch`).
- Doubly sweep `MICRO_BATCH_SIZE` (16 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256...) while setting `GRAD_ACCUMULATION_STEPS = max(1, 256 // MICRO_BATCH_SIZE)` to keep effective batch size constant.
- Stop when throughput (`samples/sec`) gain is < 5% or OOM occurs. Lock in this winning micro-batch size.

### Phase 3: BEAM Layout Search (Fast Zone ~2 Mins)
- With tensor shapes locked in, set `BEAM=2` (or `BEAM=4`). Ensure tight `@TinyJit` micro-batch scoping and `TINYCACHE=1` are enabled. Initial search completes in ~2 minutes, and subsequent re-runs load from `~/.cache/tinygrad/cache.db` in <2 seconds.

### Phase 4: SwiGLU Activation Fusion
- Test SwiGLU MLP blocks (`USE_SWIGLU=1` in `config.json` or `SwiGLUMLP` in `model.py`) to increase arithmetic intensity (`(x @ w1).silu() * (x @ w3) @ w2`).

---

## 4. Strict High-Performance Architectural Rules

### A. Tensor Core 64/128 Divisibility Alignment
- **Rule 1:** All matrix dimensions (`d_model`, `d_head`, `d_ff`, `vocab_size`) MUST be multiples of 64 or 128 (e.g. 128). Unaligned dimensions force Ada Lovelace Tensor Cores into zero-padded fallback routines.
- **Rule 2:** Automatically pad `vocab_size` to a multiple of 128 (e.g. 29,362 $\rightarrow$ 29,440).

### B. Fused Rotary Position Embeddings (RoPE)
- **Rule 3:** ALWAYS use Fused RoPE directly inside `CausalSelfAttention` (`apply_rope(q)` and `apply_rope(k)`), eliminating standalone position embedding VRAM reads/writes.

### C. Zero Intermediate Flushes & Tight @TinyJit Scoping
- **Rule 4:** NEVER call `.realize()`, `.item()`, or `.numpy()` inside `model.forward()`.
- **Rule 5:** Defer `.item()` loss evaluation in `train.py` ONLY to designated logging steps.
- **Rule 6:** ALWAYS scope `@TinyJit` cleanly and enable `TINYCACHE=1` for instant disk-cached re-runs.

### D. Code Quality & Linter Compliance
- **Rule 7:** Code MUST pass `./lint.sh` (`uv run ruff check --fix .` and `uv run ruff format .`) with zero errors or warnings before committing.

---

## 5. 125M Model Parameter Presets

| Hyperparameter | 15M Prototype | 125M Production Target | Alignment |
| :--- | :--- | :--- | :--- |
| **`D_MODEL`** | 288 | **768** | Divisible by 128 |
| **`N_LAYERS`** | 6 | **12** | - |
| **`N_HEADS`** | 6 | **12** | $d_{head} = 64$ (Divisible by 64) |
| **`D_FF`** | 1152 | **3072** | Divisible by 128 |
| **`VOCAB_SIZE`** | 29,362 | **29,440** (padded) | Divisible by 128 |

---

## 6. Execution Commands

```bash
# Stage 1: Harness Optimization Pass
uv run python harness.py --sweep-batch
uv run python optimize.py --tui

# Stage 2: 125M Production Training Pass
uv run python train_production.py --model-size 125M --total-steps 500
```
