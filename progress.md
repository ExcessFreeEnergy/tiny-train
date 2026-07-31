# progress.md - Tinygrad 15M Parameter Transformer Optimization Log

This document tracks cumulative performance benchmarks, architectural refactorings, hardware stall telemetry, and throughput improvements for training a **15M Parameter Causal Transformer** locally on an **NVIDIA GeForce RTX 4090** (24 GB VRAM, Ada Lovelace architecture).

---

## 🚀 Benchmark Performance Log

| Milestone / Iteration | Status | Step Time (`ms`) | Throughput (`smp/s`) | Total Kernels | Memory Stalled Kernels | Memory Stall (`%`) | Arithmetic Intensity (`FLOPs/byte`) | Loss | NaNs Detected? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Initial Baseline (FP32)** | `MEMORY_BOUND` | 1,086.30 ms | 58.9 | 748 | 316 | 42.2% | 2.83 | 5.96 | ❌ No |
| **1. Vocab Trim + Submodule** | `MEMORY_BOUND` | 2,106.18 ms | 60.5 | 748 | 316 | 42.2% | 2.83 | 6.13 | ❌ No |
| **2. FlashAttention + RMSNorm + BFLOAT16** | **`COMPUTE_OPTIMIZED`** | **829.06 ms** | **154.4** | **503** | **154** | **30.6%** | **7.95** | **6.13** | ❌ No |

---

## 📊 Summary of Architectural Changes & Impact

### Milestone 2: High-Performance Architecture Refactor
*Date: 2026-07-31*

#### Changes Implemented:
1. **Fused FlashAttention SDPA (`model.py`)**:
   - Replaced standard $O(T^2)$ matrix allocation ($Q K^T$) with `tinygrad.Tensor.scaled_dot_product_attention(q, k, v, is_causal=True)`.
   - Eliminates writing/reading the $T \times T$ intermediate attention matrix to global VRAM.

2. **RMSNorm Replacement (`model.py`)**:
   - Replaced `LayerNorm` with `RMSNorm` (`(x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()) * weight`).
   - Removes mean-subtraction reduction passes and fuses cleanly into single-pass activation kernels.

3. **Zero Intermediate VRAM Flushes (`model.py` & `train.py`)**:
   - Guaranteed zero `.realize()`, `.item()`, or `.numpy()` calls inside `model.forward()`.
   - Deferred `.item()` loss evaluation in `train.py` ONLY to logging steps, eliminating CPU-GPU sync stalls during training steps.

4. **Dynamic Mixed-Precision Loss Scaling (`train.py`)**:
   - Integrated dynamic loss scaling and FP32 logit stabilization for `BFLOAT16` and `HALF` precision.

5. **Linter & Code Quality (`ruff`)**:
   - Integrated `ruff` configuration in `pyproject.toml` and created `./lint.sh`. All files pass with **zero errors/warnings**.

6. **Updated Optimization Guidelines (`AGENT.md`)**:
   - Enforced strict high-performance architectural rules (FlashAttention SDPA, RMSNorm, zero flushes, linter compliance).

#### Performance Impact:
- **Execution Status**: Shifted from `MEMORY_BOUND` to **`COMPUTE_OPTIMIZED`**.
- **Kernels Launched**: Reduced from 748 to **503 kernels/step** (**245 kernels eliminated** per step).
- **Memory Stalled Kernels**: Reduced from 316 to **154 kernels/step** (**162 stalled kernels eliminated**).
- **Memory Stall Percentage**: Dropped from 42.2% down to **30.6%**.
- **Arithmetic Intensity**: Increased from 2.83 to **7.95 FLOPs/byte** (**2.81x increase**).
- **Throughput**: Increased from 58.9 to **154.4 samples/sec** (**2.62x speedup**).

---

### Milestone 1: Vocabulary Trimming & Git Submodule Setup
*Date: 2026-07-31*

#### Changes Implemented:
- Integrated `gigatoken` as a Git submodule (`gigatoken/`).
- Added vocabulary trimming in `retokenize.py` (`--trim-vocab`), removing **20,895 dead tokens** (41.58% reduction from 50,257 to 29,362).
- Reduced embedding & LM head matrix shape from `(50257, d_model)` to `(29362, d_model)`.

---

## 🔮 Optimization Roadmap & Next Steps

1. **Automated Batch Size Scaling**: Scale `BATCH_SIZE` (128 -> 256 -> 512) to maximize weight matrix reuse across parallel sequences.
2. **BEAM Compiler Layout Search**: Explore `BEAM=2` -> `BEAM=4` -> `BEAM=8` to search `tinygrad`'s `OptOps` compiler space for L1/L2 cache locality.
3. **SwiGLU Activation Fusion**: Test SwiGLU MLP blocks (`(x @ w1).silu() * (x @ w3) @ w2`) for higher arithmetic intensity.
