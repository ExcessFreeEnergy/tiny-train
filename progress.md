# progress.md - Tinygrad 15M Parameter Transformer Optimization Log

This document tracks cumulative performance benchmarks, architectural refactorings, hardware stall telemetry, and throughput improvements for training a **15M Parameter Causal Transformer** locally on an **NVIDIA GeForce RTX 4090** (24 GB VRAM, Ada Lovelace architecture).

---

## 🚀 Benchmark Performance Log

| Milestone / Iteration | Status | Step Time (`ms`) | Throughput (`smp/s`) | Total Kernels | Memory Stalled Kernels | Memory Stall (`%`) | Arithmetic Intensity (`FLOPs/byte`) | Peak GFLOPS | Loss | NaNs Detected? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Initial Baseline (FP32)** | `MEMORY_BOUND` | 1,086.30 ms | 58.9 | 748 | 316 | 42.2% | 2.83 | 1,361.3 | 5.96 | ❌ No |
| **1. Vocab Trim + Submodule** | `MEMORY_BOUND` | 2,106.18 ms | 60.5 | 748 | 316 | 42.2% | 2.83 | 1,361.3 | 6.13 | ❌ No |
| **2. FlashAttention + RMSNorm + BFLOAT16** | **`COMPUTE_OPTIMIZED`** | **829.06 ms** | **154.4** | **503** | **154** | **30.6%** | **7.95** | **3,487.6** | **6.13** | ❌ No |
| **3. Grad Accum (eff=256) + SwiGLU** | **`COMPUTE_OPTIMIZED`** | **4,598.87 ms** | **55.7** | **1,617** | **438** | **27.1%** | **7.29** | **1,416.8** | **6.05** | ❌ No |
| **4. BEAM=2 Compiler Layout Tuning** | **`COMPUTE_OPTIMIZED`** | **611.37 ms** | **418.7** | **1,617** | **565** | **34.9%** | **5.30** | **10,657.4** | **6.05** | ❌ No |

---

## 📊 Summary of Architectural Changes & Impact

### Milestone 4: BEAM Compiler Layout Tuning (`BEAM=2` + `TINYCACHE=1`)
*Date: 2026-07-31*

#### Changes Implemented:
1. **BEAM=2 Compiler Search & Disk Cache**:
   - Enabled `BEAM=2` layout search in `config.json` and explicit disk caching (`TINYCACHE=1`) in `harness.py`.
   - `tinygrad`'s `OptOps` compiler evaluated L1/L2 cache tiling, loop unrolling, and thread indexing strategies for the locked micro-batch shape (`MICRO_BATCH_SIZE=64`, `GRAD_ACCUMULATION_STEPS=4`).

2. **Performance Impact**:
   - **Step Time**: Reduced from 4,598.87 ms down to **611.37 ms** (**7.52x speedup**!).
   - **Peak Compute**: Skyrocketed to **10,657.4 GFLOPS (10.66 TFLOPS)**.
   - **Throughput**: Scaled from 55.7 to **418.7 samples/sec** (**7.52x throughput increase**).
   - **Numerical Stability**: 100% stable (`nan_detected: false`, `final_loss: 6.0488`).

---

### Milestone 3: Batch Size Decoupling & SwiGLU Activation Fusion
*Date: 2026-07-31*

#### Changes Implemented:
- Decoupled physical `MICRO_BATCH_SIZE` (64) from effective batch size (256) using `GRAD_ACCUMULATION_STEPS=4`.
- Automated OOM-safe micro-batch sweeper (`harness.py --sweep-batch`).
- SwiGLU activation fusion (`(x @ w1).silu() * (x @ w3) @ w2`).

---

### Milestone 2: High-Performance Architecture Refactor
*Date: 2026-07-31*

#### Changes Implemented:
- Fused FlashAttention SDPA (`Tensor.scaled_dot_product_attention`).
- RMSNorm replacement.
- Zero intermediate VRAM flushes.
- Dynamic loss scaling and FP32 logit stabilization.
- `ruff` linter integration (`./lint.sh`).

---

### Milestone 1: Vocabulary Trimming & Git Submodule Setup
*Date: 2026-07-31*

#### Changes Implemented:
- `gigatoken` submodule integration.
- Vocabulary trimming (**20,895 dead tokens** / 41.58% removed).

---

## 🔮 Optimization Roadmap & Next Steps

1. **Production Training Run**: Launch full dataset training pass on TinyStories using the optimal `COMPUTE_OPTIMIZED` configuration (`418.7 smp/s`, `10.66 TFLOPS`).
