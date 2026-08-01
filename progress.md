# progress.md - Tinygrad 15M & 125M Parameter Transformer Optimization Log

This document tracks cumulative performance benchmarks, architectural refactorings, hardware stall telemetry, and throughput improvements for training Transformer models locally on an **NVIDIA GeForce RTX 4090** (24 GB VRAM, Ada Lovelace architecture, ~330 TFLOPS BF16/FP16 Tensor Core peak).

---

## 🚀 Benchmark Performance Log

| Milestone / Iteration | Model Scale | Status | Step Time (`ms`) | Throughput (`smp/s`) | Total Kernels | Memory Stall (`%`) | Arithmetic Intensity (`FLOPs/byte`) | Peak GFLOPS | MFU (`%`) | Loss | NaNs Detected? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Initial Baseline (FP32)** | 15M | `MEMORY_BOUND` | 1,086.30 ms | 58.9 | 748 | 42.2% | 2.83 | 1,361.3 | 0.41% | 5.96 | ❌ No |
| **1. Vocab Trim + Submodule** | 15M | `MEMORY_BOUND` | 2,106.18 ms | 60.5 | 748 | 42.2% | 2.83 | 1,361.3 | 0.41% | 6.13 | ❌ No |
| **2. FlashAttention + RMSNorm + BFLOAT16** | 15M | **`COMPUTE_OPTIMIZED`** | **829.06 ms** | **154.4** | **503** | **30.6%** | **7.95** | **3,487.6** | **1.06%** | **6.13** | ❌ No |
| **3. Grad Accum (eff=256) + SwiGLU** | 15M | **`COMPUTE_OPTIMIZED`** | **4,598.87 ms** | **55.7** | **1,617** | **27.1%** | **7.29** | **1,416.8** | **0.43%** | **6.05** | ❌ No |
| **4. BEAM=2 Compiler Layout Tuning** | 15M | **`COMPUTE_OPTIMIZED`** | **611.37 ms** | **418.7** | **1,617** | **34.9%** | **5.30** | **10,657.4** | **3.23%** | **6.05** | ❌ No |
| **5. 64/128 Alignment + Fused RoPE** | 15M | **`COMPUTE_OPTIMIZED`** | **1,036.74 ms** | **246.9** | **1,648** | **25.6%** | **9.30** | **6,237.3** | **1.89%** | **6.04** | ❌ No |
| **6. 125M Target Model Validation** | **125M** | **`COMPUTE_OPTIMIZED`** | **2,161.10 ms** | **29.6** | **1,648** | **25.6%** | **9.30** | **6,180.7** | **1.87%** | **6.24** | ❌ No |
| **7. Fused $W_{13}$ SwiGLU 125M Target** | **125M** | **`COMPUTE_OPTIMIZED`** | **1,499.18 ms** | **42.7** | **1,248** | **22.1%** | **11.40** | **8,909.6** | **2.70%** | **6.16** | ❌ No |

---

## 📊 Summary of Architectural Changes & Impact

### Milestone 7: Fused $W_{13}$ SwiGLU MLP Linear Projection (`model.py`)
*Date: 2026-08-01*

#### Changes Implemented:
1. **Fused SwiGLU Linear Projections ($W_1$ & $W_3 \rightarrow W_{13}$)**:
   - Merged separate `self.w1` and `self.w3` matrix projections in `SwiGLUMLP` into a single `self.w13 = Tensor.glorot_uniform(d_model, 2 * d_ff)` projection.
   - Reduced GEMM kernel dispatch calls per step across 12 layers from 24 separate GEMM calls to 12 fused GEMMs.

2. **Clean Single-Step `@TinyJit` Scoping (`train_production.py`)**:
   - Kept `@TinyJit` scoped strictly to single weight update steps (`step_fn(*inputs)`), consuming micro-batch data inputs without passing 150+ individual gradient accumulators through JIT inputs.

3. **Performance Gains**:
   - 125M parameter step time dropped from **2,161 ms $\rightarrow$ 1,499 ms**.
   - Throughput increased to **42.7 samples/sec**.
   - Validation loss evaluated at **6.0737**.

---

## 🔮 Optimization Roadmap & Next Steps

1. **Dataset Expansion**: Download additional token shards (FineWeb-Edu or larger TinyStories) to scale from 469M tokens to 2.5B tokens (full 20 tokens/param Chinchilla ratio).
2. **Full Production Training**: Launch extended 125M training run when ready (`uv run python train_production.py --model-size 125M --total-steps 500`).
