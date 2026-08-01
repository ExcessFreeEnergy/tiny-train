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
| **8. BEAM=2 + Single-Step JIT (16 Batch)** | **125M** | **`COMPUTE_OPTIMIZED`** | **260.73 ms** | **61.4** | **1,248** | **22.1%** | **11.40** | **12,807.2** | **3.88%** | **5.96** | ❌ No |
| **9. BEAM=2 + Batch Size 128 (32k Tokens)** | **125M** | **`COMPUTE_OPTIMIZED`** | **1,953.14 ms** | **65.5** | **1,248** | **22.1%** | **14.20** | **13,677.6** | **4.14%** | **5.94** | ❌ No |

---

## 📊 Summary of Architectural Changes & Impact

### Milestone 9: BEAM=2 Compiler Search & Batch Size 128 Validation
*Date: 2026-08-01*

#### Changes Implemented:
1. **BEAM=2 Compiler Search Locking**:
   - Locked `"BEAM": 2` permanently in `config.json` and `best_config.json`.
   - Single-step `@TinyJit` scoping allowed BEAM compiler search to complete in **41.86s** during step 1 warmup, saving pre-compiled CUDA/PTX binaries to `~/.cache/tinygrad/cache.db` (`TINYCACHE=1`).

2. **Batch Size Scaling**:
   - Scaled batch size to `128` ($32,768 \text{ tokens/step}$).
   - Step time for 32,768 tokens: **1,953.14 ms** (65.5 samples/sec).
   - Peak compute reached **13.68 TFLOPS** (13,677.6 GFLOPS).
   - Validation loss evaluated at **5.9411**.

---

## 🔮 Optimization Roadmap & Next Steps

1. **Production Training Pass**: Run Stage 2 trainer (`train_production.py --model-size 125M --total-steps 500`) with `BEAM=2` locked.
