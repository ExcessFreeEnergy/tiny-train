# progress.md - Tinygrad vs PyTorch 125M & 350M Transformer Optimization Log

This document tracks cumulative performance benchmarks, architectural refactorings, hardware stall telemetry, and throughput improvements for training Transformer models locally on an **NVIDIA GeForce RTX 4090** (24 GB VRAM, Ada Lovelace architecture, ~330 TFLOPS BF16/FP16 Tensor Core peak).

---

## 🚀 Benchmark Performance Log

| Framework / Implementation | Model Scale | Compiler / Mode | Warmup Compile Time | Step Time (`ms`) | Throughput (`smp/s`) | Compute TFLOPS | MFU (`%`) | Validation Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **tinygrad FP32 Baseline** | 15M | Naive JIT | 420.0s | 1,086.30 ms | 58.9 | 1.36 TFLOPS | 0.41% | 5.96 |
| **tinygrad BEAM=4 Initial** | 125M | JIT + BEAM=4 | 2,400.0s | 1,953.14 ms | 65.5 | 13.68 TFLOPS | 4.14% | 5.9411 |
| 🚀 **tinygrad Refactored (Master)** | **125M** | **`@TinyJit` + BEAM=2** | **36.42s** | **466.98 ms** | **34.3** | **7.15 TFLOPS** | **2.17%** | **6.1562** |
| **PyTorch Default** | 125M | `torch.compile` | 3.7s | 159.33 ms | 401.7 | 83.83 TFLOPS | 25.40% | 5.9404 |
| **PyTorch Autotune 125M** | 125M | `max-autotune` | 21.8s | 271.91 ms | 235.4 | 98.25 TFLOPS | 29.77% | 6.0209 |
| **PyTorch 350M Scale** | 350M | `reduce-overhead` | 25.2s | 237.70 ms | 33.7 | 103.03 TFLOPS | 31.22% | 7.3492 |
| 🏆 **PyTorch 350M Autotune** | **350M** | **`max-autotune`** | **187.5s** | **219.46 ms** | **36.5** | **111.59 TFLOPS** | **33.82%** | **7.7514** |

---

## 🛠️ Code Review Audit & Refactorings (`tinygrad` Pipeline)

### Milestone 11: Critical Pipeline Fixes
*Date: 2026-08-01*

#### Key Code Refactorings Applied:
1. **Removed LM Head FP32 Cast**:
   - Replaced `logits = x.cast(dtypes.float) @ self.wte.cast(dtypes.float).T` with `logits = x @ self.wte.T`.
   - Allowed the largest matrix multiplication ($4096 \times 768 \times 29,440$) to run natively in `bfloat16` on Tensor Cores.
2. **Precomputed Static RoPE Buffers**:
   - Removed dynamic `Tensor.arange` calls from inside `apply_rope()` across all 12 transformer layers.
   - Precomputed `self.cos, self.sin` static cache buffers in `GPT.__init__`, slicing `self.cos[:, :, :t, :]` per forward pass.
3. **Corealized Parameter Updates inside `@TinyJit`**:
   - Replaced `return loss.realize()` with `Tensor.realize(loss, *params)` inside `raw_step()`.
   - Guaranteed weight parameter updates (`p.assign(...)`) are strictly compiled and realized within the `@TinyJit` graph.
4. **Warmup Compile Time Dropped to 36.42s**:
   - @TinyJit warmup compile time dropped from **40+ minutes $\rightarrow$ 36.42 seconds**.
