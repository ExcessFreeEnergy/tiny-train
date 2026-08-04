# progress.md - Tinygrad vs PyTorch 125M & 350M Transformer Optimization Log

This document tracks cumulative performance benchmarks, architectural refactorings, hardware stall telemetry, and throughput improvements for training Transformer models locally on an **NVIDIA GeForce RTX 4090** (24 GB VRAM, Ada Lovelace architecture, ~330 TFLOPS BF16/FP16 Tensor Core peak).

---

## 🚀 Benchmark Performance Log

| Framework / Implementation | Model Scale | Compiler / Mode | Warmup Compile Time | Step Time (`ms`) | Throughput (`smp/s`) | Compute TFLOPS | MFU (`%`) | Validation Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **tinygrad FP32 Baseline** | 15M | Naive JIT | 420.0s | 1,086.30 ms | 58.9 | 1.36 TFLOPS | 0.41% | 5.96 |
| **tinygrad BEAM=4 Initial** | 125M | JIT + BEAM=4 | 2,400.0s | 1,953.14 ms | 65.5 | 13.68 TFLOPS | 4.14% | 5.9411 |
| 🚀 **tinygrad Post-Audit Winner** | **125M** | **`@TinyJit` + BEAM=2** | **36.42s** | **208.95 ms** | **76.6** | **59.17 TFLOPS** | **17.93%** | **6.1562** |
| ⚡ **tinygrad Harness Suite (BEAM=4)** | **125M** | **`@TinyJit` + BEAM=4** | **100.45s** | **-** | **-** | **87.02 TFLOPS** | **26.37%** | **N/A (3-step suite)** |
| **PyTorch Default** | 125M | `torch.compile` | 3.7s | 159.33 ms | 401.7 | 83.83 TFLOPS | 25.40% | 5.9404 |
| **PyTorch Autotune 125M** | 125M | `max-autotune` | 21.8s | 271.91 ms | 235.4 | 98.25 TFLOPS | 29.77% | 6.0209 |
| **PyTorch 350M Scale** | 350M | `reduce-overhead` | 25.2s | 237.70 ms | 33.7 | 103.03 TFLOPS | 31.22% | 7.3492 |
| 🏆 **PyTorch 350M Autotune** | **350M** | **`max-autotune`** | **187.5s** | **219.46 ms** | **36.5** | **111.59 TFLOPS** | **33.82%** | **7.7514** |

---

## 📊 Transient Harness Suite Telemetry & Best Config (`BEAM=4`)

### Harness Suite Run Summary
- **Command Executed**: `BEAM_DEV_TIMEOUT=5 uv run python harness.py --run-suite --skip-batch-sweep`
- **Total Execution Time**: **100.45s**
- **Peak Compute Throughput**: **87,018.0 GFLOPS** (**87.02 TFLOPS**)
- **Model FLOPs Utilization (MFU)**: **26.37%**
- **Average Memory Bandwidth**: **2,852.4 GB/s**
- **Kernel Telemetry**:
  - Total Kernels Compiled/Executed: **767**
  - Memory-Bound Kernels: **338** (**44.1%**)
  - Arithmetic Intensity: **6.29 FLOP/Byte**
  - Hardware Status: `MEMORY_BOUND`

### Optimal Configuration (`best_config.json`)
```json
{
  "MICRO_BATCH_SIZE": 16,
  "GRAD_ACCUMULATION_STEPS": 16,
  "DEFAULT_FLOAT": "BFLOAT16",
  "ALLOW_TF32": 1,
  "BEAM": 4,
  "JIT": 1,
  "USE_SWIGLU": 0,
  "USE_ROPE": 1,
  "PAD_VOCAB_MULTIPLE": 128,
  "SEQUENCE_LENGTH": 256,
  "LEARNING_RATE": 0.001,
  "NUM_STEPS": 3,
  "VOCAB_SIZE": 29362,
  "D_MODEL": 768,
  "N_LAYERS": 12,
  "N_HEADS": 12,
  "D_FF": 3072
}
```

---

## 🛠️ Summary of Code Review Fixes (`tinygrad` Speedup)

1. **Fixed LM Head FP32 Cast**:
   - Removing `.cast(dtypes.float)` from the LM head (`logits = x @ self.wte.T`) enabled `bfloat16` Tensor Core execution for the largest MatMul in the network, boosting GEMM throughput to **59,167 GFLOPS**.
2. **Precomputed Static RoPE Cache Buffers**:
   - Replaced dynamic `Tensor.arange()` calls inside `apply_rope()` with static precomputed `cos` and `sin` buffers in `GPT.__init__`.
3. **Corealized Parameter Updates**:
   - Adding `Tensor.realize(loss, *params)` inside `raw_step()` guaranteed weight updates (`p.assign(...)`) were compiled and realized within the `@TinyJit` graph.
4. **Harness Real-Time Streaming & Timeout Guard**:
   - Streams every execution line in real-time to stdout with live status heartbeats and a strict 5-minute timeout guard (`timeout_sec=300`).

