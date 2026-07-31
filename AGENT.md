# AGENT.md: Tinygrad High-Performance Optimization Harness

## 1. Role & Objective
You are an autonomous AI Agent specializing in **hardware-level GPU optimization**, **kernel fusion**, **memory bandwidth saturation**, and **arithmetic intensity optimization** for the `tinygrad` deep learning framework.

Your target payload is a **15M Parameter Transformer** training locally on an **NVIDIA GeForce RTX 4090** (24 GB VRAM, Ada Lovelace architecture, ~1,008 GB/s peak memory bandwidth, ~82 TFLOPS FP32 / ~330 TFLOPS FP16 Tensor Core theoretical peak).

Your primary objective is to **minimize training `step_time_ms`**, **maximize `samples_per_sec` throughput**, and **eliminate memory-bound kernel stalls**, while guaranteeing numerical stability (no NaNs, decreasing training loss).

---

## 2. Workspace Architecture
The optimization harness is structured as follows:

```text
/tinygrad-tune-harness
  ├── AGENT.md           # High-performance instructions & architectural rules
  ├── model.py           # 15M Parameter Transformer (FlashAttention SDPA + RMSNorm)
  ├── train.py           # Training payload with Dynamic Loss Scaling & zero CPU sync stalls
  ├── harness.py         # Subprocess runner & kernel Arithmetic Intensity stall parser
  ├── optimize.py        # Automated tuning loop agent with live TUI Visualizer (--tui)
  ├── lint.sh            # Ruff linter & formatting script
  ├── config.json        # Active tunable configuration parameters
  ├── best_config.json   # Checkpoint of best discovered hyperparameter configuration
  └── score.json         # Standardized hardware & stall telemetry output
```

---

## 3. Strict High-Performance Architectural Rules

To maintain **state-of-the-art GPU performance** and prevent architectural regressions, all model and trainer code MUST adhere strictly to the following rules:

### A. Zero Intermediate Flushes & Pure Kernel Fusion
- **Rule 1:** NEVER call `.realize()`, `.item()`, or `.numpy()` inside `model.forward()` or intermediate layer definitions. Any intermediate call forces `tinygrad` to flush intermediate tensors to VRAM, creating a 100% memory-bound bottleneck.
- **Rule 2:** Defer `.item()` loss evaluation in `train.py` ONLY to designated logging steps. Calling `.item()` on every step causes CPU-GPU synchronization stalls.

### B. Mandatory Fused FlashAttention (SDPA)
- **Rule 3:** NEVER construct explicit $Q K^T$ matrix multiplications or manual Softmax attention masks. Standard causal attention allocates a $B \times H \times T \times T$ matrix in VRAM, causing heavy memory stalls.
- **Rule 4:** ALWAYS use `tinygrad.Tensor.scaled_dot_product_attention(q, k, v, is_causal=True)` for causal self-attention.

### C. RMSNorm over LayerNorm
- **Rule 5:** ALWAYS use `RMSNorm` (`(x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()) * weight`) instead of `LayerNorm`. `RMSNorm` eliminates mean-subtraction reduction passes and fuses cleanly into single-pass activation kernels.

### D. Dynamic Mixed-Precision Loss Scaling
- **Rule 6:** When training in FP16 (`HALF`) or BF16 (`BFLOAT16`), ALWAYS use Dynamic Loss Scaling (`loss * loss_scale`) before `.backward()` to prevent numerical underflow/NaNs while unlocking Ada Lovelace Tensor Cores.

### E. Code Quality & Linter Compliance
- **Rule 7:** Code MUST pass `./lint.sh` (`uv run ruff check --fix .` and `uv run ruff format .`) with zero errors or warnings before committing.

---

## 4. The Parameter Space (`config.json`)

### Tunable Categories
1. **Tinygrad Environment Variables:**
   - `BEAM` (integer: `0`, `2`, `4`, `8`, `16`): Controls `tinygrad`'s internal kernel layout optimizer (`OptOps`). Searches loop unrolling (`UNROLL`), upcasting (`UPCAST`), and memory indexing strategies to maximize L1/L2 cache reuse.
   - `ALLOW_TF32` (integer: `1` or `0`): Enables Tensor Cores for FP32 matrix math on Ampere/Ada GPUs.
   - `DEFAULT_FLOAT` (string: `"FLOAT"`, `"HALF"`, or `"BFLOAT16"`): Controls default tensor precision.
   - `JIT` (integer: `1` or `0`): Forces execution through `@TinyJit` to eliminate Python interpreter overhead.
2. **Model / Training Hyperparameters:**
   - `BATCH_SIZE` (integer): Scales batch dimension to reuse weight matrices loaded from VRAM, multiplying FLOPs per byte transferred.
   - `MICROBATCH_SIZE` & `GRAD_ACCUMULATION_STEPS`: Balance memory usage and arithmetic intensity.

---

## 5. Telemetry & Memory Stall Detection (`score.json`)

```json
{
  "step_time_ms": 0.564,
  "peak_gflops": 2542827.3,
  "avg_bandwidth_gbps": 205.4,
  "total_kernels": 18,
  "mem_bound_kernels": 4,
  "memory_bound_kernel_pct": 22.2,
  "arithmetic_intensity": 12.37,
  "status": "COMPUTE_OPTIMIZED",
  "final_loss": 5.9608,
  "nan_detected": false,
  "jit_active": true
}
```

- **`memory_bound_kernel_pct` (Stall Metric):** Percentage of executed kernels where memory bandwidth >600 GB/s but compute <15,000 GFLOPS. If >40%, `status` is set to `"MEMORY_BOUND"`.
- **`step_time_ms` (Primary Metric):** Lower is better.
- **`nan_detected` (Guardrail):** If `true`, configuration is invalid. Revert immediately.

---

## 6. Automated Optimization Workflow

Run the automated tuner with live TUI visualizer:
```bash
uv run python optimize.py --tui --max-steps 8
```
The decision engine will automatically scale `BATCH_SIZE`, test precision compression (`HALF`/`BFLOAT16`), explore `BEAM` compiler layouts, and save optimal checkpoints to `best_config.json`.
