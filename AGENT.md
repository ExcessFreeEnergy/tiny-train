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
  ├── AGENT.md           # High-performance instructions & 4-phase optimization rules
  ├── model.py           # 15M Parameter Transformer (FlashAttention SDPA + RMSNorm + SwiGLU)
  ├── train.py           # Training payload with Gradient Accumulation & zero CPU sync stalls
  ├── harness.py         # Subprocess runner & OOM-safe micro-batch sweeper
  ├── optimize.py        # 4-Phase automated tuning loop agent with live TUI Visualizer (--tui)
  ├── lint.sh            # Ruff linter & formatting script
  ├── config.json        # Active tunable configuration parameters
  ├── best_config.json   # Checkpoint of best discovered hyperparameter configuration
  └── score.json         # Standardized hardware & stall telemetry output
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

### Phase 3: BEAM Compiler Layout Search
- With tensor shapes locked in, increase `BEAM` (0 $\rightarrow$ 4 $\rightarrow$ 8 $\rightarrow$ 16) to let `tinygrad`'s `OptOps` compiler search for optimal L1/L2 cache tiling (`LOCAL`/`UPCAST`).

### Phase 4: SwiGLU Activation Fusion
- Test SwiGLU MLP blocks (`USE_SWIGLU=1` in `config.json` or `SwiGLUMLP` in `model.py`) to increase arithmetic intensity (`(x @ w1).silu() * (x @ w3) @ w2`).

---

## 4. Strict High-Performance Architectural Rules

To maintain **state-of-the-art GPU performance** and prevent architectural regressions, all model and trainer code MUST adhere strictly to the following rules:

### A. Zero Intermediate Flushes & Pure Kernel Fusion
- **Rule 1:** NEVER call `.realize()`, `.item()`, or `.numpy()` inside `model.forward()` or intermediate layer definitions. Any intermediate call forces `tinygrad` to flush intermediate tensors to VRAM.
- **Rule 2:** Defer `.item()` loss evaluation in `train.py` ONLY to designated logging steps. Calling `.item()` on every step causes CPU-GPU synchronization stalls.

### B. Mandatory Fused FlashAttention (SDPA)
- **Rule 3:** NEVER construct explicit $Q K^T$ matrix multiplications or manual Softmax attention masks.
- **Rule 4:** ALWAYS use `tinygrad.Tensor.scaled_dot_product_attention(q, k, v, is_causal=True)` for causal self-attention.

### C. RMSNorm over LayerNorm
- **Rule 5:** ALWAYS use `RMSNorm` (`(x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()) * weight`) instead of `LayerNorm`. `RMSNorm` eliminates mean-subtraction reduction passes and fuses cleanly into single-pass activation kernels.

### D. Code Quality & Linter Compliance
- **Rule 6:** Code MUST pass `./lint.sh` (`uv run ruff check --fix .` and `uv run ruff format .`) with zero errors or warnings before committing.

---

## 5. Telemetry & Memory Stall Detection (`score.json`)

```json
{
  "step_time_ms": 439.76,
  "peak_gflops": 3487.6,
  "avg_bandwidth_gbps": 0.2,
  "micro_batch_size": 64,
  "grad_accumulation_steps": 4,
  "effective_batch_size": 256,
  "total_kernels": 617,
  "mem_bound_kernels": 154,
  "memory_bound_kernel_pct": 28.8,
  "arithmetic_intensity": 7.95,
  "status": "COMPUTE_OPTIMIZED",
  "final_loss": 6.133,
  "nan_detected": false,
  "jit_active": true
}
```

- **`memory_bound_kernel_pct` (Stall Metric):** Percentage of executed kernels where memory bandwidth >600 GB/s but compute <15,000 GFLOPS. If >40%, `status` is set to `"MEMORY_BOUND"`.
- **`step_time_ms` (Primary Metric):** Lower is better.
- **`nan_detected` (Guardrail):** If `true`, configuration is invalid. Revert immediately.

---

## 6. Automated Optimization Workflow

Run the 4-phase automated tuner with live TUI visualizer:
```bash
uv run python optimize.py --tui --max-steps 8
```
Or run the OOM-safe micro-batch discovery sweep directly:
```bash
uv run python harness.py --sweep-batch
```
