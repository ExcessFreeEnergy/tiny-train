# Tinygrad High-Performance Transformer Optimization & Training Engine

High-performance, hardware-optimized Transformer model training engine built on [tinygrad](https://github.com/tinygrad/tinygrad). Designed for maximum Model FLOPs Utilization (MFU) on NVIDIA Ada Lovelace GPUs (e.g., RTX 4090).

## Objectives & Target Performance

- **Target Model Architecture**: 125M Parameter Transformer (`d_model=768`, `n_layers=12`, `n_heads=12`, `d_ff=3072`, `vocab_size=29,440` padded).
- **End Goal**: Complete a **1 Billion token** training run in **under 4 hours** (< 240 minutes).
- **Target MFU**: **~35% MFU (~115.5 TFLOPS)** sustained performance on NVIDIA RTX 4090 (330 TFLOPS BF16 peak).

---

## Performance Optimizations Implemented

1. **Simultaneous Loss & Weight Realization (Optimizer Graph Bloat Fix)**:
   In `@TinyJit` step functions, calling `Tensor.realize(loss, *optimizer.params)` forces lazy execution of both the loss reduction and weight assignment graphs simultaneously, preventing un-executed computation nodes from accumulating in memory.

2. **Pre-Casted Fused RoPE Buffers**:
   Static Rotary Position Embedding (RoPE) `cos` and `sin` tables are pre-cast to `dtypes.default_float` during initialization in `precompute_freqs_cis`. This removes runtime `.cast()` operations inside `apply_rope()`, eliminating hundreds of unnecessary cast nodes from the JIT compilation graph.

3. **Hardware Saturation (Micro-Batch Saturation)**:
   Micro-batch size is scaled to `MICRO_BATCH_SIZE=64` (or 128) to ensure high matrix multiplication dimensions per kernel pass, maximizing Ada Lovelace Tensor Core utilization and reducing driver dispatch overhead.

4. **Tensor Core 128 Alignment & SwiGLU Fusion**:
   All model dimensions (`d_model`, `d_head`, `d_ff`, `vocab_size`) are padded to multiples of 128 for optimal hardware alignment. SwiGLU activations boost arithmetic intensity.

---

## 2-Stage Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. HARNESS SUITE (Stage 1: harness.py)                      │
│    ├── Phase 1: Micro-Batch OOM-Safe Sweep                  │
│    ├── Phase 2: BEAM Compiler Search (BEAM=0 -> 2 -> 4)     │
│    └── Phase 3: SwiGLU Activation Fusion                    │
└──────────────────────────────┬──────────────────────────────┘
                               │ Writes best_config.json
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. MAIN PRODUCTION TRAINER (Stage 2: train_production.py)  │
│    ├── Loads best_config.json                               │
│    ├── Streams dataset via np.memmap                        │
│    ├── Cosine LR Decay + Warmup Schedule                    │
│    └── Safetensors Checkpointing & Validation Loss Evaluation │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start & Usage Instructions

### 1. Environment Setup

Ensure `uv` is installed and the environment is active:
```bash
# Install dependencies
uv sync

# Verify linting & formatting
./lint.sh
```

### 2. Standalone Benchmark Test (`train.py`)

Run `train.py` to quickly benchmark step time, GFLOPS, and MFU % over benchmark steps:
```bash
python train.py
```

### 3. Stage 1: Harness Optimization Suite (`harness.py`)

#### Run the Full Transient Optimization Suite:
Automatically sweeps micro-batch sizes, BEAM compiler levels, and SwiGLU activation fusion, locking winning parameters into `best_config.json`:
```bash
uv run python harness.py --run-suite
```

#### Run BEAM & SwiGLU Suite Directly (Skip Micro-Batch Sweep):
To keep fixed micro-batch size (`MICRO_BATCH_SIZE=64`, `GRAD_ACCUMULATION_STEPS=4`) and go directly to BEAM compiler search & SwiGLU optimization:
```bash
BEAM_DEV_TIMEOUT=5 uv run python harness.py --run-suite --skip-batch-sweep
```

#### Run Micro-Batch Sweep Only:
```bash
python harness.py --sweep-batch
```

#### Run Baseline Harness Execution:
```bash
python harness.py
```

### 4. Stage 2: Production Model Training (`train_production.py`)

To launch a full production training run using `best_config.json`:

```bash
# Train 125M Model (500 steps default)
python train_production.py --model-size 125M --total-steps 500

# Specify custom checkpoint directory and eval interval
python train_production.py --model-size 125M --total-steps 2000 --eval-interval 100 --checkpoint-dir checkpoints
```

---

## Configuration Reference (`config.json` / `best_config.json`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `MICRO_BATCH_SIZE` | `int` | `64` | Physical micro-batch size per GPU step |
| `GRAD_ACCUMULATION_STEPS` | `int` | `4` | Number of gradient accumulation steps |
| `DEFAULT_FLOAT` | `str` | `"BFLOAT16"` | Default tensor precision (`"BFLOAT16"`, `"HALF"`, or `"FLOAT"`) |
| `ALLOW_TF32` | `int` | `1` | Enable TensorFloat-32 math on Ada/Ampere GPUs |
| `BEAM` | `int` | `0` | BEAM compiler search depth (`0`, `2`, `4`) |
| `JIT` | `int` | `1` | Enable `@TinyJit` graph compilation |
| `USE_SWIGLU` | `int` | `1` | Enable SwiGLU gated activation MLP blocks |
| `USE_ROPE` | `int` | `1` | Enable Rotary Position Embeddings |
| `PAD_VOCAB_MULTIPLE` | `int` | `128` | Vocab size padding alignment multiple |
| `SEQUENCE_LENGTH` | `int` | `256` | Input token sequence length |
| `LEARNING_RATE` | `float` | `0.001` | Peak learning rate for Cosine decay schedule |
| `D_MODEL` | `int` | `768` | Transformer hidden dimension (125M model) |
| `N_LAYERS` | `int` | `12` | Number of Transformer layers |
| `N_HEADS` | `int` | `12` | Number of attention heads |
| `D_FF` | `int` | `3072` | Feed-forward inner dimension |

---

## Verification & Troubleshooting

- **Linting & Code Formatting**: Run `./lint.sh` before submitting code changes.
- **Check score telemetry**: Detailed execution telemetry is exported to `score.json` after harness runs.
