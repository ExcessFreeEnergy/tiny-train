#!/usr/bin/env python3
"""
harness.py - Subprocess execution wrapper, OOM-Safe Micro-Batch Sweeper, and Telemetry Parser.
Supports Transient 3-Phase Suite Execution: Batch Optimization -> BEAM Compiler Search -> SwiGLU Fusion.
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time


def load_config(config_path: str = "config.json") -> dict:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {
        "MICRO_BATCH_SIZE": 16,
        "GRAD_ACCUMULATION_STEPS": 4,
        "DEFAULT_FLOAT": "BFLOAT16",
        "ALLOW_TF32": 1,
        "BEAM": 0,
        "JIT": 1,
        "USE_SWIGLU": 0,
        "USE_ROPE": 1,
        "PAD_VOCAB_MULTIPLE": 128,
        "SEQUENCE_LENGTH": 256,
        "LEARNING_RATE": 1e-3,
        "NUM_STEPS": 20,
        "VOCAB_SIZE": 29362,
        "D_MODEL": 768,
        "N_LAYERS": 12,
        "N_HEADS": 12,
        "D_FF": 3072,
    }


def parse_telemetry_from_output(output_text: str) -> dict:
    """Parse JSON telemetry block and scan tinygrad DEBUG=2 kernel lines for Arithmetic Intensity."""
    base_metrics = {}

    match = re.search(r"=== HARNESS TELEMETRY METRICS ===\s*(\{.*?\})\s*=================================", output_text, re.DOTALL)
    if match:
        try:
            base_metrics = json.loads(match.group(1))
        except Exception:
            pass

    kernel_regex = re.compile(r"(\d+\.\d+|\d+)\s*ms.*?\s*(\d+\.\d+|\d+)\s*GFLOPS.*?\s*(\d+\.\d+|\d+)\s*GB/s")

    mem_bound_kernels = 0
    total_kernels = 0
    gflops_list = []
    gbps_list = []

    for line in output_text.splitlines():
        k_match = kernel_regex.search(line)
        if k_match:
            _, gflops, gbps = map(float, k_match.groups())
            total_kernels += 1
            gflops_list.append(gflops)
            gbps_list.append(gbps)

            if gbps > 600.0 and gflops < 15000.0:
                mem_bound_kernels += 1

    nan_detected = "NaN/Inf detected" in output_text or "nan" in output_text.lower()
    oom_detected = "OutOfMemory" in output_text or "CUDA error: out of memory" in output_text or "OOM" in output_text

    if "step_time_ms" not in base_metrics:
        step_times = [float(m) for m in re.findall(r"step_time=([0-9.]+)\s*ms", output_text)]
        avg_step_ms = float(sum(step_times)) / len(step_times) if step_times else 9999.0
        base_metrics["step_time_ms"] = round(avg_step_ms, 3)

    if "final_loss" not in base_metrics:
        losses = [float(m) for m in re.findall(r"loss=([0-9.]+)", output_text)]
        base_metrics["final_loss"] = round(losses[-1], 4) if losses else 999.0

    if "peak_gflops" not in base_metrics or base_metrics["peak_gflops"] == 0:
        base_metrics["peak_gflops"] = round(max(gflops_list), 1) if gflops_list else 0.0

    if "mfu_pct" not in base_metrics:
        base_metrics["mfu_pct"] = round((base_metrics.get("peak_gflops", 0) / 330000.0) * 100.0, 2)

    if "avg_bandwidth_gbps" not in base_metrics or base_metrics["avg_bandwidth_gbps"] == 0:
        base_metrics["avg_bandwidth_gbps"] = round(sum(gbps_list) / max(1, len(gbps_list)), 1) if gbps_list else 0.0

    avg_gflops = sum(gflops_list) / max(1, len(gflops_list)) if gflops_list else base_metrics.get("peak_gflops", 0)
    avg_gbps = sum(gbps_list) / max(1, len(gbps_list)) if gbps_list else max(1.0, base_metrics.get("avg_bandwidth_gbps", 1.0))
    arithmetic_intensity = round(avg_gflops / max(1.0, avg_gbps), 2)

    stall_ratio = round((mem_bound_kernels / max(1, total_kernels)) * 100.0, 1)
    status = "MEMORY_BOUND" if stall_ratio > 40.0 else "COMPUTE_OPTIMIZED"

    base_metrics.update(
        {
            "total_kernels": total_kernels,
            "mem_bound_kernels": mem_bound_kernels,
            "memory_bound_kernel_pct": stall_ratio,
            "arithmetic_intensity": arithmetic_intensity,
            "status": status,
            "nan_detected": base_metrics.get("nan_detected", nan_detected),
            "oom_detected": oom_detected,
            "jit_active": base_metrics.get("jit_active", "Warmup complete" in output_text),
        }
    )

    return base_metrics


def run_harness(config_path: str = "config.json") -> dict:
    config = load_config(config_path)
    print("=== Tinygrad Training Optimization Harness ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    env = os.environ.copy()
    env["DEBUG"] = "2"
    env["TINYCACHE"] = "1"
    env["BEAM"] = str(config.get("BEAM", 0))
    env["ALLOW_TF32"] = str(config.get("ALLOW_TF32", 1))
    env["DEFAULT_FLOAT"] = str(config.get("DEFAULT_FLOAT", "BFLOAT16"))
    env["JIT"] = str(config.get("JIT", 1))

    cmd = [sys.executable, "train.py"]
    print(f"\nExecuting payload: {' '.join(cmd)}")
    t0 = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    stdout, stderr = proc.communicate()
    t1 = time.time()

    print(f"Subprocess completed with return code {proc.returncode} in {t1 - t0:.2f}s")

    metrics = parse_telemetry_from_output(stdout + "\n" + stderr)
    if proc.returncode != 0:
        if "OutOfMemory" in stderr or "OOM" in stderr:
            metrics["oom_detected"] = True
        else:
            metrics["nan_detected"] = True

    score_file = "score.json"
    with open(score_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- SCORE TELEMETRY RESULTS ---")
    print(json.dumps(metrics, indent=2))
    print("-------------------------------\n")
    print(f"Saved results to '{score_file}'")

    return metrics


def find_optimal_batch_size(base_config: dict, target_effective_batch: int = 256) -> int:
    """Run OOM-Safe Micro-Batch Sweep to discover optimal hardware micro-batch size."""
    print("\n🔍 === Phase 1: Micro-Batch Optimization Sweep ===")
    test_batch = 16
    best_throughput = 0.0
    best_batch = 16

    while test_batch <= 256:
        config = copy.deepcopy(base_config)
        config["MICRO_BATCH_SIZE"] = test_batch
        config["GRAD_ACCUMULATION_STEPS"] = max(1, target_effective_batch // test_batch)

        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n[SWEEP] Testing MICRO_BATCH_SIZE={test_batch} (GRAD_ACCUM={config['GRAD_ACCUMULATION_STEPS']})...")
        metrics = run_harness("config.json")

        if metrics.get("oom_detected") or metrics.get("nan_detected"):
            print(f"[OOM/Instability] Hit limit at MICRO_BATCH_SIZE={test_batch}")
            break

        step_time_ms = metrics.get("step_time_ms", 9999.0)
        eff_batch = test_batch * config["GRAD_ACCUMULATION_STEPS"]
        throughput = (eff_batch / (step_time_ms / 1000.0)) if step_time_ms > 0 else 0.0
        ai = metrics.get("arithmetic_intensity", 0.0)

        print(f"[SWEEP RESULT] MICRO_BATCH_SIZE={test_batch}: {throughput:.1f} smp/s | AI: {ai:.2f} | Step Time: {step_time_ms:.2f}ms")

        if throughput > best_throughput * 1.05:
            best_throughput = throughput
            best_batch = test_batch
            test_batch *= 2
        else:
            print(f"[SATURATED] Throughput plateaued around MICRO_BATCH_SIZE={best_batch} (Gain < 5%)")
            break

    print(f"\n✅ Winning Micro-Batch Size Locked: MICRO_BATCH_SIZE={best_batch} ({best_throughput:.1f} smp/s)")
    final_cfg = copy.deepcopy(base_config)
    final_cfg["MICRO_BATCH_SIZE"] = best_batch
    final_cfg["GRAD_ACCUMULATION_STEPS"] = max(1, target_effective_batch // best_batch)
    with open("config.json", "w") as f:
        json.dump(final_cfg, f, indent=2)

    return best_batch


def run_transient_suite(base_config: dict) -> dict:
    """Run Transient Harness Suite: Batch Optimization -> BEAM Search -> SwiGLU Fusion."""
    print("\n=======================================================")
    print("🚀 STARTING TRANSIENT HARNESS OPTIMIZATION SUITE")
    print("Sequence: Batch Optimization -> BEAM Compiler Search -> SwiGLU Fusion")
    print("=======================================================\n")

    current_config = copy.deepcopy(base_config)

    # 1. Batch Optimization Phase
    winning_batch = find_optimal_batch_size(current_config)
    current_config["MICRO_BATCH_SIZE"] = winning_batch
    current_config["GRAD_ACCUMULATION_STEPS"] = max(1, 256 // winning_batch)

    # 2. BEAM Compiler Search Phase (BEAM=0 -> 2 -> 4)
    print("\n🔍 === Phase 2: BEAM Compiler Search Sweep (BEAM=0 -> 2 -> 4) ===")
    best_beam = 0
    best_step_time = 99999.0

    for beam_val in [0, 2, 4]:
        print(f"\n[BEAM SWEEP] Evaluating BEAM={beam_val}...")
        current_config["BEAM"] = beam_val
        with open("config.json", "w") as f:
            json.dump(current_config, f, indent=2)

        m = run_harness("config.json")
        step_ms = m.get("step_time_ms", 99999.0)
        gflops = m.get("peak_gflops", 0.0)
        mfu = m.get("mfu_pct", 0.0)

        print(f"[BEAM RESULT] BEAM={beam_val}: Step Time={step_ms:.2f}ms | GFLOPS={gflops:.1f} | MFU={mfu:.2f}%")

        if not m.get("nan_detected") and not m.get("oom_detected") and (step_ms < best_step_time or beam_val == 0):
            best_step_time = step_ms
            best_beam = beam_val

    current_config["BEAM"] = best_beam
    print(f"\n✅ Winning BEAM Compiler Level Locked: BEAM={best_beam} ({best_step_time:.2f}ms)")

    # 3. SwiGLU Activation Fusion Phase
    print("\n🔍 === Phase 3: SwiGLU Activation Fusion Evaluation ===")
    for swiglu_val in [1, 0]:
        current_config["USE_SWIGLU"] = swiglu_val
        with open("config.json", "w") as f:
            json.dump(current_config, f, indent=2)

        print(f"\n[SWIGLU EVAL] Testing USE_SWIGLU={swiglu_val}...")
        m = run_harness("config.json")
        step_ms = m.get("step_time_ms", 99999.0)
        gflops = m.get("peak_gflops", 0.0)
        mfu = m.get("mfu_pct", 0.0)
        print(f"[SWIGLU RESULT] USE_SWIGLU={swiglu_val}: Step Time={step_ms:.2f}ms | GFLOPS={gflops:.1f} | MFU={mfu:.2f}%")

        if step_ms <= best_step_time:
            best_step_time = step_ms
            print(f"✅ SwiGLU Option USE_SWIGLU={swiglu_val} selected!")
            break

    # Save final optimized configuration
    with open("config.json", "w") as f:
        json.dump(current_config, f, indent=2)
    with open("best_config.json", "w") as f:
        json.dump(current_config, f, indent=2)

    print("\n=======================================================")
    print("🏆 TRANSIENT HARNESS SUITE COMPLETE!")
    print("Optimal Configuration Saved to 'best_config.json':")
    print(json.dumps(current_config, indent=2))
    print("=======================================================\n")

    return current_config


def main():
    parser = argparse.ArgumentParser(description="Tinygrad Harness with Transient Optimization Suite")
    parser.add_argument("--sweep-batch", action="store_true", default=False, help="Run OOM-safe micro-batch discovery sweep")
    parser.add_argument("--run-suite", action="store_true", default=False, help="Run full transient 3-phase optimization suite")
    args = parser.parse_args()

    cfg = load_config()
    if args.run_suite:
        run_transient_suite(cfg)
    elif args.sweep_batch:
        find_optimal_batch_size(cfg)
    else:
        run_harness()


if __name__ == "__main__":
    main()
