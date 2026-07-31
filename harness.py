#!/usr/bin/env python3
"""
harness.py - Subprocess execution wrapper & telemetry parser for tinygrad training payload.
Parses kernel-level Arithmetic Intensity and Memory-Bound Stall Metrics.
"""

import json
import os
import re
import subprocess
import sys
import time


def load_config(config_path: str = "config.json") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "BEAM": 0,
        "ALLOW_TF32": 1,
        "DEFAULT_FLOAT": "FLOAT",
        "JIT": 1,
        "BATCH_SIZE": 64,
        "MICROBATCH_SIZE": 64,
        "GRAD_ACCUMULATION_STEPS": 1,
        "SEQUENCE_LENGTH": 256,
        "LEARNING_RATE": 1e-3,
        "NUM_STEPS": 20,
    }


def parse_telemetry_from_output(output_text: str) -> dict:
    """Parse JSON telemetry block and scan tinygrad DEBUG=2 kernel lines for Arithmetic Intensity."""
    base_metrics = {}
    
    # 1. Parse JSON block from train.py if present
    match = re.search(r"=== HARNESS TELEMETRY METRICS ===\s*(\{.*?\})\s*=================================", output_text, re.DOTALL)
    if match:
        try:
            base_metrics = json.loads(match.group(1))
        except Exception:
            pass

    # 2. Kernel-level DEBUG=2 parsing for Arithmetic Intensity & Stall Detection
    # Example format: "... 0.12 ms ... 1500.2 GFLOPS ... 850.4 GB/s"
    kernel_regex = re.compile(r"(\d+\.\d+|\d+)\s*ms.*?\s*(\d+\.\d+|\d+)\s*GFLOPS.*?\s*(\d+\.\d+|\d+)\s*GB/s")
    
    total_time_ms = 0.0
    mem_bound_kernels = 0
    total_kernels = 0
    gflops_list = []
    gbps_list = []

    for line in output_text.splitlines():
        k_match = kernel_regex.search(line)
        if k_match:
            time_ms, gflops, gbps = map(float, k_match.groups())
            total_time_ms += time_ms
            total_kernels += 1
            gflops_list.append(gflops)
            gbps_list.append(gbps)

            # RTX 4090 Memory-Bound Stall Signature: High Bandwidth (>600 GB/s) but Low Compute (<15,000 GFLOPS)
            if gbps > 600.0 and gflops < 15000.0:
                mem_bound_kernels += 1

    # Fallbacks for step time & loss if JSON block missing
    nan_detected = "NaN/Inf detected" in output_text or "nan" in output_text.lower()
    
    if "step_time_ms" not in base_metrics:
        step_times = [float(m) for m in re.findall(r"step_time=([0-9.]+)\s*ms", output_text)]
        avg_step_ms = float(sum(step_times[2:]) / len(step_times[2:])) if len(step_times) > 2 else (float(sum(step_times) / len(step_times)) if step_times else 9999.0)
        base_metrics["step_time_ms"] = round(avg_step_ms, 3)

    if "final_loss" not in base_metrics:
        losses = [float(m) for m in re.findall(r"loss=([0-9.]+)", output_text)]
        base_metrics["final_loss"] = round(losses[-1], 4) if losses else 999.0

    if "peak_gflops" not in base_metrics or base_metrics["peak_gflops"] == 0:
        base_metrics["peak_gflops"] = round(max(gflops_list), 1) if gflops_list else 0.0

    if "avg_bandwidth_gbps" not in base_metrics or base_metrics["avg_bandwidth_gbps"] == 0:
        base_metrics["avg_bandwidth_gbps"] = round(sum(gbps_list) / max(1, len(gbps_list)), 1) if gbps_list else 0.0

    # Arithmetic Intensity calculation
    avg_gflops = sum(gflops_list) / max(1, len(gflops_list)) if gflops_list else base_metrics.get("peak_gflops", 0)
    avg_gbps = sum(gbps_list) / max(1, len(gbps_list)) if gbps_list else max(1.0, base_metrics.get("avg_bandwidth_gbps", 1.0))
    arithmetic_intensity = round(avg_gflops / max(1.0, avg_gbps), 2)
    
    stall_ratio = round((mem_bound_kernels / max(1, total_kernels)) * 100.0, 1)
    status = "MEMORY_BOUND" if stall_ratio > 40.0 else "COMPUTE_OPTIMIZED"

    base_metrics.update({
        "total_kernels": total_kernels,
        "mem_bound_kernels": mem_bound_kernels,
        "memory_bound_kernel_pct": stall_ratio,
        "arithmetic_intensity": arithmetic_intensity,
        "status": status,
        "nan_detected": base_metrics.get("nan_detected", nan_detected),
        "jit_active": base_metrics.get("jit_active", "Warmup complete" in output_text),
    })

    return base_metrics


def run_harness(config_path: str = "config.json") -> dict:
    config = load_config(config_path)
    print("=== Tinygrad Training Optimization Harness ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    # Set environment variables
    env = os.environ.copy()
    env["DEBUG"] = "2"
    env["BEAM"] = str(config.get("BEAM", 0))
    env["ALLOW_TF32"] = str(config.get("ALLOW_TF32", 1))
    env["DEFAULT_FLOAT"] = str(config.get("DEFAULT_FLOAT", "FLOAT"))
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
    
    if proc.returncode != 0 and "NaN/Inf detected" not in stderr:
        print(f"Error during execution:\n{stderr[-2000:]}")

    metrics = parse_telemetry_from_output(stdout + "\n" + stderr)
    if proc.returncode != 0:
        metrics["nan_detected"] = True

    # Save score.json
    score_file = "score.json"
    with open(score_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- SCORE TELEMETRY RESULTS ---")
    print(json.dumps(metrics, indent=2))
    print("-------------------------------\n")
    print(f"Saved results to '{score_file}'")

    return metrics


if __name__ == "__main__":
    run_harness()
