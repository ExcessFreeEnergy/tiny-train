#!/usr/bin/env python3
"""
harness.py - Subprocess execution wrapper & telemetry parser for tinygrad training payload.
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
        "DEFAULT_FLOAT": "HALF",
        "JIT": 1,
        "BATCH_SIZE": 64,
        "MICROBATCH_SIZE": 64,
        "GRAD_ACCUMULATION_STEPS": 1,
        "SEQUENCE_LENGTH": 256,
        "LEARNING_RATE": 1e-3,
        "NUM_STEPS": 50,
    }


def parse_telemetry_from_output(output_text: str) -> dict:
    """Parse JSON telemetry block or calculate fallback from output text."""
    match = re.search(r"=== HARNESS TELEMETRY METRICS ===\s*(\{.*?\})\s*=================================", output_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Fallback parsing
    nan_detected = "NaN/Inf detected" in output_text or "nan" in output_text.lower()
    
    # Extract step times if available
    step_times = [float(m) for m in re.findall(r"step_time=([0-9.]+)\s*ms", output_text)]
    avg_step_ms = float(sum(step_times[2:]) / len(step_times[2:])) if len(step_times) > 2 else (float(sum(step_times) / len(step_times)) if step_times else 9999.0)
    
    # Extract loss values
    losses = [float(m) for m in re.findall(r"loss=([0-9.]+)", output_text)]
    final_loss = losses[-1] if losses else 999.0

    return {
        "step_time_ms": round(avg_step_ms, 3),
        "peak_gflops": 0.0,
        "avg_bandwidth_gbps": 0.0,
        "final_loss": round(final_loss, 4),
        "nan_detected": nan_detected,
        "jit_active": "Warmup complete" in output_text,
    }


def run_harness():
    config = load_config()
    print("=== Tinygrad Training Optimization Harness ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    # Set environment variables
    env = os.environ.copy()
    env["DEBUG"] = "2"
    env["BEAM"] = str(config.get("BEAM", 0))
    env["ALLOW_TF32"] = str(config.get("ALLOW_TF32", 1))
    env["DEFAULT_FLOAT"] = str(config.get("DEFAULT_FLOAT", "HALF"))
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
        sys.exit(proc.returncode)

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
