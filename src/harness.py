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


def load_config(config_path: str = "conf/config.json") -> dict:
    if not os.path.exists(config_path):
        for alt_path in ["config.json", "conf/best_config.json", "best_config.json"]:
            if os.path.exists(alt_path):
                config_path = alt_path
                break
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {
        "MICRO_BATCH_SIZE": 16,
        "GRAD_ACCUMULATION_STEPS": 4,
        "DEFAULT_FLOAT": "BFLOAT16",
        "ALLOW_TF32": 1,
        "BEAM": 2,
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
        step_times = [float(m) for m in re.findall(r"(?:time|step_time)=([0-9.]+)\s*ms", output_text)]
        avg_step_ms = float(sum(step_times)) / len(step_times) if step_times else 9999.0
        base_metrics["step_time_ms"] = round(avg_step_ms, 3)

    if "final_loss" not in base_metrics:
        losses = [float(m) for m in re.findall(r"loss=([0-9.]+)", output_text)]
        base_metrics["final_loss"] = round(losses[-1], 4) if losses else 999.0

    if "peak_gflops" not in base_metrics or base_metrics["peak_gflops"] == 0:
        gflops_vals = [float(m) for m in re.findall(r"GFLOPS=([0-9.]+)", output_text)]
        base_metrics["peak_gflops"] = round(max(gflops_vals), 1) if gflops_vals else (round(max(gflops_list), 1) if gflops_list else 0.0)

    if "mfu_pct" not in base_metrics or base_metrics["mfu_pct"] == 0:
        mfu_vals = [float(m) for m in re.findall(r"MFU=([0-9.]+)\%", output_text)]
        base_metrics["mfu_pct"] = round(max(mfu_vals), 2) if mfu_vals else round((base_metrics.get("peak_gflops", 0) / 330000.0) * 100.0, 2)

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
            "jit_active": base_metrics.get("jit_active", "Warmup complete" in output_text or "JIT Warmup complete" in output_text),
        }
    )

    return base_metrics


def run_harness(config_path: str = "conf/config.json", timeout_sec: int = 3600, beam_dev_timeout: int | None = None) -> dict:
    config = load_config(config_path)
    print("=== Tinygrad Training Optimization Harness ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    env = os.environ.copy()
    env["DEBUG"] = "2"
    env["TINYCACHE"] = "1"
    # Prevents buffer overflows on 700+ kernel queues
    env["HCQ"] = "1"
    env["BEAM"] = str(config.get("BEAM", 2))
    if beam_dev_timeout is not None:
        env["BEAM_DEV_TIMEOUT"] = str(beam_dev_timeout)
    else:
        env["BEAM_DEV_TIMEOUT"] = os.environ.get("BEAM_DEV_TIMEOUT", "60")
    env["ALLOW_TF32"] = str(config.get("ALLOW_TF32", 1))
    env["DEFAULT_FLOAT"] = str(config.get("DEFAULT_FLOAT", "BFLOAT16"))
    env["JIT"] = str(config.get("JIT", 1))

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = f"{project_root}:{os.path.dirname(__file__)}:{env.get('PYTHONPATH', '')}"

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_production.py")
    num_steps = str(config.get("NUM_STEPS", 10))
    cmd = [sys.executable, script_path, "--model-size", "125M", "--total-steps", num_steps]
    print(f"\nExecuting payload: {' '.join(cmd)}")
    print(f"⏱️ Hard Timeout Limit: {timeout_sec}s ({timeout_sec // 60} minutes max per run)\n")
    t0 = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    output_lines = []
    last_heartbeat = time.time()
    timed_out = False

    while True:
        line = proc.stdout.readline()
        now = time.time()
        elapsed = now - t0

        if line:
            output_lines.append(line)
            sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {line}")
            sys.stdout.flush()
            last_heartbeat = now

        if proc.poll() is not None:
            break

        if elapsed > timeout_sec:
            timed_out = True
            print(f"\n🚨 [TIMEOUT GUARDIAN] Subprocess exceeded {timeout_sec}s limit! Killing hanging process...")
            proc.kill()
            proc.wait()
            break

        if now - last_heartbeat > 10.0:
            sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] ⏳ Compiling / Executing... (Elapsed: {elapsed:.1f}s / {timeout_sec}s max)\n")
            sys.stdout.flush()
            last_heartbeat = now

        time.sleep(0.1)

    full_output = "".join(output_lines)
    t1 = time.time()

    if timed_out:
        print(f"❌ Subprocess TIMED OUT after {t1 - t0:.2f}s")
        metrics = {
            "step_time_ms": 99999.0,
            "final_loss": 999.0,
            "peak_gflops": 0.0,
            "mfu_pct": 0.0,
            "nan_detected": True,
            "oom_detected": True,
            "status": "TIMED_OUT",
        }
    else:
        print(f"✅ Subprocess completed with return code {proc.returncode} in {t1 - t0:.2f}s")
        metrics = parse_telemetry_from_output(full_output)
        if proc.returncode != 0:
            if "OutOfMemory" in full_output or "OOM" in full_output or "NV_ERR_NO_MEMORY" in full_output:
                metrics["oom_detected"] = True
            else:
                metrics["nan_detected"] = True

    score_file = "conf/score.json"
    os.makedirs(os.path.dirname(score_file), exist_ok=True)
    with open(score_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- SCORE TELEMETRY RESULTS ---")
    print(json.dumps(metrics, indent=2))
    print("-------------------------------\n")
    print(f"Saved results to '{score_file}'")

    return metrics


def find_optimal_batch_size(base_config: dict, target_effective_batch: int = 256, beam_dev_timeout: int | None = None) -> int:
    """Run OOM-Safe Micro-Batch Sweep to discover optimal hardware micro-batch size."""
    print("\n🔍 === Phase 1: Micro-Batch Optimization Sweep ===")
    test_batch = 16
    best_throughput = 0.0
    best_batch = 16

    while test_batch <= 256:
        config = copy.deepcopy(base_config)
        config["MICRO_BATCH_SIZE"] = test_batch
        config["GRAD_ACCUMULATION_STEPS"] = max(1, target_effective_batch // test_batch)
        config["BEAM"] = 2
        config["NUM_STEPS"] = 3

        os.makedirs("conf", exist_ok=True)
        with open("conf/config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\n[SWEEP] Testing MICRO_BATCH_SIZE={test_batch} (GRAD_ACCUM={config['GRAD_ACCUMULATION_STEPS']})...")
        metrics = run_harness("conf/config.json", beam_dev_timeout=beam_dev_timeout)

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
    os.makedirs("conf", exist_ok=True)
    with open("conf/config.json", "w") as f:
        json.dump(final_cfg, f, indent=2)

    return best_batch


def run_transient_suite(base_config: dict, skip_batch_sweep: bool = False, timeout_sec: int = 3600, beam_dev_timeout: int | None = None) -> dict:
    """Run Transient Harness Suite: Batch Optimization -> BEAM Search -> SwiGLU Fusion."""
    print("\n=======================================================")
    print("🚀 STARTING TRANSIENT HARNESS OPTIMIZATION SUITE")
    if skip_batch_sweep:
        print("Sequence: [SKIPPED Batch Sweep] -> BEAM Compiler Search -> SwiGLU Fusion")
    else:
        print("Sequence: Batch Optimization -> BEAM Compiler Search -> SwiGLU Fusion")
    print(f"⏱️ Subprocess Timeout Limit: {timeout_sec}s ({timeout_sec // 60} mins)")
    if beam_dev_timeout == 0:
        print("⚡ BEAM Compiler Timeout: DISABLED (BEAM_DEV_TIMEOUT=0)")
    elif beam_dev_timeout is not None:
        print(f"⏱️ BEAM Compiler Timeout: {beam_dev_timeout}s (BEAM_DEV_TIMEOUT={beam_dev_timeout})")
    print("=======================================================\n")

    current_config = copy.deepcopy(base_config)

    # 1. Batch Optimization Phase
    if skip_batch_sweep:
        print(f"⏩ [SKIP BATCH SWEEP] Using existing batch settings: MICRO_BATCH_SIZE={current_config.get('MICRO_BATCH_SIZE', 64)}")
    else:
        winning_batch = find_optimal_batch_size(current_config, beam_dev_timeout=beam_dev_timeout)
        current_config["MICRO_BATCH_SIZE"] = winning_batch
        current_config["GRAD_ACCUMULATION_STEPS"] = max(1, 256 // winning_batch)

    # 2. BEAM Compiler Search Phase (Skip 0, jump to 2 -> 4)
    print("\n🔍 === Phase 2: BEAM Compiler Search Sweep (BEAM=0 -> 2 -> 4) ===")
    best_beam = 2
    best_step_time = 99999.0

    for beam_val in [2, 4]:
        print(f"\n[BEAM SWEEP] Evaluating BEAM={beam_val}...")
        current_config["BEAM"] = beam_val
        current_config["NUM_STEPS"] = 3
        os.makedirs("conf", exist_ok=True)
        with open("conf/config.json", "w") as f:
            json.dump(current_config, f, indent=2)

        m = run_harness("conf/config.json", timeout_sec=timeout_sec, beam_dev_timeout=beam_dev_timeout)
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
        os.makedirs("conf", exist_ok=True)
        with open("conf/config.json", "w") as f:
            json.dump(current_config, f, indent=2)

        print(f"\n[SWIGLU EVAL] Testing USE_SWIGLU={swiglu_val}...")
        m = run_harness("conf/config.json", timeout_sec=timeout_sec, beam_dev_timeout=beam_dev_timeout)
        step_ms = m.get("step_time_ms", 99999.0)
        gflops = m.get("peak_gflops", 0.0)
        mfu = m.get("mfu_pct", 0.0)
        print(f"[SWIGLU RESULT] USE_SWIGLU={swiglu_val}: Step Time={step_ms:.2f}ms | GFLOPS={gflops:.1f} | MFU={mfu:.2f}%")

        if step_ms <= best_step_time:
            best_step_time = step_ms
            print(f"✅ SwiGLU Option USE_SWIGLU={swiglu_val} selected!")
            break

    # Save final optimized configuration
    os.makedirs("conf", exist_ok=True)
    with open("conf/config.json", "w") as f:
        json.dump(current_config, f, indent=2)
    with open("conf/best_config.json", "w") as f:
        json.dump(current_config, f, indent=2)

    print("\n=======================================================")
    print("🏆 TRANSIENT HARNESS SUITE COMPLETE!")
    print("Optimal Configuration Saved to 'conf/best_config.json':")
    print(json.dumps(current_config, indent=2))
    print("=======================================================\n")

    return current_config


def main():
    parser = argparse.ArgumentParser(description="Tinygrad Harness with Transient Optimization Suite")
    parser.add_argument("--sweep-batch", action="store_true", default=False, help="Run OOM-safe micro-batch discovery sweep")
    parser.add_argument("--run-suite", action="store_true", default=False, help="Run full transient 3-phase optimization suite")
    parser.add_argument("--skip-batch-sweep", action="store_true", default=False, help="Skip Phase 1 micro-batch sweep in suite execution")
    parser.add_argument("--timeout", type=int, default=3600, help="Subprocess timeout limit in seconds (default: 3600s / 60 mins)")
    parser.add_argument(
        "--disable-beam-timeout", "--no-beam-timeout", action="store_true", default=False, help="Disable BEAM compiler dev timeout (sets BEAM_DEV_TIMEOUT=0)"
    )
    parser.add_argument("--beam-dev-timeout", type=int, default=None, help="BEAM compiler dev timeout in seconds (default: 60, set to 0 to disable)")
    args = parser.parse_args()

    beam_timeout = 0 if args.disable_beam_timeout else args.beam_dev_timeout

    cfg = load_config()
    if args.run_suite:
        run_transient_suite(cfg, skip_batch_sweep=args.skip_batch_sweep, timeout_sec=args.timeout, beam_dev_timeout=beam_timeout)
    elif args.sweep_batch:
        find_optimal_batch_size(cfg, beam_dev_timeout=beam_timeout)
    else:
        run_harness("conf/config.json", timeout_sec=args.timeout, beam_dev_timeout=beam_timeout)


if __name__ == "__main__":
    main()
