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


def load_config(config_path: str | None = None) -> dict:
    candidates = []
    if config_path:
        candidates.append(config_path)
    candidates.extend(["conf/best_config.json", "conf/config.json", "best_config.json", "config.json"])
    for path in candidates:
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    else:
        cfg = {
            "MICRO_BATCH_SIZE": 16,
            "GRAD_ACCUMULATION_STEPS": 4,
            "DEFAULT_FLOAT": "BFLOAT16",
            "ALLOW_TF32": 1,
            "BEAM": 4,
            "TC": 1,
            "TENSOR_CORES": 1,
            "JIT": 1,
            "USE_SWIGLU": 1,
            "USE_ROPE": 1,
            "PAD_VOCAB_MULTIPLE": 128,
            "PAD_VOCAB_POWER_OF_2": 1,
            "SEQUENCE_LENGTH": 256,
            "LEARNING_RATE": 1e-3,
            "NUM_STEPS": 20,
            "VOCAB_SIZE": 13970,
            "D_MODEL": 768,
            "N_LAYERS": 12,
            "N_HEADS": 12,
            "D_FF": 3072,
        }
    if "BEAM" in os.environ:
        try:
            cfg["BEAM"] = int(os.environ["BEAM"])
        except ValueError:
            cfg["BEAM"] = os.environ["BEAM"]
    return cfg


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
    oom_detected = "OutOfMemory" in output_text or "CUDA error: out of memory" in output_text or "OOM" in output_text or "NV_ERR_NO_MEMORY" in output_text
    wait_timeout_detected = (
        "Wait timeout" in output_text
        or "signal is not set to" in output_text
        or "timeline_signal.wait" in output_text
        or "TRAINER ERROR" in output_text
        or "weakref" in output_text
    )
    recursion_detected = "RecursionError" in output_text or "recursion limit" in output_text or "recursion depth" in output_text

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
            "wait_timeout_detected": wait_timeout_detected,
            "recursion_detected": recursion_detected,
            "jit_active": base_metrics.get("jit_active", "Warmup complete" in output_text or "JIT Warmup complete" in output_text),
        }
    )

    return base_metrics


def run_harness(
    config_path: str = "conf/config.json",
    timeout_sec: int = 3600,
    beam_dev_timeout: int | None = None,
    debug_level: int | None = None,
    disable_debug: bool = False,
) -> dict:
    config = load_config(config_path)
    print("=== Tinygrad Training Optimization Harness ===")
    print(f"Configuration: {json.dumps(config, indent=2)}")

    env = os.environ.copy()
    if disable_debug or debug_level == 0:
        env["DEBUG"] = "0"
    elif debug_level is not None:
        env["DEBUG"] = str(debug_level)
    else:
        env["DEBUG"] = os.environ.get("DEBUG", "2")

    env["TINYCACHE"] = "1"
    # Prevents buffer overflows on 700+ kernel queues
    env["HCQ"] = "1"
    env["BEAM"] = os.environ.get("BEAM", str(config.get("BEAM", 4)))
    env["TC"] = str(config.get("TC", 1))
    env["TENSOR_CORES"] = str(config.get("TENSOR_CORES", 1))
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
    cmd = [sys.executable, script_path, "--model-size", "125M", "--total-steps", num_steps, "--config", config_path]
    if env["DEBUG"] == "0":
        cmd.append("--disable-debug")
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

        if not line and proc.poll() is not None:
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
            "nan_detected": False,
            "oom_detected": False,
            "wait_timeout_detected": True,
            "incompatible_config": True,
            "error_type": "HARDWARE_TIMEOUT",
            "status": "TIMED_OUT",
        }
    else:
        print(f"✅ Subprocess completed with return code {proc.returncode} in {t1 - t0:.2f}s")
        metrics = parse_telemetry_from_output(full_output)
        if proc.returncode != 0:
            metrics["incompatible_config"] = True
            if metrics.get("oom_detected"):
                metrics["error_type"] = "OOM"
            elif metrics.get("wait_timeout_detected"):
                metrics["error_type"] = "HARDWARE_WAIT_TIMEOUT"
            elif metrics.get("recursion_detected"):
                metrics["error_type"] = "AST_RECURSION_LIMIT"
            elif metrics.get("nan_detected"):
                metrics["error_type"] = "NAN_INSTABILITY"
            else:
                metrics["error_type"] = "SUBPROCESS_FAILED"

    score_file = "conf/score.json"
    os.makedirs(os.path.dirname(score_file), exist_ok=True)
    with open(score_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- SCORE TELEMETRY RESULTS ---")
    print(json.dumps(metrics, indent=2))
    print("-------------------------------\n")
    print(f"Saved results to '{score_file}'")

    return metrics


def find_optimal_batch_size(
    base_config: dict,
    target_effective_batch: int | None = None,
    grad_accum_candidates: list[int] | None = None,
    max_effective_batch: int = 512,
    full_sweep: bool = True,
    beam_dev_timeout: int | None = None,
    debug_level: int | None = None,
    disable_debug: bool = False,
) -> tuple[int, int]:
    """Run OOM-Safe Micro-Batch & Grad Accumulation Grid Sweep to discover optimal hardware settings."""
    print("\n🔍 === Phase 1: Micro-Batch & Grad Accumulation Grid Sweep ===")
    if grad_accum_candidates is None:
        grad_accum_candidates = [1, 2, 4, 8, 16, 32]

    micro_batch_candidates = [16, 32, 64, 128, 256]
    best_throughput = 0.0
    best_batch = int(base_config.get("MICRO_BATCH_SIZE", 16))
    best_accum = int(base_config.get("GRAD_ACCUMULATION_STEPS", 4))
    max_failed_accum: int | None = None

    for test_batch in micro_batch_candidates:
        batch_has_valid_run = False
        for test_accum in grad_accum_candidates:
            if max_failed_accum is not None and test_accum >= max_failed_accum:
                print(
                    f"⏩ [SKIP CUTOFF] MICRO_BATCH_SIZE={test_batch}, GRAD_ACCUM={test_accum} skipped "
                    f"(GRAD_ACCUM >= {max_failed_accum} failed at lower/equal micro-batch size)"
                )
                break

            eff_batch = test_batch * test_accum
            if eff_batch > max_effective_batch and test_accum > 1:
                continue

            config = copy.deepcopy(base_config)
            config["MICRO_BATCH_SIZE"] = test_batch
            config["GRAD_ACCUMULATION_STEPS"] = test_accum
            if "BEAM" in os.environ:
                try:
                    config["BEAM"] = int(os.environ["BEAM"])
                except ValueError:
                    config["BEAM"] = os.environ["BEAM"]
            else:
                config["BEAM"] = base_config.get("BEAM", 2)
            config["NUM_STEPS"] = 3

            os.makedirs("conf", exist_ok=True)
            with open("conf/config.json", "w") as f:
                json.dump(config, f, indent=2)

            print(f"\n[SWEEP] Testing MICRO_BATCH_SIZE={test_batch}, GRAD_ACCUM={test_accum} (Eff Batch={eff_batch})...")
            metrics = run_harness(
                "conf/config.json",
                beam_dev_timeout=beam_dev_timeout,
                debug_level=debug_level,
                disable_debug=disable_debug,
            )

            if metrics.get("incompatible_config") or metrics.get("oom_detected") or metrics.get("nan_detected") or metrics.get("wait_timeout_detected"):
                err_type = metrics.get("error_type", "INCOMPATIBLE_CONFIG")
                if max_failed_accum is None or test_accum < max_failed_accum:
                    max_failed_accum = test_accum
                print(
                    f"⚠️ [INCOMPATIBLE CONFIG] MICRO_BATCH_SIZE={test_batch}, GRAD_ACCUM={test_accum} failed ({err_type}). "
                    f"Capping max GRAD_ACCUM < {max_failed_accum} for all larger micro-batch sizes."
                )
                break

            batch_has_valid_run = True
            step_time_ms = metrics.get("step_time_ms", 9999.0)
            throughput = (eff_batch / (step_time_ms / 1000.0)) if step_time_ms > 0 else 0.0
            ai = metrics.get("arithmetic_intensity", 0.0)

            print(f"[SWEEP RESULT] MB={test_batch}, GA={test_accum} (Eff={eff_batch}): {throughput:.1f} smp/s | AI: {ai:.2f} | Step Time: {step_time_ms:.2f}ms")

            if throughput > best_throughput:
                best_throughput = throughput
                best_batch = test_batch
                best_accum = test_accum

        if not batch_has_valid_run:
            print(f"[OOM Limit] Stopping sweep at MICRO_BATCH_SIZE={test_batch}")
            break

    print(
        f"\n✅ Winning Configuration Locked: MICRO_BATCH_SIZE={best_batch}, GRAD_ACCUM={best_accum} ({best_throughput:.1f} smp/s, Eff Batch={best_batch * best_accum})"
    )
    final_cfg = copy.deepcopy(base_config)
    final_cfg["MICRO_BATCH_SIZE"] = best_batch
    final_cfg["GRAD_ACCUMULATION_STEPS"] = best_accum
    os.makedirs("conf", exist_ok=True)
    with open("conf/config.json", "w") as f:
        json.dump(final_cfg, f, indent=2)
    with open("conf/best_config.json", "w") as f:
        json.dump(final_cfg, f, indent=2)

    return best_batch, best_accum


def run_transient_suite(
    base_config: dict,
    target_effective_batch: int | None = None,
    grad_accum_candidates: list[int] | None = None,
    full_sweep: bool = True,
    skip_batch_sweep: bool = False,
    timeout_sec: int = 3600,
    beam_dev_timeout: int | None = None,
    max_beam: int | None = None,
    debug_level: int | None = None,
    disable_debug: bool = False,
) -> dict:
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
        winning_batch, winning_accum = find_optimal_batch_size(
            current_config,
            target_effective_batch=target_effective_batch,
            grad_accum_candidates=grad_accum_candidates,
            full_sweep=full_sweep,
            beam_dev_timeout=beam_dev_timeout,
            debug_level=debug_level,
            disable_debug=disable_debug,
        )
        current_config["MICRO_BATCH_SIZE"] = winning_batch
        current_config["GRAD_ACCUMULATION_STEPS"] = winning_accum

    # 2. BEAM Compiler Search Phase
    best_beam = 0
    best_step_time = 99999.0

    beam_candidates = [0, 1, 2, 4]
    if "BEAM" in os.environ:
        try:
            env_beam_val = int(os.environ["BEAM"])
            if env_beam_val not in beam_candidates:
                beam_candidates.append(env_beam_val)
                beam_candidates.sort()
        except ValueError:
            pass

    if max_beam is not None:
        beam_candidates = [b for b in beam_candidates if b <= max_beam]
        if not beam_candidates:
            beam_candidates = [0]

    beam_str = " -> ".join(map(str, beam_candidates))
    print(f"\n🔍 === Phase 2: BEAM Compiler Search Sweep (BEAM={beam_str}) ===")

    for beam_val in beam_candidates:
        print(f"\n[BEAM SWEEP] Evaluating BEAM={beam_val}...")
        current_config["BEAM"] = beam_val
        current_config["NUM_STEPS"] = 3
        os.makedirs("conf", exist_ok=True)
        with open("conf/config.json", "w") as f:
            json.dump(current_config, f, indent=2)

        m = run_harness(
            "conf/config.json",
            timeout_sec=timeout_sec,
            beam_dev_timeout=beam_dev_timeout,
            debug_level=debug_level,
            disable_debug=disable_debug,
        )
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
    for swiglu_val in [1]:
        current_config["USE_SWIGLU"] = swiglu_val
        os.makedirs("conf", exist_ok=True)
        with open("conf/config.json", "w") as f:
            json.dump(current_config, f, indent=2)

        print(f"\n[SWIGLU EVAL] Testing USE_SWIGLU={swiglu_val}...")
        m = run_harness(
            "conf/config.json",
            timeout_sec=timeout_sec,
            beam_dev_timeout=beam_dev_timeout,
            debug_level=debug_level,
            disable_debug=disable_debug,
        )
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
    parser.add_argument(
        "--grad-accum-list",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of candidate gradient accumulation steps to test (default: 1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--target-effective-batch",
        "--effective-batch",
        type=int,
        default=None,
        help="Target effective batch size for batch sweep (default: 256 or derived from config)",
    )
    parser.add_argument(
        "--no-full-sweep",
        dest="full_sweep",
        action="store_false",
        default=True,
        help="Stop batch sweep early when throughput gain < 5%%",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Subprocess timeout limit in seconds (default: 3600s / 60 mins)")
    parser.add_argument(
        "--disable-beam-timeout", "--no-beam-timeout", action="store_true", default=False, help="Disable BEAM compiler dev timeout (sets BEAM_DEV_TIMEOUT=0)"
    )
    parser.add_argument("--beam-dev-timeout", type=int, default=None, help="BEAM compiler dev timeout in seconds (default: 60, set to 0 to disable)")
    parser.add_argument(
        "--max-beam",
        "--max-beam-level",
        type=int,
        default=None,
        help="Maximum BEAM compiler level to evaluate during BEAM search sweep (e.g., 1)",
    )
    parser.add_argument("--disable-debug", "--no-debug", action="store_true", default=False, help="Disable tinygrad debug output (sets DEBUG=0)")
    parser.add_argument("--debug-level", "--debug", type=int, default=None, help="Set tinygrad DEBUG level (e.g. 0, 1, 2)")
    args = parser.parse_args()

    beam_timeout = 0 if args.disable_beam_timeout else args.beam_dev_timeout
    if beam_timeout is None and "BEAM_TIMEOUT" in os.environ:
        try:
            beam_timeout = int(os.environ["BEAM_TIMEOUT"])
        except ValueError:
            pass
    if beam_timeout is None and "BEAM_DEV_TIMEOUT" in os.environ:
        try:
            beam_timeout = int(os.environ["BEAM_DEV_TIMEOUT"])
        except ValueError:
            pass

    max_beam = args.max_beam
    if max_beam is None and "MAX_BEAM" in os.environ:
        try:
            max_beam = int(os.environ["MAX_BEAM"])
        except ValueError:
            pass

    grad_accum_candidates = [int(x.strip()) for x in args.grad_accum_list.split(",") if x.strip().isdigit()]

    cfg = load_config()
    if args.run_suite:
        run_transient_suite(
            cfg,
            target_effective_batch=args.target_effective_batch,
            grad_accum_candidates=grad_accum_candidates,
            full_sweep=args.full_sweep,
            skip_batch_sweep=args.skip_batch_sweep,
            timeout_sec=args.timeout,
            beam_dev_timeout=beam_timeout,
            max_beam=max_beam,
            debug_level=args.debug_level,
            disable_debug=args.disable_debug,
        )
    elif args.sweep_batch:
        find_optimal_batch_size(
            cfg,
            target_effective_batch=args.target_effective_batch,
            grad_accum_candidates=grad_accum_candidates,
            full_sweep=args.full_sweep,
            beam_dev_timeout=beam_timeout,
            debug_level=args.debug_level,
            disable_debug=args.disable_debug,
        )
    else:
        run_harness(
            "conf/config.json",
            timeout_sec=args.timeout,
            beam_dev_timeout=beam_timeout,
            debug_level=args.debug_level,
            disable_debug=args.disable_debug,
        )


if __name__ == "__main__":
    main()
