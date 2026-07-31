#!/usr/bin/env python3
"""
optimize.py - Automated Memory-Bound Optimization Loop with TUI Visualizer.

Follows the 4-Phase Optimization Strategy:
  - Phase 1: Precision & Tensor Core Unlock (DEFAULT_FLOAT=BFLOAT16, ALLOW_TF32=1)
  - Phase 2: Micro-Batch Saturation (OOM-Safe Micro-Batch Sweep with BEAM=0)
  - Phase 3: BEAM Layout Search (BEAM=4/8/16 on locked micro-batch shape)
  - Phase 4: SwiGLU Activation Fusion (USE_SWIGLU=1 for higher arithmetic intensity)
"""

import argparse
import copy
import json
import os
from typing import Any

from harness import find_optimal_batch_size, run_harness

# Import rich components for live TUI
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def load_json(filepath: str, fallback: dict) -> dict:
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return fallback


def save_json(filepath: str, data: dict):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def compute_throughput(config: dict, metrics: dict) -> float:
    mb = config.get("MICRO_BATCH_SIZE", config.get("BATCH_SIZE", 64))
    ga = config.get("GRAD_ACCUMULATION_STEPS", 1)
    eff_batch = mb * ga
    step_time_ms = metrics.get("step_time_ms", 9999.0)
    if step_time_ms <= 0:
        return 0.0
    return round((eff_batch / (step_time_ms / 1000.0)), 2)


def render_tui_layout(
    iteration: int,
    max_steps: int,
    current_config: dict,
    best_config: dict,
    current_metrics: dict,
    best_metrics: dict,
    history: list,
) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="middle", size=12),
        Layout(name="history"),
    )
    layout["middle"].split_row(
        Layout(name="config_panel", ratio=1),
        Layout(name="telemetry_panel", ratio=1),
    )

    header_text = Text()
    header_text.append(" 🚀 TINYGRAD 4-PHASE AUTOMATED OPTIMIZER ", style="bold white on blue")
    header_text.append(f" | RTX 4090 (24GB) | Trial {iteration}/{max_steps}", style="bold cyan")
    layout["header"].update(Panel(header_text, style="blue"))

    cfg_table = Table(title="Configuration State", show_header=True, header_style="bold yellow", expand=True)
    cfg_table.add_column("Parameter", style="cyan")
    cfg_table.add_column("Active Config", style="white")
    cfg_table.add_column("Best Config", style="bold green")

    keys = ["MICRO_BATCH_SIZE", "GRAD_ACCUMULATION_STEPS", "BEAM", "DEFAULT_FLOAT", "ALLOW_TF32", "USE_SWIGLU"]
    for k in keys:
        cur_v = str(current_config.get(k, "-"))
        best_v = str(best_config.get(k, "-"))
        style = "bold green" if cur_v != best_v else "white"
        cfg_table.add_row(k, Text(cur_v, style=style), best_v)

    layout["config_panel"].update(Panel(cfg_table, border_style="yellow"))

    tel_table = Table(title="Hardware & Stall Metrics", show_header=True, header_style="bold magenta", expand=True)
    tel_table.add_column("Metric", style="cyan")
    tel_table.add_column("Current Run", style="white")
    tel_table.add_column("Best Benchmark", style="bold green")

    c_step = f"{current_metrics.get('step_time_ms', 0):.2f} ms"
    b_step = f"{best_metrics.get('step_time_ms', 0):.2f} ms"

    c_tput = f"{compute_throughput(current_config, current_metrics):.1f} smp/s"
    b_tput = f"{compute_throughput(best_config, best_metrics):.1f} smp/s"

    c_stall = f"{current_metrics.get('memory_bound_kernel_pct', 0):.1f}%"
    b_stall = f"{best_metrics.get('memory_bound_kernel_pct', 0):.1f}%"

    status_str = current_metrics.get("status", "UNKNOWN")
    status_style = "bold red" if status_str == "MEMORY_BOUND" else "bold green"

    c_mfu = f"{current_metrics.get('mfu_pct', 0.0):.2f}%"
    b_mfu = f"{best_metrics.get('mfu_pct', 0.0):.2f}%"

    tel_table.add_row("Step Time (ms)", c_step, b_step)
    tel_table.add_row("Throughput (smp/s)", c_tput, b_tput)
    tel_table.add_row("Peak Compute (GFLOPS)", f"{current_metrics.get('peak_gflops', 0):.1f}", f"{best_metrics.get('peak_gflops', 0):.1f}")
    tel_table.add_row("Model FLOPs Util (MFU)", c_mfu, b_mfu)
    tel_table.add_row("Arithmetic Intensity", f"{current_metrics.get('arithmetic_intensity', 0):.2f}", f"{best_metrics.get('arithmetic_intensity', 0):.2f}")
    tel_table.add_row("Memory Stall Pct", c_stall, b_stall)
    tel_table.add_row("Optimizer Status", Text(status_str, style=status_style), best_metrics.get("status", "-"))

    layout["telemetry_panel"].update(Panel(tel_table, border_style="magenta"))

    hist_table = Table(title="Optimization Phase History", show_header=True, header_style="bold cyan", expand=True)
    hist_table.add_column("#", style="dim", width=4)
    hist_table.add_column("Phase & Change", style="yellow")
    hist_table.add_column("Step Time", style="white")
    hist_table.add_column("Throughput", style="bold cyan")
    hist_table.add_column("Stall Pct", style="magenta")
    hist_table.add_column("Outcome", style="bold")

    for h in history[-8:]:
        st_style = "green" if "ACCEPTED" in h["outcome"] else ("red" if "REJECTED" in h["outcome"] else "yellow")
        hist_table.add_row(
            str(h["trial"]),
            h["change"],
            f"{h['metrics'].get('step_time_ms', 0):.2f} ms",
            f"{h['throughput']:.1f} smp/s",
            f"{h['metrics'].get('memory_bound_kernel_pct', 0):.1f}%",
            Text(h["outcome"], style=st_style),
        )

    layout["history"].update(Panel(hist_table, border_style="cyan"))
    return layout


def propose_next_config(current_config: dict[str, Any], metrics: dict[str, Any], trial: int) -> tuple[dict[str, Any], str]:
    """4-Phase Decision Engine."""
    next_cfg = copy.deepcopy(current_config)

    # Phase 1: Precision Unlock
    if trial == 1:
        next_cfg["DEFAULT_FLOAT"] = "BFLOAT16"
        next_cfg["ALLOW_TF32"] = 1
        return next_cfg, "Phase 1: BFLOAT16 + ALLOW_TF32 Precision Unlock"

    # Phase 2: Micro-batch scaling handled in harness sweep or trial 2
    if trial == 2:
        return next_cfg, "Phase 2: Micro-Batch Saturation Locked"

    # Phase 3: BEAM Search on locked micro-batch shape
    cur_beam = next_cfg.get("BEAM", 0)
    if trial == 3 or cur_beam == 0:
        next_cfg["BEAM"] = 4
        return next_cfg, "Phase 3: BEAM=4 Layout Search (L1/L2 cache tiling)"
    elif trial == 4 or cur_beam == 4:
        next_cfg["BEAM"] = 8
        return next_cfg, "Phase 3: BEAM=8 Layout Search"

    # Phase 4: SwiGLU Activation Fusion
    if trial == 5 and next_cfg.get("USE_SWIGLU", 0) == 0:
        next_cfg["USE_SWIGLU"] = 1
        return next_cfg, "Phase 4: SwiGLU Activation Fusion (Higher Arithmetic Intensity)"

    return next_cfg, "Phase 4 Completed"


def main():
    parser = argparse.ArgumentParser(description="Automated Memory-Bound Optimization Loop with TUI Visualizer")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum number of optimization trials")
    parser.add_argument("--tui", "--viz", dest="use_tui", action="store_true", default=False, help="Enable interactive TUI dashboard visualizer")
    parser.add_argument("--skip-sweep", action="store_true", default=False, help="Skip Phase 2 micro-batch discovery sweep")
    args = parser.parse_args()

    config_path = "config.json"
    best_config_path = "best_config.json"

    # Baseline configuration
    active_config = load_json(
        config_path,
        {
            "MICRO_BATCH_SIZE": 64,
            "GRAD_ACCUMULATION_STEPS": 4,
            "DEFAULT_FLOAT": "BFLOAT16",
            "ALLOW_TF32": 1,
            "BEAM": 0,
            "JIT": 1,
            "USE_SWIGLU": 0,
            "SEQUENCE_LENGTH": 256,
            "LEARNING_RATE": 1e-3,
            "NUM_STEPS": 20,
        },
    )

    print("🚀 Starting 4-Phase Automated Optimization Loop...")
    print("\n--- Phase 1: Precision & Baseline Verification ---")
    active_config["DEFAULT_FLOAT"] = "BFLOAT16"
    active_config["ALLOW_TF32"] = 1
    save_json(config_path, active_config)
    baseline_metrics = run_harness(config_path)

    # Phase 2: OOM-Safe Micro-Batch Sweep
    if not args.skip_sweep:
        print("\n--- Phase 2: OOM-Safe Micro-Batch Saturation Sweep ---")
        find_optimal_batch_size(active_config, target_effective_batch=256)
        active_config = load_json(config_path, active_config)

    best_config = copy.deepcopy(active_config)
    best_metrics = copy.deepcopy(baseline_metrics)
    best_tput = compute_throughput(best_config, best_metrics)
    save_json(best_config_path, best_config)

    history = [
        {
            "trial": 0,
            "change": "Phase 1 & 2 Saturation",
            "metrics": baseline_metrics,
            "throughput": best_tput,
            "outcome": "LOCKED",
        }
    ]

    if args.use_tui and RICH_AVAILABLE:
        console = Console()
        live = Live(
            render_tui_layout(0, args.max_steps, active_config, best_config, baseline_metrics, best_metrics, history), refresh_per_second=4, console=console
        )
        live.start()
    else:
        live = None

    try:
        for trial in range(1, args.max_steps + 1):
            candidate_config, change_desc = propose_next_config(active_config, best_metrics, trial)
            save_json(config_path, candidate_config)

            if live:
                live.update(render_tui_layout(trial, args.max_steps, candidate_config, best_config, {"status": "RUNNING..."}, best_metrics, history))

            metrics = run_harness(config_path)
            candidate_tput = compute_throughput(candidate_config, metrics)
            nan_detected = metrics.get("nan_detected", False)

            is_improvement = (candidate_tput > best_tput * 1.02) and not nan_detected

            if is_improvement:
                best_tput = candidate_tput
                best_config = copy.deepcopy(candidate_config)
                best_metrics = copy.deepcopy(metrics)
                active_config = copy.deepcopy(candidate_config)
                save_json(best_config_path, best_config)
                outcome = "✅ ACCEPTED"
            else:
                outcome = "❌ REJECTED (NaN)" if nan_detected else "⚠️ REJECTED (No Speedup)"
                save_json(config_path, best_config)

            history.append(
                {
                    "trial": trial,
                    "change": change_desc,
                    "metrics": metrics,
                    "throughput": candidate_tput,
                    "outcome": outcome,
                }
            )

            if live:
                live.update(render_tui_layout(trial, args.max_steps, active_config, best_config, metrics, best_metrics, history))

    finally:
        if live:
            live.stop()

    print("\n=======================================================")
    print("🏆 4-PHASE AUTOMATED OPTIMIZATION COMPLETE!")
    print("=======================================================")
    print(f"Best Throughput: {best_tput:.1f} samples/sec")
    print(f"Best Configuration saved to '{best_config_path}':")
    print(json.dumps(best_config, indent=2))
    print("\nBest Telemetry Metrics:")
    print(json.dumps(best_metrics, indent=2))
    print("=======================================================\n")


if __name__ == "__main__":
    main()
