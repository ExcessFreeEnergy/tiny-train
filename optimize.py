#!/usr/bin/env python3
"""
optimize.py - Automated Memory-Bound Optimization Loop with TUI Visualizer.

Tunes training configurations for tinygrad 15M Parameter Transformer:
  - Detects memory-bound stalls (Arithmetic Intensity, memory_bound_kernel_pct).
  - Scales BATCH_SIZE to overcome memory bandwidth latency.
  - Explores BEAM search (0 -> 2 -> 4 -> 8) for L1/L2 cache locality.
  - Tests Precision Compression (DEFAULT_FLOAT = BFLOAT16 / HALF with ALLOW_TF32=1).
  - Provides a live TUI Visualizer Dashboard (--tui / --viz).
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

from harness import run_harness

# Import rich components for live TUI
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def load_json(filepath: str, fallback: dict) -> dict:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return fallback


def save_json(filepath: str, data: dict):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def compute_throughput(config: dict, metrics: dict) -> float:
    batch_size = config.get("BATCH_SIZE", 64)
    step_time_ms = metrics.get("step_time_ms", 9999.0)
    if step_time_ms <= 0:
        return 0.0
    return round((batch_size / (step_time_ms / 1000.0)), 2)


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

    # 1. Header
    header_text = Text()
    header_text.append(" 🚀 TINYGRAD AUTOMATED OPTIMIZATION LOOP ", style="bold white on blue")
    header_text.append(f" | RTX 4090 (24GB) | Step {iteration}/{max_steps}", style="bold cyan")
    layout["header"].update(Panel(header_text, style="blue"))

    # 2. Config Panel
    cfg_table = Table(title="Configuration State", show_header=True, header_style="bold yellow", expand=True)
    cfg_table.add_column("Parameter", style="cyan")
    cfg_table.add_column("Active Config", style="white")
    cfg_table.add_column("Best Config", style="bold green")

    keys = ["BATCH_SIZE", "BEAM", "DEFAULT_FLOAT", "ALLOW_TF32", "JIT", "SEQUENCE_LENGTH"]
    for k in keys:
        cur_v = str(current_config.get(k, "-"))
        best_v = str(best_config.get(k, "-"))
        style = "bold green" if cur_v != best_v else "white"
        cfg_table.add_row(k, Text(cur_v, style=style), best_v)

    layout["config_panel"].update(Panel(cfg_table, border_style="yellow"))

    # 3. Telemetry Panel
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

    tel_table.add_row("Step Time (ms)", c_step, b_step)
    tel_table.add_row("Throughput (smp/s)", c_tput, b_tput)
    tel_table.add_row("Peak Compute (GFLOPS)", f"{current_metrics.get('peak_gflops', 0):.1f}", f"{best_metrics.get('peak_gflops', 0):.1f}")
    tel_table.add_row("Mem Bandwidth (GB/s)", f"{current_metrics.get('avg_bandwidth_gbps', 0):.1f}", f"{best_metrics.get('avg_bandwidth_gbps', 0):.1f}")
    tel_table.add_row("Memory Stall Pct", c_stall, b_stall)
    tel_table.add_row("Optimizer Status", Text(status_str, style=status_style), best_metrics.get("status", "-"))

    layout["telemetry_panel"].update(Panel(tel_table, border_style="magenta"))

    # 4. History Table
    hist_table = Table(title="Trial History", show_header=True, header_style="bold cyan", expand=True)
    hist_table.add_column("#", style="dim", width=4)
    hist_table.add_column("Applied Change", style="yellow")
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


def propose_next_config(current_config: dict, metrics: dict, trial: int) -> Tuple_Config_Change:
    """Decision engine selecting next candidate hyperparameter configuration based on memory-bound stall analysis."""
    next_cfg = copy.deepcopy(current_config)
    status = metrics.get("status", "UNKNOWN")
    stall_pct = metrics.get("memory_bound_kernel_pct", 0.0)
    nan_detected = metrics.get("nan_detected", False)

    # Strategy 1: Precision Compression
    if trial == 1:
        next_cfg["DEFAULT_FLOAT"] = "HALF"
        next_cfg["ALLOW_TF32"] = 1
        return next_cfg, "Lever 1: Compress precision to HALF (ALLOW_TF32=1)"
    
    if trial == 2 and nan_detected:
        next_cfg["DEFAULT_FLOAT"] = "BFLOAT16"
        next_cfg["ALLOW_TF32"] = 1
        return next_cfg, "Lever 1: Switch to BFLOAT16 (numerical stability)"

    # Strategy 2: Scale BATCH_SIZE if MEMORY_BOUND or stall_pct > 30%
    cur_bs = next_cfg.get("BATCH_SIZE", 64)
    if status == "MEMORY_BOUND" or stall_pct > 30.0 or trial in [3, 4]:
        if cur_bs < 512:
            new_bs = cur_bs * 2
            next_cfg["BATCH_SIZE"] = new_bs
            next_cfg["MICROBATCH_SIZE"] = new_bs
            return next_cfg, f"Lever 2: Scale BATCH_SIZE {cur_bs} -> {new_bs} (overcome VRAM latency)"

    # Strategy 3: BEAM Compiler Search
    cur_beam = next_cfg.get("BEAM", 0)
    if cur_beam == 0:
        next_cfg["BEAM"] = 2
        return next_cfg, "Lever 3: Set BEAM=2 (search L1/L2 cache locality)"
    elif cur_beam == 2:
        next_cfg["BEAM"] = 4
        return next_cfg, "Lever 3: Set BEAM=4 (search loop unrolling & upcasting)"
    elif cur_beam == 4:
        next_cfg["BEAM"] = 8
        return next_cfg, "Lever 3: Set BEAM=8 (deep kernel layout search)"

    # Strategy 4: Microbatch / Grad Accumulation tuning
    cur_micro = next_cfg.get("MICROBATCH_SIZE", cur_bs)
    if cur_bs > 128 and cur_micro == cur_bs:
        next_cfg["MICROBATCH_SIZE"] = cur_bs // 2
        next_cfg["GRAD_ACCUMULATION_STEPS"] = 2
        return next_cfg, f"Lever 4: Split MICROBATCH_SIZE to {cur_bs // 2} (grad accum x2)"

    # Default fallback
    next_cfg["BATCH_SIZE"] = cur_bs + 32
    next_cfg["MICROBATCH_SIZE"] = next_cfg["BATCH_SIZE"]
    return next_cfg, f"Increment BATCH_SIZE to {next_cfg['BATCH_SIZE']}"


class Tuple_Config_Change:
    pass


def main():
    parser = argparse.ArgumentParser(description="Automated Memory-Bound Optimization Loop with TUI Visualizer")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum number of optimization trials")
    parser.add_argument("--tui", "--viz", dest="use_tui", action="store_true", default=False, help="Enable interactive TUI dashboard visualizer")
    args = parser.parse_args()

    config_path = "config.json"
    best_config_path = "best_config.json"

    # Baseline configuration
    active_config = load_json(config_path, {
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
    })

    print("🚀 Starting Automated Optimization Loop...")
    print("Baseline Run...")
    save_json(config_path, active_config)
    baseline_metrics = run_harness(config_path)

    best_config = copy.deepcopy(active_config)
    best_metrics = copy.deepcopy(baseline_metrics)
    best_tput = compute_throughput(best_config, best_metrics)

    save_json(best_config_path, best_config)

    history = [{
        "trial": 0,
        "change": "Baseline",
        "metrics": baseline_metrics,
        "throughput": best_tput,
        "outcome": "BASELINE",
    }]

    consecutive_non_improving = 0

    if args.use_tui and RICH_AVAILABLE:
        console = Console()
        live = Live(render_tui_layout(0, args.max_steps, active_config, best_config, baseline_metrics, best_metrics, history), refresh_per_second=4, console=console)
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

            # Decision Logic: Accept if throughput improved & no NaNs
            is_improvement = (candidate_tput > best_tput * 1.02) and not nan_detected

            if is_improvement:
                best_tput = candidate_tput
                best_config = copy.deepcopy(candidate_config)
                best_metrics = copy.deepcopy(metrics)
                active_config = copy.deepcopy(candidate_config)
                save_json(best_config_path, best_config)
                outcome = "✅ ACCEPTED"
                consecutive_non_improving = 0
            else:
                outcome = "❌ REJECTED (NaN)" if nan_detected else "⚠️ REJECTED (No Speedup)"
                save_json(config_path, best_config)  # Revert config
                consecutive_non_improving += 1

            history.append({
                "trial": trial,
                "change": change_desc,
                "metrics": metrics,
                "throughput": candidate_tput,
                "outcome": outcome,
            })

            if live:
                live.update(render_tui_layout(trial, args.max_steps, active_config, best_config, metrics, best_metrics, history))

            if consecutive_non_improving >= 3:
                print(f"\nOptimization converged. Reached plateau after {trial} iterations.")
                break

    finally:
        if live:
            live.stop()

    print("\n=======================================================")
    print("🏆 AUTOMATED OPTIMIZATION LOOP COMPLETE!")
    print("=======================================================")
    print(f"Best Throughput: {best_tput:.1f} samples/sec")
    print(f"Best Configuration saved to '{best_config_path}':")
    print(json.dumps(best_config, indent=2))
    print("\nBest Telemetry Metrics:")
    print(json.dumps(best_metrics, indent=2))
    print("=======================================================\n")


if __name__ == "__main__":
    main()
