#!/usr/bin/env python3
"""
chat.py - High-Performance Textual TUI for TinyGrad Interactive LLM Chat.

Usage:
  uv run python chat.py
  uv run python chat.py --model-size 125M --dataset tinystories
  uv run python chat.py --checkpoint checkpoints/model_125m_step_8500.safetensors
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

from rich import box
from rich.table import Table
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Input, Markdown, Static

# Ensure src path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.chat_engine import GPTEngineManager, TelemetryMetrics


def create_params_table(manager: GPTEngineManager) -> Table:
    """Create a formatted Rich Table for active sampling hyperparameters."""
    table = Table(title="⚙️ Active Generation Parameters", box=box.ROUNDED)
    table.add_column("Parameter", style="cyan bold", justify="left")
    table.add_column("Value", style="green bold", justify="left")
    table.add_column("Description", style="white", justify="left")

    table.add_row("Temperature", str(manager.temperature), "Sampling randomness threshold")
    table.add_row("Top-P", str(manager.top_p), "Nucleus sampling probability cutoff")
    table.add_row("Top-K", str(manager.top_k), "Top token filter count")
    table.add_row("Repetition Penalty", str(manager.repetition_penalty), "Logit penalty for context token repeats")
    table.add_row("Max Tokens", str(manager.max_tokens), "Max tokens generated per turn")
    table.add_row("System Prompt", manager.system_prompt or "None", "Active system persona")
    table.add_row("JIT Acceleration", "Enabled" if manager.use_jit else "Disabled", "@TinyJit graph execution")
    table.add_row("Model Scale", f"{manager.model_size} ({manager.num_params:,} params)", "Transformer size & scale")
    table.add_row("Checkpoint", os.path.basename(manager.checkpoint_path), "Safetensors weight file")
    return table


def create_help_table() -> Table:
    """Create a formatted Rich Table for slash command reference."""
    table = Table(title="📖 Slash Command Reference", box=box.ROUNDED)
    table.add_column("Category", style="magenta bold", justify="left")
    table.add_column("Command", style="cyan bold", justify="left")
    table.add_column("Description", style="white", justify="left")

    table.add_row("Generation", "/temp <float>", "Set sampling temperature (e.g. /temp 0.7)")
    table.add_row("", "/top_p <float>", "Set top-P nucleus threshold (e.g. /top_p 0.9)")
    table.add_row("", "/top_k <int>", "Set top-K filter threshold (e.g. /top_k 40)")
    table.add_row("", "/penalty <float>", "Set repetition penalty (e.g. /penalty 1.15)")
    table.add_row("", "/tokens <int>", "Set max generated tokens per turn (e.g. /tokens 256)")
    table.add_row("", "/params", "Print active hyperparameter table")

    table.add_row("Context", "/clear", "Flush KV cache & reset context to step 0")
    table.add_row("", "/system <text>", "Set dynamic system prompt & reset context")
    table.add_row("", "/pop", "Remove last user query & response from cache")
    table.add_row("", "/context", "Display KV cache token usage meter")
    table.add_row("", "/retry", "Regenerate last assistant response")

    table.add_row("File I/O", "/load <path>", "Load local file text as context for next turn")
    table.add_row("", "/save <path>", "Save chat transcript to Markdown file")
    table.add_row("", "/export", "Export raw JSON conversation history")
    table.add_row("", "/exec <cmd>", "Run shell command directly in TUI")

    table.add_row("Telemetry", "/stats", "Toggle live telemetry header overlay")
    table.add_row("", "/bench", "Run automated 100-token latency benchmark")
    table.add_row("", "/profile", "Toggle JIT compilation & execution tracing")

    table.add_row("UI & Session", "/markdown", "Toggle Markdown / Raw text rendering")
    table.add_row("", "/compact", "Toggle compact layout view mode")
    table.add_row("", "/copy", "Copy last assistant response to clipboard")
    table.add_row("", "/exit", "Safely release VRAM & exit back to bash")
    return table


def create_bench_table(res: dict) -> Table:
    """Create a formatted Rich Table for benchmark results."""
    table = Table(title="⚡ Automated Benchmark Results (100 Tokens)", box=box.ROUNDED)
    table.add_column("Metric", style="cyan bold", justify="left")
    table.add_column("Result", style="green bold", justify="left")

    table.add_row("Time To First Token (TTFT)", f"{res['ttft_ms']:.2f} ms")
    table.add_row("Average Generation Speed", f"{res['tok_per_sec']:.1f} tokens/sec")
    table.add_row("Per-Token Latency", f"{res['avg_step_ms']:.2f} ms/token")
    table.add_row("Total Duration", f"{res['total_sec']:.2f} s")
    table.add_row("Est. Memory Bandwidth", f"{res['avg_mem_bw_gbs']:.2f} GB/s")
    table.add_row("VRAM Footprint", f"{res['vram_mb']:.1f} MB")
    return table


def create_context_table(manager: GPTEngineManager) -> Table:
    """Create a formatted Rich Table for context window usage."""
    used = manager.start_pos
    total = manager.max_context
    pct = (used / total) * 100.0
    bar_len = 20
    filled = int(bar_len * (used / total))
    progress_bar = "█" * filled + "░" * (bar_len - filled)

    table = Table(title="🧠 Context Window & KV Cache Usage", box=box.ROUNDED)
    table.add_column("Metric", style="cyan bold", justify="left")
    table.add_column("Value", style="yellow bold", justify="left")

    table.add_row("Usage Meter", f"[{progress_bar}]")
    table.add_row("Tokens Used", f"{used} / {total} tokens ({pct:.1f}%)")
    table.add_row("Remaining Capacity", f"{total - used} tokens")
    table.add_row("Dialogue Turns in Cache", str(len(manager.history)))
    return table


class TelemetryHeader(Static):
    """Header widget displaying live telemetry, context usage meter, and model stats."""

    context_str = reactive("0 / 1024 (0%)")
    ttft_str = reactive("-- ms")
    speed_str = reactive("-- tok/s")
    vram_str = reactive("-- MB")
    status_str = reactive("Ready")

    def __init__(self, model_info: str, **kwargs):
        super().__init__(**kwargs)
        self.model_info = model_info

    def render(self) -> str:
        return (
            f" 🚀 [bold cyan]{self.model_info}[/bold cyan] │ "
            f"Context: [bold green]{self.context_str}[/bold green] │ "
            f"TTFT: [bold yellow]{self.ttft_str}[/bold yellow] │ "
            f"Speed: [bold magenta]{self.speed_str}[/bold magenta] │ "
            f"VRAM: [bold blue]{self.vram_str}[/bold blue] │ "
            f"Status: [bold white]{self.status_str}[/bold white]"
        )


class UserMessage(Static):
    """Widget for user queries."""

    def __init__(self, content: str, use_markdown: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.content = content
        self.use_markdown = use_markdown

    def compose(self) -> ComposeResult:
        if self.use_markdown:
            yield Markdown(f"**You:** {self.content}")
        else:
            yield Static(f"You: {self.content}")


class AssistantMessage(Static):
    """Widget for streaming LLM responses."""

    def __init__(self, title: str = "Assistant", use_markdown: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.use_markdown = use_markdown
        self.raw_text = ""
        self.markdown_widget = Markdown("")
        self.static_widget = Static("")

    def compose(self) -> ComposeResult:
        if self.use_markdown:
            yield self.markdown_widget
        else:
            yield self.static_widget

    def update_text(self, new_text: str):
        self.raw_text = new_text
        if self.use_markdown:
            self.markdown_widget.update(new_text)
        else:
            self.static_widget.update(f"{self.title}: {new_text}")

    def set_markdown_mode(self, enabled: bool):
        self.use_markdown = enabled
        self.update_text(self.raw_text)


class SystemNotice(Static):
    """Widget for system notices, command outputs, tables, and errors."""

    def __init__(self, content: str | Table, **kwargs):
        super().__init__(**kwargs)
        self.content = content

    def compose(self) -> ComposeResult:
        if isinstance(self.content, str):
            yield Markdown(self.content)
        else:
            yield Static(self.content)


class TinyChatApp(App):
    """Main Textual Application for TinyGrad LLM Chat."""

    TITLE = "TinyGrad Interactive LLM Chat"
    SUB_TITLE = "Stateful JIT Warm Engine"
    AUTO_FOCUS = "Input"

    CSS = """
    Screen {
        background: #090d16;
        color: #e2e8f0;
    }

    TelemetryHeader {
        background: #1e293b;
        color: #f8fafc;
        height: 1;
        padding: 0 1;
        border-bottom: heavy #3b82f6;
    }

    #chat-view {
        height: 1fr;
        padding: 1 2;
        scrollbar-size: 1 1;
    }

    UserMessage {
        background: #1e293b 80%;
        color: #f8fafc;
        border-left: solid #38bdf8;
        margin: 1 0 1 6;
        padding: 1 2;
    }

    AssistantMessage {
        background: #0f172a 90%;
        color: #f1f5f9;
        border-left: solid #22c55e;
        margin: 1 6 1 0;
        padding: 1 2;
    }

    SystemNotice {
        background: #1e1b4b 70%;
        color: #c084fc;
        border: solid #818cf8;
        margin: 1 2;
        padding: 1 2;
    }

    .compact-view UserMessage {
        margin: 0 0 0 2;
        padding: 0 1;
    }

    .compact-view AssistantMessage {
        margin: 0 2 0 0;
        padding: 0 1;
    }

    .hidden-header TelemetryHeader {
        display: none;
    }

    #input-container {
        height: 3;
        padding: 0 1;
        background: #0f172a;
        border-top: heavy #334155;
    }

    Input {
        background: #1e293b;
        color: #f8fafc;
        border: none;
    }

    Input:focus {
        border: none;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Exit"),
        ("ctrl+l", "clear_chat", "Clear"),
        ("ctrl+r", "retry_turn", "Retry"),
        ("tab", "focus_input", "Focus Input"),
        ("f1", "show_help", "Help"),
    ]

    def __init__(
        self,
        dataset: str = "tinystories",
        model_size: str = "125M",
        checkpoint_path: str | None = None,
        checkpoint_dir: str = "checkpoints",
        use_jit: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.model_size = model_size
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = checkpoint_dir
        self.use_jit = use_jit

        self.manager: GPTEngineManager | None = None
        self.is_generating = False
        self.use_markdown = True
        self.compact_mode = False
        self.stats_visible = True
        self.last_assistant_text = ""
        self.pending_prompt_prefix = ""

        # Single persistent thread & queue for all TinyGrad & SQLite operations
        self.work_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._inference_worker_loop, daemon=True)

    def compose(self) -> ComposeResult:
        model_label = f"TinyGrad {self.model_size} ({self.dataset})"
        yield TelemetryHeader(model_info=model_label, id="telemetry-header")
        with VerticalScroll(id="chat-view"):
            yield SystemNotice("⏳ Initializing TinyGrad Model Engine & Loading Safetensors Checkpoint...")
        with Container(id="input-container"):
            yield Input(placeholder="Loading model...", id="user-input", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        """Mount hook - start dedicated inference thread and queue load task."""
        self.query_one("#chat-view").anchor()
        self.worker_thread.start()
        self.work_queue.put(("init_model", None))

    def _inference_worker_loop(self) -> None:
        """Persistent worker thread loop processing all inference, reset, and benchmark tasks sequentially."""
        while True:
            item = self.work_queue.get()
            if item is None:
                break

            action, payload = item

            try:
                if action == "init_model":
                    manager = GPTEngineManager(
                        dataset=self.dataset,
                        model_size=self.model_size,
                        checkpoint_path=self.checkpoint_path,
                        checkpoint_dir=self.checkpoint_dir,
                        use_jit=self.use_jit,
                    )
                    self.manager = manager
                    self.call_from_thread(self._on_model_loaded_success)

                elif action == "generate":
                    prompt, asst_widget = payload
                    accumulated_text = ""
                    last_metrics = None

                    for chunk, metrics in self.manager.generate_stream(prompt):
                        accumulated_text += chunk
                        if metrics:
                            last_metrics = metrics

                        self.call_from_thread(asst_widget.update_text, accumulated_text)
                        if last_metrics:
                            self.call_from_thread(self.update_telemetry, last_metrics)
                        self.call_from_thread(self._scroll_chat_end)

                    self.last_assistant_text = accumulated_text
                    self.call_from_thread(self._on_generation_finished)

                elif action == "reset":
                    self.manager.reset_context()
                    self.call_from_thread(self._on_reset_finished)

                elif action == "system_prompt":
                    self.manager.set_system_prompt(payload)
                    self.call_from_thread(self._on_reset_finished)

                elif action == "bench":
                    results = self.manager.run_benchmark(num_tokens=100)
                    self.call_from_thread(self._on_benchmark_complete, results)

            except Exception as e:
                if action == "init_model":
                    self.call_from_thread(self._on_model_loaded_error, str(e))
                elif action == "generate":
                    asst_widget = payload[1]
                    self.call_from_thread(asst_widget.update_text, f"⚠️ Error during generation: {e}")
                    self.call_from_thread(self._on_generation_finished)

            finally:
                self.work_queue.task_done()

    def _on_model_loaded_success(self) -> None:
        """UI callback when model loading finishes."""
        chat_view = self.query_one("#chat-view")
        user_input = self.query_one("#user-input", Input)

        chat_view.mount(
            SystemNotice(
                f"✅ **Model Ready!** Loaded `{self.manager.model_size}` ({self.manager.num_params:,} params) "
                f"from `{os.path.basename(self.manager.checkpoint_path)}` in {self.manager.load_time_ms:.1f} ms.\n\n"
                "💡 Type your prompt or `/help` to view all available slash commands."
            )
        )

        user_input.placeholder = "Type message or /help..."
        user_input.disabled = False
        user_input.focus()
        self.update_telemetry()

    def _on_model_loaded_error(self, err_msg: str) -> None:
        """UI callback when model loading fails."""
        chat_view = self.query_one("#chat-view")
        chat_view.mount(SystemNotice(f"❌ **Failed to load model:** {err_msg}"))

    def _on_reset_finished(self) -> None:
        """UI callback when context reset finishes."""
        self.update_telemetry()

    def update_telemetry(self, metrics: TelemetryMetrics | None = None) -> None:
        """Update telemetry header widget values."""
        if not self.manager:
            return

        header = self.query_one("#telemetry-header", TelemetryHeader)
        pct = (self.manager.start_pos / self.manager.max_context) * 100.0
        header.context_str = f"{self.manager.start_pos} / {self.manager.max_context} ({pct:.1f}%)"

        if metrics:
            header.ttft_str = f"{metrics.ttft_ms:.1f} ms"
            header.speed_str = f"{metrics.tok_per_sec:.1f} tok/s"
            header.vram_str = f"{metrics.vram_mb:.1f} MB"
        else:
            header.vram_str = f"{self.manager.param_bytes / 1e6:.1f} MB"

        header.status_str = "Generating..." if self.is_generating else "Idle"

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle prompt or slash command submission."""
        text = event.value.strip()
        event.input.clear()

        if not text or self.is_generating or not self.manager:
            return

        if text.startswith("/"):
            self.handle_slash_command(text)
        else:
            prompt = text
            if self.pending_prompt_prefix:
                prompt = f"{self.pending_prompt_prefix}\n\n{text}"
                self.pending_prompt_prefix = ""

            self.run_generation(prompt)

    def action_clear_chat(self) -> None:
        self.handle_slash_command("/clear")

    def action_retry_turn(self) -> None:
        self.handle_slash_command("/retry")

    def action_focus_input(self) -> None:
        self.query_one("#user-input", Input).focus()

    def action_show_help(self) -> None:
        self.handle_slash_command("/help")

    def handle_slash_command(self, cmd_text: str) -> None:
        """Parse and execute slash commands."""
        parts = cmd_text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        chat_view = self.query_one("#chat-view")

        # -------------------------------------------------------------
        # Group 1: Generation & Sampling Control
        # -------------------------------------------------------------
        if cmd in ["/temp", "/temperature"]:
            try:
                val = float(arg)
                self.manager.temperature = max(0.0, val)
                chat_view.mount(SystemNotice(f"🌡️ Temperature set to `{self.manager.temperature}`"))
            except ValueError:
                chat_view.mount(SystemNotice("⚠️ Usage: `/temp <float>` (e.g. `/temp 0.7`)"))

        elif cmd == "/top_p":
            try:
                val = float(arg)
                self.manager.top_p = min(1.0, max(0.0, val))
                chat_view.mount(SystemNotice(f"🎯 Top-P (nucleus) set to `{self.manager.top_p}`"))
            except ValueError:
                chat_view.mount(SystemNotice("⚠️ Usage: `/top_p <float>` (e.g. `/top_p 0.9`)"))

        elif cmd == "/top_k":
            try:
                val = int(arg)
                self.manager.top_k = max(0, val)
                chat_view.mount(SystemNotice(f"🔢 Top-K set to `{self.manager.top_k}`"))
            except ValueError:
                chat_view.mount(SystemNotice("⚠️ Usage: `/top_k <int>` (e.g. `/top_k 40`)"))

        elif cmd in ["/penalty", "/rep_penalty", "/repetition_penalty"]:
            try:
                val = float(arg)
                self.manager.repetition_penalty = max(1.0, val)
                chat_view.mount(SystemNotice(f"🔄 Repetition penalty set to `{self.manager.repetition_penalty}`"))
            except ValueError:
                chat_view.mount(SystemNotice("⚠️ Usage: `/penalty <float>` (e.g. `/penalty 1.15`)"))

        elif cmd in ["/tokens", "/max_tokens"]:
            try:
                val = int(arg)
                self.manager.max_tokens = max(1, val)
                chat_view.mount(SystemNotice(f"📏 Max tokens per turn set to `{self.manager.max_tokens}`"))
            except ValueError:
                chat_view.mount(SystemNotice("⚠️ Usage: `/tokens <int>` (e.g. `/tokens 256`)"))

        elif cmd == "/params":
            chat_view.mount(SystemNotice(create_params_table(self.manager)))

        # -------------------------------------------------------------
        # Group 2: Context & KV-Cache Management
        # -------------------------------------------------------------
        elif cmd in ["/clear", "/reset"]:
            self.work_queue.put(("reset", None))
            chat_view.query(UserMessage).remove()
            chat_view.query(AssistantMessage).remove()
            chat_view.query(SystemNotice).remove()
            chat_view.mount(SystemNotice("🧹 **KV Cache & Context Window Cleared.** Conversation reset to step 0."))

        elif cmd == "/system":
            if not arg:
                chat_view.mount(SystemNotice(f"ℹ️ Current system prompt: `{self.manager.system_prompt or 'None'}`"))
            else:
                self.work_queue.put(("system_prompt", arg))
                chat_view.mount(SystemNotice(f"⚙️ **System Prompt Updated:** `{arg}` (Context reset)"))

        elif cmd in ["/pop", "/undo"]:
            success = self.manager.pop_last_turn()
            if success:
                user_msgs = list(chat_view.query(UserMessage))
                asst_msgs = list(chat_view.query(AssistantMessage))
                if asst_msgs:
                    asst_msgs[-1].remove()
                if user_msgs:
                    user_msgs[-1].remove()
                chat_view.mount(SystemNotice("↩️ **Popped last turn.** KV cache position rewound."))
                self.update_telemetry()
            else:
                chat_view.mount(SystemNotice("⚠️ Conversation history is empty. Nothing to pop."))

        elif cmd in ["/context", "/kv"]:
            chat_view.mount(SystemNotice(create_context_table(self.manager)))

        elif cmd in ["/retry", "/regen"]:
            prompt_to_retry = self.manager.retry_last_turn()
            if prompt_to_retry:
                asst_msgs = list(chat_view.query(AssistantMessage))
                if asst_msgs:
                    asst_msgs[-1].remove()
                chat_view.mount(SystemNotice(f'🔄 **Retrying generation for prompt:** "{prompt_to_retry}"'))
                self.update_telemetry()
                self.run_generation(prompt_to_retry, is_retry=True)
            else:
                chat_view.mount(SystemNotice("⚠️ No previous user prompt found to retry."))

        # -------------------------------------------------------------
        # Group 3: File & System I/O
        # -------------------------------------------------------------
        elif cmd == "/load":
            if not arg:
                chat_view.mount(SystemNotice("⚠️ Usage: `/load <path/to/file>`"))
            elif not os.path.exists(arg):
                chat_view.mount(SystemNotice(f"❌ File not found: `{arg}`"))
            else:
                try:
                    with open(arg, encoding="utf-8") as f:
                        file_content = f.read()
                    self.pending_prompt_prefix = f"### File Context ({os.path.basename(arg)}):\n```\n{file_content}\n```"
                    chat_view.mount(
                        SystemNotice(
                            f"📁 **Loaded `{arg}`** ({len(file_content):,} chars, {len(file_content.splitlines()):,} lines).\n"
                            "It will be attached as context to your next message!"
                        )
                    )
                except Exception as e:
                    chat_view.mount(SystemNotice(f"❌ Failed to read file `{arg}`: {e}"))

        elif cmd == "/save":
            out_path = arg or f"chat_transcript_{int(time.time())}.md"
            try:
                lines = [f"# Chat Transcript - {self.manager.model_size}\n"]
                for turn in self.manager.history:
                    lines.append(f"### {turn.role.capitalize()}\n{turn.text}\n")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                chat_view.mount(SystemNotice(f"💾 **Saved chat transcript to `{out_path}`**"))
            except Exception as e:
                chat_view.mount(SystemNotice(f"❌ Failed to save transcript: {e}"))

        elif cmd == "/export":
            out_path = arg or f"chat_export_{int(time.time())}.json"
            try:
                data = [
                    {
                        "role": t.role,
                        "text": t.text,
                        "token_count": len(t.token_ids),
                        "start_pos": t.start_pos,
                        "end_pos": t.end_pos,
                    }
                    for t in self.manager.history
                ]
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                chat_view.mount(SystemNotice(f"📦 **Exported JSON history ({len(data)} turns) to `{out_path}`**"))
            except Exception as e:
                chat_view.mount(SystemNotice(f"❌ Failed to export JSON: {e}"))

        elif cmd in ["/exec", "/sh"]:
            if not arg:
                chat_view.mount(SystemNotice("⚠️ Usage: `/exec <command>` (e.g. `/sh ls -la`)"))
            else:
                try:
                    res = subprocess.run(arg, shell=True, capture_output=True, text=True, timeout=15)
                    output = res.stdout if res.returncode == 0 else f"Error (code {res.returncode}):\n{res.stderr}"
                    chat_view.mount(SystemNotice(f"💻 **Command Output (`{arg}`):**\n```\n{output.strip()}\n```"))
                except Exception as e:
                    chat_view.mount(SystemNotice(f"❌ Execution error: {e}"))

        # -------------------------------------------------------------
        # Group 4: Telemetry & Profiling
        # -------------------------------------------------------------
        elif cmd == "/stats":
            self.stats_visible = not self.stats_visible
            if self.stats_visible:
                self.screen.remove_class("hidden-header")
                chat_view.mount(SystemNotice("📊 Telemetry header visible."))
            else:
                self.screen.add_class("hidden-header")
                chat_view.mount(SystemNotice("📊 Telemetry header hidden."))

        elif cmd == "/bench":
            chat_view.mount(SystemNotice("⚡ Running 100-token latency & memory bandwidth benchmark..."))
            self.work_queue.put(("bench", None))

        elif cmd == "/profile":
            self.manager.profile = not self.manager.profile
            self.manager.timing = not self.manager.timing
            status = "ENABLED" if self.manager.profile else "DISABLED"
            chat_view.mount(SystemNotice(f"🔍 **Graph Execution Profiling & Timing Diagnostics:** `{status}`"))

        # -------------------------------------------------------------
        # Group 5: UI & Session Controls
        # -------------------------------------------------------------
        elif cmd in ["/help", "/?"]:
            chat_view.mount(SystemNotice(create_help_table()))

        elif cmd in ["/markdown", "/raw"]:
            self.use_markdown = not self.use_markdown
            mode = "Markdown" if self.use_markdown else "Raw Text"
            for asst_widget in chat_view.query(AssistantMessage):
                asst_widget.set_markdown_mode(self.use_markdown)
            chat_view.mount(SystemNotice(f"🎨 Message rendering set to **{mode}**."))

        elif cmd == "/compact":
            self.compact_mode = not self.compact_mode
            if self.compact_mode:
                self.screen.add_class("compact-view")
                chat_view.mount(SystemNotice("📐 Compact view mode enabled."))
            else:
                self.screen.remove_class("compact-view")
                chat_view.mount(SystemNotice("📐 Standard view mode restored."))

        elif cmd == "/copy":
            if self.last_assistant_text:
                try:
                    self.copy_to_clipboard(self.last_assistant_text)
                    chat_view.mount(SystemNotice("📋 **Copied last assistant response to clipboard!**"))
                except Exception as e:
                    chat_view.mount(SystemNotice(f"⚠️ Clipboard error: {e}"))
            else:
                chat_view.mount(SystemNotice("⚠️ No assistant response available to copy."))

        elif cmd in ["/exit", "/quit"]:
            self.work_queue.put(None)
            self.exit()

        else:
            chat_view.mount(SystemNotice(f"⚠️ Unknown command `{cmd}`. Type `/help` for command reference."))

        chat_view.scroll_end(animate=False)

    def _on_benchmark_complete(self, res: dict) -> None:
        """UI callback for benchmark results."""
        chat_view = self.query_one("#chat-view")
        chat_view.mount(SystemNotice(create_bench_table(res)))
        chat_view.scroll_end(animate=False)

    def run_generation(self, prompt: str, is_retry: bool = False) -> None:
        """Mount UI widgets and queue generation task to dedicated worker thread."""
        chat_view = self.query_one("#chat-view")
        user_input = self.query_one("#user-input", Input)

        if not is_retry:
            chat_view.mount(UserMessage(prompt, use_markdown=self.use_markdown))

        asst_widget = AssistantMessage(use_markdown=self.use_markdown)
        chat_view.mount(asst_widget)
        chat_view.scroll_end(animate=False)

        self.is_generating = True
        user_input.disabled = True
        self.update_telemetry()

        self.work_queue.put(("generate", (prompt, asst_widget)))

    def _scroll_chat_end(self) -> None:
        chat_view = self.query_one("#chat-view")
        chat_view.scroll_end(animate=False)

    def _on_generation_finished(self) -> None:
        """Callback when generation finishes."""
        self.is_generating = False
        user_input = self.query_one("#user-input", Input)
        user_input.disabled = False
        user_input.focus()
        self.update_telemetry()


def main():
    parser = argparse.ArgumentParser(description="TinyGrad Textual Interactive TUI Chat")
    parser.add_argument("--dataset", type=str, choices=["tinystories", "fineweb"], default="tinystories")
    parser.add_argument("--model-size", choices=["15M", "125M"], default="125M")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--no-jit", action="store_true", default=False)
    args = parser.parse_args()

    app = TinyChatApp(
        dataset=args.dataset,
        model_size=args.model_size,
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        use_jit=not args.no_jit,
    )
    app.run()


if __name__ == "__main__":
    main()
