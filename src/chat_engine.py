"""
chat_engine.py - Stateful LLM Chat Engine and Sampling Harness for TinyGrad Checkpoints.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import tiktoken
from tinygrad import Device, Tensor, TinyJit, Variable, dtypes
from tinygrad.helpers import GlobalCounters, Profiling, Timing
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load

# Ensure src path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import GPT


def load_vocab_map(dataset_name: str = "tinystories", vocab_map_path: str | None = None):
    """Load vocabulary map for trimming/restoring original GPT-2 token IDs."""
    if not vocab_map_path:
        ds = dataset_name.lower().replace("-", "").replace("_", "")
        if "finewebedu" in ds:
            vocab_map_path = "data/FineWebEdu/vocab_map.json"
        elif "bookcorpus" in ds:
            vocab_map_path = "data/BookCorpus/vocab_map.json"
        elif "fineweb" in ds:
            vocab_map_path = "data/FineWeb/vocab_map.json"
        else:
            vocab_map_path = "data/TinyStories/vocab_map.json"

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    search_paths = [
        vocab_map_path,
        os.path.join(base_dir, vocab_map_path),
        os.path.join(os.path.dirname(__file__), vocab_map_path),
    ]

    for p in search_paths:
        if os.path.exists(p):
            with open(p) as f:
                vmap = json.load(f)
            orig_to_new = {int(k): int(v) for k, v in vmap.get("orig_to_new", {}).items()}
            new_to_orig = vmap.get("new_to_orig", [])
            return orig_to_new, new_to_orig
    return None, None


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints", model_size: str = "125M") -> str | None:
    """Scan checkpoint directory for fine-tuned or highest step count model matching model scale."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_dir = os.path.join(base_dir, checkpoint_dir) if not os.path.isabs(checkpoint_dir) else checkpoint_dir

    search_dirs = []
    if os.path.exists(full_dir):
        search_dirs.append(full_dir)

    finetuned_dir = os.path.join(base_dir, "checkpoints_finetuned")
    if os.path.exists(finetuned_dir) and finetuned_dir not in search_dirs:
        search_dirs.append(finetuned_dir)

    if not search_dirs:
        return None

    # Prioritize fused fine-tuned checkpoint if present
    for sdir in search_dirs:
        fused_ckpt = os.path.join(sdir, f"model_{model_size.lower()}_finetuned.safetensors")
        if os.path.exists(fused_ckpt):
            return fused_ckpt

    import re

    pattern = re.compile(rf"model_{model_size.lower()}_step_(\d+)\.safetensors$")
    max_step = -1
    best_ckpt = None
    for sdir in search_dirs:
        for filename in os.listdir(sdir):
            match = pattern.match(filename)
            if match:
                s = int(match.group(1))
                if s > max_step:
                    max_step = s
                    best_ckpt = os.path.join(sdir, filename)
    return best_ckpt


def apply_repetition_penalty(logits: Tensor, penalty: float = 1.15, context_mask: Tensor | None = None) -> Tensor:
    """Apply repetition penalty on-device using TinyGrad Tensor ops.

    Positive logits are divided by penalty, negative logits are multiplied by penalty for tokens in context_mask.
    """
    if penalty == 1.0 or context_mask is None:
        return logits
    penalized = (logits > 0).where(logits / penalty, logits * penalty)
    return context_mask.where(penalized, logits)


def sample_logits(logits: Tensor, temp: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> Tensor:
    """Sample next token index on device using temperature scaling, top-k, and nucleus top-p filtering."""
    assert logits.ndim == 1, "sample expects 1D logits tensor"
    if temp < 1e-6:
        return logits.argmax().cast(dtypes.int32)

    logits = logits.to(Device.DEFAULT)
    logits = (logits != logits).where(-float("inf"), logits)
    scaled_logits = logits / max(temp, 1e-5)

    # Top-K filtering
    if 0 < top_k < logits.numel():
        counter = Tensor.arange(scaled_logits.numel(), device=scaled_logits.device).contiguous()
        counter2 = Tensor.arange(scaled_logits.numel() - 1, -1, -1, device=scaled_logits.device).contiguous()
        top_k_logits = Tensor.zeros(top_k, device=scaled_logits.device).contiguous()
        top_k_indices = Tensor.zeros(top_k, device=scaled_logits.device, dtype=dtypes.int32).contiguous()
        l_copy = scaled_logits
        for i in range(top_k):
            t_argmax = (l_copy.numel() - ((l_copy == (l_max := l_copy.max())) * counter2).max() - 1).cast(dtypes.default_int)
            top_k_logits = top_k_logits + l_max.unsqueeze(0).pad(((i, top_k - i - 1),))
            top_k_indices = top_k_indices + t_argmax.unsqueeze(0).pad(((i, top_k - i - 1),))
            l_copy = (counter == t_argmax).where(-float("inf"), l_copy)

        probs = top_k_logits.softmax()

        # Apply top_p (nucleus) cutoff if top_p < 1.0 on top_k subset using pure tensor ops
        if top_p < 1.0:
            tri = Tensor.ones(top_k, top_k, device=probs.device).tril()
            cum_probs = probs @ tri
            prev_cum = cum_probs - probs
            mask = (prev_cum < top_p).cast(probs.dtype)
            probs = probs * mask
            probs = probs / (probs.sum() + 1e-10)

        output_idx = probs.multinomial()
        output_token = top_k_indices[output_idx]
    else:
        probs = scaled_logits.softmax()
        output_token = probs.multinomial().cast(dtypes.int32)

    return output_token


class GPTEngine:
    """TinyGrad Inference Engine encapsulating JIT execution graph & KV caching."""

    def __init__(self, model: GPT, max_context: int = 1024, use_jit: bool = True):
        self.model = model
        self.max_context = max_context
        self.raw_vocab_size = model.raw_vocab_size
        self.step_jit = TinyJit(self.step) if use_jit else None

    def step(
        self,
        tokens: Tensor,
        start_pos: Variable | int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float = 1.15,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        logits = self.model.forward(tokens, start_pos=start_pos)
        last_logits = logits[0, -1, : self.raw_vocab_size]
        if repetition_penalty != 1.0 and context_mask is not None:
            last_logits = apply_repetition_penalty(last_logits, penalty=repetition_penalty, context_mask=context_mask)
        return sample_logits(last_logits, temp=temperature, top_k=top_k, top_p=top_p)

    def __call__(
        self,
        tokens: Tensor,
        start_pos: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        if context_mask is None:
            context_mask = Tensor.zeros(self.raw_vocab_size, dtype=dtypes.bool, device=Device.DEFAULT).clone().realize()
        if tokens.shape == (1, 1) and self.step_jit is not None:
            v_start_pos = Variable("start_pos", 0, self.max_context - 1).bind(start_pos)
            return self.step_jit(tokens, v_start_pos, temperature, top_k, top_p, repetition_penalty, context_mask)
        return self.step(tokens, start_pos, temperature, top_k, top_p, repetition_penalty, context_mask)


@dataclass
class ChatTurn:
    role: str  # "user", "assistant", "system"
    text: str
    token_ids: list[int] = field(default_factory=list)
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class TelemetryMetrics:
    ttft_ms: float = 0.0
    tok_per_sec: float = 0.0
    avg_latency_ms: float = 0.0
    total_sec: float = 0.0
    tokens_generated: int = 0
    vram_mb: float = 0.0
    mem_bw_gbs: float = 0.0


class GPTEngineManager:
    """Stateful Manager handling model initialization, KV cache lifecycle, and multi-turn chat."""

    def __init__(
        self,
        dataset: str = "tinystories",
        model_size: str = "125M",
        checkpoint_path: str | None = None,
        checkpoint_dir: str = "checkpoints",
        max_context: int = 1024,
        use_jit: bool = True,
        engine: GPTEngine | None = None,
    ):
        self.dataset = dataset
        self.model_size = model_size
        self.max_context = max_context
        self.use_jit = use_jit

        # Hyperparameters
        self.temperature = 0.8
        self.top_p = 0.9
        self.top_k = 40
        self.repetition_penalty = 1.15
        self.max_tokens = 256
        self.system_prompt = ""
        self.profile = False
        self.timing = False

        # Session state
        self.history: list[ChatTurn] = []
        self.start_pos = 0

        # Locate checkpoint
        if not checkpoint_path:
            checkpoint_path = find_latest_checkpoint(checkpoint_dir, model_size)

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found for scale '{model_size}' in '{checkpoint_dir}'")

        self.checkpoint_path = checkpoint_path

        if engine is not None:
            self.model = engine.model
            self.engine = engine
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.orig_to_new, self.new_to_orig = load_vocab_map(dataset)
            self.param_bytes = sum(p.nbytes() for p in get_parameters(self.model))
            self.num_params = self.model.num_params()
            self.load_time_ms = 0.0
            self._warmup_jit()
            return
        t_start = time.perf_counter()
        state = safe_load(checkpoint_path)
        wte_shape = state["wte"].shape
        ckpt_padded_vocab_size = wte_shape[0]

        # Auto-detect dataset if checkpoint vocab size > 30000 (FineWeb)
        if ckpt_padded_vocab_size > 30000 and dataset.lower() == "tinystories":
            dataset = "fineweb"
            self.dataset = "fineweb"

        # Initialize Tokenizer & Vocab Map
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.orig_to_new, self.new_to_orig = load_vocab_map(dataset)

        if model_size == "125M":
            d_model, n_layers, n_heads, d_ff = 768, 12, 12, 3072
        else:
            d_model, n_layers, n_heads, d_ff = 288, 6, 6, 1152

        if self.new_to_orig:
            raw_vocab_size = len(self.new_to_orig)
        else:
            raw_vocab_size = 13970 if dataset != "fineweb" else 49685

        is_power_of_2 = (ckpt_padded_vocab_size & (ckpt_padded_vocab_size - 1) == 0) and (ckpt_padded_vocab_size >= raw_vocab_size)

        Tensor.training = False
        self.model = GPT(
            vocab_size=raw_vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_context,
            use_swiglu=True,
            use_rope=True,
            pad_vocab_multiple=128,
            pad_vocab_power_of_2=is_power_of_2,
        )

        if "freqs_cis" in state:
            state.pop("freqs_cis")

        load_state_dict(self.model, state, strict=False)
        Tensor.realize(*get_parameters(self.model))
        self.load_time_ms = (time.perf_counter() - t_start) * 1000.0

        self.param_bytes = sum(p.nbytes() for p in get_parameters(self.model))
        self.num_params = self.model.num_params()
        self.engine = GPTEngine(self.model, max_context=max_context, use_jit=use_jit)
        self._warmup_jit()

    def _warmup_jit(self):
        """Pre-compile and capture @TinyJit step graph during startup to eliminate first-turn latency."""
        if not self.use_jit:
            return
        tok = Tensor([[50256]], dtype=dtypes.int32).realize()
        dummy_mask = Tensor.zeros(self.engine.raw_vocab_size, dtype=dtypes.bool, device=Device.DEFAULT).clone().realize()
        for s in range(3):
            tok = self.engine(
                tok.reshape(1, 1),
                start_pos=s,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                context_mask=dummy_mask,
            ).realize()
        self.model.reset_cache()
        self.start_pos = 0

    def encode_text(self, text: str) -> list[int]:
        """Encode text to token IDs and map to vocabulary space."""
        orig_ids = self.tokenizer.encode(text)
        if not orig_ids:
            orig_ids = [50256]
        if self.orig_to_new:
            return [self.orig_to_new.get(tid, 0) for tid in orig_ids]
        return list(orig_ids)

    def decode_token(self, token_id: int) -> str:
        """Decode trimmed/mapped token ID back to text string."""
        if self.new_to_orig and token_id < len(self.new_to_orig):
            orig_id = self.new_to_orig[token_id]
        else:
            orig_id = token_id
        return self.tokenizer.decode([orig_id])

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode sequence of token IDs to text string."""
        if self.new_to_orig:
            orig_ids = [self.new_to_orig[tid] for tid in token_ids if tid < len(self.new_to_orig)]
        else:
            orig_ids = token_ids
        return self.tokenizer.decode(orig_ids)

    def reset_context(self):
        """Flush KV cache and clear history."""
        self.model.reset_cache()
        self.start_pos = 0
        self.history.clear()
        if self.system_prompt:
            self._apply_system_prompt()

    def set_system_prompt(self, text: str):
        """Dynamically update system prompt and reset context."""
        self.system_prompt = text.strip()
        self.reset_context()

    def _apply_system_prompt(self):
        """Pre-fill system prompt tokens into KV cache."""
        if not self.system_prompt:
            return
        sys_tokens = self.encode_text(f"System: {self.system_prompt}\n")
        x_sys = Tensor([sys_tokens], dtype=dtypes.int32).realize()
        dummy_mask = Tensor.zeros(self.engine.raw_vocab_size, dtype=dtypes.bool, device=Device.DEFAULT).clone().realize()
        self.engine(
            x_sys,
            start_pos=0,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            context_mask=dummy_mask,
        ).realize()
        start = self.start_pos
        self.start_pos += len(sys_tokens)
        self.history.append(ChatTurn("system", self.system_prompt, sys_tokens, start, self.start_pos))

    def pop_last_turn(self) -> bool:
        """Remove last user query and assistant response from history and rewind KV cache position."""
        if not self.history:
            return False

        # Remove assistant response if last
        if self.history and self.history[-1].role == "assistant":
            self.history.pop()

        # Remove user query if last
        if self.history and self.history[-1].role == "user":
            self.history.pop()

        # Rewind start_pos to end of remaining history
        if self.history:
            self.start_pos = self.history[-1].end_pos
        else:
            self.start_pos = 0
            self.model.reset_cache()
            if self.system_prompt:
                self._apply_system_prompt()

        return True

    def retry_last_turn(self) -> str | None:
        """Pop last assistant turn and return last user prompt to re-run generation."""
        if not self.history:
            return None

        last_user_prompt = None
        # Locate last user turn
        for turn in reversed(self.history):
            if turn.role == "user":
                last_user_prompt = turn.text
                break

        if last_user_prompt is None:
            return None

        # Pop assistant message
        if self.history and self.history[-1].role == "assistant":
            self.history.pop()

        # Rewind start_pos to before assistant turn (end of user turn)
        if self.history:
            self.start_pos = self.history[-1].end_pos
        else:
            self.start_pos = 0

        return last_user_prompt

    def generate_stream(self, prompt: str):
        """Generator yielding streamed text chunks and telemetry updates during generation."""
        prompt_tokens = self.encode_text(prompt)
        prompt_len = len(prompt_tokens)

        # Ensure we don't exceed max context
        if self.start_pos + prompt_len + self.max_tokens >= self.max_context:
            yield "⚠️ Warning: Context window limit approaching! Consider running /clear.", None

        # Record User Turn
        user_turn_start = self.start_pos
        user_turn_end = user_turn_start + prompt_len
        self.history.append(ChatTurn("user", prompt, prompt_tokens, user_turn_start, user_turn_end))

        # Collect all context token IDs in sequence history + prompt
        context_token_ids = set()
        for turn in self.history:
            context_token_ids.update(turn.token_ids)
        context_token_ids.update(prompt_tokens)

        # Build initial context_mask on device
        mask_np = np.zeros(self.engine.raw_vocab_size, dtype=bool)
        valid_ids = [tid for tid in context_token_ids if 0 <= tid < self.engine.raw_vocab_size]
        if valid_ids:
            mask_np[valid_ids] = True
        context_mask = Tensor(mask_np, device=Device.DEFAULT).clone().realize()

        # 1. Prompt Phase (Fill KV Cache via warm 1-token JIT steps)
        t0 = time.perf_counter()
        tok_tensor = None
        for pos, tid in enumerate(prompt_tokens):
            x_tok = Tensor([[tid]], dtype=dtypes.int32).realize()
            tok_tensor = self.engine(
                x_tok,
                start_pos=self.start_pos + pos,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                context_mask=context_mask,
            ).realize()
        next_id = tok_tensor.item()
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        current_gen_ids = [next_id]
        if 0 <= next_id < self.engine.raw_vocab_size:
            mask_np[next_id] = True
            context_mask = Tensor(mask_np, device=Device.DEFAULT).clone().realize()
        self.start_pos += prompt_len

        # Stream first token text
        first_chunk = self.decode_token(next_id)
        yield first_chunk, TelemetryMetrics(ttft_ms=ttft_ms, tokens_generated=1)

        # 2. Autoregressive generation phase
        t_gen_start = time.perf_counter()
        token_times = [ttft_ms]
        eos_id = 50256

        with Profiling(enabled=self.profile):
            for _ in range(1, self.max_tokens):
                if self.start_pos >= self.max_context - 1:
                    break

                GlobalCounters.reset()
                t_step_0 = time.perf_counter()

                with Timing("total", enabled=self.timing):
                    x_in = tok_tensor.reshape(1, 1)
                    tok_tensor = self.engine(
                        x_in,
                        start_pos=self.start_pos,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                        context_mask=context_mask,
                    ).realize()
                    next_id = tok_tensor.item()

                step_ms = (time.perf_counter() - t_step_0) * 1000.0
                token_times.append(step_ms)
                current_gen_ids.append(next_id)
                self.start_pos += 1

                if 0 <= next_id < self.engine.raw_vocab_size:
                    mask_np[next_id] = True
                    context_mask = Tensor(mask_np, device=Device.DEFAULT).clone().realize()

                chunk = self.decode_token(next_id)

                # Check EOS token or EOS string match
                if next_id == eos_id or (self.new_to_orig and next_id < len(self.new_to_orig) and self.new_to_orig[next_id] == eos_id):
                    break

                total_sec = time.perf_counter() - t_gen_start
                tok_per_sec = len(current_gen_ids) / total_sec if total_sec > 0 else 0
                avg_latency = float(sum(token_times[1:])) / len(token_times[1:]) if len(token_times) > 1 else ttft_ms

                # Estimate memory bandwidth & VRAM footprint
                mem_bw_gbs = (GlobalCounters.global_mem / (step_ms / 1000.0)) / 1e9 if step_ms > 0 else 0.0
                vram_mb = (self.param_bytes + GlobalCounters.global_mem) / 1e6

                metrics = TelemetryMetrics(
                    ttft_ms=ttft_ms,
                    tok_per_sec=tok_per_sec,
                    avg_latency_ms=avg_latency,
                    total_sec=total_sec,
                    tokens_generated=len(current_gen_ids),
                    vram_mb=vram_mb,
                    mem_bw_gbs=mem_bw_gbs,
                )

                yield chunk, metrics

        # Record Assistant Turn
        full_response_text = self.decode_tokens(current_gen_ids)
        self.history.append(ChatTurn("assistant", full_response_text, current_gen_ids, user_turn_end, self.start_pos))

    def run_benchmark(self, num_tokens: int = 100) -> dict:
        """Run automated benchmark generation pass and return detailed performance metrics."""
        self.model.reset_cache()
        dummy_prompt = "Once upon a time in a benchmark test"
        prompt_ids = self.encode_text(dummy_prompt)

        context_token_ids = set(prompt_ids)
        mask_np = np.zeros(self.engine.raw_vocab_size, dtype=bool)
        valid_ids = [tid for tid in context_token_ids if 0 <= tid < self.engine.raw_vocab_size]
        if valid_ids:
            mask_np[valid_ids] = True
        context_mask = Tensor(mask_np, device=Device.DEFAULT).clone().realize()

        t0 = time.perf_counter()
        tok_tensor = None
        for pos, tid in enumerate(prompt_ids):
            x_tok = Tensor([[tid]], dtype=dtypes.int32).realize()
            tok_tensor = self.engine(
                x_tok,
                start_pos=pos,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=self.repetition_penalty,
                context_mask=context_mask,
            ).realize()
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        step_times = []
        mem_bws = []
        start_pos = len(prompt_ids)

        t_gen_start = time.perf_counter()
        for i in range(1, num_tokens):
            GlobalCounters.reset()
            t_s = time.perf_counter()
            x_in = tok_tensor.reshape(1, 1)
            tok_tensor = self.engine(
                x_in,
                start_pos=start_pos + i - 1,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=self.repetition_penalty,
                context_mask=context_mask,
            ).realize()
            next_id = tok_tensor.item()
            if 0 <= next_id < self.engine.raw_vocab_size:
                mask_np[next_id] = True
                context_mask = Tensor(mask_np, device=Device.DEFAULT).clone().realize()
            dt = (time.perf_counter() - t_s) * 1000.0
            step_times.append(dt)
            if dt > 0:
                mem_bws.append((GlobalCounters.global_mem / (dt / 1000.0)) / 1e9)

        total_sec = time.perf_counter() - t_gen_start
        tok_per_sec = (num_tokens - 1) / total_sec if total_sec > 0 else 0.0
        avg_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
        avg_mem_bw = sum(mem_bws) / len(mem_bws) if mem_bws else 0.0

        # Reset context back to prior history state
        self.reset_context()

        return {
            "tokens": num_tokens,
            "ttft_ms": ttft_ms,
            "tok_per_sec": tok_per_sec,
            "avg_step_ms": avg_step_ms,
            "total_sec": total_sec,
            "avg_mem_bw_gbs": avg_mem_bw,
            "vram_mb": self.param_bytes / 1e6,
            "num_params": self.num_params,
        }
