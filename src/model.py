"""
model.py - High-Performance Transformer Architecture for tinygrad.
Supports Tensor Core alignment, Fused RoPE, RMSNorm, FlashAttention SDPA, and SwiGLU.
"""

import os
from tinygrad import Tensor, dtypes, getenv, nn

if "HK_FLASH_ATTENTION" not in os.environ:
    os.environ["HK_FLASH_ATTENTION"] = "1"


def pad_vocab_size(vocab_size: int, multiple: int = 128, power_of_two: bool = False) -> int:
    """Pad vocabulary size to a clean power of 2 (e.g. 16384) or multiple of 128 for optimal BEAM reduction tiling."""
    if power_of_two:
        p2 = 1 << (vocab_size - 1).bit_length()
        return max(p2, multiple)
    if vocab_size % multiple == 0:
        return vocab_size
    return ((vocab_size + multiple - 1) // multiple) * multiple


def precompute_freqs_cis(dim: int, max_len: int = 2048, theta: float = 10000.0) -> Tensor:
    """Precompute static RoPE cos/sin buffer up to max_len matching tinygrad LLM conventions."""
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2, dtype=dtypes.float) / dim))
    freqs = Tensor.arange(max_len, dtype=dtypes.float).unsqueeze(1) * freqs.unsqueeze(0)
    return freqs.cos().cat(freqs.sin(), dim=-1).cast(dtypes.default_float).is_param_(False)


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply precomputed Rotary Position Embeddings (RoPE) to Query or Key tensor."""
    b, h, t, d = x.shape
    cos, sin = freqs_cis[:t].reshape(1, 1, t, d).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)


class CausalSelfAttention:
    """Fused Scaled Dot-Product Causal Self-Attention with Static RoPE Buffers."""

    def __init__(self, d_model: int, n_heads: int, use_rope: bool = True, flash_attn: bool | None = None):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.flash_attn = bool(getenv("HK_FLASH_ATTENTION", 1)) if flash_attn is None else flash_attn
        self.c_attn = Tensor.glorot_uniform(d_model, 3 * d_model)
        self.c_proj = Tensor.glorot_uniform(d_model, d_model)

    def __call__(self, x: Tensor, freqs_cis: Tensor | None = None, start_pos: int | None = None) -> Tensor:
        b, t, c = x.shape
        qkv = (x @ self.c_attn).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        use_fa = self.flash_attn and getenv("HK_FLASH_ATTENTION", 1)

        if start_pos is None:
            if self.use_rope and freqs_cis is not None:
                q = apply_rope(q, freqs_cis[:t])
                k = apply_rope(k, freqs_cis[:t])
            if use_fa:
                try:
                    from extra.thunder.amd.fa import flash_attention
                    y, *_ = flash_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
                    y = y.transpose(1, 2)
                except Exception:
                    y = Tensor.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                y = Tensor.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            if self.use_rope and freqs_cis is not None:
                freqs = freqs_cis[start_pos : start_pos + t]
                q = apply_rope(q, freqs)
                k = apply_rope(k, freqs)

            if not hasattr(self, "cache_kv"):
                max_context = freqs_cis.shape[0] if freqs_cis is not None else 512
                self.cache_kv = Tensor.zeros(2, b, self.n_heads, max_context, self.head_dim, dtype=k.dtype).contiguous().realize()

            kv_stacked = Tensor.stack(k, v).cast(self.cache_kv.dtype)
            self.cache_kv[:, :, :, start_pos : start_pos + t, :].assign(kv_stacked).realize()
            keys = self.cache_kv[0][:, :, : start_pos + t, :]
            values = self.cache_kv[1][:, :, : start_pos + t, :]

            mask = Tensor.full((1, 1, t, start_pos + t), float("-inf"), dtype=x.dtype).triu(start_pos + 1) if t > 1 else None
            if use_fa and mask is None:
                try:
                    from extra.thunder.amd.fa import flash_attention
                    y, *_ = flash_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), is_causal=True)
                    y = y.transpose(1, 2)
                except Exception:
                    y = Tensor.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
            else:
                y = Tensor.scaled_dot_product_attention(q, keys, values, attn_mask=mask)

        y = y.transpose(1, 2).reshape(b, t, c)
        return y @ self.c_proj

    def reset_cache(self):
        if hasattr(self, "cache_kv"):
            del self.cache_kv


class GELUMLP:
    """Feed-Forward Network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int):
        self.c_fc = Tensor.glorot_uniform(d_model, d_ff)
        self.c_proj = Tensor.glorot_uniform(d_ff, d_model)

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.c_fc).gelu() @ self.c_proj


class SwiGLUMLP:
    """Fused SwiGLU Gated Feed-Forward Network for Higher Arithmetic Intensity."""

    def __init__(self, d_model: int, d_ff: int):
        self.w13 = Tensor.glorot_uniform(d_model, 2 * d_ff)
        self.w2 = Tensor.glorot_uniform(d_ff, d_model)

    def __call__(self, x: Tensor) -> Tensor:
        w13 = x @ self.w13
        w1, w3 = w13.chunk(2, dim=-1)
        return (w1.silu() * w3) @ self.w2


class Block:
    """Transformer Block with Pre-RMSNorm and Pre-Attention Residuals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_swiglu: bool = False, use_rope: bool = True, flash_attn: bool | None = None):
        self.rms_1 = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, use_rope=use_rope, flash_attn=flash_attn)
        self.rms_2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff) if use_swiglu else GELUMLP(d_model, d_ff)

    def __call__(self, x: Tensor, freqs_cis: Tensor | None = None, start_pos: int | None = None) -> Tensor:
        x = x + self.attn(self.rms_1(x), freqs_cis=freqs_cis, start_pos=start_pos)
        x = x + self.mlp(self.rms_2(x))
        return x

    def reset_cache(self):
        self.attn.reset_cache()


class GPT:
    """Causal Transformer Model supporting 15M and 125M Parameter Presets."""

    def __init__(
        self,
        vocab_size: int = 13970,
        d_model: int = 288,
        n_layers: int = 6,
        n_heads: int = 6,
        d_ff: int = 1152,
        max_len: int = 512,
        use_swiglu: bool = False,
        use_rope: bool = True,
        flash_attn: bool | None = None,
        pad_vocab_multiple: int = 128,
        pad_vocab_power_of_2: bool = False,
    ):
        self.raw_vocab_size = vocab_size
        self.vocab_size = pad_vocab_size(vocab_size, pad_vocab_multiple, power_of_two=pad_vocab_power_of_2)
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_rope = use_rope
        self.flash_attn = bool(getenv("HK_FLASH_ATTENTION", 1)) if flash_attn is None else flash_attn

        self.wte = Tensor.glorot_uniform(self.vocab_size, d_model)
        if use_rope:
            self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_len=max_len).is_param_(False)
        else:
            self.wpe = Tensor.glorot_uniform(max_len, d_model)
            self.freqs_cis = None

        self.h = [Block(d_model, n_heads, d_ff, use_swiglu=use_swiglu, use_rope=use_rope, flash_attn=self.flash_attn) for _ in range(n_layers)]
        self.rms_f = nn.RMSNorm(d_model)

    def reset_cache(self):
        for block in self.h:
            block.reset_cache()

    def forward(self, idx: Tensor, start_pos: int | None = None) -> Tensor:
        b, t = idx.shape
        x = self.wte[idx]
        if not self.use_rope:
            if start_pos is not None:
                pos = Tensor.arange(0, t, dtype=dtypes.int32) + start_pos
            else:
                pos = Tensor.arange(0, t, dtype=dtypes.int32)
            x = x + self.wpe[pos]

        for block in self.h:
            x = block(x, freqs_cis=self.freqs_cis, start_pos=start_pos)

        x = self.rms_f(x)
        logits = x @ self.wte.T
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in nn.state.get_parameters(self))


if __name__ == "__main__":
    m15 = GPT(vocab_size=13970, d_model=288, n_layers=6, n_heads=6, d_ff=1152)
    m125 = GPT(vocab_size=13970, d_model=768, n_layers=12, n_heads=12, d_ff=3072)
    print(f"15M Prototype Params: {m15.num_params():,} (Padded Vocab: {m15.vocab_size})")
    print(f"125M Target Params: {m125.num_params():,} (Padded Vocab: {m125.vocab_size})")
