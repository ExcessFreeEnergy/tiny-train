"""
model.py - High-Performance Transformer Architecture for tinygrad.
Supports Tensor Core 64/128 alignment, Fused RoPE, RMSNorm, FlashAttention SDPA, and SwiGLU.
"""

from tinygrad import Tensor, dtypes


def pad_vocab_size(vocab_size: int, multiple: int = 128) -> int:
    """Pad vocabulary size to a multiple of 128 for Tensor Core alignment."""
    if vocab_size % multiple == 0:
        return vocab_size
    return ((vocab_size + multiple - 1) // multiple) * multiple


def precompute_freqs_cis(dim: int, max_len: int = 2048) -> tuple[Tensor, Tensor]:
    """Precompute static RoPE cos and sin buffers up to max_len."""
    inv_freq = 1.0 / (10000.0 ** (Tensor.arange(0, dim, 2, dtype=dtypes.float) / dim))
    t_pos = Tensor.arange(0, max_len, dtype=dtypes.float)
    freqs = t_pos.reshape(max_len, 1) * inv_freq.reshape(1, dim // 2)
    emb = Tensor.cat(freqs, freqs, dim=-1).reshape(1, 1, max_len, dim)
    return emb.cos().cast(dtypes.default_float).realize(), emb.sin().cast(dtypes.default_float).realize()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply precomputed Rotary Position Embeddings (RoPE) to Query or Key tensor."""
    b, h, t, d = x.shape
    c, s = cos[:, :, :t, :], sin[:, :, :t, :]
    x1 = x[:, :, :, : d // 2]
    x2 = x[:, :, :, d // 2 :]
    x_rot = Tensor.cat(-x2, x1, dim=-1)
    return x * c + x_rot * s


class RMSNorm:
    """Root Mean Square Layer Normalization (fuses into single-pass reduction)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = Tensor.ones(dim)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight


class CausalSelfAttention:
    """Fused Scaled Dot-Product Causal Self-Attention with Static RoPE Buffers."""

    def __init__(self, d_model: int, n_heads: int, use_rope: bool = True):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.c_attn = Tensor.glorot_uniform(d_model, 3 * d_model)
        self.c_proj = Tensor.glorot_uniform(d_model, d_model)

    def __call__(self, x: Tensor, cos: Tensor = None, sin: Tensor = None) -> Tensor:
        b, t, c = x.shape
        qkv = x @ self.c_attn
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

        if self.use_rope and cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Fused Scaled Dot-Product Attention (SDPA / FlashAttention)
        y = Tensor.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, c)
        return y @ self.c_proj


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

    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_swiglu: bool = False, use_rope: bool = True):
        self.rms_1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, use_rope=use_rope)
        self.rms_2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff) if use_swiglu else GELUMLP(d_model, d_ff)

    def __call__(self, x: Tensor, cos: Tensor = None, sin: Tensor = None) -> Tensor:
        x = x + self.attn(self.rms_1(x), cos=cos, sin=sin)
        x = x + self.mlp(self.rms_2(x))
        return x


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
        pad_vocab_multiple: int = 128,
    ):
        self.raw_vocab_size = vocab_size
        self.vocab_size = pad_vocab_size(vocab_size, pad_vocab_multiple)
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_rope = use_rope

        self.wte = Tensor.glorot_uniform(self.vocab_size, d_model)
        if use_rope:
            self.cos, self.sin = precompute_freqs_cis(d_model // n_heads, max_len=max_len)
        else:
            self.wpe = Tensor.glorot_uniform(max_len, d_model)
            self.cos, self.sin = None, None

        self.h = [Block(d_model, n_heads, d_ff, use_swiglu=use_swiglu, use_rope=use_rope) for _ in range(n_layers)]
        self.rms_f = RMSNorm(d_model)

    def forward(self, idx: Tensor) -> Tensor:
        b, t = idx.shape
        x = self.wte[idx]
        if not self.use_rope:
            pos = Tensor.arange(0, t, dtype=dtypes.int32)
            x = x + self.wpe[pos]

        for block in self.h:
            x = block(x, cos=self.cos, sin=self.sin)

        x = self.rms_f(x)
        logits = x @ self.wte.T
        return logits

    def num_params(self) -> int:
        from tinygrad.nn.state import get_state_dict

        state = get_state_dict(self)
        return sum(p.numel() for p in state.values())


if __name__ == "__main__":
    m15 = GPT(vocab_size=13970, d_model=288, n_layers=6, n_heads=6, d_ff=1152)
    m125 = GPT(vocab_size=13970, d_model=768, n_layers=12, n_heads=12, d_ff=3072)
    print(f"15M Prototype Params: {m15.num_params():,} (Padded Vocab: {m15.vocab_size})")
    print(f"125M Target Params: {m125.num_params():,} (Padded Vocab: {m125.vocab_size})")
