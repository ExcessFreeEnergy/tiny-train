"""
model.py - High-Performance 15M Parameter Transformer Architecture in tinygrad.
Uses RMSNorm, Fused Scaled Dot-Product Attention (FlashAttention), and SwiGLU MLP options.
"""

from tinygrad import Tensor, dtypes


class RMSNorm:
    """Root Mean Square Layer Normalization (fuses into single-pass reduction)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = Tensor.ones(dim)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight


class CausalSelfAttention:
    """Fused Scaled Dot-Product Causal Self-Attention (FlashAttention)."""

    def __init__(self, d_model: int, n_heads: int):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = Tensor.glorot_uniform(d_model, 3 * d_model)
        self.c_proj = Tensor.glorot_uniform(d_model, d_model)

    def __call__(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        qkv = x @ self.c_attn
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)

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
    """SwiGLU Gated Feed-Forward Network for Higher Arithmetic Intensity."""

    def __init__(self, d_model: int, d_ff: int):
        self.w1 = Tensor.glorot_uniform(d_model, d_ff)
        self.w2 = Tensor.glorot_uniform(d_ff, d_model)
        self.w3 = Tensor.glorot_uniform(d_model, d_ff)

    def __call__(self, x: Tensor) -> Tensor:
        return ((x @ self.w1).silu() * (x @ self.w3)) @ self.w2


class Block:
    """Transformer Block with Pre-RMSNorm and Pre-Attention Residuals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, use_swiglu: bool = False):
        self.rms_1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.rms_2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff) if use_swiglu else GELUMLP(d_model, d_ff)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


class GPT:
    """15M Parameter Causal Transformer Model."""

    def __init__(
        self,
        vocab_size: int = 29362,
        d_model: int = 288,
        n_layers: int = 6,
        n_heads: int = 6,
        d_ff: int = 1152,
        max_len: int = 512,
        use_swiglu: bool = False,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.wte = Tensor.glorot_uniform(vocab_size, d_model)
        self.wpe = Tensor.glorot_uniform(max_len, d_model)
        self.h = [Block(d_model, n_heads, d_ff, use_swiglu=use_swiglu) for _ in range(n_layers)]
        self.rms_f = RMSNorm(d_model)

    def forward(self, idx: Tensor) -> Tensor:
        b, t = idx.shape
        pos = Tensor.arange(0, t, dtype=dtypes.int32)
        tok_emb = self.wte[idx]
        pos_emb = self.wpe[pos]
        x = tok_emb + pos_emb
        for block in self.h:
            x = block(x)
        x = self.rms_f(x)
        logits = x.cast(dtypes.float) @ self.wte.cast(dtypes.float).T
        return logits

    def num_params(self) -> int:
        from tinygrad.nn.state import get_state_dict

        state = get_state_dict(self)
        return sum(p.numel() for p in state.values())


if __name__ == "__main__":
    model = GPT()
    print(f"High-Performance GPT Model Initialized. Total Parameters: {model.num_params():,}")
