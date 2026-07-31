"""
model.py - 15M Parameter Transformer Architecture in tinygrad.
"""

import math
from typing import Optional
from tinygrad import Tensor, dtypes


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = Tensor.ones(dim)
        self.bias = Tensor.zeros(dim)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        return x.layernorm(eps=self.eps).mul(self.weight).add(self.bias)


class CausalSelfAttention:
    def __init__(self, d_model: int, n_heads: int):
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_attn = Tensor.glorot_uniform(d_model, 3 * d_model)
        self.c_proj = Tensor.glorot_uniform(d_model, d_model)

    def __call__(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = x @ self.c_attn
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        mask = Tensor.ones(T, T, dtype=dtypes.bool).triu(1).logical_not()
        att = att.masked_fill(mask == False, -1e9).softmax()
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return y @ self.c_proj


class MLP:
    def __init__(self, d_model: int, d_ff: int):
        self.c_fc = Tensor.glorot_uniform(d_model, d_ff)
        self.c_proj = Tensor.glorot_uniform(d_ff, d_model)

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.c_fc).gelu() @ self.c_proj


class Block:
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.ln_1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
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
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.wte = Tensor.glorot_uniform(vocab_size, d_model)
        self.wpe = Tensor.glorot_uniform(max_len, d_model)
        self.h = [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        pos = Tensor.arange(0, T, dtype=dtypes.int32)
        tok_emb = self.wte[idx]
        pos_emb = self.wpe[pos]
        x = tok_emb + pos_emb
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.T
        return logits

    def num_params(self) -> int:
        from tinygrad.nn.state import get_state_dict
        state = get_state_dict(self)
        return sum(p.numel() for p in state.values())


if __name__ == "__main__":
    model = GPT()
    print(f"GPT Model Initialized. Total Parameters: {model.num_params():,}")
