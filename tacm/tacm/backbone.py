"""
TAC-SM Transformer Backbone
Decoder-only transformer with:
  - RoPE positional embeddings
  - Grouped-Query Attention (GQA)
  - Flash Attention via PyTorch SDPA
  - Gradient checkpointing
  - bf16/fp16 support
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import TransformerConfig


# ── RoPE ─────────────────────────────────────────────────────────────────────

def precompute_freqs_cis(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    theta = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, theta)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: (B, T, H, D_head)"""
    B, T, H, D = x.shape
    x_ = x.float().reshape(B, T, H, D // 2, 2)
    x_complex = torch.view_as_complex(x_.contiguous())
    freqs = freqs_cis[:T].unsqueeze(0).unsqueeze(2)   # (1, T, 1, D//2)
    x_rotated = x_complex * freqs
    out = torch.view_as_real(x_rotated).reshape(B, T, H, D)
    return out.to(x.dtype)


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ── Grouped-Query Attention ───────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_groups = cfg.n_heads // cfg.n_kv_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.use_flash = cfg.use_flash_attn

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.d_head, cfg.d_model, bias=False)
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.d_head)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Expand KV for GQA
        k = k.repeat_interleave(self.n_groups, dim=2)
        v = v.repeat_interleave(self.n_groups, dim=2)

        # (B, H, T, D_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dp = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dp,
            is_causal=(mask is None),
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(attn_out)


# ── SwiGLU Feed-Forward ───────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.up   = nn.Linear(d_model, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn = GroupedQueryAttention(cfg)
        self.ffn  = SwiGLUFFN(cfg.d_model, cfg.ffn_dim, cfg.dropout)
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis, mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ── Backbone ──────────────────────────────────────────────────────────────────

class TransformerBackbone(nn.Module):
    """
    Decoder-only transformer backbone.
    Returns hidden states (B, T, d_model) — NOT logits.
    Logits / heads are added by the full TAC-SM model.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.use_ckpt = cfg.gradient_checkpointing

        # Pre-compute RoPE frequencies
        freqs = precompute_freqs_cis(
            cfg.d_model // cfg.n_heads,
            cfg.max_seq_len,
            cfg.rope_base,
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, f"Sequence too long: {T} > {self.cfg.max_seq_len}"

        x = self.embed_drop(self.embed(input_ids))
        freqs = self.freqs_cis[:T].to(x.device)

        for layer in self.layers:
            if self.use_ckpt and self.training:
                x = checkpoint(layer, x, freqs, mask, use_reentrant=False)
            else:
                x = layer(x, freqs, mask)

        return self.norm(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
