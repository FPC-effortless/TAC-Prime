"""
TAC-SCM-REAL001: Structure Compiler

Turns discovered latent structure candidates into typed structure objects.

Each input candidate (a latent cluster center from discovery) is compiled into
multiple typed embeddings: concept center/width, procedure, causal, trigger,
transform.  A compression head scores how compactly the structure is represented.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scm_types import StructureCompilerOutput
from .scm_config import TACSCMConfig


class StructureCompiler(nn.Module):
    """
    Compiles latent structure candidates into typed structure slot tensors.

    Input
    -----
    hidden_states       : (B, T, d_model)
    latent_state        : (B, T, d_structure)       — from discovery encoder
    structure_candidates: (B, n_candidates, d_structure)

    Output
    ------
    StructureCompilerOutput with per-candidate typed embeddings.

    Architecture
    ------------
    For each candidate embedding e ∈ R^{d_structure}:
      concept_center      = proj_center(e)
      concept_log_width   = proj_width(e)   (uncertainty / spread)
      procedure_emb       = proj_proc(e)    (how to execute)
      causal_emb          = proj_causal(e)  (cause–effect pattern)
      trigger_emb         = proj_trigger(e) (activation condition)
      transform_emb       = proj_transform(e) (state transformation)
      compression_score   = sigmoid(proj_compress(e))
      structure_token     = fusion([all above])
    """

    def __init__(self, cfg: TACSCMConfig):
        super().__init__()
        D  = cfg.d_structure
        Dm = cfg.d_model
        n  = cfg.n_structure_candidates

        # Context aggregator: compress (B, T, d_model) → (B, d_model)
        self.ctx_proj = nn.Linear(Dm, D, bias=False)
        self.ctx_attn = nn.MultiheadAttention(D, num_heads=max(1, D // 64),
                                               dropout=cfg.structure_dropout,
                                               batch_first=True)

        # Typed projection heads (candidate → typed embedding)
        self.proj_center    = _TypedHead(D, D)
        self.proj_width     = _TypedHead(D, D)    # log-width (softplus for positivity)
        self.proj_proc      = _TypedHead(D, D)
        self.proj_causal    = _TypedHead(D, D)
        self.proj_trigger   = _TypedHead(D, D)
        self.proj_transform = _TypedHead(D, D)

        # Compression scoring head
        self.proj_compress = nn.Sequential(
            nn.Linear(D, D // 2, bias=False),
            nn.GELU(),
            nn.Linear(D // 2, 1, bias=False),
        )

        # Fusion: all typed embeddings → single structure token
        # 6 typed vectors + original candidate = 7 * D → D
        self.fusion = nn.Sequential(
            nn.Linear(7 * D, D * 2, bias=False),
            nn.GELU(),
            nn.LayerNorm(D * 2),
            nn.Linear(D * 2, D, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        hidden_states:        torch.Tensor,  # (B, T, d_model)
        latent_state:         torch.Tensor,  # (B, T, d_structure)
        structure_candidates: torch.Tensor,  # (B, n_cand, d_structure)
    ) -> StructureCompilerOutput:

        B, N, D = structure_candidates.shape
        device  = structure_candidates.device

        # ── 1. Context conditioning (optional cross-attention) ────────────────
        ctx = self.ctx_proj(hidden_states)                    # (B, T, D)
        cand_ctx, _ = self.ctx_attn(
            structure_candidates, ctx, ctx                    # Q=candidates, K=V=context
        )                                                     # (B, N, D)
        cand = structure_candidates + 0.1 * cand_ctx          # residual blend

        # ── 2. Typed projections ──────────────────────────────────────────────
        center    = self.proj_center(cand)                    # (B, N, D)
        log_width = self.proj_width(cand)                     # (B, N, D) — log-space
        proc      = self.proj_proc(cand)                      # (B, N, D)
        causal    = self.proj_causal(cand)                    # (B, N, D)
        trigger   = self.proj_trigger(cand)                   # (B, N, D)
        transform = self.proj_transform(cand)                 # (B, N, D)

        # Compression score ∈ [0, 1]
        comp_score = torch.sigmoid(
            self.proj_compress(cand).squeeze(-1)              # (B, N)
        )

        # ── 3. Fusion into structure token ────────────────────────────────────
        fused_input = torch.cat([
            cand, center, log_width, proc, causal, trigger, transform
        ], dim=-1)                                            # (B, N, 7*D)
        struct_tokens = self.fusion(fused_input)              # (B, N, D)

        # ── 4. Compression loss ───────────────────────────────────────────────
        # High compression score should correlate with compact latent structure
        # Proxy: penalise low compression on high-variance candidates
        cand_var     = cand.var(dim=-1)                       # (B, N)
        # Reward: high-var candidates should have low compression score (harder to compress)
        #         low-var candidates should have high compression score
        target_comp  = (1.0 - cand_var.clamp(0, 1))
        loss_comp    = F.mse_loss(comp_score, target_comp.detach())

        return StructureCompilerOutput(
            concept_center      = center,
            concept_log_width   = log_width,
            procedure_embedding = proc,
            causal_embedding    = causal,
            trigger_embedding   = trigger,
            transform_embedding = transform,
            compression_score   = comp_score,
            structure_tokens    = struct_tokens,
            loss_compression    = loss_comp,
            loss_total          = loss_comp,
        )


class _TypedHead(nn.Module):
    """Lightweight two-layer head with layer-norm output."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
