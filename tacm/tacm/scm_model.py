"""
TAC-SCM-REAL001: Structure-Native Language Model

TACSCMLanguageModel is the real, trainable model class.

Architecture
------------
token embedding  (vocab_size × d_model)
    ↓
positional: RoPE (from backbone.py)
    ↓
[n_layers transformer layers, with SCM blocks every scm_layer_interval]
    ↓
final RMSNorm
    ↓
LM head (d_model → vocab_size)  [optionally tied to embedding]

Each SCM block:
  TransformerBlock → StructureDiscovery → StructureCompiler
  → StructureIdentityField → StructureMemoryRead → NSFSurvival
  → MemoryWrite → DPSLRefinement → fusion

State carried across calls
--------------------------
structure_state : StructureIdentityState  — active structure slots per-batch
memory_state    : StructureMemory bank snapshot (for persistence)

Generation
----------
generate_text(): minimal autoregressive decode carrying structure_state.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .scm_config  import TACSCMConfig
from .scm_types   import StructureIdentityState, TACSCMOutput
from .scm_block   import IntegratedStructureLanguageBlock, SCMBlockOutput
from .scm_memory  import StructureMemory
from .backbone    import (
    TransformerConfig, TransformerBlock,
    precompute_freqs_cis, RMSNorm,
)


class TACSCMLanguageModel(nn.Module):
    """
    Structure-Native Language Model.

    Can be trained as a standard language model (labels=input_ids shifted
    by one).  The SCM pipeline runs alongside the language model and produces
    auxiliary losses that are added to the LM loss.

    Forward signature
    -----------------
    forward(
        input_ids,
        labels=None,
        attention_mask=None,
        structure_state=None,
        memory_state=None,
        feedback=None,
        return_state=True,
        return_metrics=True,
    ) → TACSCMOutput
    """

    def __init__(self, cfg: TACSCMConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embeddings ─────────────────────────────────────────────────────────
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_drop  = nn.Dropout(cfg.dropout)

        # ── RoPE frequencies (shared across all layers) ────────────────────────
        tc = self._make_tc(cfg)
        freqs = precompute_freqs_cis(
            cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_base
        )
        self.register_buffer("freqs_cis", freqs, persistent=False)

        # ── Shared structure memory ────────────────────────────────────────────
        self.struct_memory = StructureMemory(cfg)

        # ── Layer stack ───────────────────────────────────────────────────────
        # Every scm_layer_interval transformer layers → one SCM block replaces
        # that layer.  Other layers are plain TransformerBlocks.
        self.layers = nn.ModuleList()
        self.is_scm_layer: List[bool] = []

        for i in range(cfg.n_layers):
            is_scm = cfg.enable_scm and (i % cfg.scm_layer_interval == 0)
            self.is_scm_layer.append(is_scm)
            if is_scm:
                self.layers.append(
                    IntegratedStructureLanguageBlock(cfg, self.struct_memory, i)
                )
            else:
                self.layers.append(TransformerBlock(tc))

        # ── Final norm + LM head ───────────────────────────────────────────────
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_lm_head:
            self.lm_head.weight = self.token_embed.weight

        # ── Initialise weights ─────────────────────────────────────────────────
        self._init_weights()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_tc(cfg: TACSCMConfig) -> TransformerConfig:
        return TransformerConfig(
            vocab_size  = cfg.vocab_size,
            d_model     = cfg.d_model,
            n_layers    = cfg.n_layers,
            n_heads     = cfg.n_heads,
            n_kv_heads  = cfg.n_kv_heads,
            ffn_dim     = cfg.d_ff,
            max_seq_len = cfg.max_seq_len,
            dropout     = cfg.dropout,
            rope_base   = cfg.rope_base,
            norm_eps    = cfg.norm_eps,
            use_flash_attn         = True,
            gradient_checkpointing = False,
        )

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:       torch.Tensor,
        labels:          Optional[torch.Tensor]              = None,
        attention_mask:  Optional[torch.Tensor]              = None,
        structure_state: Optional[StructureIdentityState]    = None,
        memory_state:    Optional[Dict[str, torch.Tensor]]   = None,
        feedback:        Optional[torch.Tensor]              = None,
        return_state:    bool                                = True,
        return_metrics:  bool                                = True,
    ) -> TACSCMOutput:

        B, T    = input_ids.shape
        device  = input_ids.device
        cfg     = self.cfg

        assert T <= cfg.max_seq_len, f"Sequence {T} > max_seq_len {cfg.max_seq_len}"

        # ── Restore memory state if provided ──────────────────────────────────
        if memory_state is not None:
            self.struct_memory.load_memory_state(memory_state)

        # ── Token embedding ───────────────────────────────────────────────────
        h      = self.embed_drop(self.token_embed(input_ids))   # (B, T, d_model)
        freqs  = self.freqs_cis[:T].to(device)

        # ── Layer stack ───────────────────────────────────────────────────────
        aux_losses: Dict[str, torch.Tensor] = {}
        metrics:    Dict[str, float]        = {}
        current_state = structure_state

        for i, layer in enumerate(self.layers):
            if self.is_scm_layer[i]:
                if cfg.use_gradient_checkpointing and self.training:
                    def _forward(h_, freqs_, state_, mask_, fb_):
                        return layer(h_, freqs_, state_, mask_, fb_)
                    out: SCMBlockOutput = checkpoint(
                        _forward, h, freqs, current_state,
                        attention_mask, feedback,
                        use_reentrant=False,
                    )
                else:
                    out = layer(
                        h, freqs,
                        structure_state = current_state,
                        attention_mask  = attention_mask,
                        feedback        = feedback,
                    )
                h = out.hidden_states
                if return_state and cfg.enable_state_carry:
                    current_state = out.structure_state
                    if current_state is not None:
                        current_state = current_state.detach()
                # Accumulate auxiliary losses
                for k, v in out.aux_losses.items():
                    if k in aux_losses:
                        aux_losses[k] = aux_losses[k] + v
                    else:
                        aux_losses[k] = v
                if return_metrics and out.metrics:
                    for k, v in out.metrics.items():
                        metrics[k] = v
            else:
                # Plain transformer block
                if cfg.use_gradient_checkpointing and self.training:
                    h = checkpoint(layer, h, freqs, attention_mask, use_reentrant=False)
                else:
                    h = layer(h, freqs, attention_mask)

        # ── Final norm + LM head ──────────────────────────────────────────────
        h      = self.final_norm(h)
        logits = self.lm_head(h)                               # (B, T, vocab)

        # ── Language modelling loss ───────────────────────────────────────────
        lm_loss = None
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.reshape(-1, cfg.vocab_size),
                shift_labels.reshape(-1),
                ignore_index = -100,
            )

        # ── Total loss = LM + weighted auxiliaries ────────────────────────────
        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss
            for k, v in aux_losses.items():
                if v.requires_grad or not v.requires_grad:
                    total_loss = total_loss + v

        # ── Memory state snapshot ─────────────────────────────────────────────
        new_memory_state = None
        if return_state:
            new_memory_state = self.struct_memory.save_memory_state()

        # ── Memory maintenance ────────────────────────────────────────────────
        if self.training:
            self.struct_memory.step_decay()
            if self.struct_memory._step % 100 == 0:
                self.struct_memory.prune(threshold=0.01)
            # Update discovery EMA targets
            for i, layer in enumerate(self.layers):
                if self.is_scm_layer[i] and hasattr(layer, "discovery"):
                    layer.discovery.update_target_ema()

        return TACSCMOutput(
            logits           = logits,
            loss             = total_loss,
            lm_loss          = lm_loss,
            auxiliary_losses = aux_losses,
            structure_state  = current_state if return_state else None,
            memory_state     = new_memory_state if return_state else None,
            metrics          = metrics,
            hidden_states    = h if return_metrics else None,
        )

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_text(
        self,
        input_ids:       torch.Tensor,
        max_new_tokens:  int                                  = 128,
        temperature:     float                                = 1.0,
        top_k:           int                                  = 50,
        structure_state: Optional[StructureIdentityState]    = None,
        carry_state:     bool                                 = True,
    ) -> Tuple[torch.Tensor, Optional[StructureIdentityState]]:
        """
        Minimal autoregressive generation with structure state carry.

        Returns
        -------
        generated_ids : (B, T + max_new_tokens)
        final_state   : StructureIdentityState or None
        """
        self.eval()
        cfg    = self.cfg
        ids    = input_ids.clone()
        state  = structure_state

        for _ in range(max_new_tokens):
            if ids.shape[1] >= cfg.max_seq_len:
                # Truncate to last max_seq_len tokens
                ids = ids[:, -cfg.max_seq_len:]

            out = self.forward(
                ids,
                structure_state = state,
                return_state    = carry_state,
                return_metrics  = False,
            )

            if carry_state:
                state = out.structure_state

            # Sample from next-token distribution
            next_logits = out.logits[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs  = torch.softmax(next_logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)              # (B, 1)
            ids    = torch.cat([ids, next_t], dim=1)

        return ids, state

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> Dict[str, int]:
        d = {
            "token_embed": sum(p.numel() for p in self.token_embed.parameters()),
            "lm_head":     0 if self.cfg.tie_lm_head else sum(p.numel() for p in self.lm_head.parameters()),
            "struct_memory_projections": sum(p.numel() for p in self.struct_memory.parameters()),
        }
        n_scm = n_plain = 0
        scm_params = plain_params = 0
        for i, layer in enumerate(self.layers):
            n = sum(p.numel() for p in layer.parameters())
            if self.is_scm_layer[i]:
                n_scm      += 1
                scm_params += n
            else:
                n_plain      += 1
                plain_params += n
        d["scm_blocks"]   = scm_params
        d["plain_blocks"] = plain_params
        d["n_scm_layers"] = n_scm
        d["n_plain_layers"] = n_plain
        return d

    def memory_stats(self) -> Dict[str, float]:
        return self.struct_memory.stats()

    def reset_structure_state(self, batch_size: int) -> StructureIdentityState:
        """Create a blank structure state for a new batch."""
        from .scm_identity import StructureIdentityFieldLayer
        # Find any SCM layer to get the identity module
        for i, layer in enumerate(self.layers):
            if self.is_scm_layer[i] and hasattr(layer, "identity"):
                return layer.identity.init_state(
                    batch_size, next(self.parameters()).device
                )
        # Fallback: construct directly
        return StructureIdentityState.zeros(
            batch_size,
            self.cfg.n_identity_slots,
            self.cfg.d_structure,
            device = next(self.parameters()).device,
        )

    # ── Checkpoint save / load ────────────────────────────────────────────────

    def save_pretrained(self, out_dir: str):
        """Save model weights, config, and memory state."""
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), path / "model.pt")
        memory_state = self.struct_memory.save_memory_state()
        torch.save(memory_state, path / "memory.pt")

        cfg_dict = self.cfg.__dict__
        (path / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    @classmethod
    def load_pretrained(cls, dir_path: str, device: str = "cpu") -> "TACSCMLanguageModel":
        path = Path(dir_path)
        cfg_dict = json.loads((path / "config.json").read_text())
        cfg  = TACSCMConfig(**cfg_dict)
        model = cls(cfg)
        model.load_state_dict(
            torch.load(path / "model.pt", map_location=device), strict=False
        )
        if (path / "memory.pt").exists():
            mem_state = torch.load(path / "memory.pt", map_location=device)
            model.struct_memory.load_memory_state(mem_state)
        model = model.to(device)
        return model
