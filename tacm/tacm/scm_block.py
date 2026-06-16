"""
TAC-SCM-REAL001: Integrated Structure Language Block

Wraps a standard TransformerBlock and inserts the full SCM pipeline:

  hidden_states
    ↓ TransformerBlock (attention + FFN)
    ↓ StructureDiscoveryLayer  (JEPA latents + candidates)
    ↓ StructureCompiler        (latents → typed slots)
    ↓ StructureIdentityFieldLayer (route + read + update state)
    ↓ StructureMemory.read()   (retrieve from long-term memory)
    ↓ NSFSurvivalScorer        (score structure candidates)
    ↓ StructureMemory.write()  (write survivors to memory)
    ↓ DPSLRefinementLayer      (refine embeddings)
    ↓ fusion projection
    ↓ residual merge
    ↓ output hidden_states

When enable_scm=False the block is a transparent TransformerBlock wrapper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scm_config  import TACSCMConfig
from .scm_types   import (
    StructureDiscoveryOutput, StructureCompilerOutput,
    StructureIdentityState, StructureMemoryOutput,
    SurvivalOutput, DPSLRefinementOutput,
)
from .scm_discovery  import StructureDiscoveryLayer
from .scm_compiler   import StructureCompiler
from .scm_identity   import StructureIdentityFieldLayer
from .scm_memory     import StructureMemory
from .scm_survival   import NSFSurvivalScorer
from .scm_refinement import DPSLRefinementLayer
from .backbone       import TransformerBlock, TransformerConfig


class SCMBlockOutput:
    """Output container for IntegratedStructureLanguageBlock."""
    __slots__ = [
        "hidden_states", "structure_state", "aux_losses", "metrics",
        "discovery_out", "compiler_out", "survival_out", "refinement_out",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        for s in self.__slots__:
            if not hasattr(self, s):
                setattr(self, s, None)


class IntegratedStructureLanguageBlock(nn.Module):
    """
    One interleaved structure–language block.

    Parameters
    ----------
    cfg        : TACSCMConfig
    memory     : StructureMemory — shared across blocks (passed in, not owned)
    layer_idx  : int — which transformer layer this wraps

    The memory is shared so all SCM blocks in the model contribute to and
    read from the same long-term structure store.
    """

    def __init__(
        self,
        cfg:       TACSCMConfig,
        memory:    StructureMemory,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.cfg       = cfg
        self.memory    = memory
        self.layer_idx = layer_idx

        # ── Language block ─────────────────────────────────────────────────────
        tc = TransformerConfig(
            vocab_size  = cfg.vocab_size,
            d_model     = cfg.d_model,
            n_layers    = 1,             # only need the block params
            n_heads     = cfg.n_heads,
            n_kv_heads  = cfg.n_kv_heads,
            ffn_dim     = cfg.d_ff,
            max_seq_len = cfg.max_seq_len,
            dropout     = cfg.dropout,
            rope_base   = cfg.rope_base,
            norm_eps    = cfg.norm_eps,
            use_flash_attn         = True,
            gradient_checkpointing = False,  # handled by model
        )
        self.transformer_block = TransformerBlock(tc)

        if not cfg.enable_scm:
            return  # no SCM modules needed

        # ── SCM modules ────────────────────────────────────────────────────────
        if cfg.enable_structure_discovery:
            self.discovery = StructureDiscoveryLayer(cfg)

        if cfg.enable_structure_compiler:
            self.compiler = StructureCompiler(cfg)

        if cfg.enable_structure_identity:
            self.identity = StructureIdentityFieldLayer(cfg)

        if cfg.enable_nsf_survival:
            self.survival = NSFSurvivalScorer(cfg)

        if cfg.enable_dpsl_refinement:
            self.refinement = DPSLRefinementLayer(cfg)

        # ── Fusion ─────────────────────────────────────────────────────────────
        if cfg.enable_language_structure_fusion:
            self.fusion_proj = nn.Sequential(
                nn.Linear(cfg.d_model + cfg.d_structure, cfg.d_model, bias=False),
                nn.LayerNorm(cfg.d_model),
            )
            nn.init.normal_(self.fusion_proj[0].weight, std=0.02)

        # ── Memory read projection: d_structure → d_model ─────────────────────
        self.mem_read_proj = nn.Linear(cfg.d_structure, cfg.d_model, bias=False)
        nn.init.normal_(self.mem_read_proj.weight, std=0.02)

        # Query projection: d_model → d_structure (for memory read)
        self.hidden_to_query = nn.Linear(cfg.d_model, cfg.d_structure, bias=False)
        nn.init.normal_(self.hidden_to_query.weight, std=0.02)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states:    torch.Tensor,
        freqs_cis:        torch.Tensor,
        structure_state:  Optional[StructureIdentityState] = None,
        attention_mask:   Optional[torch.Tensor]           = None,
        feedback:         Optional[torch.Tensor]           = None,
    ) -> SCMBlockOutput:

        cfg    = self.cfg
        B, T, D = hidden_states.shape
        device = hidden_states.device

        aux_losses: Dict[str, torch.Tensor] = {}
        metrics:    Dict[str, float]        = {}

        # ── 1. Language block ─────────────────────────────────────────────────
        h = self.transformer_block(hidden_states, freqs_cis, attention_mask)

        if not cfg.enable_scm:
            return SCMBlockOutput(
                hidden_states   = h,
                structure_state = structure_state,
                aux_losses      = aux_losses,
                metrics         = metrics,
            )

        # ── 2. Structure Discovery ────────────────────────────────────────────
        discovery_out = None
        structure_candidates = None

        if cfg.enable_structure_discovery and hasattr(self, "discovery"):
            discovery_out = self.discovery(h, attention_mask)
            structure_candidates = discovery_out.structure_candidates  # (B, n_cand, d_str)
            aux_losses["discovery"] = (
                cfg.discovery_loss_weight * discovery_out.loss_total
            )
            metrics["discovery_collapse"] = discovery_out.collapse_metric.item()
            metrics["discovery_loss"]     = discovery_out.loss_total.item()

        # ── 3. Structure Compiler ─────────────────────────────────────────────
        compiler_out = None
        struct_tokens = None

        if (cfg.enable_structure_compiler
                and hasattr(self, "compiler")
                and structure_candidates is not None
                and discovery_out is not None):
            compiler_out = self.compiler(h, discovery_out.latent_state, structure_candidates)
            struct_tokens = compiler_out.structure_tokens  # (B, n_cand, d_str)
            aux_losses["compiler"] = compiler_out.loss_total

        candidates_for_identity = struct_tokens if struct_tokens is not None else structure_candidates

        # ── 4. Structure Identity Field ───────────────────────────────────────
        identity_readout = None
        new_structure_state = structure_state

        if cfg.enable_structure_identity and hasattr(self, "identity"):
            (h_updated, new_structure_state,
             route_logits, route_weights,
             identity_readout, id_aux) = self.identity(
                h,
                structure_candidates = candidates_for_identity,
                state                = structure_state,
                attention_mask       = attention_mask,
            )
            h = h_updated
            aux_losses.update(id_aux)

            # Compute route entropy metric
            ent = -(route_weights * (route_weights + 1e-9).log()).sum(-1).mean()
            metrics["route_entropy"] = ent.item()

        # ── 5. Structure Memory Read ──────────────────────────────────────────
        memory_context = None

        if cfg.enable_structure_memory:
            # Query: prefer structure-space candidates, else project hidden mean
            if struct_tokens is not None:
                query = struct_tokens.mean(dim=1)              # (B, d_str)
            else:
                query = self.hidden_to_query(h.mean(dim=1))   # (B, d_str)

            mem_out      = self.memory.read(query)           # StructureMemoryOutput
            memory_context = self.mem_read_proj(
                mem_out.context_vector
            )                                                # (B, d_model)
            metrics["memory_retrieval_score"] = mem_out.retrieval_scores.mean().item()

        # ── 6. NSF Survival Scoring ───────────────────────────────────────────
        survival_out = None

        if (cfg.enable_nsf_survival
                and hasattr(self, "survival")
                and struct_tokens is not None):
            # Flatten batch × candidates for scoring
            N_total = B * struct_tokens.shape[1]
            flat_embs = struct_tokens.reshape(N_total, -1)  # (B*n, d_str)

            # Reuse signals: proxy from compiler compression score
            if compiler_out is not None:
                comp_signal = compiler_out.compression_score.reshape(N_total)
            else:
                comp_signal = None

            survival_out = self.survival(
                flat_embs,
                compression_signal = comp_signal,
            )
            aux_losses["survival"] = survival_out.loss_total
            metrics["mean_survival"] = survival_out.survival_score.mean().item()

            # ── 7. Memory Write ─────────────────────────────────────────────
            if cfg.enable_memory_write and self.training:
                write_candidates = flat_embs.detach()
                write_survival   = survival_out.survival_score.detach()
                write_mask       = survival_out.write_gate.detach() > 0.5
                self.memory.write(write_candidates, write_survival, write_mask)

        # ── 8. DPSL Refinement ────────────────────────────────────────────────
        refinement_out = None

        if (cfg.enable_dpsl_refinement
                and hasattr(self, "refinement")
                and struct_tokens is not None
                and survival_out is not None):
            N_total   = B * struct_tokens.shape[1]
            flat_embs = struct_tokens.reshape(N_total, -1)

            # Feedback: from identity readout if available
            if identity_readout is not None:
                # (B, T, d_model) → (B, d_model) → project to d_str
                fb_raw = identity_readout.mean(dim=1)        # (B, d_model)
                fb_proj = self.mem_read_proj(fb_raw)[:, :self.cfg.d_structure]
                if fb_proj.shape[-1] == self.cfg.d_structure:
                    feedback_flat = fb_proj.unsqueeze(1).expand(
                        B, struct_tokens.shape[1], -1
                    ).reshape(N_total, -1)
                else:
                    feedback_flat = None
            else:
                feedback_flat = feedback.reshape(N_total, -1) if feedback is not None else None

            refinement_out = self.refinement(
                flat_embs,
                survival_out.survival_score,
                feedback_flat,
            )
            aux_losses["refinement"] = refinement_out.loss_total

        # ── 9. Language–Structure Fusion ──────────────────────────────────────
        if (cfg.enable_language_structure_fusion
                and hasattr(self, "fusion_proj")
                and struct_tokens is not None):
            # Summary of structure tokens: mean pool
            struct_summary = struct_tokens.mean(dim=1, keepdim=True).expand(-1, T, -1)  # (B, T, d_str)
            fused = self.fusion_proj(
                torch.cat([h, struct_summary], dim=-1)
            )                                                # (B, T, d_model)
            h = h + fused

        # ── 10. Memory context injection ──────────────────────────────────────
        if memory_context is not None:
            h = h + 0.1 * memory_context.unsqueeze(1)

        # ── 11. Gradient checkpointing handled by parent model ─────────────────

        return SCMBlockOutput(
            hidden_states   = h,
            structure_state = new_structure_state,
            aux_losses      = aux_losses,
            metrics         = metrics,
            discovery_out   = discovery_out,
            compiler_out    = compiler_out,
            survival_out    = survival_out,
            refinement_out  = refinement_out,
        )
