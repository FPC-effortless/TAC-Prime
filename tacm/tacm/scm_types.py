"""
TAC-SCM-REAL001: Core Type Definitions

All dataclasses used by the SCM pipeline.  Every class that holds tensors
supports .to(device), .detach(), and optional shape validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to(tensor_or_none: Optional[torch.Tensor], device) -> Optional[torch.Tensor]:
    return tensor_or_none.to(device) if tensor_or_none is not None else None


def _detach(tensor_or_none: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return tensor_or_none.detach() if tensor_or_none is not None else None


# ── StructureObject ────────────────────────────────────────────────────────────

@dataclass
class StructureObject:
    """
    A single reusable computational structure discovered during forward passes.

    Fields
    ------
    embedding      : (d_structure,)  — the canonical embedding of this structure
    concept_center : (d_structure,)  — concept-volume center for this structure
    concept_log_width: (d_structure,) — log width of the volume (uncertainty)
    procedure_emb  : (d_structure,)  — procedure aspect of the structure
    causal_emb     : (d_structure,)  — causal aspect (what triggers outcomes)
    trigger_emb    : (d_structure,)  — trigger aspect (when to activate)
    transform_emb  : (d_structure,)  — transform aspect (how to change state)
    compression_score: float scalar tensor
    survival_score : float scalar tensor
    structure_id   : int identifier (index in memory bank)
    family_id      : int family/cluster assignment
    """
    embedding:          torch.Tensor
    concept_center:     torch.Tensor
    concept_log_width:  torch.Tensor
    procedure_emb:      torch.Tensor
    causal_emb:         torch.Tensor
    trigger_emb:        torch.Tensor
    transform_emb:      torch.Tensor
    compression_score:  torch.Tensor
    survival_score:     torch.Tensor
    structure_id:       int = -1
    family_id:          int = -1

    def to(self, device) -> "StructureObject":
        return StructureObject(
            embedding         = self.embedding.to(device),
            concept_center    = self.concept_center.to(device),
            concept_log_width = self.concept_log_width.to(device),
            procedure_emb     = self.procedure_emb.to(device),
            causal_emb        = self.causal_emb.to(device),
            trigger_emb       = self.trigger_emb.to(device),
            transform_emb     = self.transform_emb.to(device),
            compression_score = self.compression_score.to(device),
            survival_score    = self.survival_score.to(device),
            structure_id      = self.structure_id,
            family_id         = self.family_id,
        )

    def detach(self) -> "StructureObject":
        return StructureObject(
            embedding         = self.embedding.detach(),
            concept_center    = self.concept_center.detach(),
            concept_log_width = self.concept_log_width.detach(),
            procedure_emb     = self.procedure_emb.detach(),
            causal_emb        = self.causal_emb.detach(),
            trigger_emb       = self.trigger_emb.detach(),
            transform_emb     = self.transform_emb.detach(),
            compression_score = self.compression_score.detach(),
            survival_score    = self.survival_score.detach(),
            structure_id      = self.structure_id,
            family_id         = self.family_id,
        )

    def validate_shapes(self, d_structure: int):
        for name in ("embedding", "concept_center", "concept_log_width",
                     "procedure_emb", "causal_emb", "trigger_emb", "transform_emb"):
            t = getattr(self, name)
            assert t.shape == (d_structure,), \
                f"StructureObject.{name}: expected ({d_structure},), got {t.shape}"


# ── StructureBatch ─────────────────────────────────────────────────────────────

@dataclass
class StructureBatch:
    """
    Batched version of multiple structure objects.

    Fields
    ------
    embeddings       : (N, d_structure)
    concept_centers  : (N, d_structure)
    concept_log_widths: (N, d_structure)
    procedure_embs   : (N, d_structure)
    causal_embs      : (N, d_structure)
    trigger_embs     : (N, d_structure)
    transform_embs   : (N, d_structure)
    compression_scores: (N,)
    survival_scores  : (N,)
    structure_ids    : (N,) int64 or None
    family_ids       : (N,) int64 or None
    """
    embeddings:          torch.Tensor
    concept_centers:     torch.Tensor
    concept_log_widths:  torch.Tensor
    procedure_embs:      torch.Tensor
    causal_embs:         torch.Tensor
    trigger_embs:        torch.Tensor
    transform_embs:      torch.Tensor
    compression_scores:  torch.Tensor
    survival_scores:     torch.Tensor
    structure_ids:       Optional[torch.Tensor] = None
    family_ids:          Optional[torch.Tensor] = None

    def to(self, device) -> "StructureBatch":
        return StructureBatch(
            embeddings         = self.embeddings.to(device),
            concept_centers    = self.concept_centers.to(device),
            concept_log_widths = self.concept_log_widths.to(device),
            procedure_embs     = self.procedure_embs.to(device),
            causal_embs        = self.causal_embs.to(device),
            trigger_embs       = self.trigger_embs.to(device),
            transform_embs     = self.transform_embs.to(device),
            compression_scores = self.compression_scores.to(device),
            survival_scores    = self.survival_scores.to(device),
            structure_ids      = _to(self.structure_ids, device),
            family_ids         = _to(self.family_ids, device),
        )

    def detach(self) -> "StructureBatch":
        return StructureBatch(
            embeddings         = self.embeddings.detach(),
            concept_centers    = self.concept_centers.detach(),
            concept_log_widths = self.concept_log_widths.detach(),
            procedure_embs     = self.procedure_embs.detach(),
            causal_embs        = self.causal_embs.detach(),
            trigger_embs       = self.trigger_embs.detach(),
            transform_embs     = self.transform_embs.detach(),
            compression_scores = self.compression_scores.detach(),
            survival_scores    = self.survival_scores.detach(),
            structure_ids      = _detach(self.structure_ids),
            family_ids         = _detach(self.family_ids),
        )

    @property
    def n(self) -> int:
        return self.embeddings.shape[0]

    def validate_shapes(self, d_structure: int):
        N = self.n
        for name in ("embeddings", "concept_centers", "concept_log_widths",
                     "procedure_embs", "causal_embs", "trigger_embs", "transform_embs"):
            t = getattr(self, name)
            assert t.shape == (N, d_structure), \
                f"StructureBatch.{name}: expected ({N}, {d_structure}), got {t.shape}"
        for name in ("compression_scores", "survival_scores"):
            t = getattr(self, name)
            assert t.shape == (N,), f"StructureBatch.{name}: expected ({N},), got {t.shape}"


# ── StructureDiscoveryOutput ───────────────────────────────────────────────────

@dataclass
class StructureDiscoveryOutput:
    """
    Output of StructureDiscoveryLayer (JEPA-inspired).

    Fields
    ------
    latent_state            : (B, T, d_structure)  — online encoder output
    predicted_latent_state  : (B, T, d_structure)  — predictor output
    target_latent_state     : (B, T, d_structure)  — stop-gradient target
    structure_candidates    : (B, n_candidates, d_structure)  — extracted candidates
    loss_prediction         : scalar — JEPA prediction loss
    loss_variance           : scalar — spread/variance loss
    loss_covariance         : scalar — decorrelation loss
    loss_total              : scalar — weighted sum
    collapse_metric         : scalar — std of latents (0 = collapsed)
    """
    latent_state:           torch.Tensor
    predicted_latent_state: torch.Tensor
    target_latent_state:    torch.Tensor
    structure_candidates:   torch.Tensor
    loss_prediction:        torch.Tensor
    loss_variance:          torch.Tensor
    loss_covariance:        torch.Tensor
    loss_total:             torch.Tensor
    collapse_metric:        torch.Tensor

    def to(self, device) -> "StructureDiscoveryOutput":
        return StructureDiscoveryOutput(**{
            k: v.to(device) for k, v in self.__dict__.items()
        })

    def detach(self) -> "StructureDiscoveryOutput":
        return StructureDiscoveryOutput(**{
            k: v.detach() for k, v in self.__dict__.items()
        })


# ── StructureCompilerOutput ────────────────────────────────────────────────────

@dataclass
class StructureCompilerOutput:
    """
    Output of StructureCompiler.

    The compiler turns latent structure candidates into typed structure slots.

    Fields
    ------
    concept_center      : (B, n_slots, d_structure)
    concept_log_width   : (B, n_slots, d_structure)
    procedure_embedding : (B, n_slots, d_structure)
    causal_embedding    : (B, n_slots, d_structure)
    trigger_embedding   : (B, n_slots, d_structure)
    transform_embedding : (B, n_slots, d_structure)
    compression_score   : (B, n_slots)
    structure_tokens    : (B, n_slots, d_structure)  — final compiled embedding
    loss_compression    : scalar
    loss_total          : scalar
    """
    concept_center:       torch.Tensor
    concept_log_width:    torch.Tensor
    procedure_embedding:  torch.Tensor
    causal_embedding:     torch.Tensor
    trigger_embedding:    torch.Tensor
    transform_embedding:  torch.Tensor
    compression_score:    torch.Tensor
    structure_tokens:     torch.Tensor
    loss_compression:     torch.Tensor
    loss_total:           torch.Tensor

    def to(self, device) -> "StructureCompilerOutput":
        return StructureCompilerOutput(**{
            k: v.to(device) for k, v in self.__dict__.items()
        })

    def detach(self) -> "StructureCompilerOutput":
        return StructureCompilerOutput(**{
            k: v.detach() for k, v in self.__dict__.items()
        })


# ── StructureIdentityState ─────────────────────────────────────────────────────

@dataclass
class StructureIdentityState:
    """
    Stateful carrier of active computational structures across forward passes.

    This is conceptually different from the old IdentityState in identity.py:
    - Old: carries identity IDs and stability scores (symbolic)
    - SCM: carries actual structure embeddings (computational)

    Fields
    ------
    slot_embeddings   : (B, n_identity_slots, d_structure)  — active structure embeddings
    slot_weights      : (B, n_identity_slots)  — how active / strong each slot is
    route_history     : (B, n_identity_slots)  — EMA of routing frequency
    stability_scores  : (B, n_identity_slots)  — how stable each slot's content is
    decision_memory   : (B, n_identity_slots, d_structure)  — accumulated context
    step_count        : int — how many forward passes have been carried
    """
    slot_embeddings:  torch.Tensor
    slot_weights:     torch.Tensor
    route_history:    torch.Tensor
    stability_scores: torch.Tensor
    decision_memory:  torch.Tensor
    step_count:       int = 0

    def to(self, device) -> "StructureIdentityState":
        return StructureIdentityState(
            slot_embeddings  = self.slot_embeddings.to(device),
            slot_weights     = self.slot_weights.to(device),
            route_history    = self.route_history.to(device),
            stability_scores = self.stability_scores.to(device),
            decision_memory  = self.decision_memory.to(device),
            step_count       = self.step_count,
        )

    def detach(self) -> "StructureIdentityState":
        return StructureIdentityState(
            slot_embeddings  = self.slot_embeddings.detach(),
            slot_weights     = self.slot_weights.detach(),
            route_history    = self.route_history.detach(),
            stability_scores = self.stability_scores.detach(),
            decision_memory  = self.decision_memory.detach(),
            step_count       = self.step_count,
        )

    def reset(self) -> "StructureIdentityState":
        """Return zeroed state with same shape and device."""
        device = self.slot_embeddings.device
        return StructureIdentityState.zeros(
            batch_size        = self.slot_embeddings.shape[0],
            n_identity_slots  = self.slot_embeddings.shape[1],
            d_structure       = self.slot_embeddings.shape[2],
            device            = device,
        )

    @staticmethod
    def zeros(
        batch_size: int,
        n_identity_slots: int,
        d_structure: int,
        device=None,
    ) -> "StructureIdentityState":
        kw = {} if device is None else {"device": device}
        return StructureIdentityState(
            slot_embeddings  = torch.zeros(batch_size, n_identity_slots, d_structure, **kw),
            slot_weights     = torch.ones(batch_size, n_identity_slots, **kw) / n_identity_slots,
            route_history    = torch.zeros(batch_size, n_identity_slots, **kw),
            stability_scores = torch.ones(batch_size, n_identity_slots, **kw),
            decision_memory  = torch.zeros(batch_size, n_identity_slots, d_structure, **kw),
            step_count       = 0,
        )


# ── StructureMemoryOutput ──────────────────────────────────────────────────────

@dataclass
class StructureMemoryOutput:
    """
    Output of StructureMemory.read().

    Fields
    ------
    retrieved_keys      : (B, top_k, d_structure)
    retrieved_values    : (B, top_k, d_structure)
    retrieval_scores    : (B, top_k) — cosine similarities
    retrieved_ids       : (B, top_k) — memory slot indices
    retrieved_survival  : (B, top_k) — survival scores of retrieved structures
    context_vector      : (B, d_structure) — weighted combination
    """
    retrieved_keys:     torch.Tensor
    retrieved_values:   torch.Tensor
    retrieval_scores:   torch.Tensor
    retrieved_ids:      torch.Tensor
    retrieved_survival: torch.Tensor
    context_vector:     torch.Tensor

    def to(self, device) -> "StructureMemoryOutput":
        return StructureMemoryOutput(**{
            k: v.to(device) for k, v in self.__dict__.items()
        })

    def detach(self) -> "StructureMemoryOutput":
        return StructureMemoryOutput(**{
            k: v.detach() for k, v in self.__dict__.items()
        })


# ── SurvivalOutput ─────────────────────────────────────────────────────────────

@dataclass
class SurvivalOutput:
    """
    Output of NSFSurvivalScorer.

    Fields
    ------
    survival_score : (N,)  — scalar ∈ [0, 1] per structure
    decay_gate     : (N,)  — whether to decay (low = strong survival)
    write_gate     : (N,)  — whether to write to memory
    refine_gate    : (N,)  — whether to trigger refinement
    keep_mask      : (N,)  bool — True if structure should be kept
    loss_survival  : scalar
    loss_total     : scalar
    """
    survival_score: torch.Tensor
    decay_gate:     torch.Tensor
    write_gate:     torch.Tensor
    refine_gate:    torch.Tensor
    keep_mask:      torch.Tensor
    loss_survival:  torch.Tensor
    loss_total:     torch.Tensor

    def to(self, device) -> "SurvivalOutput":
        return SurvivalOutput(**{
            k: v.to(device) for k, v in self.__dict__.items()
        })

    def detach(self) -> "SurvivalOutput":
        return SurvivalOutput(**{
            k: v.detach() for k, v in self.__dict__.items()
        })


# ── DPSLRefinementOutput ───────────────────────────────────────────────────────

@dataclass
class DPSLRefinementOutput:
    """
    Output of DPSLRefinementLayer.

    Fields
    ------
    refined_embeddings : (N, d_structure)  — updated structure embeddings
    gate_values        : (N,)  — how much refinement was applied
    merge_mask         : (N,)  bool — which pairs were merged
    loss_refinement    : scalar
    loss_total         : scalar
    """
    refined_embeddings: torch.Tensor
    gate_values:        torch.Tensor
    merge_mask:         torch.Tensor
    loss_refinement:    torch.Tensor
    loss_total:         torch.Tensor

    def to(self, device) -> "DPSLRefinementOutput":
        return DPSLRefinementOutput(**{
            k: v.to(device) for k, v in self.__dict__.items()
        })

    def detach(self) -> "DPSLRefinementOutput":
        return DPSLRefinementOutput(**{
            k: v.detach() for k, v in self.__dict__.items()
        })


# ── TACSCMOutput ───────────────────────────────────────────────────────────────

@dataclass
class TACSCMOutput:
    """
    Full output of TACSCMLanguageModel.forward().

    Fields
    ------
    logits           : (B, T, vocab_size)
    loss             : scalar total loss (None if no labels)
    lm_loss          : scalar cross-entropy language loss
    auxiliary_losses : dict of str → scalar tensor
    structure_state  : updated StructureIdentityState (or None)
    memory_state     : StructureMemory internal state snapshot dict
    metrics          : dict of str → float for logging
    hidden_states    : (B, T, d_model) final hidden states
    """
    logits:           torch.Tensor
    loss:             Optional[torch.Tensor]
    lm_loss:          Optional[torch.Tensor]
    auxiliary_losses: Dict[str, torch.Tensor]
    structure_state:  Optional[StructureIdentityState]
    memory_state:     Optional[Dict[str, torch.Tensor]]
    metrics:          Dict[str, float]
    hidden_states:    Optional[torch.Tensor] = None

    def token_ids(self) -> torch.Tensor:
        return self.logits.argmax(-1)
