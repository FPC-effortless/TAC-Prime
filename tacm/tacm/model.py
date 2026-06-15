"""
TAC-SM — Full Model

Wires all components together:
  Input → TransformerBackbone → IdentityField → ConceptVolume → StructureRouter
        → MoELayer → MemoryReadHead → MultiTokenPrediction
        → VerifierHead → output

Also manages:
  - Structure Memory (read + write)
  - Procedural Memory (read)
  - Neural Survival Field
  - Lifecycle Tracker
  - Identity Field (TAC-Prime-ID001)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone    import TransformerBackbone
from .concept_volume import ConceptVolume, ConceptVolumeLoss
from .router      import StructureRouter, RouterLoss
from .experts     import MoELayer
from .memory      import StructureMemory, MemoryReadHead
from .procedural_memory import ProceduralMemory
from .survival    import SurvivalField, StructureLifecycleTracker
from .verifier    import VerifierHead, VerifierLoss, RewardBridge
from .multi_token import MultiTokenPredictionModule
from .losses      import StructureMemoryLoss, TransferLoss, SurvivalLoss, TotalLoss
from .config      import TACSMConfig
from .identity    import IdentityFieldLayer, IdentityState


class TACSM(nn.Module):
    """
    TAC-SM: Token–Algorithm–Coherence with Structure Memory.

    The model learns reusable computational structures and transfers them across tasks.

    TAC-Prime-ID001: IdentityFieldLayer is wired immediately after the backbone
    and before ConceptVolume.  When cfg.identity.use_identity_field=False, the
    identity path is fully bypassed and the model behaves identically to the
    original TAC-SM.

    Forward path:
      backbone → IdentityField → ConceptVolume
               → identity-biased StructureRouter → MoE
               → identity-biased StructureMemory
               → identity-biased ProceduralMemory
               → SurvivalField → Verifier
    """

    def __init__(self, cfg: TACSMConfig):
        super().__init__()
        self.cfg = cfg
        tc  = cfg.transformer
        cvc = cfg.concept_volume
        rc  = cfg.router
        ec  = cfg.expert
        mc  = cfg.memory
        sc  = cfg.survival
        vc  = cfg.verifier
        mtc = cfg.multi_token
        idc = cfg.identity

        # ── Core Components ─────────────────────────────────────────────────
        self.backbone      = TransformerBackbone(tc)
        self.concept_vol   = ConceptVolume(tc.d_model, cvc)
        self.router        = StructureRouter(tc.d_model, cvc.volume_dim, rc)
        self.moe           = MoELayer(tc.d_model, ec)
        self.mem_read_head = MemoryReadHead(tc.d_model, mc)
        self.multi_token   = MultiTokenPredictionModule(tc.d_model, tc, mtc)
        self.verifier      = VerifierHead(tc.d_model, vc)

        # ── Identity Field (TAC-Prime-ID001) ─────────────────────────────────
        if idc.use_identity_field:
            self.identity_field: Optional[IdentityFieldLayer] = IdentityFieldLayer(
                d_model               = tc.d_model,
                n_identities          = idc.n_identities,
                identity_energy_budget= idc.identity_energy_budget,
                identity_state_decay  = idc.identity_state_decay,
                identity_loss_weight  = idc.identity_loss_weight,
            )
        else:
            self.identity_field = None

        # ── Memory ──────────────────────────────────────────────────────────
        self.struct_memory = StructureMemory(mc)
        self.proc_memory   = ProceduralMemory(mc)

        # ── Survival ────────────────────────────────────────────────────────
        self.survival_field = SurvivalField(mc.embedding_dim, sc)
        self.lifecycle      = StructureLifecycleTracker()

        # ── Losses ──────────────────────────────────────────────────────────
        self.vol_loss      = ConceptVolumeLoss(cvc)
        self.router_loss   = RouterLoss(rc)
        self.verif_loss    = VerifierLoss()
        self.mem_loss      = StructureMemoryLoss()
        self.transfer_loss = TransferLoss()
        self.survival_loss = SurvivalLoss()
        self.total_loss    = TotalLoss(cfg.training)
        self.reward_bridge = RewardBridge()

        # ── Task embedding projector (backbone → memory dim) ────────────────
        self.task_proj = nn.Linear(tc.d_model, mc.embedding_dim, bias=False)
        nn.init.normal_(self.task_proj.weight, std=0.02)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,                       # (B, T)
        labels:         Optional[torch.Tensor]      = None, # (B, T)
        concept_labels:  Optional[torch.Tensor]     = None, # (B, T)
        parent_labels:   Optional[torch.Tensor]     = None, # (B, T)
        family_labels:   Optional[torch.Tensor]     = None, # (B, T)
        expert_labels:   Optional[torch.Tensor]     = None, # (B, T)
        action_labels:   Optional[torch.Tensor]     = None, # (B, T, n_actions)
        success_labels:  Optional[torch.Tensor]     = None, # (B,)
        failure_labels:  Optional[torch.Tensor]     = None, # (B,)
        identity_state:  Optional[IdentityState]    = None, # previous state
        return_all:     bool                        = False,
    ) -> "TACSMOutput":

        B, T = input_ids.shape
        idc  = self.cfg.identity

        # ── 1. Backbone ───────────────────────────────────────────────────
        hidden = self.backbone(input_ids)          # (B, T, d_model)

        # ── 2. Identity Field (TAC-Prime-ID001) ───────────────────────────
        identity_out        = None
        identity_context    = None
        new_identity_state  = None
        active_identity     = None
        identity_coherence  = None

        if self.identity_field is not None and idc.use_identity_field:
            identity_out = self.identity_field(hidden, identity_state)
            identity_context   = identity_out.identity_context     # (B, T, d_model)
            new_identity_state = identity_out.identity_state
            active_identity    = identity_out.active_identity       # (B,)
            identity_coherence = identity_out.identity_coherence    # (B, n_id, n_id)

            # Residual: blend identity context into hidden before ConceptVolume
            hidden = hidden + idc.identity_residual_scale * identity_context

        # ── 3. Concept Volume ─────────────────────────────────────────────
        vol_out = self.concept_vol(hidden)         # ConceptVolumeOutput

        # ── 4. Two-Level Router (identity-biased) ─────────────────────────
        routing = self.router(
            hidden,
            vol_out.center,
            identity_context        = identity_context,
            identity_router_bias_scale = idc.identity_router_bias_scale,
        )

        # ── 5. MoE Layer ──────────────────────────────────────────────────
        moe_out, expert_stats = self.moe(hidden, routing)  # (B, T, d_model)

        # ── 6. Structure Memory Read (identity-biased) ────────────────────
        task_emb    = self.task_proj(moe_out.mean(1))      # (B, emb_dim)
        family_ids  = routing.family_ids[:, 0].tolist()    # leading token family

        active_identity_ids = (
            active_identity.tolist()
            if active_identity is not None
            else None
        )
        retrieved = self.struct_memory.retrieve_batch(
            task_emb,
            top_k                      = self.cfg.memory.retrieval_top_k,
            active_identity_ids        = active_identity_ids,
            identity_memory_bias_scale = idc.identity_memory_bias_scale,
        )
        mem_context = self.mem_read_head(moe_out, retrieved)  # (B, T, d_model)

        # Add memory context
        enriched = moe_out + mem_context                   # (B, T, d_model)

        # ── 7. Multi-Token Prediction ─────────────────────────────────────
        mt_out  = self.multi_token(enriched)               # MultiTokenOutput

        # ── 8. Verifier ───────────────────────────────────────────────────
        verif_out = self.verifier(enriched)                # VerifierOutput

        # ── 9. Loss Computation ───────────────────────────────────────────
        loss_dict = {}

        if labels is not None:
            mt_losses = self.multi_token.compute_loss(
                enriched, input_ids, labels, action_labels
            )
            loss_dict["next_token"]  = mt_losses["next_token"]
            loss_dict["multi_token"] = mt_losses["multi_token"]
            if action_labels is not None:
                loss_dict["procedure"] = mt_losses["procedure"]

        vol_losses = self.vol_loss(
            vol_out, concept_labels, parent_labels,
            self.concept_vol.ema_centers if self.concept_vol.ema_initialized.any() else None,
            self.concept_vol.ema_initialized if self.concept_vol.ema_initialized.any() else None,
        )
        loss_dict["volume"] = vol_losses["total"]

        rout_losses = self.router_loss(routing, family_labels, expert_labels)
        loss_dict["family_route"] = rout_losses["family"]
        loss_dict["expert_route"] = rout_losses["expert"]

        verif_losses = self.verif_loss(verif_out, success_labels, failure_labels)
        loss_dict["verifier"] = verif_losses["total"]

        # Structure memory loss placeholder (triplet training via agent loop)
        loss_dict["structure_memory"] = torch.tensor(0.0, device=hidden.device)
        loss_dict["transfer"]         = torch.tensor(0.0, device=hidden.device)
        loss_dict["survival"]         = torch.tensor(0.0, device=hidden.device)

        # Identity auxiliary losses (TAC-Prime-ID001)
        if identity_out is not None:
            for k, v in identity_out.aux_losses.items():
                loss_dict[k] = v

        # Aggregate
        if any(v.requires_grad or not v.requires_grad for v in loss_dict.values()):
            total = self.total_loss(loss_dict)
        else:
            total = torch.tensor(0.0, device=hidden.device)

        # ── 10. Memory Write (during training, based on verifier) ─────────
        if self.training:
            self._maybe_write_memory(
                task_emb, routing, verif_out,
                active_identity_ids = active_identity_ids,
                identity_context    = identity_context,
            )
            self.survival_field.step()
            if self.survival_field.should_decay():
                self.struct_memory.decay_all()
            if self.survival_field.should_prune():
                self.struct_memory.prune_weak()
            # Update EMA for concept volumes
            self.concept_vol.update_ema(
                vol_out.center.reshape(B * T, -1),
                vol_out.family_ids.reshape(B * T),
            )

        return TACSMOutput(
            hidden             = enriched,
            lm_logits          = mt_out.lm_logits,
            verifier_out       = verif_out,
            routing            = routing,
            vol_out            = vol_out,
            expert_stats       = expert_stats,
            loss               = total,
            loss_dict          = loss_dict,
            task_emb           = task_emb,
            identity_state     = new_identity_state,
            identity_aux       = identity_out.aux_losses if identity_out is not None else None,
            active_identity    = active_identity,
            identity_coherence = identity_coherence,
        )

    @torch.no_grad()
    def _maybe_write_memory(
        self,
        task_emb:   torch.Tensor,
        routing,
        verif_out,
        active_identity_ids: Optional[List[int]]  = None,
        identity_context:    Optional[torch.Tensor] = None,
    ):
        """Write successful structures to memory after each forward pass."""
        B = task_emb.shape[0]
        for b in range(B):
            sp  = verif_out.success_prob[b].item()
            fid = routing.family_ids[b, 0].item()
            eid = routing.topk_ids[b, 0, 0].item()

            # Prepare optional identity fields
            aid = active_identity_ids[b] if active_identity_ids is not None else None
            id_emb = (
                identity_context[b].mean(0)  # mean-pool over T
                if identity_context is not None else None
            )

            sid = self.struct_memory.write(
                embedding          = task_emb[b],
                family_id          = fid,
                expert_id          = eid,
                task_type          = "training",
                success_score      = sp,
                survival_score     = 1.0,
                identity_id        = aid,
                identity_embedding = id_emb,
            )
            if sid is not None:
                self.lifecycle.register(sid)

    # ── Inference helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids:   torch.Tensor,
        max_new_tokens: int = 128,
        identity_state: Optional[IdentityState] = None,
    ) -> torch.Tensor:
        """Simple greedy decoding. Returns full sequence including prompt."""
        self.eval()
        tc     = self.cfg.transformer
        device = input_ids.device
        ids    = input_ids.clone()
        state  = identity_state

        for _ in range(max_new_tokens):
            if ids.shape[1] >= tc.max_seq_len:
                break
            out   = self.forward(ids, identity_state=state)
            # Carry identity state across generation steps
            state = out.identity_state
            logit = out.lm_logits[:, -1, :]   # (B, vocab)
            nxt   = logit.argmax(-1, keepdim=True)
            ids   = torch.cat([ids, nxt], dim=1)

        return ids

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict:
        d = {
            "backbone":      sum(p.numel() for p in self.backbone.parameters()),
            "concept_vol":   sum(p.numel() for p in self.concept_vol.parameters()),
            "router":        sum(p.numel() for p in self.router.parameters()),
            "moe":           sum(p.numel() for p in self.moe.parameters()),
            "multi_token":   sum(p.numel() for p in self.multi_token.parameters()),
            "verifier":      sum(p.numel() for p in self.verifier.parameters()),
            "mem_read_head": sum(p.numel() for p in self.mem_read_head.parameters()),
            "task_proj":     sum(p.numel() for p in self.task_proj.parameters()),
        }
        if self.identity_field is not None:
            d["identity_field"] = sum(p.numel() for p in self.identity_field.parameters())
        return d


class TACSMOutput:
    __slots__ = [
        "hidden", "lm_logits", "verifier_out", "routing",
        "vol_out", "expert_stats", "loss", "loss_dict", "task_emb",
        # TAC-Prime-ID001 additions
        "identity_state", "identity_aux", "active_identity", "identity_coherence",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Ensure identity slots are always present (None if not set)
        for slot in ("identity_state", "identity_aux", "active_identity", "identity_coherence"):
            if not hasattr(self, slot):
                setattr(self, slot, None)

    def token_ids(self) -> torch.Tensor:
        return self.lm_logits.argmax(-1)
