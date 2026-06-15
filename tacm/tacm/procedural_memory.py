"""
TAC-SM Procedural Memory Extension (TAC-S200)

Extends Structure Memory to store full procedures:
  Procedure = ordered sequence of steps with success/transfer rates.

The model retrieves procedures BEFORE generating solutions.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryConfig


@dataclass
class ProcedureStep:
    step_index:   int
    description:  str
    embedding:    Optional[torch.Tensor] = None  # (emb_dim,) on CPU

    def to_dict(self) -> dict:
        return {"step_index": self.step_index, "description": self.description}


@dataclass
class ProcedureRecord:
    """
    A reusable procedure stored in Procedural Memory.

    Example (Python Import Error):
      steps = [
        "inspect environment",
        "verify package",
        "inspect version",
        "reinstall",
        "rerun tests",
      ]
    """
    procedure_id:   str
    family:         str                     # e.g. "CodeRepair"
    task_type:      str                     # e.g. "PythonImportError"
    steps:          List[ProcedureStep]
    embedding:      torch.Tensor            # mean-pool of step embeddings (CPU)
    success_rate:   float  = 0.0
    transfer_rate:  float  = 0.0
    reuse_count:    int    = 0
    timestamp:      float  = field(default_factory=time.time)
    # Identity fields (TAC-Prime-ID001) — optional; retrieval degrades gracefully
    identity_id:        Optional[int]          = None
    identity_embedding: Optional[torch.Tensor] = None  # (d_model,) on CPU

    def overall_score(self) -> float:
        return 0.5 * self.success_rate + 0.3 * self.transfer_rate + 0.2 * min(self.reuse_count / 10, 1.0)


class ProceduralMemory(nn.Module):
    """
    Memory bank for reusable procedures.
    Retrieval: query embedding → cosine similarity over procedure embeddings.
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.cfg    = cfg
        self._store: Dict[str, ProcedureRecord] = {}

        self.query_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        nn.init.eye_(self.query_proj.weight)

    # ── READ ──────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        family: Optional[str] = None,
        top_k: int = 4,
        active_identity_id: Optional[int] = None,
        identity_memory_bias_scale: float = 0.25,
    ) -> List[ProcedureRecord]:
        """
        Retrieve top-k procedures by semantic similarity + overall score.
        active_identity_id: optional — adds identity_match_bonus for procedures
            associated with the currently active identity.  Degrades gracefully
            when records have no identity_id.
        """
        if not self._store:
            return []

        q = F.normalize(self.query_proj(query_embedding.unsqueeze(0)), dim=-1)

        records = list(self._store.values())
        if family is not None:
            fam_records = [r for r in records if r.family == family]
            if fam_records:
                records = fam_records

        embs = torch.stack([r.embedding for r in records]).to(q.device)
        embs = F.normalize(embs, dim=-1)
        sims = (embs @ q.T).squeeze(-1)

        scores = torch.tensor([r.overall_score() for r in records], device=q.device)

        # Identity match bonus — gracefully degrades when no identity on record
        if active_identity_id is not None and identity_memory_bias_scale > 0.0:
            identity_bonus = torch.tensor(
                [
                    identity_memory_bias_scale
                    if (r.identity_id is not None and r.identity_id == active_identity_id)
                    else 0.0
                    for r in records
                ],
                device=q.device,
            )
        else:
            identity_bonus = torch.zeros(len(records), device=q.device)

        combined = 0.6 * sims + 0.4 * scores + identity_bonus

        k      = min(top_k, len(records))
        topk_i = combined.topk(k).indices.tolist()
        return [records[i] for i in topk_i]

    # ── WRITE ─────────────────────────────────────────────────────────────────

    def write(
        self,
        family:       str,
        task_type:    str,
        steps:        List[str],
        step_embeddings: Optional[List[torch.Tensor]] = None,
        success_rate: float = 0.0,
        identity_id:        Optional[int]          = None,
        identity_embedding: Optional[torch.Tensor] = None,
    ) -> str:
        if len(self._store) >= self.cfg.max_structures:
            self._prune()

        pid = str(uuid.uuid4())[:16]
        proc_steps = [
            ProcedureStep(
                step_index  = i,
                description = s,
                embedding   = step_embeddings[i].cpu() if step_embeddings else None,
            )
            for i, s in enumerate(steps)
        ]

        # Build mean-pool embedding for the procedure
        if step_embeddings:
            emb = torch.stack(step_embeddings).mean(0).cpu()
        else:
            emb = torch.zeros(self.cfg.embedding_dim)

        record = ProcedureRecord(
            procedure_id       = pid,
            family             = family,
            task_type          = task_type,
            steps              = proc_steps,
            embedding          = emb,
            success_rate       = success_rate,
            identity_id        = identity_id,
            identity_embedding = identity_embedding.detach().cpu().float()
                                 if identity_embedding is not None else None,
        )
        self._store[pid] = record
        return pid

    # ── UPDATE ────────────────────────────────────────────────────────────────

    def update(
        self,
        procedure_id:   str,
        success_delta:  float = 0.0,
        transfer_delta: float = 0.0,
    ):
        if procedure_id not in self._store:
            return
        r = self._store[procedure_id]
        r.reuse_count   += 1
        r.success_rate   = min(1.0, r.success_rate  + success_delta)
        r.transfer_rate  = min(1.0, r.transfer_rate + transfer_delta)
        r.timestamp      = time.time()

    # ── PRUNE ─────────────────────────────────────────────────────────────────

    def _prune(self):
        target  = int(self.cfg.max_structures * 0.9)
        scored  = sorted(self._store.items(), key=lambda kv: kv[1].overall_score())
        remove_n = len(self._store) - target
        for pid, _ in scored[:remove_n]:
            del self._store[pid]

    # ── Encode helper ──────────────────────────────────────────────────────────

    def encode_steps_from_model(
        self,
        steps: List[str],
        encoder_fn,                    # callable: str → Tensor(emb_dim)
    ) -> List[torch.Tensor]:
        return [encoder_fn(s) for s in steps]

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        if not self._store:
            return {"size": 0}
        records = list(self._store.values())
        return {
            "size":          len(records),
            "avg_success":   sum(r.success_rate  for r in records) / len(records),
            "avg_transfer":  sum(r.transfer_rate for r in records) / len(records),
            "avg_reuse":     sum(r.reuse_count   for r in records) / len(records),
            "family_dist":   {r.family: 0 for r in records},
        }

    def format_procedure(self, record: ProcedureRecord) -> str:
        lines = [
            f"Procedure: {record.task_type} ({record.family})",
            f"Success: {record.success_rate:.2f}  Transfer: {record.transfer_rate:.2f}  Uses: {record.reuse_count}",
        ]
        for s in record.steps:
            lines.append(f"  {s.step_index + 1}. {s.description}")
        return "\n".join(lines)


# ── Procedural Context Encoder ─────────────────────────────────────────────────

class ProceduralContextEncoder(nn.Module):
    """
    Encodes retrieved procedures into a context vector the model can condition on.
    """

    def __init__(self, d_model: int, max_steps: int = 16):
        super().__init__()
        self.max_steps   = max_steps
        self.step_proj   = nn.Linear(d_model, d_model, bias=False)
        self.pool        = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.out_proj    = nn.Linear(d_model, d_model, bias=False)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(
        self,
        step_embeddings: torch.Tensor,    # (B, n_steps, d_model)
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns a single context vector per batch item: (B, d_model)."""
        B = step_embeddings.shape[0]
        q = self.query_token.expand(B, -1, -1)   # (B, 1, d_model)
        kv = self.step_proj(step_embeddings)       # (B, n_steps, d_model)

        ctx, _ = self.pool(q, kv, kv, key_padding_mask=key_padding_mask)
        return self.out_proj(ctx.squeeze(1))       # (B, d_model)
