"""
TAC-SM Structure Memory

Primary innovation: stores reusable computational structures rather than text.

Each record tracks:
  - what worked (embedding, task_type)
  - where it worked (family_id, expert_id)
  - why it worked (success_score)
  - whether it survived perturbation (survival_score)
  - whether it transferred (transfer_score)

Operations: READ / WRITE / UPDATE / PRUNE
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryConfig


@dataclass
class StructureRecord:
    """Single entry in Structure Memory."""
    structure_id:   str
    family_id:      int
    expert_id:      int
    task_type:      str
    embedding:      torch.Tensor    # (embedding_dim,) on CPU
    success_score:  float = 0.0
    transfer_score: float = 0.0
    survival_score: float = 1.0
    usage_count:    int   = 0
    timestamp:      float = field(default_factory=time.time)
    # Identity fields (TAC-Prime-ID001) — optional; retrieval degrades gracefully
    identity_id:        Optional[int]          = None
    identity_embedding: Optional[torch.Tensor] = None  # (d_model,) on CPU

    def overall_score(self) -> float:
        return (
            0.4 * self.success_score
            + 0.3 * self.transfer_score
            + 0.3 * self.survival_score
        )


class StructureMemory(nn.Module):
    """
    In-memory vector store for reusable structures.
    Retrieval uses cosine similarity + family matching.
    Write threshold: only persist structures above success threshold.
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.cfg  = cfg
        self._store: Dict[str, StructureRecord] = {}
        self._step = 0

        # Learnable read projection (query → memory key space)
        self.query_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim, bias=False)
        nn.init.eye_(self.query_proj.weight)

    # ── READ ──────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        family_id: Optional[int] = None,
        top_k: Optional[int] = None,
        active_identity_id: Optional[int] = None,
        identity_memory_bias_scale: float = 0.25,
    ) -> List[StructureRecord]:
        """
        query_embedding : (embedding_dim,)
        active_identity_id : optional int — identity currently active; adds a
            small identity_match_bonus controlled by identity_memory_bias_scale.
            Gracefully degrades when records have no identity_id.
        Returns top-k records sorted by combined score.
        """
        if not self._store:
            return []

        top_k = top_k or self.cfg.retrieval_top_k
        q     = F.normalize(self.query_proj(query_embedding.unsqueeze(0)), dim=-1)

        records  = list(self._store.values())
        if family_id is not None:
            family_candidates = [r for r in records if r.family_id == family_id]
            if family_candidates:
                records = family_candidates

        # Stack embeddings for batched cosine similarity
        embs = torch.stack([r.embedding for r in records]).to(q.device)
        embs = F.normalize(embs, dim=-1)
        sims = (embs @ q.T).squeeze(-1)    # (N,)

        # Combine similarity with survival score
        survival = torch.tensor([r.survival_score for r in records], device=q.device)

        # Identity match bonus — small; degrades gracefully when no identity on record
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

        combined = 0.7 * sims + 0.3 * survival + identity_bonus

        k      = min(top_k, len(records))
        topk_i = combined.topk(k).indices.tolist()
        return [records[i] for i in topk_i]

    def retrieve_batch(
        self,
        query_embeddings: torch.Tensor,
        top_k: Optional[int] = None,
        active_identity_ids: Optional[List[int]] = None,
        identity_memory_bias_scale: float = 0.25,
    ) -> List[List[StructureRecord]]:
        """
        query_embeddings    : (B, embedding_dim)
        active_identity_ids : optional list of length B — identity per item
        Returns list of top-k results per query.
        """
        if active_identity_ids is None:
            return [self.retrieve(q, top_k=top_k) for q in query_embeddings]
        return [
            self.retrieve(
                q, top_k=top_k,
                active_identity_id=aid,
                identity_memory_bias_scale=identity_memory_bias_scale,
            )
            for q, aid in zip(query_embeddings, active_identity_ids)
        ]

    # ── WRITE ─────────────────────────────────────────────────────────────────

    def write(
        self,
        embedding:     torch.Tensor,
        family_id:     int,
        expert_id:     int,
        task_type:     str,
        success_score: float,
        survival_score: float = 1.0,
        transfer_score: float = 0.0,
        identity_id:        Optional[int]           = None,
        identity_embedding: Optional[torch.Tensor]  = None,
    ) -> Optional[str]:
        """
        Write a new structure. Returns structure_id if written, else None.
        Only persists if success_score >= write_threshold.
        identity_id / identity_embedding are optional (TAC-Prime-ID001).
        """
        if success_score < self.cfg.write_threshold:
            return None

        # Prune if at capacity
        if len(self._store) >= self.cfg.max_structures:
            self._prune()

        sid = str(uuid.uuid4())[:16]
        record = StructureRecord(
            structure_id       = sid,
            family_id          = family_id,
            expert_id          = expert_id,
            task_type          = task_type,
            embedding          = embedding.detach().cpu().float(),
            success_score      = success_score,
            transfer_score     = transfer_score,
            survival_score     = survival_score,
            usage_count        = 0,
            timestamp          = time.time(),
            identity_id        = identity_id,
            identity_embedding = identity_embedding.detach().cpu().float()
                                 if identity_embedding is not None else None,
        )
        self._store[sid] = record
        return sid

    # ── UPDATE ────────────────────────────────────────────────────────────────

    def update(
        self,
        structure_id:  str,
        success_delta: float = 0.0,
        transfer_delta: float = 0.0,
        survival_delta: float = 0.0,
    ):
        """Update metrics for an existing structure (after reuse or transfer)."""
        if structure_id not in self._store:
            return
        r = self._store[structure_id]
        r.usage_count    += 1
        r.success_score   = min(1.0, r.success_score  + success_delta)
        r.transfer_score  = min(1.0, r.transfer_score + transfer_delta)
        r.survival_score  = min(1.0, r.survival_score + survival_delta)
        r.timestamp       = time.time()

    def decay_all(self):
        """Apply survival decay to all structures. Call periodically."""
        for r in self._store.values():
            r.survival_score *= self.cfg.decay_rate

    # ── PRUNE ─────────────────────────────────────────────────────────────────

    def _prune(self):
        """Remove weakest structures until below capacity."""
        target = int(self.cfg.max_structures * 0.9)
        scored = sorted(
            self._store.items(),
            key=lambda kv: kv[1].overall_score(),
        )
        remove_n = len(self._store) - target
        for sid, _ in scored[:remove_n]:
            del self._store[sid]

    def prune_weak(self):
        """Remove structures below survival threshold."""
        to_remove = [
            sid for sid, r in self._store.items()
            if r.survival_score < self.cfg.prune_threshold
        ]
        for sid in to_remove:
            del self._store[sid]

    # ── Stats ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        if not self._store:
            return {"size": 0}
        records = list(self._store.values())
        return {
            "size":          len(records),
            "avg_success":   sum(r.success_score  for r in records) / len(records),
            "avg_transfer":  sum(r.transfer_score for r in records) / len(records),
            "avg_survival":  sum(r.survival_score for r in records) / len(records),
            "avg_usage":     sum(r.usage_count    for r in records) / len(records),
            "family_dist":   self._family_dist(records),
        }

    def _family_dist(self, records: List[StructureRecord]) -> Dict[int, int]:
        dist: Dict[int, int] = {}
        for r in records:
            dist[r.family_id] = dist.get(r.family_id, 0) + 1
        return dist


# ── Differentiable Memory Read ─────────────────────────────────────────────────

class MemoryReadHead(nn.Module):
    """
    Produces a differentiable read from Structure Memory.
    Uses soft attention over retrieved embeddings so gradients flow through.

    Returns a memory context vector (B, T, embedding_dim).
    """

    def __init__(self, d_model: int, cfg: MemoryConfig):
        super().__init__()
        self.cfg       = cfg
        self.attn_proj = nn.Linear(d_model, cfg.embedding_dim, bias=False)
        self.out_proj  = nn.Linear(cfg.embedding_dim, d_model, bias=False)
        nn.init.normal_(self.attn_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight,  std=0.02)

    def forward(
        self,
        hidden: torch.Tensor,
        retrieved: List[List[StructureRecord]],
    ) -> torch.Tensor:
        """
        hidden    : (B, T, d_model)
        retrieved : list of B lists, each containing StructureRecords (or empty)
        Returns   : (B, T, d_model)
        """
        B, T, D = hidden.shape
        device  = hidden.device

        queries = self.attn_proj(hidden)  # (B, T, emb_dim)
        out     = torch.zeros_like(queries)

        for b in range(B):
            recs = retrieved[b]
            if not recs:
                continue
            mem_embs = torch.stack([r.embedding for r in recs]).to(device)  # (K, emb_dim)
            q        = F.normalize(queries[b], dim=-1)                       # (T, emb_dim)
            keys     = F.normalize(mem_embs, dim=-1)                         # (K, emb_dim)
            scores   = q @ keys.T                                             # (T, K)
            attn     = F.softmax(scores, dim=-1)                              # (T, K)
            out[b]   = attn @ mem_embs                                        # (T, emb_dim)

        return self.out_proj(out)  # (B, T, d_model)
