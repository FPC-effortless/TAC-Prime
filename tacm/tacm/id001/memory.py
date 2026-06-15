"""
TAC-Prime-ID001: Identity-Biased Memory (NumPy simulation)

Pure-Python / NumPy implementations of StructureMemory and ProceduralMemory
with identity-match bonus scoring.  Used by the benchmark and unit tests
without requiring PyTorch.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ── Structure Records ─────────────────────────────────────────────────────────

@dataclass
class StructureRecordNP:
    structure_id:   str
    family_id:      int
    expert_id:      int
    task_type:      str
    embedding:      np.ndarray      # (embedding_dim,)
    success_score:  float = 0.0
    transfer_score: float = 0.0
    survival_score: float = 1.0
    usage_count:    int   = 0
    timestamp:      float = field(default_factory=time.time)
    identity_id:    Optional[int] = None

    def overall_score(self) -> float:
        return (
            0.4 * self.success_score
            + 0.3 * self.transfer_score
            + 0.3 * self.survival_score
        )


@dataclass
class ProceduralRecordNP:
    procedure_id:   str
    family:         str
    task_type:      str
    steps:          List[str]
    embedding:      np.ndarray      # (embedding_dim,)  mean-pool of steps
    success_rate:   float = 0.0
    transfer_rate:  float = 0.0
    reuse_count:    int   = 0
    timestamp:      float = field(default_factory=time.time)
    identity_id:    Optional[int] = None

    def overall_score(self) -> float:
        return (
            0.5 * self.success_rate
            + 0.3 * self.transfer_rate
            + 0.2 * min(self.reuse_count / 10, 1.0)
        )


# ── Identity-Biased Structure Memory ─────────────────────────────────────────

class IdentityStructureMemory:
    """
    NumPy-based structure memory with identity-match bonus.

    Retrieval score:
        score = 0.7 * cosine_sim + 0.3 * survival + identity_match_bonus
    where identity_match_bonus = identity_memory_bias_scale
                                 if record.identity_id == active_identity_id
                                 else 0.0
    """

    def __init__(
        self,
        embedding_dim:    int   = 64,
        max_structures:   int   = 1024,
        write_threshold:  float = 0.0,
    ):
        self.embedding_dim   = embedding_dim
        self.max_structures  = max_structures
        self.write_threshold = write_threshold
        self._store: Dict[str, StructureRecordNP] = {}

    def write(
        self,
        embedding:     np.ndarray,
        family_id:     int,
        expert_id:     int,
        task_type:     str,
        success_score: float,
        survival_score: float = 1.0,
        transfer_score: float = 0.0,
        identity_id:   Optional[int] = None,
    ) -> Optional[str]:
        if success_score < self.write_threshold:
            return None
        if len(self._store) >= self.max_structures:
            self._prune()
        sid = str(uuid.uuid4())[:16]
        self._store[sid] = StructureRecordNP(
            structure_id   = sid,
            family_id      = family_id,
            expert_id      = expert_id,
            task_type      = task_type,
            embedding      = embedding.copy().astype(np.float32),
            success_score  = success_score,
            transfer_score = transfer_score,
            survival_score = survival_score,
            identity_id    = identity_id,
        )
        return sid

    def retrieve(
        self,
        query_embedding:            np.ndarray,
        top_k:                      int  = 4,
        family_id:                  Optional[int] = None,
        active_identity_id:         Optional[int] = None,
        identity_memory_bias_scale: float = 0.25,
    ) -> List[StructureRecordNP]:
        if not self._store:
            return []
        records = list(self._store.values())
        if family_id is not None:
            fam = [r for r in records if r.family_id == family_id]
            if fam:
                records = fam

        embs = np.stack([r.embedding for r in records])     # (N, D)
        q    = _normalize(query_embedding)
        embs_n = _normalize(embs)
        sims = embs_n @ q                                    # (N,)

        survival = np.array([r.survival_score for r in records])

        if active_identity_id is not None and identity_memory_bias_scale > 0.0:
            id_bonus = np.array([
                identity_memory_bias_scale
                if (r.identity_id is not None and r.identity_id == active_identity_id)
                else 0.0
                for r in records
            ])
        else:
            id_bonus = np.zeros(len(records))

        combined = 0.7 * sims + 0.3 * survival + id_bonus
        k        = min(top_k, len(records))
        top_idx  = combined.argsort()[::-1][:k]
        return [records[i] for i in top_idx]

    def clear(self):
        self._store.clear()

    def _prune(self):
        target  = int(self.max_structures * 0.9)
        scored  = sorted(self._store.items(), key=lambda kv: kv[1].overall_score())
        remove_n = len(self._store) - target
        for sid, _ in scored[:remove_n]:
            del self._store[sid]

    def __len__(self) -> int:
        return len(self._store)


# ── Identity-Biased Procedural Memory ────────────────────────────────────────

class IdentityProceduralMemory:
    """
    NumPy-based procedural memory with identity-match bonus.

    Retrieval score:
        score = 0.6 * cosine_sim + 0.4 * overall_score + identity_match_bonus
    """

    def __init__(
        self,
        embedding_dim:  int = 64,
        max_procedures: int = 1024,
    ):
        self.embedding_dim   = embedding_dim
        self.max_procedures  = max_procedures
        self._store: Dict[str, ProceduralRecordNP] = {}

    def write(
        self,
        family:        str,
        task_type:     str,
        steps:         List[str],
        embedding:     np.ndarray,
        success_rate:  float = 0.0,
        identity_id:   Optional[int] = None,
    ) -> str:
        if len(self._store) >= self.max_procedures:
            self._prune()
        pid = str(uuid.uuid4())[:16]
        self._store[pid] = ProceduralRecordNP(
            procedure_id = pid,
            family       = family,
            task_type    = task_type,
            steps        = list(steps),
            embedding    = embedding.copy().astype(np.float32),
            success_rate = success_rate,
            identity_id  = identity_id,
        )
        return pid

    def retrieve(
        self,
        query_embedding:            np.ndarray,
        top_k:                      int  = 4,
        family:                     Optional[str] = None,
        active_identity_id:         Optional[int] = None,
        identity_memory_bias_scale: float = 0.25,
    ) -> List[ProceduralRecordNP]:
        if not self._store:
            return []
        records = list(self._store.values())
        if family is not None:
            fam = [r for r in records if r.family == family]
            if fam:
                records = fam

        embs   = np.stack([r.embedding for r in records])
        q      = _normalize(query_embedding)
        embs_n = _normalize(embs)
        sims   = embs_n @ q

        scores = np.array([r.overall_score() for r in records])

        if active_identity_id is not None and identity_memory_bias_scale > 0.0:
            id_bonus = np.array([
                identity_memory_bias_scale
                if (r.identity_id is not None and r.identity_id == active_identity_id)
                else 0.0
                for r in records
            ])
        else:
            id_bonus = np.zeros(len(records))

        combined = 0.6 * sims + 0.4 * scores + id_bonus
        k        = min(top_k, len(records))
        top_idx  = combined.argsort()[::-1][:k]
        return [records[i] for i in top_idx]

    def clear(self):
        self._store.clear()

    def _prune(self):
        target   = int(self.max_procedures * 0.9)
        scored   = sorted(self._store.items(), key=lambda kv: kv[1].overall_score())
        remove_n = len(self._store) - target
        for pid, _ in scored[:remove_n]:
            del self._store[pid]

    def __len__(self) -> int:
        return len(self._store)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalise along last axis. Safe against zero vectors."""
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / (n + 1e-8)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + 1e-8)
