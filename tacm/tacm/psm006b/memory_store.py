"""
TAC-PSM-006B: Lightweight Procedural Memory Store
===================================================

A self-contained numpy-backed procedural memory store for PSM-006B.
Wraps PSM-001's ProceduralMemoryStore with a simplified interface and adds
PSM-006B-specific methods (write, augment, reinforce) so the repair agent
can use a clean API without requiring FAISS.

Record schema kept in a simple list + numpy matrix (fast for 60-fixture scale).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ProcedureRecord:
    """A single stored procedure record."""
    proc_id:      str
    family:       str
    task_type:    str
    steps:        List[str]
    embedding:    np.ndarray     # float32, unit-normed
    success_rate: float = 0.8
    reuse_count:  int   = 0
    retired:      bool  = False

    def to_dict(self) -> dict:
        return {
            "proc_id":      self.proc_id,
            "family":       self.family,
            "task_type":    self.task_type,
            "steps":        self.steps,
            "success_rate": self.success_rate,
            "reuse_count":  self.reuse_count,
        }


class SimpleProceduralMemoryStore:
    """
    Pure-numpy procedural memory store for PSM-006B.

    API
    ---
    write(family, task_type, steps, embedding, success_rate) -> str  (proc_id)
    retrieve(query_embedding, top_k)                         -> List[ProcedureRecord]
    augment(proc_id, extra_steps)                            -> None
    reinforce(proc_id, delta)                                -> None
    clear()                                                  -> None
    clone()                                                  -> SimpleProceduralMemoryStore
    """

    def __init__(self):
        self._records: List[ProcedureRecord] = []

    # ── Write ─────────────────────────────────────────────────────────────

    def write(
        self,
        family:       str,
        task_type:    str,
        steps:        List[str],
        embedding:    np.ndarray,
        success_rate: float = 0.8,
    ) -> str:
        """Store a procedure and return its proc_id."""
        pid = str(uuid.uuid4())[:12]
        emb = embedding.astype(np.float32)
        nrm = np.linalg.norm(emb)
        if nrm > 0:
            emb = emb / nrm
        rec = ProcedureRecord(
            proc_id      = pid,
            family       = family,
            task_type    = task_type,
            steps        = list(steps),
            embedding    = emb,
            success_rate = success_rate,
        )
        self._records.append(rec)
        return pid

    # ── Retrieve ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k:           int = 1,
    ) -> List[ProcedureRecord]:
        """Return top_k records by cosine similarity, excluding retired records."""
        active = [r for r in self._records if not r.retired]
        if not active:
            return []

        q    = query_embedding.astype(np.float32)
        nrm  = np.linalg.norm(q)
        if nrm > 0:
            q = q / nrm

        mat  = np.stack([r.embedding for r in active])          # (N, D)
        sims = mat @ q                                           # (N,)

        # Blend cosine sim with success_rate for better ranking
        scores = 0.7 * sims + 0.3 * np.array([r.success_rate for r in active])
        idxs   = np.argsort(scores)[::-1][:top_k]
        return [active[i] for i in idxs]

    # ── Update ────────────────────────────────────────────────────────────

    def augment(self, proc_id: str, extra_steps: List[str]) -> None:
        """Add oracle steps to a procedure that failed (helps next retrieval)."""
        rec = self._get(proc_id)
        if rec is None:
            return
        existing_set = set(rec.steps)
        for s in extra_steps:
            if s not in existing_set:
                rec.steps.append(s)
                existing_set.add(s)
        # Slightly penalise success_rate (procedure needed correction)
        rec.success_rate = max(0.0, rec.success_rate - 0.05)

    def reinforce(self, proc_id: str, delta: float = 0.05) -> None:
        """Increase success_rate for a procedure that led to a pytest pass."""
        rec = self._get(proc_id)
        if rec is None:
            return
        rec.success_rate = min(1.0, rec.success_rate + delta)
        rec.reuse_count += 1

    def retire(self, proc_id: str) -> None:
        """Mark a procedure as retired (excluded from retrieval)."""
        rec = self._get(proc_id)
        if rec:
            rec.retired = True

    # ── Helpers ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all records (used by reset baseline)."""
        self._records.clear()

    def clone(self) -> "SimpleProceduralMemoryStore":
        """Deep copy (used to create independent baseline instances)."""
        import copy
        new_store = SimpleProceduralMemoryStore()
        new_store._records = [copy.deepcopy(r) for r in self._records]
        return new_store

    def __len__(self) -> int:
        return len(self._records)

    def _get(self, proc_id: str) -> Optional[ProcedureRecord]:
        for r in self._records:
            if r.proc_id == proc_id:
                return r
        return None
