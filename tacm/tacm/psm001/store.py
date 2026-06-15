"""
TAC-PSM-001: ProceduralMemoryStore

Full CRUD for ProcedureTrace records, backed by FAISS (or numpy fallback).

Operations:
  CREATE  — build()       writes a new procedure
  RETRIEVE — retrieve()   returns ranked candidates
  UPDATE  — update()      modifies scores/steps after verification
  RETIRE  — retire()      marks procedure as retired
  PRUNE   — prune()       removes retired / low-survival entries
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .records import (
    ProcedureTrace,
    ProcedureStep,
    FailureMode,
    RecoveryStrategy,
    ProcedureLifecycleState,
    StructureMemoryRecordV2,
)


# ── Internal vector index (same as memory_faiss.py but local) ─────────────────

class _VecIndex:
    """Thin wrapper that has the same API whether backed by FAISS or numpy."""

    def __init__(self, dim: int):
        self.dim = dim
        if FAISS_AVAILABLE:
            self._faiss = faiss.IndexFlatIP(dim)
            self._vecs  = None
        else:
            self._faiss = None
            self._vecs: List[np.ndarray] = []

    def add(self, vec: np.ndarray):
        v = vec.astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        if FAISS_AVAILABLE:
            self._faiss.add(v.reshape(1, -1))
        else:
            self._vecs.append(v)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        ntotal = self._faiss.ntotal if FAISS_AVAILABLE else len(self._vecs)
        if ntotal == 0:
            return np.array([]), np.array([], dtype=np.int64)
        k = min(k, ntotal)
        q = query.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        if FAISS_AVAILABLE:
            D, I = self._faiss.search(q.reshape(1, -1), k)
            return D[0], I[0]
        else:
            mat  = np.stack(self._vecs)
            sims = mat @ q
            idxs = np.argsort(sims)[::-1][:k]
            return sims[idxs], idxs.astype(np.int64)

    def reset(self):
        if FAISS_AVAILABLE:
            self._faiss = faiss.IndexFlatIP(self.dim)
        else:
            self._vecs = []

    @property
    def ntotal(self) -> int:
        return self._faiss.ntotal if FAISS_AVAILABLE else len(self._vecs)


# ── ProceduralMemoryStore ─────────────────────────────────────────────────────

class ProceduralMemoryStore:
    """
    Persistent procedural memory store.

    Disk layout (save_dir):
      procedures/
        <procedure_id>.json   — full ProcedureTrace JSON
      index_meta.json         — ordered list of (procedure_id, embedding_dim) for index rebuild
    """

    PRUNE_THRESHOLD  = 0.05   # survival_score below → eligible for retirement
    MAX_PROCEDURES   = 8192

    def __init__(
        self,
        embedding_dim: int = 512,
        save_dir:      Optional[str] = None,
        max_size:      int = MAX_PROCEDURES,
    ):
        self.dim      = embedding_dim
        self.max_size = max_size
        self.save_dir = Path(save_dir) if save_dir else None

        self._index:  _VecIndex                  = _VecIndex(embedding_dim)
        self._procs:  List[ProcedureTrace]        = []   # ordered by insertion
        self._id_map: Dict[str, int]              = {}   # procedure_id → list index

        if self.save_dir and (self.save_dir / "index_meta.json").exists():
            self._load()

        backend = "FAISS" if FAISS_AVAILABLE else "numpy"
        print(f"[ProceduralMemoryStore] backend={backend}  dim={embedding_dim}"
              f"  loaded={len(self._procs)} procedures")

    # ── CREATE ────────────────────────────────────────────────────────────────

    def build(
        self,
        problem_family:    str,
        task_signature:    str,
        steps:             List[str],
        embedding:         Optional[np.ndarray] = None,
        failure_modes:     Optional[List[FailureMode]] = None,
        recovery_strategies: Optional[List[RecoveryStrategy]] = None,
        success_score:     float = 0.0,
        parent_id:         Optional[str] = None,
        selection_reason:  str = "",
    ) -> ProcedureTrace:
        """
        Create and store a new ProcedureTrace.
        Returns the stored record (with assigned procedure_id).
        """
        if len(self._procs) >= self.max_size:
            self._prune()

        pid = str(uuid.uuid4())[:16]
        proc_steps = [
            ProcedureStep(step_index=i, action=s)
            for i, s in enumerate(steps)
        ]

        emb_list: Optional[List[float]] = None
        if embedding is not None:
            emb_list = embedding.astype(float).tolist()
        else:
            # Random placeholder embedding (replaced when model encodes the trace)
            emb_list = np.random.randn(self.dim).tolist()

        trace = ProcedureTrace(
            procedure_id        = pid,
            problem_family      = problem_family,
            task_signature      = task_signature,
            steps               = proc_steps,
            failure_modes       = failure_modes or [],
            recovery_strategies = recovery_strategies or [],
            success_score       = success_score,
            parent_id           = parent_id,
            selection_reason    = selection_reason,
            lifecycle_state     = ProcedureLifecycleState.CREATED,
            embedding           = emb_list,
        )

        self._register(trace)
        if self.save_dir:
            self._save_one(trace)
        return trace

    def _register(self, trace: ProcedureTrace):
        idx = len(self._procs)
        self._procs.append(trace)
        self._id_map[trace.procedure_id] = idx
        if trace.embedding is not None:
            vec = np.array(trace.embedding, dtype=np.float32)
            self._index.add(vec)

    # ── RETRIEVE ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: np.ndarray,
        family:          Optional[str] = None,
        top_k:           int = 5,
        min_score:       float = 0.0,
    ) -> List[Tuple[float, ProcedureTrace]]:
        """
        Returns list of (combined_score, ProcedureTrace) sorted descending.
        combined_score = 0.6 * cosine_sim + 0.4 * overall_score
        """
        if not self._procs:
            return []

        fetch_k  = min(top_k * 6 if family else top_k * 2, len(self._procs))
        sims, idxs = self._index.search(query_embedding.astype(np.float32), fetch_k)

        results: List[Tuple[float, ProcedureTrace]] = []
        for sim, idx in zip(sims, idxs):
            if idx < 0 or idx >= len(self._procs):
                continue
            p = self._procs[idx]
            if p.lifecycle_state == ProcedureLifecycleState.RETIRED:
                continue
            if family is not None and p.problem_family != family:
                continue
            combined = 0.6 * float(sim) + 0.4 * p.overall_score()
            if combined >= min_score:
                results.append((combined, p))
            if len(results) == top_k:
                break

        return sorted(results, key=lambda x: -x[0])

    def get(self, procedure_id: str) -> Optional[ProcedureTrace]:
        idx = self._id_map.get(procedure_id)
        return self._procs[idx] if idx is not None else None

    def get_by_family(self, family: str) -> List[ProcedureTrace]:
        return [p for p in self._procs if p.problem_family == family
                and p.lifecycle_state != ProcedureLifecycleState.RETIRED]

    # ── UPDATE ────────────────────────────────────────────────────────────────

    def update(
        self,
        procedure_id:   str,
        success_delta:  float = 0.0,
        transfer_delta: float = 0.0,
        survival_delta: float = 0.0,
        new_step:       Optional[ProcedureStep] = None,
        new_failure:    Optional[FailureMode] = None,
        new_recovery:   Optional[RecoveryStrategy] = None,
        task_signature: Optional[str] = None,
        version_bump:   bool = False,
    ) -> Optional[ProcedureTrace]:
        p = self.get(procedure_id)
        if p is None:
            return None

        p.success_score  = min(1.0, max(0.0, p.success_score  + success_delta))
        p.transfer_score = min(1.0, max(0.0, p.transfer_score + transfer_delta))
        p.survival_score = min(1.0, max(0.0, p.survival_score + survival_delta))
        p.reuse_count   += 1
        p.last_used      = time.time()

        if task_signature and task_signature not in p.used_by_tasks:
            p.used_by_tasks.append(task_signature)

        if new_step is not None:
            p.steps.append(new_step)

        if new_failure is not None:
            existing = {f.failure_id for f in p.failure_modes}
            if new_failure.failure_id in existing:
                for f in p.failure_modes:
                    if f.failure_id == new_failure.failure_id:
                        f.frequency += 1
            else:
                p.failure_modes.append(new_failure)

        if new_recovery is not None:
            p.recovery_strategies.append(new_recovery)

        if version_bump:
            p.version += 1
            p.lifecycle_state = ProcedureLifecycleState.UPDATED

        self._advance_lifecycle(p)

        if self.save_dir:
            self._save_one(p)
        return p

    def _advance_lifecycle(self, p: ProcedureTrace):
        s = p.survival_score
        r = p.reuse_count
        t = p.transfer_score
        if s < self.PRUNE_THRESHOLD:
            p.lifecycle_state = ProcedureLifecycleState.RETIRING
        elif t >= 0.5:
            p.lifecycle_state = ProcedureLifecycleState.TRANSFERRED
        elif r >= 5 and s >= 0.5:
            p.lifecycle_state = ProcedureLifecycleState.SPECIALISED
        elif r >= 1:
            p.lifecycle_state = ProcedureLifecycleState.ACTIVE

    # ── RETIRE ────────────────────────────────────────────────────────────────

    def retire(self, procedure_id: str):
        p = self.get(procedure_id)
        if p:
            p.lifecycle_state = ProcedureLifecycleState.RETIRED
            if self.save_dir:
                self._save_one(p)

    # ── PRUNE ─────────────────────────────────────────────────────────────────

    def prune(self):
        """Remove RETIRED and low-survival entries, rebuild index."""
        keep = [
            p for p in self._procs
            if p.lifecycle_state != ProcedureLifecycleState.RETIRED
            and p.survival_score >= self.PRUNE_THRESHOLD
        ]
        self._rebuild(keep)

    def _prune(self):
        """Capacity-driven prune: drop weakest 10%."""
        target = int(self.max_size * 0.9)
        scored = sorted(self._procs, key=lambda p: p.overall_score())
        keep   = scored[len(scored) - target:]
        self._rebuild(keep)

    def _rebuild(self, keep: List[ProcedureTrace]):
        self._index.reset()
        self._procs  = []
        self._id_map = {}
        for p in keep:
            self._register(p)

    def decay_all(self, rate: float = 0.99):
        for p in self._procs:
            p.survival_score = max(0.0, p.survival_score * rate)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_one(self, trace: ProcedureTrace):
        proc_dir = self.save_dir / "procedures"
        proc_dir.mkdir(parents=True, exist_ok=True)
        path = proc_dir / f"{trace.procedure_id}.json"
        with open(path, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)
        self._save_index_meta()

    def _save_index_meta(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        meta = [{"procedure_id": p.procedure_id} for p in self._procs]
        with open(self.save_dir / "index_meta.json", "w") as f:
            json.dump(meta, f)

    def save(self):
        if not self.save_dir:
            return
        for p in self._procs:
            self._save_one(p)
        print(f"[ProceduralMemoryStore] Saved {len(self._procs)} procedures → {self.save_dir}")

    def _load(self):
        with open(self.save_dir / "index_meta.json") as f:
            meta = json.load(f)
        proc_dir = self.save_dir / "procedures"
        for entry in meta:
            pid  = entry["procedure_id"]
            path = proc_dir / f"{pid}.json"
            if not path.exists():
                continue
            with open(path) as f:
                d = json.load(f)
            trace = ProcedureTrace.from_dict(d)
            self._register(trace)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._procs)

    def stats(self) -> dict:
        procs = self._procs
        if not procs:
            return {"size": 0}
        lifecycle_dist: Dict[str, int] = {}
        for p in procs:
            k = p.lifecycle_state.value
            lifecycle_dist[k] = lifecycle_dist.get(k, 0) + 1
        family_dist: Dict[str, int] = {}
        for p in procs:
            family_dist[p.problem_family] = family_dist.get(p.problem_family, 0) + 1
        return {
            "size":           len(procs),
            "backend":        "FAISS" if FAISS_AVAILABLE else "numpy",
            "avg_success":    sum(p.success_score  for p in procs) / len(procs),
            "avg_transfer":   sum(p.transfer_score for p in procs) / len(procs),
            "avg_survival":   sum(p.survival_score for p in procs) / len(procs),
            "avg_reuse":      sum(p.reuse_count    for p in procs) / len(procs),
            "avg_steps":      sum(p.n_steps()      for p in procs) / len(procs),
            "lifecycle_dist": lifecycle_dist,
            "family_dist":    family_dist,
        }
