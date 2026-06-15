"""
TAC-SM FAISS-Backed Structure Memory

Drop-in replacement for StructureMemory that:
  - Uses FAISS IndexFlatIP for O(1) approximate nearest-neighbour retrieval
  - Persists the index and metadata to disk (survives training restarts)
  - Supports millions of structures without loading all embeddings onto GPU
  - Falls back to pure-numpy exact cosine search when FAISS is unavailable

API is identical to StructureMemory so it can be swapped in config.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryConfig
from .memory import StructureRecord  # reuse the same record dataclass

# ── FAISS availability ────────────────────────────────────────────────────────

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ── Index wrappers ────────────────────────────────────────────────────────────

class _NumpyIndex:
    """Pure-numpy fallback: exact cosine similarity via inner product on L2-normalised vecs."""

    def __init__(self, dim: int):
        self.dim    = dim
        self._vecs: List[np.ndarray] = []   # each (dim,) float32

    def add(self, vec: np.ndarray):
        v = vec.astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        self._vecs.append(v)

    def search(self, query: np.ndarray, k: int):
        if not self._vecs:
            return np.array([]), np.array([], dtype=np.int64)
        q = query.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        mat  = np.stack(self._vecs)           # (N, dim)
        sims = mat @ q                        # (N,)
        k    = min(k, len(self._vecs))
        idxs = np.argsort(sims)[::-1][:k]
        return sims[idxs], idxs.astype(np.int64)

    def __len__(self):
        return len(self._vecs)

    def reset(self):
        self._vecs = []

    @property
    def ntotal(self):
        return len(self._vecs)


class _FAISSIndex:
    """FAISS IndexFlatIP — exact inner-product search on L2-normalised vecs = cosine."""

    def __init__(self, dim: int):
        self.dim   = dim
        self._idx  = faiss.IndexFlatIP(dim)

    def add(self, vec: np.ndarray):
        v = vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(v)
        self._idx.add(v)

    def search(self, query: np.ndarray, k: int):
        if self._idx.ntotal == 0:
            return np.array([]), np.array([], dtype=np.int64)
        q = query.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        k   = min(k, self._idx.ntotal)
        D, I = self._idx.search(q, k)
        return D[0], I[0]

    def __len__(self):
        return self._idx.ntotal

    @property
    def ntotal(self):
        return self._idx.ntotal

    def save(self, path: str):
        faiss.write_index(self._idx, path)

    def load(self, path: str):
        self._idx = faiss.read_index(path)

    def reset(self):
        self._idx = faiss.IndexFlatIP(self.dim)


def _make_index(dim: int):
    if FAISS_AVAILABLE:
        return _FAISSIndex(dim)
    return _NumpyIndex(dim)


# ── FAISS Structure Memory ────────────────────────────────────────────────────

class FAISSStructureMemory(nn.Module):
    """
    Large-scale Structure Memory backed by FAISS (or numpy fallback).

    Persistence layout (save_dir):
      structures.json   — metadata for all records
      index.faiss       — FAISS vector index  (if FAISS available)
      index.npy         — numpy vector array  (fallback)

    Thread safety: not thread-safe. Use a single-process training loop.
    """

    def __init__(self, cfg: MemoryConfig, save_dir: Optional[str] = None):
        super().__init__()
        self.cfg      = cfg
        self.dim      = cfg.embedding_dim
        self.save_dir = Path(save_dir) if save_dir else None

        # Vector index
        self._index = _make_index(self.dim)

        # Ordered list of records (index i → record)
        self._records: List[StructureRecord] = []

        # Metadata map structure_id → list-index (for O(1) update)
        self._id_to_idx: Dict[str, int] = {}

        # Learnable read projection
        self.query_proj = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.eye_(self.query_proj.weight)

        # Load from disk if save_dir exists and has data
        if self.save_dir and (self.save_dir / "structures.json").exists():
            self._load()

        backend = "FAISS" if FAISS_AVAILABLE else "numpy (FAISS not installed)"
        print(f"[FAISSStructureMemory] backend={backend}  dim={self.dim}"
              f"  loaded={len(self._records)} records")

    # ── READ ─────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        family_id:       Optional[int] = None,
        top_k:           Optional[int] = None,
    ) -> List[StructureRecord]:
        if not self._records:
            return []

        top_k  = top_k or self.cfg.retrieval_top_k
        q_np   = self._project(query_embedding)
        # Request more candidates when family filtering
        fetch_k = min(top_k * 4 if family_id is not None else top_k,
                      len(self._records))

        sims, idxs = self._index.search(q_np, fetch_k)
        if len(idxs) == 0:
            return []

        results = []
        for sim, idx in zip(sims, idxs):
            if idx < 0 or idx >= len(self._records):
                continue
            r = self._records[idx]
            if family_id is not None and r.family_id != family_id:
                continue
            results.append((float(sim) * 0.7 + r.survival_score * 0.3, r))
            if len(results) == top_k:
                break

        return [r for _, r in sorted(results, key=lambda x: -x[0])]

    def retrieve_batch(
        self,
        query_embeddings: torch.Tensor,
        top_k:            Optional[int] = None,
    ) -> List[List[StructureRecord]]:
        return [self.retrieve(q, top_k=top_k) for q in query_embeddings]

    # ── WRITE ─────────────────────────────────────────────────────────────────

    def write(
        self,
        embedding:      torch.Tensor,
        family_id:      int,
        expert_id:      int,
        task_type:      str,
        success_score:  float,
        survival_score: float = 1.0,
        transfer_score: float = 0.0,
    ) -> Optional[str]:
        if success_score < self.cfg.write_threshold:
            return None

        if len(self._records) >= self.cfg.max_structures:
            self._prune()

        sid = str(uuid.uuid4())[:16]
        emb = embedding.detach().cpu().float()

        record = StructureRecord(
            structure_id   = sid,
            family_id      = family_id,
            expert_id      = expert_id,
            task_type      = task_type,
            embedding      = emb,
            success_score  = success_score,
            transfer_score = transfer_score,
            survival_score = survival_score,
            usage_count    = 0,
            timestamp      = time.time(),
        )

        idx = len(self._records)
        self._records.append(record)
        self._id_to_idx[sid] = idx
        self._index.add(emb.numpy())

        return sid

    # ── UPDATE ────────────────────────────────────────────────────────────────

    def update(
        self,
        structure_id:   str,
        success_delta:  float = 0.0,
        transfer_delta: float = 0.0,
        survival_delta: float = 0.0,
    ):
        idx = self._id_to_idx.get(structure_id)
        if idx is None:
            return
        r = self._records[idx]
        r.usage_count   += 1
        r.success_score  = min(1.0, r.success_score  + success_delta)
        r.transfer_score = min(1.0, r.transfer_score + transfer_delta)
        r.survival_score = min(1.0, r.survival_score + survival_delta)
        r.timestamp      = time.time()

    def decay_all(self):
        for r in self._records:
            r.survival_score *= self.cfg.decay_rate

    # ── PRUNE ─────────────────────────────────────────────────────────────────

    def prune_weak(self):
        keep = [r for r in self._records
                if r.survival_score >= self.cfg.prune_threshold]
        self._rebuild(keep)

    def _prune(self):
        target = int(self.cfg.max_structures * 0.9)
        scored = sorted(self._records, key=lambda r: r.overall_score())
        keep   = scored[len(scored) - target:]
        self._rebuild(keep)

    def _rebuild(self, keep: List[StructureRecord]):
        self._index.reset()
        self._records   = []
        self._id_to_idx = {}
        for r in keep:
            idx = len(self._records)
            self._records.append(r)
            self._id_to_idx[r.structure_id] = idx
            self._index.add(r.embedding.numpy())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        if not self.save_dir:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Metadata
        meta = []
        for r in self._records:
            d = {k: getattr(r, k) for k in
                 ["structure_id", "family_id", "expert_id", "task_type",
                  "success_score", "transfer_score", "survival_score",
                  "usage_count", "timestamp"]}
            d["embedding"] = r.embedding.tolist()
            meta.append(d)

        with open(self.save_dir / "structures.json", "w") as f:
            json.dump(meta, f)

        # Index
        if FAISS_AVAILABLE and isinstance(self._index, _FAISSIndex):
            self._index.save(str(self.save_dir / "index.faiss"))
        else:
            vecs = np.stack([r.embedding.numpy() for r in self._records]) \
                   if self._records else np.zeros((0, self.dim), dtype=np.float32)
            np.save(str(self.save_dir / "index.npy"), vecs)

        print(f"[FAISSStructureMemory] Saved {len(self._records)} records → {self.save_dir}")

    def _load(self):
        with open(self.save_dir / "structures.json") as f:
            meta = json.load(f)

        self._records   = []
        self._id_to_idx = {}

        for d in meta:
            emb = torch.tensor(d.pop("embedding"), dtype=torch.float32)
            r   = StructureRecord(embedding=emb, **d)
            idx = len(self._records)
            self._records.append(r)
            self._id_to_idx[r.structure_id] = idx

        faiss_path = self.save_dir / "index.faiss"
        npy_path   = self.save_dir / "index.npy"

        if FAISS_AVAILABLE and faiss_path.exists():
            self._index.load(str(faiss_path))
        elif npy_path.exists():
            vecs = np.load(str(npy_path))
            for v in vecs:
                self._index.add(v)
        else:
            # Rebuild from record embeddings
            for r in self._records:
                self._index.add(r.embedding.numpy())

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _project(self, emb: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            e = self.query_proj(emb.float().unsqueeze(0)).squeeze(0)
        return e.numpy()

    def __len__(self):
        return len(self._records)

    def stats(self) -> dict:
        if not self._records:
            return {"size": 0, "backend": "FAISS" if FAISS_AVAILABLE else "numpy"}
        r = self._records
        return {
            "size":         len(r),
            "backend":      "FAISS" if FAISS_AVAILABLE else "numpy",
            "index_ntotal": self._index.ntotal,
            "avg_success":  sum(x.success_score  for x in r) / len(r),
            "avg_transfer": sum(x.transfer_score for x in r) / len(r),
            "avg_survival": sum(x.survival_score for x in r) / len(r),
            "avg_usage":    sum(x.usage_count    for x in r) / len(r),
        }
