"""
TAC-PSM-001: Procedure Retrieval

retrieve_procedure(task_signature, store, ...) — the canonical retrieval function.

Supports five retrieval modes used in the experiment:
  CORRECT  — cosine similarity + overall score (normal operation)
  DISABLED — returns nothing (memory-disabled baseline)
  RANDOM   — random selection from store
  WRONG    — deliberately retrieves worst-ranked results (wrong-retrieval baseline)
  ORACLE   — uses ground-truth family label for perfect family filtering
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

from .records import ProcedureTrace
from .store import ProceduralMemoryStore


class RetrievalMode(Enum):
    CORRECT  = "correct"    # normal cosine + score retrieval
    DISABLED = "disabled"   # returns [] (memory disabled)
    RANDOM   = "random"     # random selection
    WRONG    = "wrong"      # anti-ranked (worst first)
    ORACLE   = "oracle"     # perfect family-filtered retrieval


@dataclass
class RetrievalResult:
    """Output of a single retrieval call."""
    mode:             RetrievalMode
    candidates:       List[Tuple[float, ProcedureTrace]]   # (score, trace)
    top1:             Optional[ProcedureTrace]
    top1_score:       float
    query_signature:  str
    correct_family:   Optional[str]
    family_matched:   bool            # top-1 family == correct_family
    retrieval_conf:   float           # score of top-1 (proxy for confidence)
    n_candidates:     int

    # Ground-truth label (filled by experiment runner)
    is_correct:       Optional[bool] = None   # top-1 is the correct procedure

    def precision_at_k(self, k: int, correct_id: str) -> float:
        hits = sum(1 for _, p in self.candidates[:k] if p.procedure_id == correct_id)
        return hits / max(k, 1)

    def recall_at_k(self, k: int, correct_id: str) -> float:
        return 1.0 if any(p.procedure_id == correct_id
                          for _, p in self.candidates[:k]) else 0.0

    def to_dict(self) -> dict:
        return {
            "mode":            self.mode.value,
            "top1_id":         self.top1.procedure_id if self.top1 else None,
            "top1_family":     self.top1.problem_family if self.top1 else None,
            "top1_score":      self.top1_score,
            "query_signature": self.query_signature,
            "correct_family":  self.correct_family,
            "family_matched":  self.family_matched,
            "retrieval_conf":  self.retrieval_conf,
            "n_candidates":    self.n_candidates,
            "is_correct":      self.is_correct,
        }


def retrieve_procedure(
    task_signature:  str,
    query_embedding: np.ndarray,
    store:           ProceduralMemoryStore,
    mode:            RetrievalMode = RetrievalMode.CORRECT,
    top_k:           int = 5,
    correct_family:  Optional[str] = None,
    rng:             Optional[random.Random] = None,
) -> RetrievalResult:
    """
    Retrieve procedures for a task.

    Parameters
    ----------
    task_signature  : canonical string fingerprint of the task
    query_embedding : (embedding_dim,) float32 array
    store           : ProceduralMemoryStore
    mode            : RetrievalMode — controls retrieval strategy
    top_k           : number of candidates to return
    correct_family  : ground-truth task family (used for ORACLE mode + evaluation)
    rng             : seeded RNG for RANDOM mode reproducibility

    Returns
    -------
    RetrievalResult
    """
    rng = rng or random.Random()

    if mode == RetrievalMode.DISABLED or len(store) == 0:
        return RetrievalResult(
            mode            = mode,
            candidates      = [],
            top1            = None,
            top1_score      = 0.0,
            query_signature = task_signature,
            correct_family  = correct_family,
            family_matched  = False,
            retrieval_conf  = 0.0,
            n_candidates    = 0,
        )

    if mode == RetrievalMode.RANDOM:
        all_procs = [p for p in store._procs
                     if p.lifecycle_state.value != "retired"]
        chosen = rng.sample(all_procs, min(top_k, len(all_procs)))
        candidates = [(0.5, p) for p in chosen]

    elif mode == RetrievalMode.WRONG:
        # Return the WORST ranked results (anti-retrieval baseline)
        ranked = store.retrieve(query_embedding, top_k=len(store._procs))
        candidates = list(reversed(ranked))[:top_k]

    elif mode == RetrievalMode.ORACLE:
        # Perfect family-filtered retrieval
        candidates = store.retrieve(
            query_embedding,
            family  = correct_family,
            top_k   = top_k,
        )

    else:  # CORRECT
        candidates = store.retrieve(
            query_embedding,
            top_k = top_k,
        )

    top1       = candidates[0][1] if candidates else None
    top1_score = candidates[0][0] if candidates else 0.0
    fam_match  = (top1 is not None and correct_family is not None
                  and top1.problem_family == correct_family)

    return RetrievalResult(
        mode            = mode,
        candidates      = candidates,
        top1            = top1,
        top1_score      = top1_score,
        query_signature = task_signature,
        correct_family  = correct_family,
        family_matched  = fam_match,
        retrieval_conf  = top1_score,
        n_candidates    = len(candidates),
    )


def retrieve_batch(
    task_signatures:  List[str],
    query_embeddings: np.ndarray,               # (N, dim)
    store:            ProceduralMemoryStore,
    mode:             RetrievalMode = RetrievalMode.CORRECT,
    top_k:            int = 5,
    correct_families: Optional[List[str]] = None,
    rng:              Optional[random.Random] = None,
) -> List[RetrievalResult]:
    """Batch version of retrieve_procedure."""
    results = []
    for i, (sig, emb) in enumerate(zip(task_signatures, query_embeddings)):
        family = correct_families[i] if correct_families else None
        results.append(retrieve_procedure(
            task_signature  = sig,
            query_embedding = emb,
            store           = store,
            mode            = mode,
            top_k           = top_k,
            correct_family  = family,
            rng             = rng,
        ))
    return results


# ── Retrieval metrics ──────────────────────────────────────────────────────────

def compute_retrieval_metrics(
    results:       List[RetrievalResult],
    correct_ids:   List[str],
    k_values:      List[int] = [1, 3, 5],
) -> dict:
    """
    Compute retrieval accuracy, precision@k, recall@k over a list of results.

    correct_ids[i] = ground-truth procedure_id for results[i]
    """
    N = len(results)
    assert len(correct_ids) == N

    metrics: dict = {
        "retrieval_accuracy":   0.0,
        "family_match_rate":    0.0,
        "avg_confidence":       0.0,
    }
    for k in k_values:
        metrics[f"precision@{k}"] = 0.0
        metrics[f"recall@{k}"]    = 0.0

    if N == 0:
        return metrics

    for r, cid in zip(results, correct_ids):
        if r.top1 and r.top1.procedure_id == cid:
            metrics["retrieval_accuracy"] += 1.0
        if r.family_matched:
            metrics["family_match_rate"]  += 1.0
        metrics["avg_confidence"] += r.retrieval_conf
        for k in k_values:
            metrics[f"precision@{k}"] += r.precision_at_k(k, cid)
            metrics[f"recall@{k}"]    += r.recall_at_k(k, cid)

    for key in metrics:
        metrics[key] /= N

    return metrics
