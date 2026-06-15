"""
TAC-PSM-006C: Online Embedding Adaptation
==========================================

Implements the single mechanical change that distinguishes PSM-006C from
PSM-006B: online updates to retrieval embeddings after repair outcomes.

PSM-006B updated procedure *text* (steps) but left embedding vectors
unchanged, so wrong-family retrievals repeated on retry.  PSM-006C also
updates the embedding vectors:

  On failure (wrong family retrieved):
    - Push retrieved record's embedding AWAY from the task embedding.
    - Pull every correct-family record's embedding TOWARD the task embedding.

  On success (correct family retrieved):
    - Reinforce retrieved record's embedding (gentle pull TOWARD task).

All updates use a simple learning-rate rule:
    new = unit(old + lr * direction)

This is the minimum viable online metric learning step.

API
---
OnlineEmbeddingAdapter
    adapt_on_failure(store, proc_id, task_embedding, correct_family, lr)
        -> EmbeddingUpdateRecord
    adapt_on_success(store, proc_id, task_embedding, lr)
        -> EmbeddingUpdateRecord
    check_retrieval_change(store, task_embedding, old_proc_id, old_family)
        -> (proc_changed: bool, family_changed: bool)
    summary()
        -> dict  {update_count, mean_shift_norm, retrieval_changed_frac,
                  family_changed_frac, successful_recovery_frac}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Record ────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingUpdateRecord:
    """Record of a single online embedding update event."""
    applied:                bool
    proc_id:                Optional[str]  = None
    update_type:            str            = "none"   # "failure" | "success" | "none"
    embedding_shift_norm:   float          = 0.0
    retrieval_changed:      bool           = False
    family_changed:         bool           = False
    successful_recovery:    bool           = False    # wrong→right after update
    n_correct_family_nudged: int           = 0

    def to_dict(self) -> dict:
        return {
            "applied":                   self.applied,
            "proc_id":                   self.proc_id,
            "update_type":               self.update_type,
            "embedding_shift_norm":      self.embedding_shift_norm,
            "retrieval_changed":         self.retrieval_changed,
            "family_changed":            self.family_changed,
            "successful_recovery":       self.successful_recovery,
            "n_correct_family_nudged":   self.n_correct_family_nudged,
        }


# ── Adapter ───────────────────────────────────────────────────────────────

class OnlineEmbeddingAdapter:
    """
    Online embedding adapter for PSM-006C.

    Parameters
    ----------
    lr_fail    : learning rate applied when retrieval fails
                 (push away from wrong; pull correct families toward task)
    lr_success : learning rate applied when retrieval succeeds
                 (reinforce retrieved record toward task)
    """

    def __init__(self, lr_fail: float = 0.10, lr_success: float = 0.05):
        self.lr_fail    = lr_fail
        self.lr_success = lr_success
        self._log: List[EmbeddingUpdateRecord] = []

    # ── Core update calls ─────────────────────────────────────────────────

    def adapt_on_failure(
        self,
        store,                   # SimpleProceduralMemoryStore
        proc_id:        str,
        task_embedding: np.ndarray,
        correct_family: str,
    ) -> EmbeddingUpdateRecord:
        """
        Update embeddings after a wrong-family retrieval.

        Moves the retrieved (wrong) record away from the task and moves
        all correct-family records toward the task.

        Parameters
        ----------
        store          : the live SimpleProceduralMemoryStore
        proc_id        : the proc_id of the wrongly retrieved record
        task_embedding : unit-normed query embedding of the current fixture
        correct_family : the fixture's true repair family
        """
        task_emb = _unit(task_embedding)

        rec = store._get(proc_id)
        if rec is None:
            ur = EmbeddingUpdateRecord(applied=False, proc_id=proc_id,
                                       update_type="failure")
            self._log.append(ur)
            return ur

        # Push retrieved record AWAY from task direction
        old_emb            = rec.embedding.copy()
        new_emb            = _unit(old_emb - self.lr_fail * task_emb)
        shift_norm         = float(np.linalg.norm(new_emb - old_emb))
        rec.embedding      = new_emb

        # Pull all correct-family records TOWARD task
        n_nudged = 0
        for r in store._records:
            if r.family == correct_family and not r.retired:
                r.embedding = _unit(
                    r.embedding + self.lr_fail * (task_emb - r.embedding)
                )
                n_nudged += 1

        ur = EmbeddingUpdateRecord(
            applied                = True,
            proc_id                = proc_id,
            update_type            = "failure",
            embedding_shift_norm   = shift_norm,
            n_correct_family_nudged = n_nudged,
        )
        self._log.append(ur)
        return ur

    def adapt_on_success(
        self,
        store,
        proc_id:        str,
        task_embedding: np.ndarray,
    ) -> EmbeddingUpdateRecord:
        """
        Reinforce embedding after a correct retrieval + successful repair.

        Gently moves the retrieved record toward the task embedding.
        """
        task_emb = _unit(task_embedding)
        rec      = store._get(proc_id)
        if rec is None:
            ur = EmbeddingUpdateRecord(applied=False, proc_id=proc_id,
                                       update_type="success")
            self._log.append(ur)
            return ur

        old_emb       = rec.embedding.copy()
        new_emb       = _unit(old_emb + self.lr_success * (task_emb - old_emb))
        shift_norm    = float(np.linalg.norm(new_emb - old_emb))
        rec.embedding = new_emb

        ur = EmbeddingUpdateRecord(
            applied              = True,
            proc_id              = proc_id,
            update_type          = "success",
            embedding_shift_norm = shift_norm,
        )
        self._log.append(ur)
        return ur

    # ── Post-update retrieval probe ───────────────────────────────────────

    def check_retrieval_change(
        self,
        store,
        task_embedding: np.ndarray,
        old_proc_id:    str,
        old_family:     str,
    ) -> Tuple[bool, bool]:
        """
        After an embedding update, query the store again to see if the
        top-1 result has changed.

        Returns
        -------
        (proc_changed, family_changed) — both are False if store is empty.
        """
        noisy = _unit(task_embedding)   # use clean emb for the probe
        records = store.retrieve(noisy, top_k=1)
        if not records:
            return False, False
        new_rec        = records[0]
        proc_changed   = new_rec.proc_id != old_proc_id
        family_changed = new_rec.family  != old_family
        return proc_changed, family_changed

    def annotate_record(
        self,
        update_record:  EmbeddingUpdateRecord,
        proc_changed:   bool,
        family_changed: bool,
        new_family:     str,
        correct_family: str,
    ) -> None:
        """Attach retrieval-change fields to an already-created update record."""
        update_record.retrieval_changed   = proc_changed
        update_record.family_changed      = family_changed
        update_record.successful_recovery = (
            family_changed and new_family == correct_family
        )

    # ── Summary stats ─────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Aggregate statistics over all recorded updates in this run."""
        failure_updates = [u for u in self._log if u.update_type == "failure" and u.applied]
        success_updates = [u for u in self._log if u.update_type == "success" and u.applied]
        all_applied     = failure_updates + success_updates

        def _mean(vals):
            return float(np.mean(vals)) if vals else 0.0

        shift_norms     = [u.embedding_shift_norm for u in all_applied]
        retrievals      = [u.retrieval_changed for u in failure_updates]
        family_changes  = [u.family_changed for u in failure_updates]
        recoveries      = [u.successful_recovery for u in failure_updates]

        return {
            "embedding_update_count":            len(all_applied),
            "failure_update_count":              len(failure_updates),
            "success_update_count":              len(success_updates),
            "embedding_shift_norm_mean":         _mean(shift_norms),
            "retrieval_changed_after_update":    _mean(retrievals),
            "family_changed_after_update":       _mean(family_changes),
            "successful_retrieval_recovery":     _mean(recoveries),
        }

    def reset(self) -> None:
        """Clear the update log (call between seeds)."""
        self._log.clear()


# ── Helper ────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    """Return unit-normalised float32 copy of v."""
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-8)
