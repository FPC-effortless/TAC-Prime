"""
TAC-PSM-003: Lifecycle Engine

Manages procedure evolution over time:
  strengthen   → success_score increases with each reuse
  specialize   → a new child procedure emerges for a sub-task variant
  merge        → two frequently co-used procedures combine
  split        → a procedure that serves two purposes forks into two
  retire       → low-fitness procedure is marked RETIRED and pruned

All lifecycle decisions are score-driven and deterministic given the same state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from ..psm001.records import ProcedureTrace, ProcedureStep, ProcedureLifecycleState
from ..psm001.store import ProceduralMemoryStore


class LifecycleEventType(Enum):
    STRENGTHENED  = "strengthened"
    SPECIALIZED   = "specialized"
    MERGED        = "merged"
    SPLIT         = "split"
    RETIRED       = "retired"
    DECLINED      = "declined"


@dataclass
class LifecycleEvent:
    event_type:   LifecycleEventType
    procedure_id: str
    timestamp:    float = field(default_factory=time.time)
    details:      dict  = field(default_factory=dict)


@dataclass
class SpecializationResult:
    parent_id:  str
    child_id:   str
    sub_type:   str
    score_gain: float       # child.success_score - parent.success_score on sub-tasks
    event:      LifecycleEvent = None


@dataclass
class MergeResult:
    parent_ids:   List[str]
    merged_id:    str
    quality_gain: float    # merged.overall_score - max(parent.overall_scores)
    n_steps_merged: int
    event:        LifecycleEvent = None


@dataclass
class SplitResult:
    parent_id:   str
    child_ids:   List[str]
    reason:      str
    event:       LifecycleEvent = None


@dataclass
class RetirementResult:
    procedure_id: str
    reason:       str
    final_score:  float
    reuse_count:  int
    event:        LifecycleEvent = None


# ── Lifecycle Engine ──────────────────────────────────────────────────────────

class LifecycleEngine:
    """
    Observes the ProceduralMemoryStore and applies lifecycle transitions.

    Configuration:
      strengthen_threshold  — success_score above which we record "strengthened"
      specialize_threshold  — reuse_count above which specialization is considered
      merge_similarity      — step-overlap above which two procedures can merge
      split_diversity       — step divergence above which split is triggered
      retire_threshold      — survival_score below which procedure is retired
    """

    def __init__(
        self,
        store:                 ProceduralMemoryStore,
        strengthen_threshold:  float = 0.70,
        specialize_threshold:  int   = 3,
        merge_similarity:      float = 0.50,
        split_diversity:       float = 0.70,
        retire_threshold:      float = 0.10,
    ):
        self.store                = store
        self.strengthen_threshold = strengthen_threshold
        self.specialize_threshold = specialize_threshold
        self.merge_similarity     = merge_similarity
        self.split_diversity      = split_diversity
        self.retire_threshold     = retire_threshold
        self.event_log:  List[LifecycleEvent] = []

    # ── Strengthen ────────────────────────────────────────────────────────────

    def apply_strengthening(self, procedure_id: str, delta: float = 0.05) -> Optional[LifecycleEvent]:
        p = self.store.get(procedure_id)
        if p is None:
            return None
        self.store.update(procedure_id, success_delta=delta, survival_delta=0.01)
        p_updated = self.store.get(procedure_id)
        if p_updated and p_updated.success_score >= self.strengthen_threshold:
            ev = LifecycleEvent(
                LifecycleEventType.STRENGTHENED, procedure_id,
                details={"score": p_updated.success_score},
            )
            self.event_log.append(ev)
            return ev
        return None

    # ── Specialize ────────────────────────────────────────────────────────────

    def specialize(
        self,
        procedure_id: str,
        sub_type:     str,
        extra_steps:  List[str],
        seed:         int = 0,
    ) -> Optional[SpecializationResult]:
        """
        Create a child procedure for a sub-type variant.
        Child inherits parent steps + adds sub-type specific steps.
        """
        import numpy as np
        parent = self.store.get(procedure_id)
        if parent is None:
            return None
        if parent.reuse_count < self.specialize_threshold:
            return None   # not enough reuse to justify specialization

        child_steps = [s.action for s in parent.steps] + extra_steps
        emb = np.array(parent.embedding, dtype=np.float32) if parent.embedding else None
        # Add small perturbation so child has distinct embedding
        if emb is not None:
            rng  = np.random.default_rng(seed)
            emb  = emb + rng.standard_normal(len(emb)).astype(np.float32) * 0.05
            emb  = emb / (np.linalg.norm(emb) + 1e-9)

        child = self.store.build(
            problem_family   = parent.problem_family,
            task_signature   = parent.task_signature + f"::{sub_type}",
            steps            = child_steps,
            embedding        = emb,
            success_score    = parent.success_score,
            parent_id        = parent.procedure_id,
            selection_reason = f"Specialized from {parent.procedure_id} for {sub_type}",
        )

        score_gain = child.success_score - parent.success_score
        ev = LifecycleEvent(
            LifecycleEventType.SPECIALIZED, child.procedure_id,
            details={"parent": procedure_id, "sub_type": sub_type, "gain": score_gain},
        )
        self.event_log.append(ev)
        return SpecializationResult(
            parent_id  = procedure_id,
            child_id   = child.procedure_id,
            sub_type   = sub_type,
            score_gain = score_gain,
            event      = ev,
        )

    # ── Merge ─────────────────────────────────────────────────────────────────

    def find_merge_candidates(self) -> List[Tuple[str, str, float]]:
        """
        Scan store for pairs of procedures with high step overlap.
        Returns list of (pid_a, pid_b, overlap_score) sorted descending.
        """
        procs   = [p for p in self.store._procs
                   if p.lifecycle_state != ProcedureLifecycleState.RETIRED]
        pairs   = []
        for i, pa in enumerate(procs):
            for pb in procs[i + 1:]:
                overlap = _step_overlap(pa, pb)
                if overlap >= self.merge_similarity:
                    pairs.append((pa.procedure_id, pb.procedure_id, overlap))
        return sorted(pairs, key=lambda x: -x[2])

    def merge(
        self,
        proc_id_a: str,
        proc_id_b: str,
        seed:      int = 0,
    ) -> Optional[MergeResult]:
        """
        Merge two procedures into a new combined procedure.
        The merged procedure contains the union of steps in canonical order.
        """
        import numpy as np
        pa = self.store.get(proc_id_a)
        pb = self.store.get(proc_id_b)
        if pa is None or pb is None:
            return None

        # Union of steps, deduplicated, ordered by source then unique additions
        steps_a   = [s.action for s in pa.steps]
        steps_b   = [s.action for s in pb.steps]
        seen      = set()
        merged_steps = []
        for s in steps_a + steps_b:
            key = s.lower().strip()
            if key not in seen:
                merged_steps.append(s)
                seen.add(key)

        # Average embeddings
        emb_a = np.array(pa.embedding, dtype=np.float32) if pa.embedding else np.zeros(self.store.dim)
        emb_b = np.array(pb.embedding, dtype=np.float32) if pb.embedding else np.zeros(self.store.dim)
        emb_m = (emb_a + emb_b) / 2.0
        norm  = np.linalg.norm(emb_m)
        if norm > 0:
            emb_m = emb_m / norm

        # Merged procedure inherits best scores
        merged = self.store.build(
            problem_family   = pa.problem_family,
            task_signature   = f"{pa.task_signature}+{pb.task_signature[:8]}",
            steps            = merged_steps,
            embedding        = emb_m,
            success_score    = max(pa.success_score, pb.success_score),
            selection_reason = f"Merged from {proc_id_a} + {proc_id_b}",
        )

        quality_gain = merged.success_score - max(pa.overall_score(), pb.overall_score())

        # Retire parents
        self.store.retire(proc_id_a)
        self.store.retire(proc_id_b)

        ev = LifecycleEvent(
            LifecycleEventType.MERGED, merged.procedure_id,
            details={"parents": [proc_id_a, proc_id_b], "n_steps": len(merged_steps)},
        )
        self.event_log.append(ev)
        return MergeResult(
            parent_ids     = [proc_id_a, proc_id_b],
            merged_id      = merged.procedure_id,
            quality_gain   = quality_gain,
            n_steps_merged = len(merged_steps),
            event          = ev,
        )

    # ── Split ─────────────────────────────────────────────────────────────────

    def split(
        self,
        procedure_id: str,
        split_point:  int,
        labels:       Tuple[str, str] = ("part-A", "part-B"),
        seed:         int = 0,
    ) -> Optional[SplitResult]:
        """
        Split a procedure at split_point into two child procedures.
        First child = steps[:split_point], second = steps[split_point:]
        """
        import numpy as np
        proc = self.store.get(procedure_id)
        if proc is None or len(proc.steps) < 2:
            return None

        split_point = max(1, min(split_point, len(proc.steps) - 1))
        steps_a     = [s.action for s in proc.steps[:split_point]]
        steps_b     = [s.action for s in proc.steps[split_point:]]

        base_emb = np.array(proc.embedding, dtype=np.float32) if proc.embedding else np.zeros(self.store.dim)
        rng      = np.random.default_rng(seed)

        child_ids = []
        for i, (steps, label) in enumerate(zip([steps_a, steps_b], labels)):
            emb = base_emb + rng.standard_normal(len(base_emb)).astype(np.float32) * 0.1
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            child = self.store.build(
                problem_family   = proc.problem_family,
                task_signature   = proc.task_signature + f"::{label}",
                steps            = steps,
                embedding        = emb,
                success_score    = proc.success_score * 0.9,
                parent_id        = procedure_id,
                selection_reason = f"Split from {procedure_id} ({label})",
            )
            child_ids.append(child.procedure_id)

        self.store.retire(procedure_id)

        ev = LifecycleEvent(
            LifecycleEventType.SPLIT, procedure_id,
            details={"children": child_ids, "split_point": split_point},
        )
        self.event_log.append(ev)
        return SplitResult(parent_id=procedure_id, child_ids=child_ids, reason="manual split", event=ev)

    # ── Retire ────────────────────────────────────────────────────────────────

    def apply_retirement_sweep(self) -> List[RetirementResult]:
        """Retire all procedures below survival threshold."""
        retired = []
        for p in list(self.store._procs):
            if (p.lifecycle_state != ProcedureLifecycleState.RETIRED
                    and p.survival_score < self.retire_threshold):
                result = RetirementResult(
                    procedure_id = p.procedure_id,
                    reason       = f"survival_score={p.survival_score:.4f} < threshold={self.retire_threshold}",
                    final_score  = p.overall_score(),
                    reuse_count  = p.reuse_count,
                )
                self.store.retire(p.procedure_id)
                ev = LifecycleEvent(
                    LifecycleEventType.RETIRED, p.procedure_id,
                    details={"survival": p.survival_score, "score": p.overall_score()},
                )
                result.event = ev
                self.event_log.append(ev)
                retired.append(result)
        return retired

    # ── Stats ─────────────────────────────────────────────────────────────────

    def lifecycle_stats(self) -> dict:
        from collections import Counter
        counts = Counter(ev.event_type.value for ev in self.event_log)
        return {
            "total_events": len(self.event_log),
            "event_counts": dict(counts),
            "store_size":   len(self.store),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _step_overlap(pa: ProcedureTrace, pb: ProcedureTrace) -> float:
    sa = set(s.action.lower().strip() for s in pa.steps)
    sb = set(s.action.lower().strip() for s in pb.steps)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)
