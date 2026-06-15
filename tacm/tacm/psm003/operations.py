"""
TAC-PSM-003: High-level lifecycle operation wrappers

Thin wrappers that run a full lifecycle operation scenario and return
structured results with before/after metrics for benchmarking.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from ..psm001.records import ProcedureTrace
from ..psm001.store import ProceduralMemoryStore
from ..psm001.benchmark_families import TaskInstance, evaluate_procedure_on_task, oracle_steps
from .lifecycle import (
    LifecycleEngine,
    MergeResult,
    SplitResult,
    SpecializationResult,
    RetirementResult,
)


class MergeStrategy(Enum):
    UNION        = "union"          # all steps from both
    INTERSECTION = "intersection"   # only shared steps
    INTERLEAVED  = "interleaved"    # alternate steps


def merge_procedures(
    store:     ProceduralMemoryStore,
    proc_id_a: str,
    proc_id_b: str,
    task_a:    Optional[TaskInstance] = None,
    task_b:    Optional[TaskInstance] = None,
    seed:      int = 0,
) -> Tuple[Optional[MergeResult], dict]:
    """
    Run a merge operation and measure before/after quality on both tasks.
    Returns (MergeResult, metrics_dict).
    """
    engine = LifecycleEngine(store)

    # Measure quality before merge
    pa = store.get(proc_id_a)
    pb = store.get(proc_id_b)
    before_a = _eval(pa, task_a, seed) if (pa and task_a) else 0.0
    before_b = _eval(pb, task_b, seed) if (pb and task_b) else 0.0

    result = engine.merge(proc_id_a, proc_id_b, seed=seed)
    if result is None:
        return None, {"error": "merge failed"}

    # Measure quality after merge
    merged = store.get(result.merged_id)
    after_a = _eval(merged, task_a, seed) if (merged and task_a) else 0.0
    after_b = _eval(merged, task_b, seed) if (merged and task_b) else 0.0

    metrics = {
        "before_a":     before_a,
        "before_b":     before_b,
        "after_a":      after_a,
        "after_b":      after_b,
        "max_before":   max(before_a, before_b),
        "avg_after":    (after_a + after_b) / 2.0,
        "quality_gain": result.quality_gain,
        "merged_beats_best_parent": (after_a + after_b) / 2.0 > max(before_a, before_b),
    }
    return result, metrics


def split_procedure(
    store:        ProceduralMemoryStore,
    procedure_id: str,
    split_point:  int,
    task_a:       Optional[TaskInstance] = None,
    task_b:       Optional[TaskInstance] = None,
    seed:         int = 0,
) -> Tuple[Optional[SplitResult], dict]:
    """
    Run a split operation and measure child quality on respective tasks.
    """
    engine = LifecycleEngine(store)
    parent = store.get(procedure_id)
    before = _eval(parent, task_a, seed) if (parent and task_a) else 0.0

    result = engine.split(procedure_id, split_point, seed=seed)
    if result is None:
        return None, {"error": "split failed"}

    child_a = store.get(result.child_ids[0]) if len(result.child_ids) > 0 else None
    child_b = store.get(result.child_ids[1]) if len(result.child_ids) > 1 else None

    after_a = _eval(child_a, task_a, seed) if (child_a and task_a) else 0.0
    after_b = _eval(child_b, task_b, seed) if (child_b and task_b) else 0.0

    metrics = {
        "before":     before,
        "after_a":    after_a,
        "after_b":    after_b,
        "sum_children": after_a + after_b,
        "children_beat_parent": (after_a + after_b) / 2.0 >= before * 0.8,
    }
    return result, metrics


def specialize_procedure(
    store:        ProceduralMemoryStore,
    procedure_id: str,
    sub_type:     str,
    extra_steps:  List[str],
    task:         Optional[TaskInstance] = None,
    seed:         int = 0,
) -> Tuple[Optional[SpecializationResult], dict]:
    """
    Trigger specialization and measure gain.
    """
    # Ensure procedure has enough reuse
    store.update(procedure_id, survival_delta=0.0)
    for _ in range(5):   # simulate reuse
        store.update(procedure_id, success_delta=0.01)

    engine = LifecycleEngine(store, specialize_threshold=1)  # low threshold for test
    parent = store.get(procedure_id)
    before = _eval(parent, task, seed) if (parent and task) else 0.0

    result = engine.specialize(procedure_id, sub_type, extra_steps, seed=seed)
    if result is None:
        return None, {"error": "specialization failed"}

    child = store.get(result.child_id)
    after = _eval(child, task, seed) if (child and task) else 0.0

    metrics = {
        "before":      before,
        "after":       after,
        "score_gain":  result.score_gain,
        "n_steps_added": len(extra_steps),
        "child_beats_parent": after >= before,
    }
    return result, metrics


def retire_procedure(
    store:        ProceduralMemoryStore,
    procedure_id: str,
    decay_rounds: int = 20,
    rate:         float = 0.5,
) -> Tuple[Optional[RetirementResult], dict]:
    """
    Simulate decay and trigger retirement sweep.
    """
    for _ in range(decay_rounds):
        store.decay_all(rate=rate)

    engine = LifecycleEngine(store, retire_threshold=0.10)
    before_count = len(store)
    retired_list = engine.apply_retirement_sweep()
    after_count  = len([p for p in store._procs
                        if p.lifecycle_state.value != "retired"])

    target = next((r for r in retired_list if r.procedure_id == procedure_id), None)
    metrics = {
        "before_count": before_count,
        "after_count":  after_count,
        "n_retired":    len(retired_list),
        "target_retired": target is not None,
    }
    return target, metrics


# ── Internal helper ───────────────────────────────────────────────────────────

def _eval(proc: ProcedureTrace, task: TaskInstance, seed: int) -> float:
    steps = [s.action for s in proc.steps] if proc else []
    _, quality, _ = evaluate_procedure_on_task(task, steps, seed=seed)
    return quality
