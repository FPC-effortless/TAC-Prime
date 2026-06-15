"""
TAC-PSM-006C: Metrics
======================

Extends PSM-006B metrics with 5 embedding-update-specific metrics and
replaces the gate set with 7 PSM-006C gates focused on the ablation question:

  Does online embedding adaptation improve procedural retrieval and repair?

New metrics (on top of all PSM-006B metrics):
  embedding_update_count           mean embedding updates per run
  embedding_shift_norm_mean        mean ||Δemb|| per update
  retrieval_changed_after_update   frac updates that changed top-1 record
  family_changed_after_update      frac updates that changed retrieved family
  successful_retrieval_recovery    frac updates where family went wrong→correct

PSM-006C Gates (7 total):
  1. retry_after_update_gt_0              retry_after_update_success > 0
  2. embedding_update_beats_full_memory   emb_update rate > full_memory rate
  3. embedding_update_beats_reset         emb_update rate > reset rate
  4. embedding_update_beats_no_update     emb_update rate > no_update rate
  5. reuse_gain_positive                  emb_update rate − reset rate > 0
  6. retrieval_changed_after_update_gt_0  at least some updates changed retrieval
  7. oracle_above_tac                     oracle rate >= emb_update rate
"""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Optional

from .agent import RepairTrace006C


# ── Helpers ───────────────────────────────────────────────────────────────

def _pass_rate(traces: List[RepairTrace006C]) -> float:
    if not traces:
        return 0.0
    return mean(1.0 if t.pytest_pass else 0.0 for t in traces)


def _retrieval_acc(traces: List[RepairTrace006C]) -> float:
    if not traces:
        return 0.0
    return mean(1.0 if t.retrieval_correct else 0.0 for t in traces)


def _emb_mean(traces: List[RepairTrace006C], attr: str) -> float:
    vals = [getattr(t, attr) for t in traces]
    return mean(float(v) for v in vals) if vals else 0.0


# ── Core metric computation ───────────────────────────────────────────────

def compute_metrics_006c(
    results: Dict[str, List[RepairTrace006C]],
    reference_variant: str = "full_memory_embedding_update",
) -> Dict[str, float]:
    """
    Compute all PSM-006C metrics.

    Parameters
    ----------
    results           : {variant_name: [RepairTrace006C]}
    reference_variant : the TAC+embedding variant (default: full_memory_embedding_update)
    """
    ref    = results.get(reference_variant, [])
    fm     = results.get("full_memory", [])
    reset  = results.get("reset", [])
    no_upd = results.get("no_update", [])
    oracle = results.get("oracle", [])

    # ── PSM-006B metrics (on reference variant) ───────────────────────────
    pytest_pass_rate = _pass_rate(ref)

    first_attempt = mean(
        1.0 if (t.pytest_pass and t.n_retries == 0) else 0.0
        for t in ref
    ) if ref else 0.0

    retry_after_update = mean(
        1.0 if t.update_improved else 0.0 for t in ref
    ) if ref else 0.0

    retrieval_acc = _retrieval_acc(ref)

    reuse_gain = _pass_rate(ref) - _pass_rate(reset) if reset else 0.0

    transfer_traces = [t for t in ref if t.fixture_id and
                       any(g in t.fixture_id for g in ("near_transfer", "far_transfer"))]
    cross_fixture = _pass_rate(transfer_traces) if transfer_traces else _pass_rate(ref)

    cross_family_traces = [t for t in ref if not t.retrieval_correct and t.pytest_pass]
    cross_family = len(cross_family_traces) / max(len(ref), 1)

    wrong_harm = _pass_rate(oracle) - _pass_rate(no_upd) if (oracle and no_upd) else 0.0

    patch_correct = mean(
        1.0 if t.patch_result.get("success", False) else 0.0
        for t in ref
    ) if ref else 0.0

    steps_mean = mean(t.steps_to_repair for t in ref) if ref else 0.0
    time_mean  = mean(t.time_to_repair_s for t in ref) if ref else 0.0

    survival = mean(
        1.0 if t.failure_class is None else 0.0 for t in ref
    ) if ref else 0.0

    confusion = len([t for t in ref if not t.retrieval_correct]) / max(len(ref), 1)

    # ── PSM-006C new metrics (embedding-specific) ─────────────────────────
    emb_traces = [t for t in ref if t.embedding_update_applied]

    embedding_update_count = sum(1 for t in ref if t.embedding_update_applied)

    embedding_shift_norm_mean = (
        mean(t.embedding_shift_norm for t in emb_traces) if emb_traces else 0.0
    )

    retrieval_changed_after_update = (
        mean(1.0 if t.retrieval_changed_after_update else 0.0 for t in emb_traces)
        if emb_traces else 0.0
    )

    family_changed_after_update = (
        mean(1.0 if t.family_changed_after_update else 0.0 for t in emb_traces)
        if emb_traces else 0.0
    )

    successful_retrieval_recovery = (
        mean(1.0 if t.successful_retrieval_recovery else 0.0 for t in emb_traces)
        if emb_traces else 0.0
    )

    # Comparative: does embedding update beat text-only?
    emb_vs_full_memory_gain = _pass_rate(ref) - _pass_rate(fm) if fm else 0.0

    return {
        # PSM-006B metrics
        "pytest_pass_rate":                     pytest_pass_rate,
        "first_attempt_repair_success":         first_attempt,
        "retry_after_update_success":           retry_after_update,
        "procedure_retrieval_accuracy":         retrieval_acc,
        "procedure_reuse_gain":                 reuse_gain,
        "cross_fixture_transfer_success":       cross_fixture,
        "cross_family_transfer_success":        cross_family,
        "wrong_procedure_harm":                 wrong_harm,
        "patch_correctness":                    patch_correct,
        "steps_to_repair":                      steps_mean,
        "time_to_repair_s":                     time_mean,
        "procedure_survival_stability":         survival,
        "family_confusion_rate":                confusion,
        # PSM-006C new metrics
        "embedding_update_count":               float(embedding_update_count),
        "embedding_shift_norm_mean":            embedding_shift_norm_mean,
        "retrieval_changed_after_update":       retrieval_changed_after_update,
        "family_changed_after_update":          family_changed_after_update,
        "successful_retrieval_recovery":        successful_retrieval_recovery,
        "emb_update_vs_full_memory_gain":       emb_vs_full_memory_gain,
    }


# ── Gate evaluation ───────────────────────────────────────────────────────

def evaluate_success_gates_006c(
    metrics: Dict[str, float],
    results: Dict[str, List[RepairTrace006C]],
) -> Dict[str, bool]:
    """
    Evaluate all 7 PSM-006C success gates.

    Gate 1: retry_after_update_gt_0
        retry_after_update_success > 0  (PSM-006B had 0.000)
    Gate 2: embedding_update_beats_full_memory
        full_memory_embedding_update pass rate > full_memory pass rate
    Gate 3: embedding_update_beats_reset
        full_memory_embedding_update pass rate > reset pass rate
    Gate 4: embedding_update_beats_no_update
        full_memory_embedding_update pass rate > no_update pass rate
    Gate 5: reuse_gain_positive
        procedure_reuse_gain > 0  (emb_update beats reset)
    Gate 6: retrieval_changed_after_update_gt_0
        retrieval_changed_after_update > 0  (updates are doing something)
    Gate 7: oracle_above_tac
        oracle rate >= full_memory_embedding_update rate
    """
    emb     = results.get("full_memory_embedding_update", [])
    fm      = results.get("full_memory", [])
    reset   = results.get("reset", [])
    no_upd  = results.get("no_update", [])
    oracle  = results.get("oracle", [])

    emb_rate    = _pass_rate(emb)
    fm_rate     = _pass_rate(fm)
    reset_rate  = _pass_rate(reset)
    no_upd_rate = _pass_rate(no_upd)
    oracle_rate = _pass_rate(oracle)

    return {
        "retry_after_update_gt_0": (
            metrics.get("retry_after_update_success", 0.0) > 0.0
        ),
        "embedding_update_beats_full_memory": (
            emb_rate > fm_rate if fm else False
        ),
        "embedding_update_beats_reset": (
            emb_rate > reset_rate if reset else False
        ),
        "embedding_update_beats_no_update": (
            emb_rate > no_upd_rate if no_upd else False
        ),
        "reuse_gain_positive": (
            metrics.get("procedure_reuse_gain", 0.0) > 0.0
        ),
        "retrieval_changed_after_update_gt_0": (
            metrics.get("retrieval_changed_after_update", 0.0) > 0.0
        ),
        "oracle_above_tac": (
            oracle_rate >= emb_rate if oracle else False
        ),
    }


# ── Confusion matrix ──────────────────────────────────────────────────────

def compute_family_confusion_matrix_006c(
    traces:   List[RepairTrace006C],
    families: List[str],
) -> Dict[str, Dict[str, int]]:
    matrix = {f: {g: 0 for g in families} for f in families}
    for t in traces:
        if t.family in matrix and t.retrieved_family in matrix:
            matrix[t.family][t.retrieved_family] += 1
    return matrix


# ── Failure classification ────────────────────────────────────────────────

def classify_failures_006c(
    traces: List[RepairTrace006C],
) -> Dict[str, int]:
    from ..psm006b.fixture_schema import FAILURE_CLASSES
    counts: Dict[str, int] = {fc: 0 for fc in FAILURE_CLASSES}
    counts["none"] = 0
    for t in traces:
        fc = t.failure_class or "none"
        counts[fc] = counts.get(fc, 0) + 1
    return counts
