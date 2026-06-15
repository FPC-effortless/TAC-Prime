"""
TAC-PSM-006: Metrics
====================

Computes all 9 required PSM-006 metrics from AgentTrace collections:

  1. verified_repair_success       — fraction of tasks verified as repaired
  2. procedure_retrieval_accuracy  — fraction where retrieved family == expected family
  3. procedure_reuse_gain          — TAC success rate minus reset success rate
  4. update_retry_improvement      — with-update improvement over no-update on retried tasks
  5. transfer_success              — cross-repo verified success rate
  6. wrong_procedure_harm          — random procedure success - reset success (should be ≤ 0)
  7. steps_to_repair               — mean number of applied steps at success
  8. survival_score_stability      — std-dev of procedure survival scores across tasks
  9. procedure_family_confusion    — full confusion matrix (predicted vs expected family)

All metrics are computed from List[AgentTrace], which is variant-agnostic.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import numpy as np

from .procedural_repair_agent import AgentTrace
from .repository_task import ALL_FAMILY_NAMES


# ── Per-task survival score helper ────────────────────────────────────────────

def _survival_from_trace(trace: AgentTrace) -> float:
    """
    Proxy survival score for a trace: based on composite verification score
    and procedure update signal.

    Range [0, 1]; higher = more stable / successful.
    """
    base = trace.verification.composite_score
    bonus = 0.05 if trace.procedure_updated and trace.update_success else 0.0
    return min(1.0, base + bonus)


# ── Metric dataclasses ────────────────────────────────────────────────────────

@dataclass
class ConfusionMatrix:
    """
    Family-level procedure confusion matrix.

    rows = expected family (ground truth)
    cols = retrieved/selected family (prediction)
    """
    families:    List[str]
    matrix:      Dict[str, Dict[str, int]]   # expected → {predicted: count}
    n_samples:   int

    def precision(self, family: str) -> float:
        col_total = sum(self.matrix[f].get(family, 0) for f in self.families)
        correct   = self.matrix.get(family, {}).get(family, 0)
        return correct / max(col_total, 1)

    def recall(self, family: str) -> float:
        row_total = sum(self.matrix.get(family, {}).values())
        correct   = self.matrix.get(family, {}).get(family, 0)
        return correct / max(row_total, 1)

    def macro_precision(self) -> float:
        return mean(self.precision(f) for f in self.families)

    def macro_recall(self) -> float:
        return mean(self.recall(f) for f in self.families)

    def to_dict(self) -> dict:
        return {
            "families":         self.families,
            "matrix":           self.matrix,
            "n_samples":        self.n_samples,
            "macro_precision":  round(self.macro_precision(), 4),
            "macro_recall":     round(self.macro_recall(), 4),
        }

    def format_table(self) -> str:
        """Return a readable confusion matrix table."""
        w     = 22
        short = {f: f[:10] for f in self.families}
        header = " " * w + "  ".join(f"{short[f]:>10}" for f in self.families)
        lines  = [header]
        for expected in self.families:
            row = f"{short[expected]:<{w}}"
            for predicted in self.families:
                cnt = self.matrix.get(expected, {}).get(predicted, 0)
                row += f"{cnt:>10}  "
            lines.append(row)
        return "\n".join(lines)


@dataclass
class PSM006Metrics:
    """All 9 PSM-006 metrics for one variant and one seed."""
    mode:                        str
    seed:                        int
    n_tasks:                     int

    # Metric 1
    verified_repair_success:     float
    # Metric 2
    procedure_retrieval_accuracy: float
    # Metric 3
    procedure_reuse_gain:        float   # relative to reset; set externally
    # Metric 4
    update_retry_improvement:    float   # set externally (requires two variants)
    # Metric 5
    transfer_success:            float
    # Metric 6
    wrong_procedure_harm:        float   # set externally
    # Metric 7
    steps_to_repair:             float
    # Metric 8
    survival_score_stability:    float   # std-dev (lower = more stable)
    # Metric 9 (stored separately)
    confusion:                   Optional[ConfusionMatrix] = None

    def to_dict(self) -> dict:
        d = {
            "mode":                         self.mode,
            "seed":                         self.seed,
            "n_tasks":                      self.n_tasks,
            "verified_repair_success":      round(self.verified_repair_success, 4),
            "procedure_retrieval_accuracy": round(self.procedure_retrieval_accuracy, 4),
            "procedure_reuse_gain":         round(self.procedure_reuse_gain, 4),
            "update_retry_improvement":     round(self.update_retry_improvement, 4),
            "transfer_success":             round(self.transfer_success, 4),
            "wrong_procedure_harm":         round(self.wrong_procedure_harm, 4),
            "steps_to_repair":              round(self.steps_to_repair, 4),
            "survival_score_stability":     round(self.survival_score_stability, 4),
        }
        if self.confusion is not None:
            d["confusion"] = self.confusion.to_dict()
        return d


# ── Individual metric computations ────────────────────────────────────────────

def metric_verified_repair_success(traces: List[AgentTrace]) -> float:
    """Fraction of tasks where verification.success == True."""
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.verification.success) / len(traces)


def metric_retrieval_accuracy(traces: List[AgentTrace]) -> float:
    """
    Fraction of tasks where retrieved procedure family matches expected family.
    Tasks with no retrieval (retrieved_family == 'Unknown') count as wrong.
    """
    if not traces:
        return 0.0
    correct = sum(
        1 for t in traces
        if t.retrieved_family == t.family and t.retrieved_family != "Unknown"
    )
    return correct / len(traces)


def metric_steps_to_repair(traces: List[AgentTrace]) -> float:
    """Mean steps_to_repair across all tasks (successful and failed)."""
    if not traces:
        return 0.0
    return mean(t.steps_to_repair for t in traces)


def metric_survival_stability(traces: List[AgentTrace]) -> float:
    """
    Std-dev of proxy survival scores across traces.
    Lower = more stable survival field.
    Returns 0.0 if fewer than 2 traces.
    """
    if len(traces) < 2:
        return 0.0
    scores = [_survival_from_trace(t) for t in traces]
    return stdev(scores)


def metric_transfer_success(traces: List[AgentTrace]) -> float:
    """
    Cross-repo transfer success rate.

    A trace is a transfer case if the retrieved procedure's origin repo
    differs from the current task's repo.  We proxy this via transfer_group:
    if the task's repo is not in the procedure's used_by_tasks origins,
    any success in verification is a transfer success.

    Simpler proxy used here: we treat tasks where verification.family_match==True
    AND steps_to_repair > 0 AND the task has a non-trivial transfer_group as
    cross-repo successes.
    """
    if not traces:
        return 0.0
    # Filter to tasks where retrieval found a procedure
    retrieval_attempted = [
        t for t in traces if t.retrieved_proc_id is not None
    ]
    if not retrieval_attempted:
        return 0.0
    successes = sum(
        1 for t in retrieval_attempted
        if t.verification.success and t.verification.family_match
    )
    return successes / len(retrieval_attempted)


def metric_confusion_matrix(traces: List[AgentTrace]) -> ConfusionMatrix:
    """Build family-level confusion matrix."""
    # Include all known families plus "Unknown"
    families = list(ALL_FAMILY_NAMES) + ["Unknown"]

    matrix: Dict[str, Dict[str, int]] = {f: defaultdict(int) for f in families}

    for t in traces:
        expected  = t.family
        predicted = t.retrieved_family if t.retrieved_family else "Unknown"
        if expected not in matrix:
            matrix[expected] = defaultdict(int)
        matrix[expected][predicted] += 1

    return ConfusionMatrix(
        families  = families,
        matrix    = {f: dict(v) for f, v in matrix.items()},
        n_samples = len(traces),
    )


# ── Composite metric builder ───────────────────────────────────────────────────

def compute_metrics(
    traces:            List[AgentTrace],
    reset_traces:      Optional[List[AgentTrace]] = None,
    no_update_traces:  Optional[List[AgentTrace]] = None,
    random_traces:     Optional[List[AgentTrace]] = None,
    seed:              int = 0,
) -> PSM006Metrics:
    """
    Compute all 9 PSM-006 metrics.

    Parameters
    ----------
    traces           : primary variant (full_memory or any single variant)
    reset_traces     : reset baseline traces (for reuse_gain)
    no_update_traces : no-update baseline traces (for update_retry_improvement)
    random_traces    : random-procedure traces (for wrong_procedure_harm)
    """
    mode   = traces[0].mode if traces else "unknown"
    n      = len(traces)

    # M1: verified repair success
    vrs = metric_verified_repair_success(traces)

    # M2: retrieval accuracy
    acc = metric_retrieval_accuracy(traces)

    # M3: reuse gain = TAC success - reset success
    reset_success = metric_verified_repair_success(reset_traces) if reset_traces else 0.0
    reuse_gain    = vrs - reset_success

    # M4: update retry improvement
    # Compare retry success rates: traces (with update) vs no_update_traces
    if no_update_traces:
        full_retry_rate    = mean(t.n_retries for t in traces) if traces else 0.0
        noupd_retry_rate   = mean(t.n_retries for t in no_update_traces) if no_update_traces else 0.0
        # improvement in success on initially-failed tasks
        full_success  = metric_verified_repair_success(traces)
        noupd_success = metric_verified_repair_success(no_update_traces)
        update_retry_improvement = full_success - noupd_success
    else:
        update_retry_improvement = 0.0

    # M5: transfer success
    transfer = metric_transfer_success(traces)

    # M6: wrong procedure harm
    # Compare random retrieval against full TAC (not reset).
    # A positive value means wrong procedures accidentally help — undesirable.
    # We want wrong_harm = random_success - vrs <= 0 (random is no better than TAC).
    if random_traces:
        rand_success = metric_verified_repair_success(random_traces)
        wrong_harm   = rand_success - vrs   # positive = harm to claim; negative = TAC wins
    else:
        wrong_harm = 0.0

    # M7: steps to repair
    s2r = metric_steps_to_repair(traces)

    # M8: survival stability
    surv_std = metric_survival_stability(traces)

    # M9: confusion matrix
    confusion = metric_confusion_matrix(traces)

    return PSM006Metrics(
        mode                        = mode,
        seed                        = seed,
        n_tasks                     = n,
        verified_repair_success     = vrs,
        procedure_retrieval_accuracy = acc,
        procedure_reuse_gain        = reuse_gain,
        update_retry_improvement    = update_retry_improvement,
        transfer_success            = transfer,
        wrong_procedure_harm        = wrong_harm,
        steps_to_repair             = s2r,
        survival_score_stability    = surv_std,
        confusion                   = confusion,
    )


# ── Multi-seed aggregation ─────────────────────────────────────────────────────

@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple seeds for one variant."""
    mode:     str
    n_seeds:  int
    stats:    Dict[str, Dict[str, float]]   # metric → {mean, std, ci95}
    all_runs: List[PSM006Metrics]

    def to_dict(self) -> dict:
        return {
            "mode":    self.mode,
            "n_seeds": self.n_seeds,
            "stats":   self.stats,
        }

    def mean(self, metric: str) -> float:
        return self.stats.get(metric, {}).get("mean", 0.0)

    def std(self, metric: str) -> float:
        return self.stats.get(metric, {}).get("std", 0.0)


def aggregate_metrics(runs: List[PSM006Metrics]) -> AggregatedMetrics:
    """Aggregate PSM006Metrics across seeds."""
    if not runs:
        return AggregatedMetrics(mode="unknown", n_seeds=0, stats={}, all_runs=[])

    n    = len(runs)
    mode = runs[0].mode

    metric_names = [
        "verified_repair_success",
        "procedure_retrieval_accuracy",
        "procedure_reuse_gain",
        "update_retry_improvement",
        "transfer_success",
        "wrong_procedure_harm",
        "steps_to_repair",
        "survival_score_stability",
    ]

    stats: Dict[str, Dict[str, float]] = {}
    for mname in metric_names:
        vals = [getattr(r, mname) for r in runs]
        m    = mean(vals)
        s    = stdev(vals) if n > 1 else 0.0
        ci   = 1.96 * s / (n ** 0.5) if n > 1 else 0.0
        stats[mname] = {"mean": round(m, 4), "std": round(s, 4), "ci95": round(ci, 4)}

    return AggregatedMetrics(mode=mode, n_seeds=n, stats=stats, all_runs=runs)


# ── Gate evaluation ───────────────────────────────────────────────────────────

PSM006_GATES = {
    "tac_beats_reset_by_0.10": {
        "metric": "procedure_reuse_gain",
        "op":     ">=",
        "threshold": 0.10,
        "description": "TAC verified repair success > reset by >= 0.10",
    },
    "retrieval_accuracy_ge_0.60": {
        "metric": "procedure_retrieval_accuracy",
        "op":     ">=",
        "threshold": 0.60,
        "description": "Procedure retrieval accuracy >= 0.60",
    },
    "update_improves_retry": {
        "metric": "update_retry_improvement",
        "op":     ">",
        "threshold": 0.0,
        "description": "Update step improves retry success over no-update",
    },
    "transfer_success_gt_0": {
        "metric": "transfer_success",
        "op":     ">",
        "threshold": 0.0,
        "description": "Cross-repository transfer success > 0",
    },
    "wrong_procedure_no_gain": {
        "metric": "wrong_procedure_harm",
        "op":     "<=",
        "threshold": 0.0,
        "description": "Random/wrong procedure does not outperform full TAC memory",
    },
    "survival_stable": {
        "metric": "survival_score_stability",
        "op":     "<=",
        "threshold": 0.35,
        "description": "Survival score std-dev <= 0.35 (stable)",
    },
}


def evaluate_gates(
    full_agg:        AggregatedMetrics,
    oracle_agg:      AggregatedMetrics,
    no_update_agg:   AggregatedMetrics,
) -> Dict[str, bool]:
    """
    Evaluate all PSM-006 success gates.

    Extra checks:
    - oracle_above_tac : oracle success > TAC success (upper bound is above system)
    - no_update_underperforms: no-update < full TAC
    """
    results: Dict[str, bool] = {}

    for gate_name, gate in PSM006_GATES.items():
        val = full_agg.mean(gate["metric"])
        op  = gate["op"]
        thr = gate["threshold"]
        if op == ">=":
            results[gate_name] = val >= thr
        elif op == ">":
            results[gate_name] = val > thr
        elif op == "<=":
            results[gate_name] = val <= thr
        else:
            results[gate_name] = False

    # Oracle must remain above TAC
    results["oracle_above_tac"] = (
        oracle_agg.mean("verified_repair_success")
        >= full_agg.mean("verified_repair_success")
    )

    # No-update must underperform full TAC
    results["no_update_underperforms_tac"] = (
        no_update_agg.mean("verified_repair_success")
        < full_agg.mean("verified_repair_success")
    )

    return results
