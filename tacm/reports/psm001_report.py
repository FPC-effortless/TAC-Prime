"""
TAC-PSM-001 Research Report Generator

Produces:
  1. Research report        (Markdown)
  2. Failure analysis       (Markdown)
  3. Replication summary    (Markdown)

Reads from JSON outputs produced by benchmark_tac_psm001.py,
run_ablations.py, and replicate_psm001.py.

Usage:
  # First run experiments:
  python scripts/benchmark_tac_psm001.py --seeds 5 --output reports/psm001_benchmark.json
  python scripts/run_ablations.py --output reports/psm001_ablations.json
  python scripts/replicate_psm001.py --output_dir reports/replication

  # Then generate report:
  python reports/psm001_report.py
  python reports/psm001_report.py --benchmark reports/psm001_benchmark.json \\
                                   --ablations reports/psm001_ablations.json \\
                                   --replication reports/replication/replication_summary.json \\
                                   --output    reports/TAC_PSM001_Research_Report.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [warn] Not found: {path}")
        return None
    with open(p) as f:
        return json.load(f)


def _stat(d: Optional[dict], key: str) -> str:
    if d is None:
        return "—"
    v = d.get("agg", {}).get(key, None) or d.get(key, None)
    if v is None:
        return "—"
    if isinstance(v, dict):
        m  = v.get("mean", 0.0)
        s  = v.get("std",  0.0)
        ci = v.get("ci95", 0.0)
        return f"{m:.4f} ± {s:.4f}  (95% CI ±{ci:.4f})"
    return f"{v:.4f}"


def _gate_row(gates: dict) -> str:
    passed = sum(gates.values())
    total  = len(gates)
    sym    = "✓" if passed == total else "⚠"
    return f"{sym} {passed}/{total} gates passed"


# ── Report sections ────────────────────────────────────────────────────────────

def research_report(
    benchmark: Optional[dict],
    ablations: Optional[dict],
    replication: Optional[dict],
) -> str:

    ts = time.strftime("%Y-%m-%d %H:%M UTC")
    B  = benchmark
    A  = ablations
    R  = replication

    lines = [
        "# TAC-PSM-001: Procedural Memory Build / Retrieve / Update",
        f"**Research Report**  |  Generated: {ts}",
        "",
        "---",
        "",
        "## 1. Experiment Overview",
        "",
        "**Hypothesis:** TAC can learn, store, retrieve, update, and reuse procedures,",
        "producing measurable gains over reset, retrieval-disabled, and incorrect-procedure baselines.",
        "",
        "**Null hypothesis:** Procedural memory provides no measurable advantage over",
        "reset systems, retrieval-disabled systems, random procedure retrieval, or structure-only memory.",
        "",
        "**Central claim:**",
        "> \"TAC can remember, reuse, adapt, and improve procedures, producing measurable",
        "> gains over memory-disabled and reset baselines.\"",
        "",
        "---",
        "",
        "## 2. Experimental Design",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Seeds | {B.get('n_seeds', '?') if B else '?'} |",
        "| Task families | 4 (ImportErrors, DependencyConflicts, VersionMismatch, PathResolution) |",
        "| Baselines | 5 (Reset, Disabled, Random, Structure-only, Oracle) |",
        "| Ablations | 5 (failure modes, recovery, update, transfer, survival) |",
        "| Embedding dim | 64 (benchmark), 512 (production model) |",
        "| Evaluation sequence | A1 → A2 → B1 → C1 → D1 |",
        "",
        "### Evaluation Sequence",
        "",
        "| Step | Description | Measurement |",
        "|---|---|---|",
        "| A1 | Solve initial import error; store procedure | Baseline success rate |",
        "| A2 | Retrieve + reuse procedure on similar import error | Retrieval accuracy, reuse gain |",
        "| B1 | Transfer A→B: dependency conflict | Transfer accuracy, transfer gain |",
        "| C1 | Transfer A→C: version mismatch | Broader transfer quality |",
        "| D1 | Force failure → update → retry | Retry improvement, failure recovery |",
        "",
        "---",
        "",
        "## 3. Results",
        "",
        "### 3.1 Primary Metrics",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| A1 — Initial success | {_stat(B, 'a1_success')} |",
        f"| A2 — Procedure reuse success | {_stat(B, 'a2_reuse')} |",
        f"| B1 — Transfer (A→B) | {_stat(B, 'b1_transfer')} |",
        f"| C1 — Transfer (A→C) | {_stat(B, 'c1_transfer')} |",
        f"| D1 — Retry after update | {_stat(B, 'd1_retry')} |",
        f"| Retrieval accuracy | {_stat(B, 'retrieval_accuracy')} |",
        f"| Family match rate | {_stat(B, 'family_match_rate')} |",
        f"| Procedure reuse gain | {_stat(B, 'reuse_gain')} |",
        f"| Retry improvement | {_stat(B, 'retry_improvement')} |",
        f"| Transfer gain | {_stat(B, 'transfer_gain')} |",
        f"| Reset deficit | {_stat(B, 'reset_deficit')} |",
        f"| Memory survival (final) | {_stat(B, 'final_survival')} |",
        "",
        "### 3.2 Baseline Comparison",
        "",
        "| Condition | Reuse Success Rate |",
        "|---|---|",
        f"| TAC-PSM (correct retrieval) | {_stat(B, 'a2_reuse')} |",
        f"| Memory disabled | {_stat(B, 'disabled_rate')} |",
        f"| Random retrieval | {_stat(B, 'random_rate')} |",
        f"| Oracle retrieval | {_stat(B, 'oracle_rate')} |",
        "",
        "### 3.3 Success Gates",
        "",
        "| Gate | Threshold | Result |",
        "|---|---|---|",
    ]

    gates_data = B.get("gates", {}) if B else {}
    gate_thresholds = {
        "retrieval_accuracy_ge_0.70":   "≥ 0.70",
        "reuse_gain_ge_0.10":           "≥ 0.10",
        "update_improves_retry":        "> 0.0",
        "reset_deficit_ge_0.20":        "≥ 0.20",
        "random_worse_than_correct":    "< 0.0 (random worse)",
        "transfer_gain_gt_0":           "> 0.0",
        "survival_stable_across_seeds": "CV < 0.30",
    }
    for gname, threshold in gate_thresholds.items():
        passed = gates_data.get(gname, None)
        if passed is None:
            sym = "—"
        else:
            sym = "✓ PASS" if passed else "✗ FAIL"
        lines.append(f"| {gname} | {threshold} | {sym} |")

    all_pass = all(gates_data.values()) if gates_data else False
    lines += [
        "",
        f"**Overall gate verdict: {'ALL PASS ✓' if all_pass else 'PARTIAL / FAIL ✗'}**",
        "",
        "---",
        "",
        "## 4. Ablation Study",
        "",
        "Ablation study measures performance degradation when each component is removed.",
        "Positive degradation = component contributes positively to performance.",
        "",
        "| Ablation | Reuse Δ | Transfer Δ | Retry Δ | Retrieval Δ |",
        "|---|---|---|---|---|",
    ]

    abl_results = A.get("results", {}) if A else {}
    for abl_key, abl_label in {
        "remove_failure_modes":       "A: No failure modes",
        "remove_recovery_strategies": "B: No recovery strategies",
        "remove_update_mechanism":    "C: No update mechanism",
        "remove_transfer_metadata":   "D: No transfer metadata",
        "remove_survival_scoring":    "E: No survival scoring",
    }.items():
        abl = abl_results.get(abl_key, {})
        deg = abl.get("degradation", {})
        lines.append(
            f"| {abl_label} "
            f"| {deg.get('a2_reuse', '—')} "
            f"| {deg.get('b1_transfer', '—')} "
            f"| {deg.get('d1_retry', '—')} "
            f"| {deg.get('retrieval_accuracy', '—')} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Replication",
        "",
        f"| Parameter | Value |",
        "|---|---|",
        f"| Seeds | {R.get('n_seeds', '?') if R else '?'} |",
        f"| Verdict | {R.get('replication_verdict', '—') if R else '—'} |",
        f"| Gates passed | {(str(R['n_gates_pass']) + '/' + str(R['n_gates_pass'] + R['n_gates_fail'])) if R else '—'} |",
        "",
        "---",
        "",
        "## 6. Discussion",
        "",
        "### What this experiment tests",
        "",
        "TAC-PSM-001 validates whether a procedural memory system — storing ordered action",
        "sequences rather than raw text or embeddings — provides measurable improvements",
        "over memory-disabled baselines.",
        "",
        "### What counts as success",
        "",
        "- **Retrieval accuracy ≥ 0.70:** The system must retrieve the correct procedure",
        "  for at least 70% of tasks — otherwise memory is not useful.",
        "",
        "- **Reuse gain ≥ 0.10:** Using retrieved procedures must improve success rate",
        "  by at least 10 percentage points over not using memory.",
        "",
        "- **Retry improvement > 0:** After a failure and update, the retry must succeed",
        "  more often than the original attempt.",
        "",
        "- **Transfer gain > 0:** Procedures from family A must help on families B/C.",
        "",
        "### Limitations",
        "",
        "- The benchmark uses synthetic task families. Transfer to real codebases",
        "  requires integration with actual code execution (stage 3, agent loop).",
        "",
        "- The embedding dimension (64) used in benchmarks is smaller than the",
        "  production model (512). Performance may differ at production scale.",
        "",
        "- Procedure quality is measured by step-set Jaccard similarity to canonical",
        "  procedures, not by actual code execution success.",
        "",
        "---",
        "",
        "## 7. Next Experiments",
        "",
        "If PSM-001 validates successfully:",
        "",
        "| Experiment | Hypothesis |",
        "|---|---|",
        "| TAC-PSM-002 | Procedural transfer improves with chain length A→B→C |",
        "| TAC-PSM-003 | Procedure lifecycle transitions improve long-run performance |",
        "| TAC-PSM-004 | Survival fields correctly identify high-value procedures |",
        "| TAC-PSM-005 | Autonomous procedure discovery from unlabelled traces |",
        "",
        "---",
        "",
        "*Report generated by `tacm/reports/psm001_report.py`*",
    ]

    return "\n".join(lines)


def failure_analysis(benchmark: Optional[dict]) -> str:
    B     = benchmark
    gates = B.get("gates", {}) if B else {}
    failed_gates = [g for g, p in gates.items() if not p]

    lines = [
        "# TAC-PSM-001: Failure Analysis",
        "",
        f"Gates failed: {len(failed_gates)} / {len(gates)}",
        "",
    ]

    if not failed_gates:
        lines += [
            "**No gate failures detected.** All success criteria were met.",
            "",
            "If unexpected failures occur in future runs, common root causes are:",
            "",
            "| Failure pattern | Most likely cause | Corrective experiment |",
            "|---|---|---|",
            "| Low retrieval accuracy | Embedding space poorly separated by family | Increase embedding dim; add family-contrastive loss |",
            "| Low reuse gain | Retrieved procedures too generic | Increase task-signature specificity |",
            "| No retry improvement | Fork threshold too high | Lower `fork_threshold`; improve recovery step injection |",
            "| No transfer gain | Family embeddings too distant | Add cross-family similarity loss during store build |",
            "| Survival instability | High decay rate | Lower `decay_rate` in training config |",
        ]
        return "\n".join(lines)

    cause_map = {
        "retrieval_accuracy_ge_0.70":   (
            "Incorrect retrieval",
            "Embedding space not separated; family filter not applied",
            "Increase embedding dim; train with family-contrastive objective",
        ),
        "reuse_gain_ge_0.10":           (
            "Poor adaptation",
            "Retrieved steps too generic or semantically mismatched",
            "Increase task signature specificity; add step-level similarity",
        ),
        "update_improves_retry":         (
            "Insufficient update",
            "Fork threshold too high; recovery steps not appended correctly",
            "Lower fork_threshold; verify recovery_steps injection in _fork_procedure",
        ),
        "reset_deficit_ge_0.20":         (
            "Reset baseline too strong",
            "Task difficulty too low — tasks solvable without memory",
            "Increase task difficulty; verify distractor steps are distinct from canonical",
        ),
        "random_worse_than_correct":     (
            "Family confusion",
            "Correct retrieval not outperforming random — embedding space collapsed",
            "Verify FAISS index is populated; check embedding normalisation",
        ),
        "transfer_gain_gt_0":            (
            "Weak transfer",
            "Family embeddings too far apart; no cross-family adaptation",
            "Add family-overlap embedding; test partial step reuse",
        ),
        "survival_stable_across_seeds":  (
            "Memory collapse",
            "Survival CV too high — procedures pruned aggressively across seeds",
            "Reduce decay rate; increase prune threshold margin",
        ),
    }

    lines += [
        "## Failed Gates",
        "",
        "| Gate | Root Cause | Likely Mechanism | Corrective Action |",
        "|---|---|---|---|",
    ]

    for gname in failed_gates:
        if gname in cause_map:
            cause, mech, fix = cause_map[gname]
            lines.append(f"| {gname} | {cause} | {mech} | {fix} |")
        else:
            lines.append(f"| {gname} | Unknown | — | Investigate manually |")

    lines += [
        "",
        "## Proposed Corrective Experiments",
        "",
        "1. **Re-run with higher embedding dim** (128 or 256) to improve family separation.",
        "2. **Add explicit family-contrastive loss** during procedure encoding.",
        "3. **Lower fork_threshold to 1** to ensure all failures trigger a recovery branch.",
        "4. **Increase task difficulty parameters** (`difficulty` field in TaskInstance) "
        "to ensure memory provides measurable gain.",
        "5. **Inspect per-seed variance** — if variance is high, increase seed count.",
        "",
        "*Generated by `tacm/reports/psm001_report.py`*",
    ]

    return "\n".join(lines)


def replication_summary_md(replication: Optional[dict]) -> str:
    R = replication
    if R is None:
        return "# Replication Summary\n\nNo replication data available.\n"

    verdict   = R.get("replication_verdict", "UNKNOWN")
    n_seeds   = R.get("n_seeds", 0)
    n_pass    = R.get("n_gates_pass", 0)
    n_fail    = R.get("n_gates_fail", 0)
    elapsed   = R.get("elapsed_total_s", 0)
    gates     = R.get("aggregate", {})

    lines = [
        "# TAC-PSM-001: Replication Summary",
        "",
        f"| Parameter | Value |",
        "|---|---|",
        f"| Seeds | {n_seeds} |",
        f"| Verdict | **{verdict}** |",
        f"| Gates passed | {n_pass}/{n_pass + n_fail} |",
        f"| Elapsed | {elapsed:.1f}s |",
        "",
        "## Gate Results",
        "",
        "| Gate | Pass? |",
        "|---|---|",
    ]

    gate_results = R.get("gates", {})
    for gname, passed in gate_results.items():
        sym = "✓" if passed else "✗"
        lines.append(f"| {gname} | {sym} |")

    lines += [
        "",
        f"## Replication Verdict: {verdict}",
        "",
        ("The experiment was replicated successfully across all seeds."
         if verdict == "REPLICATED"
         else "Some success gates failed. See failure analysis for root causes."),
        "",
        "*Generated by `tacm/reports/psm001_report.py`*",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TAC-PSM-001 Report Generator")
    parser.add_argument("--benchmark",   type=str, default="./reports/psm001_benchmark.json")
    parser.add_argument("--ablations",   type=str, default="./reports/psm001_ablations.json")
    parser.add_argument("--replication", type=str, default="./reports/replication/replication_summary.json")
    parser.add_argument("--output",      type=str, default="./reports/TAC_PSM001_Research_Report.md")
    parser.add_argument("--failure_output", type=str, default="./reports/TAC_PSM001_Failure_Analysis.md")
    parser.add_argument("--replication_output", type=str, default="./reports/TAC_PSM001_Replication_Summary.md")
    args = parser.parse_args()

    bench = _load_json(args.benchmark)
    abl   = _load_json(args.ablations)
    rep   = _load_json(args.replication)

    # Research report
    report_md = research_report(bench, abl, rep)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report_md)
    print(f"Research report  → {args.output}")

    # Failure analysis
    fail_md = failure_analysis(bench)
    with open(args.failure_output, "w") as f:
        f.write(fail_md)
    print(f"Failure analysis → {args.failure_output}")

    # Replication summary
    rep_md = replication_summary_md(rep)
    with open(args.replication_output, "w") as f:
        f.write(rep_md)
    print(f"Replication summary → {args.replication_output}")


if __name__ == "__main__":
    main()
