"""
TAC-PSM-006 Replication Script
================================

Runs the full 5-seed replication of PSM-006 and produces:
  - reports/psm006_replication.json    (full data)
  - reports/psm006_replication_summary.txt  (human-readable summary)

Usage:
  python scripts/run_psm006_replication.py
  python scripts/run_psm006_replication.py --seeds 0 1 2 3 4
  python scripts/run_psm006_replication.py --quick    (seed 0 only, 5 tasks/family)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_tac_psm006_repository_memory import run_benchmark, TASKS_PER_FAMILY
from tacm.psm006 import ALL_FAMILY_NAMES, BASELINE_NAMES, PSM006_GATES


REPORT_DIR = Path(__file__).parent.parent / "reports"


# ── Summary builder ───────────────────────────────────────────────────────────

def build_summary_text(report: Dict[str, Any]) -> str:
    agg   = report["aggregate"]
    claim = report["research_claim"]
    seeds = report["seeds"]

    lines = [
        "=" * 70,
        "TAC-PSM-006 REPLICATION REPORT",
        "Repository-Grounded Procedural Memory",
        "=" * 70,
        "",
        f"Seeds run:    {seeds}",
        f"N seeds:      {len(seeds)}",
        f"Tasks/family: {TASKS_PER_FAMILY}  (6 families × {TASKS_PER_FAMILY} = {TASKS_PER_FAMILY*6} total)",
        f"Elapsed:      {report['elapsed']:.1f}s",
        "",
        "─" * 70,
        "VERIFIED REPAIR SUCCESS — All Variants (mean ± std)",
        "─" * 70,
    ]

    for name in BASELINE_NAMES:
        v   = agg["variant_agg"].get(name, {})
        vrs = v.get("verified_repair_success", {})
        m   = vrs.get("mean", 0.0)
        s   = vrs.get("std",  0.0)
        tag = " ← PRIMARY SYSTEM" if name == "full_memory" else \
              " ← UPPER BOUND"     if name == "oracle"      else \
              " ← LOWER BOUND"     if name == "reset"       else ""
        lines.append(f"  {name:<26} {m:>7.4f} ± {s:.4f}{tag}")

    lines += [
        "",
        "─" * 70,
        "KEY METRICS (full_memory variant)",
        "─" * 70,
    ]
    fm = agg["variant_agg"].get("full_memory", {})
    metric_labels = [
        ("verified_repair_success",      "Verified repair success"),
        ("procedure_retrieval_accuracy", "Procedure retrieval accuracy"),
        ("procedure_reuse_gain",         "Reuse gain over reset"),
        ("update_retry_improvement",     "Update retry improvement"),
        ("transfer_success",             "Cross-repo transfer success"),
        ("wrong_procedure_harm",         "Wrong-procedure harm"),
        ("steps_to_repair",              "Avg steps to repair"),
        ("survival_score_stability",     "Survival score std-dev"),
    ]
    for key, label in metric_labels:
        v = fm.get(key, {})
        lines.append(f"  {label:<36} {v.get('mean', 0.0):>7.4f} ± {v.get('std', 0.0):.4f}")

    lines += [
        "",
        "─" * 70,
        "PER-FAMILY SUCCESS (full_memory, mean ± std)",
        "─" * 70,
    ]
    for family in ALL_FAMILY_NAMES:
        fs = agg["family_agg"].get(family, {})
        lines.append(f"  {family:<28} {fs.get('mean', 0.0):.4f} ± {fs.get('std', 0.0):.4f}")

    lines += [
        "",
        "─" * 70,
        "SUCCESS GATES (pass rate across seeds)",
        "─" * 70,
    ]
    gate_rates = agg.get("gate_pass_rates", {})
    all_pass = True
    for gate, gate_def in PSM006_GATES.items():
        rate = gate_rates.get(gate, 0.0)
        sym  = "✓" if rate >= 1.0 else ("~" if rate >= 0.6 else "✗")
        lines.append(f"  [{sym}] {gate:<44} {rate:.2f}/1.0")
        if rate < 1.0:
            all_pass = False
    for extra in ["oracle_above_tac", "no_update_underperforms_tac"]:
        rate = gate_rates.get(extra, 0.0)
        sym  = "✓" if rate >= 1.0 else "✗"
        lines.append(f"  [{sym}] {extra:<44} {rate:.2f}/1.0")
        if rate < 1.0:
            all_pass = False

    lines += [
        "",
        "─" * 70,
        "RESEARCH CLAIM VERDICT",
        "─" * 70,
        f"  TAC full memory success:  {claim['tac_success']:.4f}",
        f"  Reset baseline success:   {claim['reset_success']:.4f}",
        f"  Oracle upper bound:       {claim['oracle_success']:.4f}",
        f"  Gain over reset:          {claim['gain']:+.4f}  "
        f"(threshold >= 0.10)",
        "",
        "  VERDICT: " + (
            "✓ PSM-006 VALIDATES repository-grounded procedural memory."
            if claim["validated"]
            else "✗ PSM-006 does NOT yet validate claim (gain < 0.10)."
        ),
        "",
        "  Overall gate pass rate: "
        f"{agg['overall_pass']:.2f}/1.0",
        "",
        "=" * 70,
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TAC-PSM-006 Replication: 5-seed benchmark"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: seed 0 only, 5 tasks/family")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tasks", type=int, default=TASKS_PER_FAMILY)
    args = parser.parse_args()

    seeds = [0] if args.quick else args.seeds
    tpf   = 5   if args.quick else args.tasks

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = str(REPORT_DIR / "psm006_replication.json")
    txt_path  = REPORT_DIR / "psm006_replication_summary.txt"

    print(f"\nTAC-PSM-006 Replication")
    print(f"Seeds: {seeds}  |  Tasks/family: {tpf}")
    print(f"Output: {json_path}")

    report = run_benchmark(
        seeds            = seeds,
        verbose          = args.verbose,
        output           = json_path,
        tasks_per_family = tpf,
    )

    summary = build_summary_text(report)
    txt_path.write_text(summary, encoding="utf-8")

    print(f"\n{'─'*68}")
    print(summary)
    print(f"\n  Summary → {txt_path}")


if __name__ == "__main__":
    main()
