"""
TAC-PSM-001 Ablation Runner

Runs 5 ablations against the full model and measures performance degradation.

Ablations:
  A  Remove failure modes         — ProcedureTrace.failure_modes always empty
  B  Remove recovery strategies   — no recovery strategy logged or applied
  C  Remove update mechanism      — procedures never updated after verification
  D  Remove transfer metadata     — transfer scores not updated; no cross-family bonus
  E  Remove survival scoring      — no decay; all survival scores stay at 1.0

Usage:
  python scripts/run_ablations.py
  python scripts/run_ablations.py --seeds 0 1 2 --output ./reports/ablations.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_tac_psm001 import (
    run_one_seed,
    aggregate,
    evaluate_gates,
    GATES,
    SeedResult,
)

ABLATION_LABELS = {
    "remove_failure_modes":       "A: No failure modes",
    "remove_recovery_strategies": "B: No recovery strategies",
    "remove_update_mechanism":    "C: No update mechanism",
    "remove_transfer_metadata":   "D: No transfer metadata",
    "remove_survival_scoring":    "E: No survival scoring",
}

KEY_METRICS = [
    "a2_reuse",
    "b1_transfer",
    "c1_transfer",
    "d1_retry",
    "retrieval_accuracy",
    "reuse_gain",
    "retry_improvement",
    "transfer_gain",
    "final_survival",
]


def run_ablation_study(
    seeds:      List[int],
    output:     Optional[str] = None,
    verbose:    bool = False,
) -> dict:

    print(f"\n{'='*65}")
    print(f"TAC-PSM-001 Ablation Study")
    print(f"Seeds: {seeds}  (n={len(seeds)})")
    print(f"{'='*65}")

    t_start  = time.time()
    all_results: Dict[str, Dict[str, Any]] = {}

    # ── Full model baseline ────────────────────────────────────────────────────
    print("\n[FULL] Full model (no ablation)...")
    full_results = [run_one_seed(s, ablation=None, verbose=verbose) for s in seeds]
    full_agg     = aggregate(full_results)
    full_gates   = evaluate_gates(full_agg)
    all_results["FULL"] = {
        "label":     "Full Model",
        "ablation":  None,
        "agg":       full_agg,
        "gates":     full_gates,
        "all_pass":  all(full_gates.values()),
    }
    _print_row("FULL (baseline)", full_agg)

    # ── Each ablation ──────────────────────────────────────────────────────────
    for ablation_key, ablation_label in ABLATION_LABELS.items():
        print(f"\n[{ablation_key}] {ablation_label}...")
        abl_results = [run_one_seed(s, ablation=ablation_key, verbose=verbose) for s in seeds]
        abl_agg     = aggregate(abl_results)
        abl_gates   = evaluate_gates(abl_agg)

        # Degradation = full_metric - ablation_metric (positive = ablation hurts)
        degradation = {}
        for k in KEY_METRICS:
            full_v = full_agg.get(k, {})
            abl_v  = abl_agg.get(k, {})
            fm = full_v["mean"] if isinstance(full_v, dict) else float(full_v or 0)
            am = abl_v["mean"]  if isinstance(abl_v, dict)  else float(abl_v or 0)
            degradation[k] = round(fm - am, 4)

        all_results[ablation_key] = {
            "label":       ablation_label,
            "ablation":    ablation_key,
            "agg":         abl_agg,
            "gates":       abl_gates,
            "all_pass":    all(abl_gates.values()),
            "degradation": degradation,
        }
        _print_row(ablation_label, abl_agg)

    elapsed = time.time() - t_start

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("ABLATION DEGRADATION TABLE")
    print(f"  Positive = ablation hurts performance (full - ablation)")
    print(f"{'='*65}")

    col_w = 10
    header = f"  {'Metric':<28}" + "".join(
        f"{k[:col_w]:>{col_w}}" for k in ["FULL"] + list(ABLATION_LABELS.keys())
    )
    print(header)
    print("  " + "-" * (28 + col_w * (1 + len(ABLATION_LABELS))))

    for metric in KEY_METRICS:
        full_v = all_results["FULL"]["agg"].get(metric, {})
        fm     = full_v["mean"] if isinstance(full_v, dict) else 0.0
        row    = f"  {metric:<28}{fm:>{col_w}.4f}"
        for abl_key in ABLATION_LABELS:
            deg = all_results[abl_key].get("degradation", {}).get(metric, 0.0)
            mark = "↓" if deg > 0.02 else (" " if abs(deg) <= 0.02 else "↑")
            row += f"{deg:>{col_w - 1}.4f}{mark}"
        print(row)

    print(f"\n  Gates passed per condition:")
    for key in ["FULL"] + list(ABLATION_LABELS.keys()):
        res  = all_results[key]
        n_p  = sum(res["gates"].values())
        n_t  = len(res["gates"])
        lbl  = res["label"]
        sym  = "✓" if res["all_pass"] else "✗"
        print(f"  [{sym}] {lbl:<40} {n_p}/{n_t} gates")

    print(f"\n  Elapsed: {elapsed:.1f}s")

    # ── Save ───────────────────────────────────────────────────────────────────
    report = {
        "experiment": "TAC-PSM-001 Ablations",
        "seeds":      seeds,
        "elapsed_s":  elapsed,
        "results":    {k: {
            "label":       v["label"],
            "all_pass":    v["all_pass"],
            "degradation": v.get("degradation", {}),
            "gates":       v["gates"],
        } for k, v in all_results.items()},
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved → {output}")

    return report


def _print_row(label: str, agg: dict):
    parts = []
    for m in ["a2_reuse", "b1_transfer", "d1_retry", "retrieval_accuracy", "reuse_gain"]:
        v = agg.get(m, {})
        vm = v["mean"] if isinstance(v, dict) else 0.0
        parts.append(f"{m}={vm:.3f}")
    print("  " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="TAC-PSM-001 Ablation Runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=str, default="./reports/psm001_ablations.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_ablation_study(seeds=args.seeds, output=args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
