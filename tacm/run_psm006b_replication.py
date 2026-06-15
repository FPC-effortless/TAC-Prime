"""
TAC-PSM-006B: Replication Runner
==================================

Standalone replication script. Runs the full PSM-006B benchmark, writes
results to reports/, and prints a summary matching the format expected in
the PSM-006B report.

Usage:
  python run_psm006b_replication.py [--seeds 0 1 2 3 4] [--quick] [--out reports/]

This script is the single entry point for external verification of
TAC-PSM-006B results.  It produces:
  - reports/psm006b_results.json    — machine-readable full results
  - reports/psm006b_summary.txt     — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from tacm.psm006b.fixture_builder import build_all_fixtures
from tacm.psm006b.fixture_schema import FAMILY_NAMES
from tacm.psm006b.baselines import run_all_baselines, VARIANT_NAMES
from tacm.psm006b.metrics import (
    compute_metrics,
    evaluate_success_gates,
    compute_family_confusion_matrix,
    classify_failures,
)
from tacm.psm006b.procedural_repair_agent import RepairTrace006B


def _pass_rate(traces: List[RepairTrace006B]) -> float:
    if not traces:
        return 0.0
    return mean(1.0 if t.pytest_pass else 0.0 for t in traces)


def run_seed(seed: int, fixtures: list, timeout_s: float) -> dict:
    print(f"  Seed {seed}: running {len(fixtures)} fixtures × 7 variants ...",
          flush=True)
    t0 = time.time()

    results = run_all_baselines(
        fixtures  = fixtures,
        seed      = seed,
        timeout_s = timeout_s,
    )

    metrics = compute_metrics(results, reference_variant="full_memory")
    gates   = evaluate_success_gates(metrics, results)
    failures = classify_failures(results.get("full_memory", []))
    confusion = compute_family_confusion_matrix(
        results.get("full_memory", []), FAMILY_NAMES
    )
    variant_rates = {v: _pass_rate(results.get(v, [])) for v in VARIANT_NAMES}

    elapsed = time.time() - t0
    print(f"  Seed {seed} done in {elapsed:.1f}s  "
          f"TAC={variant_rates['full_memory']:.3f}  "
          f"oracle={variant_rates['oracle']:.3f}  "
          f"reset={variant_rates['reset']:.3f}",
          flush=True)

    return {
        "seed":           seed,
        "metrics":        metrics,
        "gates":          gates,
        "variant_rates":  variant_rates,
        "failures":       failures,
        "confusion":      {k: dict(v) for k, v in confusion.items()},
        "n_fixtures":     len(fixtures),
        "elapsed_s":      elapsed,
    }


def aggregate_seeds(seed_results: List[dict]) -> dict:
    keys = list(seed_results[0]["metrics"].keys())
    agg_metrics = {k: mean(r["metrics"][k] for r in seed_results) for k in keys}

    gate_names = list(seed_results[0]["gates"].keys())
    gate_pass_rates = {}
    for g in gate_names:
        n_pass = sum(1 for r in seed_results if r["gates"].get(g, False))
        gate_pass_rates[g] = {"pass": n_pass, "total": len(seed_results)}

    variant_names = list(seed_results[0]["variant_rates"].keys())
    agg_variant = {v: mean(r["variant_rates"][v] for r in seed_results)
                   for v in variant_names}

    return {
        "metrics":        agg_metrics,
        "gate_pass_rates": gate_pass_rates,
        "variant_rates":  agg_variant,
    }


def compute_verdict(seed_results: List[dict]) -> str:
    gate_names  = list(seed_results[0]["gates"].keys())
    n_seeds     = len(seed_results)
    gates_all   = sum(
        1 for g in gate_names
        if sum(1 for r in seed_results if r["gates"].get(g, False)) == n_seeds
    )
    gates_most  = sum(
        1 for g in gate_names
        if sum(1 for r in seed_results if r["gates"].get(g, False)) >= n_seeds * 0.6
    )
    if gates_all == len(gate_names):
        return "VALIDATES"
    elif gates_most >= 5:
        return "PARTIALLY_VALIDATES"
    else:
        return "DOES_NOT_VALIDATE"


def write_summary(
    agg:          dict,
    seed_results: List[dict],
    verdict:      str,
    out_path:     str,
) -> None:
    lines = [
        "TAC-PSM-006B Replication Summary",
        "=" * 60,
        "",
        f"Verdict: TAC-PSM-006B {verdict}",
        "",
        "Variant Pass Rates (mean across seeds):",
    ]
    for v, rate in agg["variant_rates"].items():
        lines.append(f"  {v:<26s}  {rate:.3f}")
    lines.append("")
    lines.append("Key Metrics (mean across seeds):")
    for k, v in agg["metrics"].items():
        lines.append(f"  {k:<42s}  {v:.4f}")
    lines.append("")
    lines.append("Gate Results (n_seeds_passing / total):")
    for gate, info in agg["gate_pass_rates"].items():
        passed = info["pass"] == info["total"]
        sym = "[PASS]" if passed else "[FAIL]"
        lines.append(f"  {sym} {gate}  {info['pass']}/{info['total']}")
    lines.append("")
    lines.append(
        "TAC-PSM-006B validates / partially validates / does not validate "
        "semi-real pytest-grounded procedural memory."
    )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="PSM-006B replication runner")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick",   action="store_true",
                        help="Run 12 fixtures only (3 families × 4) for fast iteration")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out",     type=str, default="reports")
    args = parser.parse_args()

    all_fixtures = build_all_fixtures()
    if args.quick:
        fixtures = []
        for fam in FAMILY_NAMES[:3]:
            fx_fam = [fx for fx in all_fixtures if fx.family == fam]
            fixtures.extend(fx_fam[:4])
    else:
        fixtures = all_fixtures

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTAC-PSM-006B Replication Run")
    print(f"  Fixtures: {len(fixtures)}  |  Seeds: {args.seeds}  |  "
          f"Timeout: {args.timeout}s  |  Output: {args.out}/")
    print()

    t_total = time.time()
    seed_results = [run_seed(s, fixtures, args.timeout) for s in args.seeds]
    elapsed = time.time() - t_total

    agg     = aggregate_seeds(seed_results)
    verdict = compute_verdict(seed_results)

    # Write JSON results
    json_path = out_dir / "psm006b_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "seed_results": [
                {k: v for k, v in sr.items() if k != "confusion"}
                for sr in seed_results
            ],
            "aggregate": agg,
            "verdict":   verdict,
        }, f, indent=2)

    # Write human-readable summary
    txt_path = out_dir / "psm006b_summary.txt"
    write_summary(agg, seed_results, verdict, str(txt_path))

    # Print summary
    print(f"\nVERDICT: TAC-PSM-006B {verdict}")
    print(f"Total elapsed: {elapsed:.1f}s")
    print(f"Results: {json_path}  |  Summary: {txt_path}\n")

    n_gates = len(seed_results[0]["gates"])
    n_all_pass = sum(
        1 for g in seed_results[0]["gates"]
        if all(r["gates"].get(g, False) for r in seed_results)
    )
    print(f"  {n_all_pass}/{n_gates} gates pass on all seeds\n")

    sys.exit(0 if verdict == "VALIDATES" else 1)


if __name__ == "__main__":
    main()
