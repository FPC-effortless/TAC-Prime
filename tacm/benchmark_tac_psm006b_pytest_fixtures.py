"""
TAC-PSM-006B: Benchmark — Semi-Real Pytest Repository Repair Fixtures
=======================================================================

Core claim:
  TAC can reuse procedural repair memory to improve real pytest-verified
  repository repair over reset, retrieval-disabled, random-procedure,
  structure-only, and no-update baselines.

This benchmark runs all 7 variants on all 60 fixtures across N seeds and
evaluates 8 success gates.  Because PytestVerifier spawns real subprocess
invocations, results are grounded in actual pytest exit codes — not heuristics.

Usage:
  python benchmark_tac_psm006b_pytest_fixtures.py [--seeds 0 1 2 3 4] [--quick]

  --quick  runs only 3 families × 2 fixtures (12 fixtures) for fast iteration.
  --seeds  specifies which seeds to run (default: 0 1 2 3 4).

Exit codes:
  0 — all 8 gates pass on every seed
  1 — at least one gate fails on at least one seed

Important calibration rule:
  Do NOT over-tune to make TAC pass. If a gate fails, record the failure
  class and keep going. Partial validation is still informative.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from statistics import mean
from typing import Dict, List

sys.path.insert(0, __file__.rsplit("/benchmark", 1)[0])

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


# ── Gate thresholds (from PSM-006B spec) ────────────────────────────────

GATE_NAMES = [
    "tac_beats_reset_by_0.10",
    "retrieval_accuracy_ge_0.55",
    "update_improves_retry",
    "no_update_underperforms_tac",
    "random_procedure_no_benefit",
    "oracle_above_tac",
    "cross_fixture_transfer_positive",
    "reuse_gain_positive",
]


# ── Per-seed result ──────────────────────────────────────────────────────

def run_seed(
    seed:     int,
    fixtures: list,
    timeout_s: float,
    variants: List[str],
) -> dict:
    """Run all variants on all fixtures for one seed. Return metrics + gates."""
    results = run_all_baselines(
        fixtures  = fixtures,
        seed      = seed,
        timeout_s = timeout_s,
        variants  = variants,
    )

    metrics = compute_metrics(results, reference_variant="full_memory")
    gates   = evaluate_success_gates(metrics, results)

    full_traces = results.get("full_memory", [])
    failures    = classify_failures(full_traces)
    confusion   = compute_family_confusion_matrix(full_traces, FAMILY_NAMES)

    # Per-variant pass rates for reporting
    variant_rates = {v: _pass_rate(results.get(v, [])) for v in variants}

    return {
        "seed":          seed,
        "metrics":       metrics,
        "gates":         gates,
        "variant_rates": variant_rates,
        "failures":      failures,
        "confusion":     confusion,
        "n_fixtures":    len(fixtures),
    }


def _pass_rate(traces: List[RepairTrace006B]) -> float:
    if not traces:
        return 0.0
    return mean(1.0 if t.pytest_pass else 0.0 for t in traces)


# ── Aggregation ──────────────────────────────────────────────────────────

def aggregate_seeds(seed_results: List[dict]) -> dict:
    def avg_metric(key: str) -> float:
        return mean(r["metrics"][key] for r in seed_results)

    agg_metrics = {
        k: avg_metric(k) for k in seed_results[0]["metrics"]
    }

    gate_pass_rates = {}
    for gate in GATE_NAMES:
        passed = sum(1 for r in seed_results if r["gates"].get(gate, False))
        gate_pass_rates[gate] = f"{passed}/{len(seed_results)}"

    agg_variant_rates = {}
    for v in seed_results[0]["variant_rates"]:
        agg_variant_rates[v] = mean(r["variant_rates"][v] for r in seed_results)

    return {
        "metrics":       agg_metrics,
        "gate_pass_rates": gate_pass_rates,
        "variant_rates": agg_variant_rates,
    }


# ── Verdict ───────────────────────────────────────────────────────────────

def verdict(seed_results: List[dict]) -> str:
    """
    Classify overall validation status.

    VALIDATES          — all 8 gates pass on all seeds
    PARTIALLY_VALIDATES — 5–7 gates pass consistently
    DOES_NOT_VALIDATE  — fewer than 5 gates pass
    """
    per_gate_pass = {}
    for gate in GATE_NAMES:
        n_pass = sum(1 for r in seed_results if r["gates"].get(gate, False))
        per_gate_pass[gate] = n_pass

    n_seeds    = len(seed_results)
    gates_all  = sum(1 for c in per_gate_pass.values() if c == n_seeds)
    gates_most = sum(1 for c in per_gate_pass.values() if c >= n_seeds * 0.6)

    if gates_all == len(GATE_NAMES):
        return "VALIDATES"
    elif gates_most >= 5:
        return "PARTIALLY_VALIDATES"
    else:
        return "DOES_NOT_VALIDATE"


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TAC-PSM-006B benchmark")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick",   action="store_true",
                        help="Run only 12 fixtures (3 families × 4) for fast iteration")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Per-fixture pytest timeout in seconds (default 10)")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variants to run (default: all 7)")
    parser.add_argument("--save",    type=str, default=None,
                        help="Path to save JSON results (optional)")
    args = parser.parse_args()

    all_fixtures = build_all_fixtures()
    if args.quick:
        # Take first 2 fixtures from each of the first 3 families
        quick_fixtures = []
        for fam in FAMILY_NAMES[:3]:
            fam_fx = [fx for fx in all_fixtures if fx.family == fam]
            quick_fixtures.extend(fam_fx[:4])
        fixtures = quick_fixtures
    else:
        fixtures = all_fixtures

    variants = args.variants or VARIANT_NAMES

    print("\n" + "=" * 72)
    print("  TAC-PSM-006B: Semi-Real Pytest Repository Repair Fixtures")
    print(f"  Seeds: {args.seeds}  |  Fixtures: {len(fixtures)}  |  "
          f"Variants: {len(variants)}  |  Timeout: {args.timeout}s/fixture")
    print("=" * 72)

    t_start = time.time()
    seed_results = []

    # Ensure full_memory is included for reference metrics unless explicitly excluded
    if "full_memory" not in variants and len(variants) < len(VARIANT_NAMES):
        print(f"  NOTE: 'full_memory' not in variants; adding it for reference metrics.")
        variants = ["full_memory"] + [v for v in variants if v != "full_memory"]

    for seed in args.seeds:
        print(f"\n  ── Seed {seed} ──────────────────────────────────────────────")
        t_seed = time.time()
        sr = run_seed(seed, fixtures, args.timeout, variants)
        seed_results.append(sr)
        elapsed = time.time() - t_seed

        # Print per-seed variant rates
        for v, rate in sr["variant_rates"].items():
            print(f"    {v:<26s}  pass_rate={rate:.3f}")
        print(f"    retrieval_acc={sr['metrics']['procedure_retrieval_accuracy']:.3f}  "
              f"reuse_gain={sr['metrics']['procedure_reuse_gain']:.3f}  "
              f"time={elapsed:.1f}s")

        # Gate summary for this seed
        n_pass = sum(1 for g in sr["gates"].values() if g)
        print(f"    Gates: {n_pass}/{len(GATE_NAMES)} pass")
        for gate, passed in sr["gates"].items():
            sym = "[✓]" if passed else "[✗]"
            print(f"      {sym} {gate}")

    total_elapsed = time.time() - t_start
    agg = aggregate_seeds(seed_results)
    v_dict = verdict(seed_results)

    print("\n" + "─" * 72)
    print("  AGGREGATE METRICS (mean across seeds)")
    print("─" * 72)
    for k, v in agg["metrics"].items():
        print(f"  {k:<42s}  {v:.4f}")

    print("\n" + "─" * 72)
    print("  VARIANT PASS RATES (mean across seeds)")
    print("─" * 72)
    for v, rate in agg["variant_rates"].items():
        print(f"  {v:<26s}  {rate:.3f}")

    print("\n" + "─" * 72)
    print("  SUCCESS GATES (seeds passing / total seeds)")
    print("─" * 72)
    n_fully = sum(1 for v in agg["gate_pass_rates"].values() if v.split("/")[0] == v.split("/")[1])
    for gate, frac in agg["gate_pass_rates"].items():
        passed = frac.split("/")[0] == frac.split("/")[1]
        sym = "[✓]" if passed else "[✗]"
        print(f"  {sym} {gate:<40s}  {frac}")

    print("\n" + "─" * 72)
    print(f"  VERDICT: TAC-PSM-006B {v_dict}")
    print(f"  ({n_fully}/{len(GATE_NAMES)} gates pass on all seeds)")
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print("=" * 72 + "\n")

    if args.save:
        with open(args.save, "w") as f:
            json.dump({
                "seed_results": [
                    {k: v for k, v in sr.items() if k != "confusion"}
                    for sr in seed_results
                ],
                "aggregate": agg,
                "verdict":   v_dict,
            }, f, indent=2)
        print(f"  Results saved to {args.save}")

    sys.exit(0 if v_dict == "VALIDATES" else 1)


if __name__ == "__main__":
    main()
