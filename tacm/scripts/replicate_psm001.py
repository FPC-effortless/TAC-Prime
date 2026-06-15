"""
TAC-PSM-001 Replication Runner

Runs the full benchmark across multiple independent seeds, saves per-seed
results, and produces a replication summary report.

Usage:
  python scripts/replicate_psm001.py
  python scripts/replicate_psm001.py --seeds 42 123 456 789 1337
  python scripts/replicate_psm001.py --output_dir ./reports/replication
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev, median
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_tac_psm001 import (
    run_one_seed,
    aggregate,
    evaluate_gates,
    GATES,
    SeedResult,
)


DEFAULT_SEEDS = [0, 1, 2, 3, 4, 10, 20, 42, 100, 200]


def run_replication(
    seeds:      List[int],
    output_dir: str,
    verbose:    bool = False,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"TAC-PSM-001 Replication Runner")
    print(f"Seeds: {seeds}  (n={len(seeds)})")
    print(f"Output: {out}")
    print(f"{'='*65}\n")

    t_start  = time.time()
    results: List[SeedResult] = []

    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] seed={seed} ...", flush=True)
        t0 = time.time()
        r  = run_one_seed(seed, verbose=verbose)
        elapsed = time.time() - t0
        results.append(r)

        # Save per-seed result
        seed_data = {
            "seed":             seed,
            "a1_success":       r.a1_success,
            "a2_reuse":         r.a2_reuse,
            "b1_transfer":      r.b1_transfer,
            "c1_transfer":      r.c1_transfer,
            "d1_retry":         r.d1_retry,
            "d1_pre_retry":     r.d1_pre_retry,
            "retrieval_accuracy": r.retrieval_accuracy,
            "family_match_rate": r.family_match_rate,
            "final_survival":   r.final_survival,
            "store_size_end":   r.store_size_end,
            "results_by_mode":  r.results_by_mode,
            "elapsed_s":        elapsed,
        }
        with open(out / f"seed_{seed:06d}.json", "w") as f:
            json.dump(seed_data, f, indent=2)

        print(f"  → A1={r.a1_success}  A2={r.a2_reuse}  "
              f"B1={r.b1_transfer}  C1={r.c1_transfer}  "
              f"D1={r.d1_retry}  ({elapsed:.1f}s)")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    agg   = aggregate(results)
    gates = evaluate_gates(agg)

    # ── Replication summary ────────────────────────────────────────────────────
    all_gate_pass = all(gates.values())
    n_pass = sum(gates.values())
    n_fail = len(gates) - n_pass

    summary = {
        "experiment":     "TAC-PSM-001",
        "seeds":          seeds,
        "n_seeds":        len(seeds),
        "elapsed_total_s": time.time() - t_start,
        "aggregate":      agg,
        "gates":          gates,
        "all_gates_pass": all_gate_pass,
        "n_gates_pass":   n_pass,
        "n_gates_fail":   n_fail,
        "replication_verdict": "REPLICATED" if all_gate_pass else "PARTIAL / FAILED",
    }

    with open(out / "replication_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Print report ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("REPLICATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Seeds evaluated:   {len(seeds)}")
    print(f"  Elapsed:           {summary['elapsed_total_s']:.1f}s")
    print()

    print(f"  {'Metric':<30} {'Mean':>8} {'Std':>8} {'Median':>8} {'95%CI':>8}")
    print(f"  {'-'*62}")
    key_metrics = [
        "a1_success", "a2_reuse", "b1_transfer", "c1_transfer",
        "d1_retry", "retrieval_accuracy", "reuse_gain",
        "retry_improvement", "reset_deficit", "transfer_gain", "final_survival"
    ]
    for k in key_metrics:
        v = agg.get(k, {})
        if isinstance(v, dict):
            vals = []   # reconstruct raw values for median
            for r in results:
                attr = getattr(r, k, None)
                if attr is not None:
                    vals.append(float(attr))
            med = median(vals) if vals else float("nan")
            print(f"  {k:<30} {v['mean']:>8.4f} {v['std']:>8.4f} {med:>8.4f} {v['ci95']:>8.4f}")

    print(f"\n  {'Gate':<45} {'Result':>8}")
    print(f"  {'-'*55}")
    for gname, gpass in gates.items():
        sym = "PASS ✓" if gpass else "FAIL ✗"
        print(f"  {gname:<45} {sym:>8}")

    print(f"\n  Verdict: {summary['replication_verdict']}")
    print(f"  Gates:   {n_pass}/{len(gates)} passed")
    print(f"\n  Summary saved → {out / 'replication_summary.json'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="TAC-PSM-001 Replication Runner")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=DEFAULT_SEEDS,
                        help="Seed list for replication runs")
    parser.add_argument("--n_seeds", type=int, default=None,
                        help="Use first N seeds from default list")
    parser.add_argument("--output_dir", type=str,
                        default="./reports/replication")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    seeds = args.seeds
    if args.n_seeds:
        seeds = DEFAULT_SEEDS[:args.n_seeds]

    run_replication(seeds=seeds, output_dir=args.output_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
