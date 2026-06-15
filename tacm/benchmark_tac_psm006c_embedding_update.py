"""
TAC-PSM-006C: Single-Seed Embedding Update Benchmark
=====================================================

Runs the PSM-006C ablation on 5 variants for a single seed.

  python benchmark_tac_psm006c_embedding_update.py [--seed 0] [--quick] [--out reports/]

Output
------
  reports/psm006c_seed<N>_results.json
  reports/psm006c_seed<N>_summary.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, os.path.dirname(__file__))

from tacm.psm006b.fixture_builder import build_all_fixtures
from tacm.psm006b.fixture_schema import FAMILY_NAMES
from tacm.psm006c.baselines import (
    VARIANT_NAMES_006C,
    run_all_baselines_006c,
)
from tacm.psm006c.metrics import (
    compute_metrics_006c,
    evaluate_success_gates_006c,
    compute_family_confusion_matrix_006c,
    classify_failures_006c,
)


def _pass_rate(traces) -> float:
    if not traces:
        return 0.0
    return mean(1.0 if t.pytest_pass else 0.0 for t in traces)


def run_single_seed(
    seed:      int,
    fixtures,
    timeout_s: float = 10.0,
) -> dict:
    t0 = time.time()
    print(f"\n[PSM-006C] Seed {seed}: {len(fixtures)} fixtures × 5 variants ...", flush=True)

    results = run_all_baselines_006c(fixtures, seed=seed, timeout_s=timeout_s)

    metrics  = compute_metrics_006c(results)
    gates    = evaluate_success_gates_006c(metrics, results)
    failures = classify_failures_006c(results.get("full_memory_embedding_update", []))
    confusion = compute_family_confusion_matrix_006c(
        results.get("full_memory_embedding_update", []), FAMILY_NAMES
    )

    variant_rates = {v: _pass_rate(results.get(v, [])) for v in VARIANT_NAMES_006C}
    elapsed = time.time() - t0

    # Print summary
    print(f"\n[PSM-006C] Seed {seed} results ({elapsed:.1f}s):")
    print(f"  Variant pass rates:")
    for v, r in variant_rates.items():
        marker = " ← NEW" if v == "full_memory_embedding_update" else ""
        print(f"    {v:<36s}  {r:.3f}{marker}")
    print(f"\n  Key metrics:")
    for k in ["retry_after_update_success", "procedure_retrieval_accuracy",
              "retrieval_changed_after_update", "family_changed_after_update",
              "successful_retrieval_recovery", "embedding_update_count",
              "embedding_shift_norm_mean", "procedure_reuse_gain"]:
        print(f"    {k:<44s}  {metrics.get(k, 0.0):.4f}")
    print(f"\n  Gates:")
    for g, v in gates.items():
        sym = "PASS" if v else "FAIL"
        print(f"    [{sym}] {g}")

    return {
        "seed":          seed,
        "metrics":       metrics,
        "gates":         gates,
        "variant_rates": variant_rates,
        "failures":      failures,
        "confusion":     {k: dict(v) for k, v in confusion.items()},
        "n_fixtures":    len(fixtures),
        "elapsed_s":     elapsed,
    }


def write_seed_json(result: dict, path: str) -> None:
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")


def write_seed_summary(result: dict, path: str) -> None:
    seed = result["seed"]
    elapsed = result["elapsed_s"]
    gates = result["gates"]
    n_pass = sum(1 for v in gates.values() if v)
    lines = [
        "TAC-PSM-006C Single-Seed Benchmark",
        "=" * 60,
        f"Seed: {seed}",
        f"Fixtures: {result['n_fixtures']}",
        f"Elapsed: {elapsed:.1f}s",
        f"Gates passed: {n_pass}/{len(gates)}",
        "",
        "Variant Pass Rates:",
    ]
    for v, r in result["variant_rates"].items():
        lines.append(f"  {v:<36s}  {r:.3f}")
    lines += ["", "Key Metrics:"]
    for k, v in result["metrics"].items():
        lines.append(f"  {k:<44s}  {v:.4f}")
    lines += ["", "Gate Results:"]
    for g, v in gates.items():
        sym = "PASS" if v else "FAIL"
        lines.append(f"  [{sym}] {g}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="PSM-006C single-seed embedding update benchmark"
    )
    parser.add_argument("--seed",    type=int,   default=0)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out",     type=str,   default="reports")
    parser.add_argument("--quick",   action="store_true",
                        help="Run only 12 fixtures (smoke test)")
    args = parser.parse_args()

    all_fixtures = build_all_fixtures()
    if args.quick:
        fixtures = []
        for fam in FAMILY_NAMES[:3]:
            fxs = [fx for fx in all_fixtures if fx.family == fam]
            fixtures.extend(fxs[:4])
    else:
        fixtures = all_fixtures

    result = run_single_seed(args.seed, fixtures, timeout_s=args.timeout)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_seed_json(result, str(out_dir / f"psm006c_seed{args.seed}_results.json"))
    write_seed_summary(result, str(out_dir / f"psm006c_seed{args.seed}_summary.txt"))
    print(f"\n[PSM-006C] Results saved to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
