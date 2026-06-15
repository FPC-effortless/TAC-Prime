"""
TAC-PSM-002 Benchmark: Procedural Transfer

Evaluates whether procedures learned in family A transfer to families B and C.

Transfer chains: A→B, A→C, A→B→C

Controls: Fresh Learning, Random, Reset, Wrong Procedure, Oracle

Success gates:
  - transfer_gain > 0       (adapted > reset)
  - outperforms fresh       (adapted > fresh)
  - outperforms random      (adapted > random)
  - chain retention > 0.50  (A→B→C doesn't collapse to zero)

Usage:
  python scripts/benchmark_tac_psm002.py --seeds 5
  python scripts/benchmark_tac_psm002.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tacm.psm001 import (
    ProceduralMemoryStore,
    ProcedureTrace,
    ProcedureStep,
    oracle_steps,
    get_all_tasks,
    FAMILY_A_IMPORT_ERRORS,
    FAMILY_B_DEPENDENCY_CONFLICTS,
    FAMILY_C_VERSION_MISMATCH,
    FAMILY_D_PATH_RESOLUTION,
)
from tacm.psm002 import (
    TransferMode,
    TransferResult,
    TransferChainResult,
    run_transfer,
    run_transfer_chain,
    compute_transfer_metrics,
)

EMBEDDING_DIM = 64

GATES = {
    "transfer_gain_gt_0":          ("transfer_gain",       ">",  0.0),
    "outperforms_fresh_learning":  ("gain_vs_fresh",       ">",  0.0),
    "outperforms_random":          ("gain_vs_random",      ">",  0.0),
    "outperforms_reset":           ("gain_vs_reset",       ">",  0.05),
    "chain_retention_ge_0.50":     ("chain_retention",     ">=", 0.50),
}


def _make_source_proc(family_tasks, family_name: str, seed: int) -> ProcedureTrace:
    """Build a source procedure for a family using oracle steps."""
    task   = family_tasks[0]
    steps  = oracle_steps(task)
    emb    = task.query_embedding(EMBEDDING_DIM)
    return ProcedureTrace(
        procedure_id   = f"source-{family_name[:3]}-{seed}",
        problem_family = family_name,
        task_signature = task.task_signature,
        steps          = [ProcedureStep(i, s) for i, s in enumerate(steps)],
        success_score  = 0.9,
        transfer_score = 0.3,
        survival_score = 1.0,
        embedding      = emb.tolist(),
    )


def run_one_seed(seed: int, verbose: bool = False) -> Dict[str, Any]:
    store = ProceduralMemoryStore(embedding_dim=EMBEDDING_DIM)

    task_a = FAMILY_A_IMPORT_ERRORS.tasks[0]
    task_b = FAMILY_B_DEPENDENCY_CONFLICTS.tasks[0]
    task_c = FAMILY_C_VERSION_MISMATCH.tasks[0]
    task_d = FAMILY_D_PATH_RESOLUTION.tasks[0]

    source_a = _make_source_proc(FAMILY_A_IMPORT_ERRORS.tasks,    "ImportErrors", seed)
    source_b = _make_source_proc(FAMILY_B_DEPENDENCY_CONFLICTS.tasks, "DependencyConflicts", seed)

    # ── A→B transfer ──────────────────────────────────────────────────────────
    results_ab: Dict[TransferMode, List[TransferResult]] = {}
    for mode in [TransferMode.ADAPTED, TransferMode.FRESH, TransferMode.RANDOM,
                 TransferMode.RESET, TransferMode.ORACLE, TransferMode.DIRECT]:
        r = run_transfer(source_a, task_b, mode, seed=seed, store=store)
        results_ab[mode] = [r]
        if verbose:
            print(f"  A→B [{mode.value}] success={r.success} quality={r.quality:.3f} cost={r.adaptation_cost:.2f}")

    # ── A→C transfer ──────────────────────────────────────────────────────────
    results_ac: Dict[TransferMode, List[TransferResult]] = {}
    for mode in [TransferMode.ADAPTED, TransferMode.FRESH, TransferMode.RANDOM,
                 TransferMode.RESET, TransferMode.ORACLE]:
        r = run_transfer(source_a, task_c, mode, seed=seed, store=store)
        results_ac[mode] = [r]

    # ── A→B→C chain transfer ──────────────────────────────────────────────────
    procedures = {
        "ImportErrors":        source_a,
        "DependencyConflicts": source_b,
    }
    chain_abc = run_transfer_chain(
        procedures  = procedures,
        task_chain  = [task_a, task_b, task_c],
        mode        = TransferMode.ADAPTED,
        seed        = seed,
        store       = store,
    )
    chain_reset = run_transfer_chain(
        procedures  = procedures,
        task_chain  = [task_a, task_b, task_c],
        mode        = TransferMode.RESET,
        seed        = seed,
    )

    if verbose:
        print(f"  Chain A→B→C: quality={chain_abc.chain_quality:.3f} retention={chain_abc.retention:.3f}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Pool A→B and A→C adapted results for overall transfer metrics
    adapted_all = results_ab[TransferMode.ADAPTED] + results_ac[TransferMode.ADAPTED]
    controls = {
        TransferMode.FRESH:  results_ab.get(TransferMode.FRESH, []) + results_ac.get(TransferMode.FRESH, []),
        TransferMode.RANDOM: results_ab.get(TransferMode.RANDOM, []) + results_ac.get(TransferMode.RANDOM, []),
        TransferMode.RESET:  results_ab.get(TransferMode.RESET, []) + results_ac.get(TransferMode.RESET, []),
        TransferMode.ORACLE: results_ab.get(TransferMode.ORACLE, []) + results_ac.get(TransferMode.ORACLE, []),
    }

    metrics = compute_transfer_metrics(adapted_all, controls, [chain_abc])

    return {
        "seed":              seed,
        "transfer_success":  metrics.transfer_success,
        "transfer_gain":     metrics.transfer_gain,
        "transfer_retention": metrics.transfer_retention,
        "adaptation_cost":   metrics.adaptation_cost,
        "transfer_efficiency": metrics.transfer_efficiency,
        "gain_vs_fresh":     metrics.gain_vs_fresh,
        "gain_vs_random":    metrics.gain_vs_random,
        "gain_vs_reset":     metrics.gain_vs_reset,
        "mode_success":      metrics.mode_success,
        "chain_ab_quality":  chain_abc.chain_quality,
        "chain_retention":   chain_abc.retention,
        "chain_reset_quality": chain_reset.chain_quality,
        "transfer_gain_over_chain_reset": chain_abc.chain_quality - chain_reset.chain_quality,
    }


def aggregate(results: List[dict]) -> dict:
    n = len(results)
    def _stat(key: str) -> dict:
        vals = [r[key] for r in results if key in r]
        m    = mean(vals) if vals else 0.0
        s    = stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": m, "std": s, "ci95": 1.96 * s / (n ** 0.5) if n > 1 else 0.0}
    keys = ["transfer_success", "transfer_gain", "transfer_retention", "adaptation_cost",
            "transfer_efficiency", "gain_vs_fresh", "gain_vs_random", "gain_vs_reset",
            "chain_ab_quality", "chain_retention", "chain_reset_quality",
            "transfer_gain_over_chain_reset"]
    return {k: _stat(k) for k in keys}


def evaluate_gates(agg: dict) -> dict:
    passed = {}
    for gate, (metric, op, threshold) in GATES.items():
        v = agg.get(metric, {})
        val = v["mean"] if isinstance(v, dict) else v
        if op == ">":
            passed[gate] = val > threshold
        elif op == ">=":
            passed[gate] = val >= threshold
        else:
            passed[gate] = False
    return passed


def run_benchmark(seeds: List[int], verbose: bool = False, output: str = None) -> dict:
    print(f"\n{'='*60}")
    print(f"TAC-PSM-002 Benchmark: Procedural Transfer")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")

    t0      = time.time()
    results = []
    for s in seeds:
        print(f"\n  --- seed={s} ---")
        r = run_one_seed(s, verbose=verbose)
        results.append(r)
        print(f"  transfer={r['transfer_success']:.3f}  gain={r['transfer_gain']:.3f}"
              f"  chain={r['chain_ab_quality']:.3f}  ret={r['chain_retention']:.3f}")

    agg   = aggregate(results)
    gates = evaluate_gates(agg)

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    for k in ["transfer_success", "transfer_gain", "gain_vs_fresh", "gain_vs_random",
              "gain_vs_reset", "chain_retention", "adaptation_cost"]:
        v = agg[k]
        print(f"  {k:<32} {v['mean']:.4f} ± {v['std']:.4f}")

    print(f"\nSUCCESS GATES")
    all_pass = True
    for g, p in gates.items():
        sym = "✓" if p else "✗"
        print(f"  [{sym}] {g}")
        if not p:
            all_pass = False
    elapsed = time.time() - t0
    print(f"\n{'ALL GATES PASS ✓' if all_pass else 'SOME GATES FAIL ✗'}  ({elapsed:.1f}s)")

    report = {"experiment": "TAC-PSM-002", "seeds": seeds, "agg": agg,
              "gates": gates, "all_pass": all_pass, "elapsed": elapsed}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report → {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="TAC-PSM-002 Benchmark")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick",   action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output",  type=str, default="./reports/psm002_benchmark.json")
    args = parser.parse_args()
    seeds = [0] if args.quick else args.seeds
    run_benchmark(seeds=seeds, verbose=args.verbose, output=args.output)


if __name__ == "__main__":
    main()
