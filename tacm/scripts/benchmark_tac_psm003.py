"""
TAC-PSM-003 Benchmark: Procedure Lifecycle

Evaluates whether procedures can evolve over time through:
  - Strengthening (reuse → higher success score)
  - Specialization (sub-type child emerges)
  - Merging (two co-used procedures combine into a better one)
  - Splitting (one procedure forks into two specialised ones)
  - Retirement (low-fitness procedures decay and are removed)

Success gates:
  - merge_beats_best_parent  (merged quality > max(parent quality))
  - specialization_gain >= 0 (child >= parent)
  - children_beat_parent     (split children combined >= parent)
  - retirement_accuracy >= 0.5 (low-fitness procs actually get retired)
  - lifecycle_stability      (strengthening monotonically increases score)

Usage:
  python scripts/benchmark_tac_psm003.py --seeds 5
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
    ProcedureStep,
    ProcedureTrace,
    oracle_steps,
    FAMILY_A_IMPORT_ERRORS,
    FAMILY_B_DEPENDENCY_CONFLICTS,
    FAMILY_C_VERSION_MISMATCH,
    FAMILY_D_PATH_RESOLUTION,
)
from tacm.psm003 import (
    LifecycleEngine,
    merge_procedures,
    split_procedure,
    specialize_procedure,
    retire_procedure,
)

EMBEDDING_DIM = 64

GATES = {
    "merge_quality_gain_gt_0":     ("merge_quality_gain",       ">",  0.0),
    "specialization_non_negative": ("spec_score_gain",          ">=", 0.0),
    "split_children_viable":       ("split_children_viable",    ">",  0.0),
    "retirement_works":            ("retirement_accuracy",      ">=", 0.5),
    "strengthening_monotone":      ("strengthen_monotone_rate", ">=", 0.8),
}


def _build_proc(store, task, steps=None, success=0.8, seed=0) -> ProcedureTrace:
    emb = task.query_embedding(EMBEDDING_DIM)
    s   = steps or oracle_steps(task)
    return store.build(
        problem_family = task.family,
        task_signature = task.task_signature,
        steps          = s,
        embedding      = emb,
        success_score  = success,
    )


def run_one_seed(seed: int, verbose: bool = False) -> dict:
    task_a  = FAMILY_A_IMPORT_ERRORS.tasks[0]
    task_a2 = FAMILY_A_IMPORT_ERRORS.tasks[1]
    task_b  = FAMILY_B_DEPENDENCY_CONFLICTS.tasks[0]
    task_c  = FAMILY_C_VERSION_MISMATCH.tasks[0]
    task_d  = FAMILY_D_PATH_RESOLUTION.tasks[0]

    # ── Strengthen ────────────────────────────────────────────────────────────
    store  = ProceduralMemoryStore(embedding_dim=EMBEDDING_DIM)
    proc_s = _build_proc(store, task_a, success=0.5, seed=seed)
    engine = LifecycleEngine(store, strengthen_threshold=0.65)
    scores_before = [store.get(proc_s.procedure_id).success_score]
    for i in range(10):
        engine.apply_strengthening(proc_s.procedure_id, delta=0.05)
        scores_before.append(store.get(proc_s.procedure_id).success_score)
    # Monotone check: each score >= previous
    diffs    = [scores_before[i+1] - scores_before[i] for i in range(len(scores_before)-1)]
    monotone = sum(1 for d in diffs if d >= -1e-6) / max(len(diffs), 1)
    if verbose:
        print(f"  Strengthen: scores {scores_before[0]:.3f}→{scores_before[-1]:.3f} monotone={monotone:.2f}")

    # ── Specialize ────────────────────────────────────────────────────────────
    store2 = ProceduralMemoryStore(embedding_dim=EMBEDDING_DIM)
    proc_p = _build_proc(store2, task_a, success=0.7, seed=seed)
    extra_steps = [f"Verify {task_a2.sub_type} pattern", f"Apply {task_a2.sub_type} fix"]
    spec_result, spec_metrics = specialize_procedure(
        store2, proc_p.procedure_id, task_a2.sub_type, extra_steps, task=task_a2, seed=seed
    )
    spec_gain = spec_metrics.get("score_gain", 0.0)
    if verbose:
        print(f"  Specialize: gain={spec_gain:.3f}  child_beats_parent={spec_metrics.get('child_beats_parent')}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    store3  = ProceduralMemoryStore(embedding_dim=EMBEDDING_DIM)
    proc_a  = _build_proc(store3, task_a, success=0.7, seed=seed)
    proc_b  = _build_proc(store3, task_b,
                          steps=oracle_steps(task_a)[:3] + oracle_steps(task_b)[-2:],
                          success=0.65, seed=seed)
    merge_result, merge_metrics = merge_procedures(
        store3, proc_a.procedure_id, proc_b.procedure_id,
        task_a=task_a, task_b=task_b, seed=seed
    )
    merge_beats = merge_metrics.get("merged_beats_best_parent", False)
    if verbose:
        print(f"  Merge: beats_parent={merge_beats}  gain={merge_metrics.get('quality_gain', 0):.3f}")

    # ── Split ─────────────────────────────────────────────────────────────────
    store4   = ProceduralMemoryStore(embedding_dim=EMBEDDING_DIM)
    combined = oracle_steps(task_a) + oracle_steps(task_d)
    proc_big = _build_proc(store4, task_a, steps=combined, success=0.6, seed=seed)
    split_pt  = len(oracle_steps(task_a))
    split_result, split_metrics = split_procedure(
        store4, proc_big.procedure_id, split_point=split_pt,
        task_a=task_a, task_b=task_d, seed=seed,
    )
    split_viable = float(split_metrics.get("children_beat_parent", False))
    if verbose:
        print(f"  Split: children_beat_parent={split_viable}  after_a={split_metrics.get('after_a',0):.3f}")

    # ── Retire ────────────────────────────────────────────────────────────────
    store5  = ProceduralMemoryStore(embedding_dim=EMBEDDING_DIM)
    # Build one high-fitness and one low-fitness procedure
    proc_hi = _build_proc(store5, task_a, success=0.9, seed=seed)
    proc_lo = _build_proc(store5, task_c, success=0.2, seed=seed)
    store5.update(proc_hi.procedure_id, survival_delta=0.0)   # keep alive
    _, ret_metrics = retire_procedure(store5, proc_lo.procedure_id, decay_rounds=30, rate=0.5)
    retirement_acc = float(ret_metrics.get("target_retired", False))
    if verbose:
        print(f"  Retire: target_retired={retirement_acc}  n_retired={ret_metrics.get('n_retired',0)}")

    return {
        "seed":                    seed,
        "strengthen_monotone_rate": monotone,
        "spec_score_gain":         spec_gain,
        "spec_child_ok":           float(spec_metrics.get("child_beats_parent", False)),
        "merge_beats_best_parent": float(merge_beats),
        "merge_quality_gain":      merge_metrics.get("quality_gain", 0.0),
        "split_children_viable":   split_viable,
        "retirement_accuracy":     retirement_acc,
    }


def aggregate(results: List[dict]) -> dict:
    n = len(results)
    def _stat(key):
        vals = [r[key] for r in results if key in r]
        m    = mean(vals) if vals else 0.0
        s    = stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": m, "std": s, "ci95": 1.96 * s / (n ** 0.5) if n > 1 else 0.0}
    keys = ["strengthen_monotone_rate", "spec_score_gain", "spec_child_ok",
            "merge_beats_best_parent", "merge_quality_gain",
            "split_children_viable", "retirement_accuracy"]
    return {k: _stat(k) for k in keys}


def evaluate_gates(agg: dict) -> dict:
    passed = {}
    for gate, (metric, op, threshold) in GATES.items():
        v   = agg.get(metric, {})
        val = v["mean"] if isinstance(v, dict) else float(v or 0)
        if op == ">":
            passed[gate] = val > threshold
        elif op == ">=":
            passed[gate] = val >= threshold
        else:
            passed[gate] = False
    return passed


def run_benchmark(seeds: List[int], verbose: bool = False, output: str = None) -> dict:
    print(f"\n{'='*60}")
    print(f"TAC-PSM-003 Benchmark: Procedure Lifecycle")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")

    t0      = time.time()
    results = []
    for s in seeds:
        print(f"\n  --- seed={s} ---")
        r = run_one_seed(s, verbose=verbose)
        results.append(r)
        print(f"  strengthen={r['strengthen_monotone_rate']:.2f}  "
              f"merge_gain={r['merge_quality_gain']:.3f}  "
              f"split={r['split_children_viable']:.0f}  "
              f"retire={r['retirement_accuracy']:.0f}")

    agg   = aggregate(results)
    gates = evaluate_gates(agg)

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    for k, v in agg.items():
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

    report = {"experiment": "TAC-PSM-003", "seeds": seeds, "agg": agg,
              "gates": gates, "all_pass": all_pass, "elapsed": elapsed}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report → {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="TAC-PSM-003 Benchmark")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick",   action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output",  type=str, default="./reports/psm003_benchmark.json")
    args = parser.parse_args()
    seeds = [0] if args.quick else args.seeds
    run_benchmark(seeds=seeds, verbose=args.verbose, output=args.output)


if __name__ == "__main__":
    main()
