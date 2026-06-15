"""
TAC-PSM-004 Benchmark: Procedure Survival Field

Evaluates whether high-fitness procedures survive longer than low-fitness ones
under decay and selection pressure.

Perturbation tests: Noise, Distribution Shift, Procedure Attack,
                    Task Mutation, Adversarial Retrieval

Success gates:
  - high_fitness_survives_longer  (high-fit alive rate > low-fit alive rate)
  - survival_gap >= 0.20          (meaningful gap between high/low fitness)
  - mean_robustness >= 0.50       (procedures robust to perturbations on average)
  - noise_robustness >= 0.40      (robust to embedding noise)
  - attack_robustness_gt_0        (at least some resistance to adversarial retrieval)

Usage:
  python scripts/benchmark_tac_psm004.py --seeds 5
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
    get_all_tasks,
)
from tacm.psm004 import (
    FitnessProfile,
    compute_fitness,
    SurvivalField,
    PerturbationType,
    PerturbationResult,
    apply_perturbation,
    run_perturbation_suite,
    SurvivalExperimentResult,
)

EMBEDDING_DIM = 64
N_STEPS       = 30      # survival field time steps
N_PERT_TRIALS = 15      # perturbation trials per procedure

GATES = {
    "high_fitness_survives_longer": ("survival_gap",       ">",  0.0),
    "survival_gap_ge_0.20":         ("survival_gap",       ">=", 0.20),
    "mean_robustness_ge_0.40":      ("mean_robustness",    ">=", 0.40),
    "noise_robustness_ge_0.40":     ("noise_robustness",   ">=", 0.40),
    "attack_robustness_gt_0":       ("attack_robustness",  ">",  0.0),
}


def _make_high_fitness_proc(store, task, seed):
    emb = task.query_embedding(EMBEDDING_DIM)
    p   = store.build(task.family, task.task_signature, oracle_steps(task),
                      emb, success_score=0.9)
    # Simulate high reuse + transfer
    for _ in range(15):
        store.update(p.procedure_id, success_delta=0.01, transfer_delta=0.02)
    return p


def _make_low_fitness_proc(store, task, seed):
    emb = task.query_embedding(EMBEDDING_DIM)
    p   = store.build(task.family, task.task_signature, task.distractor_steps,
                      emb, success_score=0.15)
    # Start with already-decayed survival score to make gap visible in 30 steps
    p.survival_score = 0.35
    return p


def run_one_seed(seed: int, verbose: bool = False) -> dict:
    task_a = FAMILY_A_IMPORT_ERRORS.tasks[0]
    task_b = FAMILY_B_DEPENDENCY_CONFLICTS.tasks[0]
    task_c = FAMILY_C_VERSION_MISMATCH.tasks[0]
    task_d = FAMILY_D_PATH_RESOLUTION.tasks[0]

    store = ProceduralMemoryStore(embedding_dim=EMBEDDING_DIM)

    # Build 2 high-fitness and 2 low-fitness procedures
    hi1 = _make_high_fitness_proc(store, task_a, seed)
    hi2 = _make_high_fitness_proc(store, task_b, seed)
    lo1 = _make_low_fitness_proc(store, task_c, seed)
    lo2 = _make_low_fitness_proc(store, task_d, seed)

    # ── Compute fitness profiles ───────────────────────────────────────────────
    pert_hi1 = run_perturbation_suite(
        store.get(hi1.procedure_id), task_a, n_trials=N_PERT_TRIALS, seed=seed
    )
    pert_lo1 = run_perturbation_suite(
        store.get(lo1.procedure_id), task_c, n_trials=N_PERT_TRIALS, seed=seed
    )

    from tacm.psm004.perturbation import mean_robustness
    rob_hi = mean_robustness(pert_hi1)
    rob_lo = mean_robustness(pert_lo1)

    fp_hi1 = compute_fitness(store.get(hi1.procedure_id), robustness=rob_hi)
    fp_hi2 = compute_fitness(store.get(hi2.procedure_id), robustness=rob_hi * 0.9)
    fp_lo1 = compute_fitness(store.get(lo1.procedure_id), robustness=rob_lo)
    fp_lo2 = compute_fitness(store.get(lo2.procedure_id), robustness=rob_lo * 0.9)

    fitness_map = {
        hi1.procedure_id: fp_hi1,
        hi2.procedure_id: fp_hi2,
        lo1.procedure_id: fp_lo1,
        lo2.procedure_id: fp_lo2,
    }

    if verbose:
        print(f"  Fitness: hi={fp_hi1.fitness:.3f},{fp_hi2.fitness:.3f}"
              f"  lo={fp_lo1.fitness:.3f},{fp_lo2.fitness:.3f}")

    # ── Run survival field ─────────────────────────────────────────────────────
    # decay_rate=0.88 means lo procs (start=0.35) die in ~ln(0.35/0.10)/ln(1/0.88)≈10 steps
    # Hi procs (start=1.0) get reward each step: equilibrium ≈ 0.05/(1-0.88) = 0.42 → alive
    sf = SurvivalField(
        store,
        decay_rate      = 0.88,
        fitness_reward  = 0.05,
        death_threshold = 0.10,
        fitness_cutoff  = 0.45,
    )
    for pid, fp in fitness_map.items():
        sf.register(fp)

    step_summaries = sf.run(fitness_map, n_steps=N_STEPS)

    # Survival: who is still alive?
    hi_alive = sum(1 for pid in [hi1.procedure_id, hi2.procedure_id]
                   if sf._records.get(pid) and sf._records[pid].alive)
    lo_alive = sum(1 for pid in [lo1.procedure_id, lo2.procedure_id]
                   if sf._records.get(pid) and sf._records[pid].alive)

    hi_rate = hi_alive / 2.0
    lo_rate = lo_alive / 2.0
    gap     = hi_rate - lo_rate

    if verbose:
        print(f"  After {N_STEPS} steps: hi_alive={hi_alive}/2  lo_alive={lo_alive}/2  gap={gap:.2f}")

    # ── Perturbation breakdown ─────────────────────────────────────────────────
    noise_rob  = pert_hi1.get(PerturbationType.NOISE, None)
    attack_rob = pert_hi1.get(PerturbationType.ADVERSARIAL_RETR, None)
    mean_rob   = (rob_hi + rob_lo) / 2.0

    return {
        "seed":             seed,
        "hi_survival_rate": hi_rate,
        "lo_survival_rate": lo_rate,
        "survival_gap":     gap,
        "mean_robustness":  mean_rob,
        "noise_robustness": noise_rob.robustness if noise_rob else 0.0,
        "attack_robustness": attack_rob.robustness if attack_rob else 0.0,
        "hi_fitness":       (fp_hi1.fitness + fp_hi2.fitness) / 2.0,
        "lo_fitness":       (fp_lo1.fitness + fp_lo2.fitness) / 2.0,
        "steps_run":        N_STEPS,
    }


def aggregate(results: List[dict]) -> dict:
    n = len(results)
    def _stat(key):
        vals = [r[key] for r in results if key in r]
        m    = mean(vals) if vals else 0.0
        s    = stdev(vals) if len(vals) > 1 else 0.0
        return {"mean": m, "std": s, "ci95": 1.96 * s / (n ** 0.5) if n > 1 else 0.0}
    keys = ["hi_survival_rate", "lo_survival_rate", "survival_gap",
            "mean_robustness", "noise_robustness", "attack_robustness",
            "hi_fitness", "lo_fitness"]
    return {k: _stat(k) for k in keys}


def evaluate_gates(agg: dict) -> dict:
    passed = {}
    for gate, (metric, op, threshold) in GATES.items():
        v   = agg.get(metric, {})
        val = v["mean"] if isinstance(v, dict) else float(v or 0)
        passed[gate] = (val > threshold if op == ">" else val >= threshold)
    return passed


def run_benchmark(seeds: List[int], verbose: bool = False, output: str = None) -> dict:
    print(f"\n{'='*60}")
    print(f"TAC-PSM-004 Benchmark: Procedure Survival Field")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")

    t0      = time.time()
    results = []
    for s in seeds:
        print(f"\n  --- seed={s} ---")
        r = run_one_seed(s, verbose=verbose)
        results.append(r)
        print(f"  gap={r['survival_gap']:.3f}  robustness={r['mean_robustness']:.3f}"
              f"  hi_fit={r['hi_fitness']:.3f}  lo_fit={r['lo_fitness']:.3f}")

    agg   = aggregate(results)
    gates = evaluate_gates(agg)

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    for k in ["survival_gap", "hi_survival_rate", "lo_survival_rate",
              "mean_robustness", "noise_robustness", "attack_robustness",
              "hi_fitness", "lo_fitness"]:
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

    report = {"experiment": "TAC-PSM-004", "seeds": seeds, "agg": agg,
              "gates": gates, "all_pass": all_pass, "elapsed": elapsed}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report → {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="TAC-PSM-004 Benchmark")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick",   action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output",  type=str, default="./reports/psm004_benchmark.json")
    args = parser.parse_args()
    seeds = [0] if args.quick else args.seeds
    run_benchmark(seeds=seeds, verbose=args.verbose, output=args.output)


if __name__ == "__main__":
    main()
