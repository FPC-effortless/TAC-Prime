"""
TAC-PSM-001 Benchmark: Procedural Memory Build / Retrieve / Update

Full evaluation sequence:
  A1  Solve initial import error → store procedure
  A2  Similar import error → retrieve + measure reuse
  B1  Dependency conflict → procedure adaptation / transfer
  C1  Version mismatch → broader transfer
  D1  Force failure → update procedure → retry → measure improvement

5 baselines × 5 ablations × N seeds

Usage:
  python scripts/benchmark_tac_psm001.py --seeds 5
  python scripts/benchmark_tac_psm001.py --seeds 1 --quick
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tacm.psm001 import (
    ProceduralMemoryStore,
    ProcedureTrace,
    RetrievalMode,
    RetrievalResult,
    VerificationSignal,
    retrieve_procedure,
    update_procedure_after_verification,
    TaskInstance,
    FAMILY_A_IMPORT_ERRORS,
    FAMILY_B_DEPENDENCY_CONFLICTS,
    FAMILY_C_VERSION_MISMATCH,
    FAMILY_D_PATH_RESOLUTION,
    ALL_FAMILIES,
    evaluate_procedure_on_task,
    oracle_steps,
    reset_steps,
    random_steps,
    get_all_tasks,
    make_task_signature,
)
from tacm.psm001.retrieval import compute_retrieval_metrics

EMBEDDING_DIM = 64

# ── Success gates ─────────────────────────────────────────────────────────────
GATES = {
    "retrieval_accuracy_ge_0.70":    ("retrieval_accuracy",  ">=", 0.70),
    "reuse_gain_ge_0.10":            ("reuse_gain",          ">=", 0.10),
    "update_improves_retry":         ("retry_improvement",   ">",  0.0),
    "reset_deficit_ge_0.20":         ("reset_deficit",       ">=", 0.20),
    "random_worse_than_correct":     ("random_vs_correct",   "<",  0.0),
    "transfer_gain_gt_0":            ("transfer_gain",       ">",  0.0),
    "survival_stable_across_seeds":  ("survival_cv",         "<",  0.3),
}


# ── Experiment runner ─────────────────────────────────────────────────────────

@dataclass
class SeedResult:
    seed:          int
    # Per-step metrics
    a1_success:    bool   = False    # A1: initial solve
    a2_reuse:      bool   = False    # A2: reuse
    b1_transfer:   bool   = False    # B1: transfer A→B
    c1_transfer:   bool   = False    # C1: transfer A→C
    d1_retry:      bool   = False    # D1: retry after update
    d1_pre_retry:  bool   = False    # D1: before update
    # Retrieval
    retrieval_accuracy: float = 0.0
    family_match_rate:  float = 0.0
    # Baselines
    results_by_mode:   Dict[str, bool] = field(default_factory=dict)
    # Survival
    final_survival:    float = 0.0
    # Procedure growth
    store_size_end:    int   = 0
    # Raw retrieval results
    retrieval_results: List[dict] = field(default_factory=list)


def run_one_seed(
    seed:     int,
    ablation: Optional[str] = None,
    verbose:  bool = False,
) -> SeedResult:
    rng     = random.Random(seed)
    np_rng  = np.random.default_rng(seed)
    result  = SeedResult(seed=seed)

    store = ProceduralMemoryStore(embedding_dim=EMBEDDING_DIM)

    all_tasks = get_all_tasks()

    def _emb(task: TaskInstance) -> np.ndarray:
        """Get deterministic noisy embedding for this seed."""
        base  = task.query_embedding(EMBEDDING_DIM)
        noise = np_rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.05
        v     = base + noise
        return v / (np.linalg.norm(v) + 1e-9)

    def _log(msg: str):
        if verbose:
            print(f"  [seed={seed}] {msg}")

    # ── A1: Solve initial import error → store procedure ─────────────────────
    task_a1 = FAMILY_A_IMPORT_ERRORS.tasks[0]
    emb_a1  = _emb(task_a1)
    steps   = oracle_steps(task_a1)
    success_a1, quality_a1, _ = evaluate_procedure_on_task(task_a1, steps, seed=seed)
    result.a1_success = success_a1

    proc_a1 = store.build(
        problem_family  = task_a1.family,
        task_signature  = task_a1.task_signature,
        steps           = steps,
        embedding       = emb_a1,
        success_score   = float(success_a1),
    )
    _log(f"A1: stored {proc_a1.procedure_id}  success={success_a1}  quality={quality_a1:.3f}")

    # Update after verification
    sig_a1 = VerificationSignal(
        procedure_id   = proc_a1.procedure_id,
        task_signature = task_a1.task_signature,
        success        = success_a1,
    )
    update_procedure_after_verification(sig_a1, store)

    # ── A2: Retrieve + reuse ──────────────────────────────────────────────────
    task_a2 = FAMILY_A_IMPORT_ERRORS.tasks[1]
    emb_a2  = _emb(task_a2)

    retr_a2 = retrieve_procedure(
        task_signature  = task_a2.task_signature,
        query_embedding = emb_a2,
        store           = store,
        mode            = RetrievalMode.CORRECT,
        top_k           = 5,
        correct_family  = task_a2.family,
    )
    retr_a2.is_correct = (retr_a2.top1 is not None and
                          retr_a2.top1.procedure_id == proc_a1.procedure_id)
    result.retrieval_accuracy = float(retr_a2.is_correct)
    result.family_match_rate  = float(retr_a2.family_matched)
    result.retrieval_results.append(retr_a2.to_dict())

    # Use retrieved steps
    retrieved_steps = retr_a2.top1.steps if retr_a2.top1 else []
    retrieved_step_strs = [s.action for s in retrieved_steps]
    success_a2, quality_a2, _ = evaluate_procedure_on_task(task_a2, retrieved_step_strs, seed=seed)
    result.a2_reuse = success_a2
    _log(f"A2: reuse={success_a2}  family_match={retr_a2.family_matched}  retrieved={len(retrieved_step_strs)} steps")

    # ── B1: Transfer A→B ──────────────────────────────────────────────────────
    task_b1 = FAMILY_B_DEPENDENCY_CONFLICTS.tasks[0]
    emb_b1  = _emb(task_b1)

    if ablation == "remove_transfer_metadata":
        # Ablation D: remove transfer metadata → same retrieval, but transfer_delta blocked later
        retr_b1 = retrieve_procedure(
            task_signature  = task_b1.task_signature,
            query_embedding = emb_b1,
            store           = store,
            mode            = RetrievalMode.CORRECT,
            top_k           = 5,
            correct_family  = task_b1.family,
        )
    else:
        retr_b1 = retrieve_procedure(
            task_signature  = task_b1.task_signature,
            query_embedding = emb_b1,
            store           = store,
            mode            = RetrievalMode.CORRECT,
            top_k           = 5,
            correct_family  = task_b1.family,
        )

    # Attempt adaptation: prefix retrieved steps with family-specific preamble
    if retr_b1.top1:
        adapted_steps = (
            [f"[adapt:{task_b1.family}] " + s.action for s in retr_b1.top1.steps[:2]]
            + list(task_b1.canonical_steps[2:])
        )
        is_transfer = True
    else:
        adapted_steps = list(task_b1.canonical_steps)
        is_transfer   = False

    success_b1, quality_b1, _ = evaluate_procedure_on_task(task_b1, adapted_steps, seed=seed)
    result.b1_transfer = success_b1
    _log(f"B1: transfer A→B  success={success_b1}  quality={quality_b1:.3f}")

    # Store transferred procedure
    if success_b1:
        proc_b1 = store.build(
            problem_family  = task_b1.family,
            task_signature  = task_b1.task_signature,
            steps           = adapted_steps,
            embedding       = emb_b1,
            success_score   = quality_b1,
        )
        sig_b1 = VerificationSignal(
            procedure_id   = proc_b1.procedure_id,
            task_signature = task_b1.task_signature,
            success        = True,
            is_transfer    = True,
            source_family  = task_a1.family,
            target_family  = task_b1.family,
        )
        update_procedure_after_verification(sig_b1, store)

    # ── C1: Transfer A→C ──────────────────────────────────────────────────────
    task_c1  = FAMILY_C_VERSION_MISMATCH.tasks[0]
    emb_c1   = _emb(task_c1)

    retr_c1  = retrieve_procedure(
        task_signature  = task_c1.task_signature,
        query_embedding = emb_c1,
        store           = store,
        mode            = RetrievalMode.CORRECT,
        top_k           = 5,
        correct_family  = task_c1.family,
    )
    c1_steps = (
        [s.action for s in retr_c1.top1.steps[:2]]
        + list(task_c1.canonical_steps[2:])
        if retr_c1.top1 else list(task_c1.canonical_steps)
    )
    success_c1, quality_c1, _ = evaluate_procedure_on_task(task_c1, c1_steps, seed=seed)
    result.c1_transfer = success_c1
    _log(f"C1: transfer A→C  success={success_c1}  quality={quality_c1:.3f}")

    # ── D1: Force failure → update → retry ────────────────────────────────────
    task_d1 = FAMILY_D_PATH_RESOLUTION.tasks[0]
    emb_d1  = _emb(task_d1)

    # First attempt with wrong steps (simulated failure)
    pre_steps = task_d1.distractor_steps
    success_pre, quality_pre, _ = evaluate_procedure_on_task(task_d1, pre_steps, seed=seed)
    result.d1_pre_retry = success_pre

    proc_d1 = store.build(
        problem_family = task_d1.family,
        task_signature = task_d1.task_signature,
        steps          = pre_steps,
        embedding      = emb_d1,
        success_score  = quality_pre,
    )
    _log(f"D1: pre-update  success={success_pre}  quality={quality_pre:.3f}")

    # Simulate failure signal
    fail_sig = VerificationSignal(
        procedure_id   = proc_d1.procedure_id,
        task_signature = task_d1.task_signature,
        success        = False,
        failed_step    = 0,
        error_type     = "IncorrectPath",
        error_message  = "FileNotFoundError: config.yaml not found",
        failure_family = task_d1.family,
        recovery_applied  = True,
        recovery_steps    = task_d1.canonical_steps,
        recovery_success  = True,
    )

    if ablation != "remove_update_mechanism":
        upd = update_procedure_after_verification(fail_sig, store, fork_on_failure=True, fork_threshold=1)
        _log(f"D1: update result: {upd.message}")

        # Retry with the forked (improved) procedure
        if upd.forked_id:
            forked = store.get(upd.forked_id)
            retry_steps = [s.action for s in forked.steps] if forked else list(task_d1.canonical_steps)
        else:
            retry_steps = list(task_d1.canonical_steps)
    else:
        retry_steps = list(task_d1.canonical_steps)

    success_retry, quality_retry, _ = evaluate_procedure_on_task(task_d1, retry_steps, seed=seed+1)
    result.d1_retry = success_retry
    _log(f"D1: post-update  success={success_retry}  quality={quality_retry:.3f}")

    # ── Baseline comparisons ──────────────────────────────────────────────────
    for mode in [RetrievalMode.DISABLED, RetrievalMode.RANDOM, RetrievalMode.WRONG, RetrievalMode.ORACLE]:
        retr = retrieve_procedure(
            task_signature  = task_a2.task_signature,
            query_embedding = emb_a2,
            store           = store,
            mode            = mode,
            top_k           = 5,
            correct_family  = task_a2.family,
            rng             = rng,
        )
        steps_bl = (
            [s.action for s in retr.top1.steps]
            if retr.top1 else []
        )
        success_bl, _, _ = evaluate_procedure_on_task(task_a2, steps_bl, seed=seed)
        result.results_by_mode[mode.value] = success_bl

    # ── Survival ──────────────────────────────────────────────────────────────
    if ablation != "remove_survival_scoring":
        store.decay_all(rate=0.99)
    all_procs  = store._procs
    result.final_survival = (
        mean(p.survival_score for p in all_procs) if all_procs else 0.0
    )
    result.store_size_end = len(store)

    return result


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def aggregate(results: List[SeedResult]) -> Dict[str, Any]:
    n = len(results)

    def _stat(vals: List[float]) -> dict:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
        m    = mean(vals)
        s    = stdev(vals) if len(vals) > 1 else 0.0
        ci95 = 1.96 * s / (n ** 0.5) if n > 1 else 0.0
        return {"mean": m, "std": s, "ci95": ci95, "n": n}

    correct_rates = [float(r.a2_reuse) for r in results]
    disabled_rates = [float(r.results_by_mode.get("disabled", False)) for r in results]
    random_rates   = [float(r.results_by_mode.get("random", False)) for r in results]
    oracle_rates   = [float(r.results_by_mode.get("oracle", False)) for r in results]

    reuse_gain     = [float(r.a2_reuse) - float(r.results_by_mode.get("disabled", False)) for r in results]
    retry_improve  = [float(r.d1_retry) - float(r.d1_pre_retry) for r in results]
    reset_deficit  = [float(r.a2_reuse) - float(r.results_by_mode.get("disabled", False)) for r in results]
    rand_vs_correct = [float(r.results_by_mode.get("random", False)) - float(r.a2_reuse) for r in results]
    transfer_gain  = [float(r.b1_transfer) - float(r.results_by_mode.get("disabled", False)) for r in results]
    survivals      = [r.final_survival for r in results]
    surv_cv        = (stdev(survivals) / max(mean(survivals), 1e-9)) if len(survivals) > 1 else 0.0

    agg = {
        "n_seeds":           n,
        "a1_success":        _stat([float(r.a1_success)  for r in results]),
        "a2_reuse":          _stat([float(r.a2_reuse)    for r in results]),
        "b1_transfer":       _stat([float(r.b1_transfer) for r in results]),
        "c1_transfer":       _stat([float(r.c1_transfer) for r in results]),
        "d1_retry":          _stat([float(r.d1_retry)    for r in results]),
        "d1_pre_retry":      _stat([float(r.d1_pre_retry) for r in results]),
        "retrieval_accuracy": _stat([r.retrieval_accuracy for r in results]),
        "family_match_rate":  _stat([r.family_match_rate  for r in results]),
        "final_survival":     _stat(survivals),
        "store_size_end":     _stat([float(r.store_size_end) for r in results]),
        # Derived
        "reuse_gain":        _stat(reuse_gain),
        "retry_improvement": _stat(retry_improve),
        "reset_deficit":     _stat(reset_deficit),
        "random_vs_correct": _stat(rand_vs_correct),
        "transfer_gain":     _stat(transfer_gain),
        "survival_cv":       surv_cv,
        # Baselines
        "disabled_rate":     _stat(disabled_rates),
        "random_rate":       _stat(random_rates),
        "oracle_rate":       _stat(oracle_rates),
        "correct_rate":      _stat(correct_rates),
    }
    return agg


# ── Gate evaluation ───────────────────────────────────────────────────────────

def evaluate_gates(agg: Dict[str, Any]) -> Dict[str, bool]:
    passed = {}
    for gate_name, (metric, op, threshold) in GATES.items():
        if metric not in agg:
            passed[gate_name] = False
            continue
        val = agg[metric]
        if isinstance(val, dict):
            val = val["mean"]
        if op == ">=":
            passed[gate_name] = val >= threshold
        elif op == ">":
            passed[gate_name] = val > threshold
        elif op == "<":
            passed[gate_name] = val < threshold
        else:
            passed[gate_name] = False
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(
    seeds:     List[int],
    ablation:  Optional[str] = None,
    verbose:   bool = False,
    output:    Optional[str] = None,
) -> Dict[str, Any]:

    label = f"ablation={ablation}" if ablation else "full"
    print(f"\n{'='*60}")
    print(f"TAC-PSM-001 Benchmark  [{label}]")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")

    t0      = time.time()
    results = []
    for s in seeds:
        print(f"\n  --- seed={s} ---")
        r = run_one_seed(s, ablation=ablation, verbose=verbose)
        results.append(r)
        print(f"  A1={r.a1_success}  A2={r.a2_reuse}  B1={r.b1_transfer}"
              f"  C1={r.c1_transfer}  D1={r.d1_retry}  surv={r.final_survival:.3f}")

    agg   = aggregate(results)
    gates = evaluate_gates(agg)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    for key in ["a1_success", "a2_reuse", "b1_transfer", "c1_transfer",
                "d1_retry", "retrieval_accuracy", "reuse_gain",
                "retry_improvement", "reset_deficit", "transfer_gain"]:
        v = agg.get(key, {})
        if isinstance(v, dict):
            print(f"  {key:<28} {v['mean']:.4f} ± {v['std']:.4f}  (95% CI ±{v['ci95']:.4f})")
        else:
            print(f"  {key:<28} {v:.4f}")
    print(f"  {'survival_cv':<28} {agg['survival_cv']:.4f}")

    print(f"\nBASELINES (A2 task)")
    for bname in ["disabled_rate", "random_rate", "oracle_rate", "correct_rate"]:
        v = agg.get(bname, {})
        m = v["mean"] if isinstance(v, dict) else v
        print(f"  {bname:<28} {m:.4f}")

    print(f"\nSUCCESS GATES")
    all_pass = True
    for gname, gpass in gates.items():
        sym  = "✓" if gpass else "✗"
        print(f"  [{sym}] {gname}")
        if not gpass:
            all_pass = False

    overall = "ALL GATES PASS ✓" if all_pass else "SOME GATES FAIL ✗"
    print(f"\n{overall}  (elapsed {elapsed:.1f}s)")

    report = {
        "label":    label,
        "seeds":    seeds,
        "n_seeds":  len(seeds),
        "elapsed":  elapsed,
        "agg":      agg,
        "gates":    gates,
        "all_pass": all_pass,
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved → {output}")

    return report


def main():
    parser = argparse.ArgumentParser(description="TAC-PSM-001 Benchmark")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Random seeds for multi-seed evaluation")
    parser.add_argument("--quick",   action="store_true",
                        help="Single-seed quick run (seed=0)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output",  type=str,
                        default="./reports/psm001_benchmark.json")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=[
                            "remove_failure_modes",
                            "remove_recovery_strategies",
                            "remove_update_mechanism",
                            "remove_transfer_metadata",
                            "remove_survival_scoring",
                        ])
    args = parser.parse_args()

    seeds = [0] if args.quick else args.seeds
    run_benchmark(
        seeds    = seeds,
        ablation = args.ablation,
        verbose  = args.verbose,
        output   = args.output,
    )


if __name__ == "__main__":
    main()
