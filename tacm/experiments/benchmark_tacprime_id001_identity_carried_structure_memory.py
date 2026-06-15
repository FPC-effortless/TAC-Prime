"""
TAC-Prime-ID001: Identity-Carried Structure Memory Benchmark

Pure-Python / NumPy simulation — no PyTorch required.
Follows the TAC-PSM benchmark pattern.

Research hypothesis:
  TAC-Prime reuses, routes, and transfers structures better when reusable
  structures/procedures are carried by persistent computational identities
  rather than only stored as detached memory records.

Conditions:
  A. identity_carried  — IdentityState persists across all tasks
  B. identity_reset    — IdentityState is re-zeroed before every task
  C. identity_shuffled — IdentityState from the NEXT task (mismatched)
  D. memory_knockout   — StructureMemory cleared; retrieval always empty

Key mechanism:
  active_identity blends current query signal with accumulated route_history
  (history_blend=0.65).  A carried state builds a stable route_history that
  consistently activates the same identity for the same family.  A reset state
  starts from zero, so its active_identity is noisier on high-noise queries.
  Memory records are pre-tagged with each family's natural identity so the
  identity bonus fires correctly only when state and memory agree.

Metrics:
  route_consistency        — 1 − normalised_entropy of routing per family
  structure_retrieval_acc  — top-1 family match fraction
  procedure_retrieval_acc  — top-1 family match fraction
  carried_vs_reset_gain    — struct_acc[carried] − struct_acc[reset]
  carried_vs_shuffled_gain — struct_acc[carried] − struct_acc[shuffled]
  memory_knockout_drop     — struct_acc[carried] − struct_acc[knockout]
  identity_specialization  — concentration of active identities per family
  benchmark_score          — weighted aggregate ≥ 0.60 to validate

Validation gates (6):
  1. carried_route_consistency > reset_route_consistency
  2. carried_structure_retrieval > reset_structure_retrieval
  3. carried_procedure_retrieval > reset_procedure_retrieval
  4. carried_vs_shuffled_gain > 0.0
  5. memory_knockout_drop > 0.0
  6. benchmark_score ≥ 0.60
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List

import numpy as np

sys.path.insert(0, __file__.rsplit("/experiments", 1)[0])

from tacm.id001.memory  import IdentityStructureMemory, IdentityProceduralMemory
from tacm.id001.routing import (
    IdentityRouter, compute_route_consistency, compute_identity_specialization,
)
from tacm.id001.simulation import (
    make_tasks, seed_memory,
    run_condition_carried, run_condition_reset,
    run_condition_shuffled, run_condition_memory_knockout,
)


# ── Config ──────────────────────────────────────────────────────────────────

N_FAMILIES    = 4
TASKS_PER_FAM = 20
D_MODEL       = 32
N_IDENTITIES  = N_FAMILIES * 2
ENERGY_BUDGET = float(N_FAMILIES)
STATE_DECAY   = 0.7
HISTORY_BLEND = 0.65
BIAS_SCALE    = 0.25
N_SEEDS       = 5

GATE_THRESHOLDS = {
    "carried_gt_reset_route":   ("carried_route_consistency",   "reset_route_consistency",   "gt"),
    "carried_gt_reset_struct":  ("carried_structure_retrieval", "reset_structure_retrieval", "gt"),
    "carried_gt_reset_proc":    ("carried_procedure_retrieval", "reset_procedure_retrieval", "gt"),
    "shuffled_hurts":           ("carried_vs_shuffled_gain",    None,                        "gt_zero"),
    "memory_knockout":          ("memory_knockout_drop",        None,                        "gt_zero"),
    "benchmark_score_ge_0.60":  ("benchmark_score",             None,                        "ge_0.60"),
}


# ── Per-seed result ──────────────────────────────────────────────────────────

@dataclass
class SeedResult:
    carried_route_consistency:    float = 0.0
    reset_route_consistency:      float = 0.0
    carried_structure_retrieval:  float = 0.0
    reset_structure_retrieval:    float = 0.0
    carried_procedure_retrieval:  float = 0.0
    reset_procedure_retrieval:    float = 0.0
    shuffled_structure_retrieval: float = 0.0
    memory_knockout_retrieval:    float = 0.0
    identity_specialization:      float = 0.0


def run_seed(seed: int) -> SeedResult:
    random.seed(seed)
    np.random.seed(seed)

    router = IdentityRouter(
        d_model               = D_MODEL,
        n_identities          = N_IDENTITIES,
        identity_energy_budget = ENERGY_BUDGET,
        identity_state_decay  = STATE_DECAY,
        history_blend         = HISTORY_BLEND,
        seed                  = seed,
    )
    struct_mem = IdentityStructureMemory(embedding_dim=D_MODEL)
    proc_mem   = IdentityProceduralMemory(embedding_dim=D_MODEL)

    tasks, centroids = make_tasks(N_FAMILIES, TASKS_PER_FAM, D_MODEL, seed=seed)
    seed_memory(struct_mem, proc_mem, centroids, N_FAMILIES,
                router=router, rng_seed=seed)

    # ── A: carried ───────────────────────────────────────────────────────
    s_c, p_c, carried_states, routes_c, actids_c = run_condition_carried(
        tasks, router, struct_mem, proc_mem,
        bias_scale=BIAS_SCALE, n_families=N_FAMILIES,
    )
    route_c = compute_route_consistency(routes_c, n_families=N_FAMILIES)
    spec    = compute_identity_specialization(actids_c, n_identities=N_IDENTITIES)

    # ── B: reset ──────────────────────────────────────────────────────────
    s_r, p_r, routes_r = run_condition_reset(
        tasks, router, struct_mem, proc_mem,
        bias_scale=BIAS_SCALE, n_families=N_FAMILIES,
    )
    route_r = compute_route_consistency(routes_r, n_families=N_FAMILIES)

    # ── C: shuffled ───────────────────────────────────────────────────────
    s_s = run_condition_shuffled(
        tasks, router, struct_mem, carried_states, bias_scale=BIAS_SCALE,
    )

    # ── D: memory knockout ────────────────────────────────────────────────
    s_ko = run_condition_memory_knockout(tasks[:TASKS_PER_FAM], router, struct_mem)

    return SeedResult(
        carried_route_consistency    = route_c,
        reset_route_consistency      = route_r,
        carried_structure_retrieval  = s_c,
        reset_structure_retrieval    = s_r,
        carried_procedure_retrieval  = p_c,
        reset_procedure_retrieval    = p_r,
        shuffled_structure_retrieval = s_s,
        memory_knockout_retrieval    = s_ko,
        identity_specialization      = spec,
    )


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(results: List[SeedResult]) -> Dict[str, float]:
    def avg(attr: str) -> float:
        return float(mean(getattr(r, attr) for r in results))

    c_s  = avg("carried_structure_retrieval")
    c_p  = avg("carried_procedure_retrieval")
    r_s  = avg("reset_structure_retrieval")
    r_p  = avg("reset_procedure_retrieval")
    s_s  = avg("shuffled_structure_retrieval")
    ko_s = avg("memory_knockout_retrieval")

    cvr_gain = c_s - r_s
    cvs_gain = c_s - s_s
    ko_drop  = c_s - ko_s

    benchmark_score = (
        0.20 * avg("carried_route_consistency")
        + 0.25 * c_s
        + 0.25 * c_p
        + 0.10 * max(cvr_gain, 0.0) / 0.5
        + 0.10 * max(cvs_gain, 0.0) / 0.5
        + 0.05 * avg("identity_specialization")
        + 0.05 * max(ko_drop,  0.0) / 0.5
    )

    return {
        "carried_route_consistency":    avg("carried_route_consistency"),
        "reset_route_consistency":      avg("reset_route_consistency"),
        "carried_structure_retrieval":  c_s,
        "reset_structure_retrieval":    r_s,
        "carried_procedure_retrieval":  c_p,
        "reset_procedure_retrieval":    r_p,
        "shuffled_structure_retrieval": s_s,
        "memory_knockout_retrieval":    ko_s,
        "carried_vs_reset_gain":        cvr_gain,
        "carried_vs_shuffled_gain":     cvs_gain,
        "memory_knockout_drop":         ko_drop,
        "identity_specialization":      avg("identity_specialization"),
        "benchmark_score":              benchmark_score,
    }


# ── Gate evaluation ──────────────────────────────────────────────────────────

def evaluate_gates(metrics: Dict[str, float]) -> Dict[str, bool]:
    results = {}
    for gate, (m_a, m_b, cond) in GATE_THRESHOLDS.items():
        a = metrics.get(m_a, 0.0)
        if cond == "gt":
            results[gate] = a > metrics.get(m_b, 0.0)
        elif cond == "gt_zero":
            results[gate] = a > 0.0
        elif cond == "ge_0.60":
            results[gate] = a >= 0.60
        else:
            results[gate] = False
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TAC-Prime-ID001 benchmark")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(N_SEEDS)))
    args  = parser.parse_args()
    seeds = args.seeds

    print("\n" + "=" * 70)
    print("  TAC-Prime-ID001: Identity-Carried Structure Memory Benchmark")
    print(f"  Seeds: {seeds}  |  Families: {N_FAMILIES}  |  "
          f"Tasks/fam: {TASKS_PER_FAM}  |  D={D_MODEL}  |  "
          f"history_blend={HISTORY_BLEND}")
    print("=" * 70)

    t0 = time.time()
    seed_results: List[SeedResult] = []

    for s in seeds:
        print(f"\n  Seed {s} ...", flush=True)
        r = run_seed(s)
        seed_results.append(r)
        print(f"    struct  carried={r.carried_structure_retrieval:.3f}  "
              f"reset={r.reset_structure_retrieval:.3f}  "
              f"shuffled={r.shuffled_structure_retrieval:.3f}  "
              f"ko={r.memory_knockout_retrieval:.3f}")
        print(f"    proc    carried={r.carried_procedure_retrieval:.3f}  "
              f"reset={r.reset_procedure_retrieval:.3f}")
        print(f"    route   carried={r.carried_route_consistency:.3f}  "
              f"reset={r.reset_route_consistency:.3f}  "
              f"id_spec={r.identity_specialization:.3f}")

    elapsed = time.time() - t0
    metrics = aggregate(seed_results)
    gates   = evaluate_gates(metrics)
    n_pass  = sum(gates.values())
    n_total = len(gates)

    print("\n" + "─" * 70)
    print("  METRICS (mean across seeds)")
    print("─" * 70)
    for k, v in metrics.items():
        print(f"  {k:<42s}  {v:.4f}")

    print("\n" + "─" * 70)
    print("  VALIDATION GATES")
    print("─" * 70)
    for gate, passed in gates.items():
        status = "[✓]" if passed else "[✗]"
        print(f"  {status} {gate}")

    print("\n" + "─" * 70)
    verdict = "✓ VALIDATES" if n_pass == n_total else "✗ FAILS"
    print(f"  VERDICT: {n_pass}/{n_total} gates  ({verdict})")
    print(f"  benchmark_score = {metrics['benchmark_score']:.4f}  "
          f"(threshold ≥ 0.60)")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
