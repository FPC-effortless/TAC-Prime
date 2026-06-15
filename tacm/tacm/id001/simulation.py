"""
TAC-Prime-ID001: Benchmark Simulation (NumPy)

Generates synthetic structure-reuse tasks and runs the three experimental
conditions (carried / reset / shuffled) using the NumPy simulation modules.

No PyTorch required.

Key design: memory records are tagged with the identity the router NATURALLY
activates for each family centroid after warm-up — not with the family index.
This ensures the identity bonus fires correctly when the carried state agrees
with the memory tags.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .state   import IdentityStateNP, identity_state_zeros
from .memory  import IdentityStructureMemory, IdentityProceduralMemory, _normalize
from .routing import (
    IdentityRouter, map_families_to_identities,
    compute_route_consistency, compute_identity_specialization,
)


# ── Synthetic task ─────────────────────────────────────────────────────────

@dataclass
class SyntheticTask:
    family_id:  int
    embedding:  np.ndarray     # (d_model,)  surface-noised query
    family_emb: np.ndarray     # (d_model,)  canonical centroid


# ── Factory helpers ─────────────────────────────────────────────────────────

def make_family_centroids(n_families: int, d: int, seed: int) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    raw  = rng.standard_normal((n_families, d)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / (norms + 1e-8)


def make_tasks(
    n_families:    int,
    tasks_per_fam: int,
    d:             int,
    seed:          int,
    noise:         float = 0.50,   # higher noise → retrieval harder → identity bias matters
) -> Tuple[List[SyntheticTask], np.ndarray]:
    centroids = make_family_centroids(n_families, d, seed)
    rng = np.random.default_rng(seed + 1)
    tasks: List[SyntheticTask] = []
    for fid in range(n_families):
        for _ in range(tasks_per_fam):
            noised = centroids[fid] + rng.standard_normal(d).astype(np.float32) * noise
            tasks.append(SyntheticTask(
                family_id  = fid,
                embedding  = _normalize(noised),
                family_emb = centroids[fid].copy(),
            ))
    rng_py = random.Random(seed + 2)
    rng_py.shuffle(tasks)
    return tasks, centroids


def seed_memory(
    struct_mem:       IdentityStructureMemory,
    proc_mem:         IdentityProceduralMemory,
    centroids:        np.ndarray,
    n_families:       int,
    router:           IdentityRouter,
    rng_seed:         int = 0,
):
    """
    Pre-populate memory with canonical embeddings tagged to each family.

    identity_id is set to the identity the router naturally activates for
    that family's centroid after warm-up.  This ensures the identity bonus
    fires correctly when carried state aligns with memory tags.
    """
    family_identity_map = map_families_to_identities(router, centroids, n_warmup=12)

    rng = np.random.default_rng(rng_seed)
    d   = centroids.shape[1]
    for fid in range(n_families):
        emb      = centroids[fid].copy()
        nat_id   = family_identity_map[fid]
        for _ in range(3):
            noised = emb + rng.standard_normal(d).astype(np.float32) * 0.05
            struct_mem.write(
                embedding      = _normalize(noised),
                family_id      = fid,
                expert_id      = fid % 4,
                task_type      = f"synth_family_{fid}",
                success_score  = 0.9,
                survival_score = 1.0,
                identity_id    = nat_id,
            )
        step_emb = emb + rng.standard_normal(d).astype(np.float32) * 0.05
        proc_mem.write(
            family       = f"Family{fid}",
            task_type    = f"synth_proc_{fid}",
            steps        = [f"step_{i}" for i in range(3)],
            embedding    = _normalize(step_emb),
            success_rate = 0.8,
            identity_id  = nat_id,
        )


# ── Condition A: Carried ────────────────────────────────────────────────────

def run_condition_carried(
    tasks:       List[SyntheticTask],
    router:      IdentityRouter,
    struct_mem:  IdentityStructureMemory,
    proc_mem:    IdentityProceduralMemory,
    bias_scale:  float = 0.25,
    n_families:  int   = 4,
) -> Tuple[float, float, List[Optional[IdentityStateNP]],
           Dict[int, List[int]], Dict[int, List[int]]]:
    state: Optional[IdentityStateNP] = None
    struct_accs   = []
    proc_accs     = []
    per_task_states: List[Optional[IdentityStateNP]] = []
    family_routes: Dict[int, List[int]]   = {f: [] for f in range(n_families)}
    family_actids: Dict[int, List[int]]   = {f: [] for f in range(n_families)}

    for task in tasks:
        new_state, active_id, weights = router.forward(task.embedding, state)
        state = new_state
        per_task_states.append(state.copy())

        s_recs = struct_mem.retrieve(
            task.family_emb, top_k=1,
            active_identity_id         = active_id,
            identity_memory_bias_scale = bias_scale,
        )
        p_recs = proc_mem.retrieve(
            task.family_emb, top_k=1,
            active_identity_id         = active_id,
            identity_memory_bias_scale = bias_scale,
        )

        s_acc  = 1.0 if s_recs and s_recs[0].family_id == task.family_id else 0.0
        p_name = f"Family{task.family_id}"
        p_acc  = 1.0 if p_recs and p_recs[0].family == p_name else 0.0

        struct_accs.append(s_acc)
        proc_accs.append(p_acc)
        family_routes[task.family_id].append(active_id)
        family_actids[task.family_id].append(active_id)

    return (
        float(sum(struct_accs) / len(struct_accs)),
        float(sum(proc_accs)   / len(proc_accs)),
        per_task_states,
        family_routes,
        family_actids,
    )


# ── Condition B: Reset ──────────────────────────────────────────────────────

def run_condition_reset(
    tasks:       List[SyntheticTask],
    router:      IdentityRouter,
    struct_mem:  IdentityStructureMemory,
    proc_mem:    IdentityProceduralMemory,
    bias_scale:  float = 0.25,
    n_families:  int   = 4,
) -> Tuple[float, float, Dict[int, List[int]]]:
    struct_accs   = []
    proc_accs     = []
    family_routes: Dict[int, List[int]] = {f: [] for f in range(n_families)}

    for task in tasks:
        _, active_id, _ = router.forward(task.embedding, state=None)

        s_recs = struct_mem.retrieve(
            task.embedding, top_k=1,
            active_identity_id         = active_id,
            identity_memory_bias_scale = bias_scale,
        )
        p_recs = proc_mem.retrieve(
            task.embedding, top_k=1,
            active_identity_id         = active_id,
            identity_memory_bias_scale = bias_scale,
        )

        s_acc  = 1.0 if s_recs and s_recs[0].family_id == task.family_id else 0.0
        p_name = f"Family{task.family_id}"
        p_acc  = 1.0 if p_recs and p_recs[0].family == p_name else 0.0

        struct_accs.append(s_acc)
        proc_accs.append(p_acc)
        family_routes[task.family_id].append(active_id)

    return (
        float(sum(struct_accs) / len(struct_accs)),
        float(sum(proc_accs)   / len(proc_accs)),
        family_routes,
    )


# ── Condition C: Shuffled ───────────────────────────────────────────────────

def run_condition_shuffled(
    tasks:            List[SyntheticTask],
    router:           IdentityRouter,
    struct_mem:       IdentityStructureMemory,
    per_task_states:  List[Optional[IdentityStateNP]],
    bias_scale:       float = 0.25,
) -> float:
    """Supply identity state from the NEXT task (index+1 mod N) — identity mismatch."""
    struct_accs = []
    N = len(tasks)
    for i, task in enumerate(tasks):
        wrong_state = per_task_states[(i + 1) % N]
        _, active_id, _ = router.forward(task.embedding, state=wrong_state)

        s_recs = struct_mem.retrieve(
            task.embedding, top_k=1,
            active_identity_id         = active_id,
            identity_memory_bias_scale = bias_scale,
        )
        s_acc = 1.0 if s_recs and s_recs[0].family_id == task.family_id else 0.0
        struct_accs.append(s_acc)

    return float(sum(struct_accs) / len(struct_accs))


# ── Condition D: Memory Knockout ─────────────────────────────────────────────

def run_condition_memory_knockout(
    tasks:      List[SyntheticTask],
    router:     IdentityRouter,
    struct_mem: IdentityStructureMemory,
) -> float:
    """Clear memory — all retrievals return empty → acc = 0."""
    saved = dict(struct_mem._store)
    struct_mem.clear()
    struct_accs = []
    for task in tasks:
        _, _, _ = router.forward(task.embedding, state=None)
        s_recs  = struct_mem.retrieve(task.embedding, top_k=1)
        s_acc   = 1.0 if s_recs and s_recs[0].family_id == task.family_id else 0.0
        struct_accs.append(s_acc)
    struct_mem._store.update(saved)
    return float(sum(struct_accs) / len(struct_accs))
