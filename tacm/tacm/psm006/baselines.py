"""
TAC-PSM-006: Baseline Variants
================================

Implements all 7 required system variants for PSM-006:

  1. TAC-PSM full memory       — retrieve + apply + verify + update
  2. Reset baseline            — empty memory; no retrieval
  3. Retrieval-disabled        — memory populated but retrieval off; random steps
  4. Random-procedure          — random procedure from memory (not similarity-based)
  5. Structure-memory-only     — uses family embedding only; ignores step content
  6. Oracle-procedure          — always uses ground-truth oracle steps (upper bound)
  7. No-update baseline        — retrieves correctly but never updates memory

Each variant exposes a single function:
  run_baseline(tasks, fixtures, store, seed) -> List[AgentTrace]
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..psm001.store import ProceduralMemoryStore
from .repository_task import RepoTask, EMBEDDING_DIM
from .repo_fixture_builder import RepoFixture, build_fixture
from .verifier import verify_repair, verify_with_retry, VerificationResult
from .procedural_repair_agent import AgentTrace, ProceduralRepairAgent, make_agent


# ── Helper: build fixtures if not provided ────────────────────────────────────

def _ensure_fixtures(
    tasks:    List[RepoTask],
    fixtures: Optional[Dict[str, RepoFixture]],
) -> Dict[str, RepoFixture]:
    if fixtures is not None:
        return fixtures
    return {t.task_id: build_fixture(t) for t in tasks}


# ── 1. TAC-PSM Full Memory ────────────────────────────────────────────────────

def run_full_memory(
    tasks:       List[RepoTask],
    fixtures:    Optional[Dict[str, RepoFixture]],
    store:       ProceduralMemoryStore,
    seed:        int = 0,
    max_retries: int = 2,
) -> List[AgentTrace]:
    """
    Full TAC-PSM system: retrieve → apply → verify → update.
    This is the primary system being evaluated.
    """
    fix = _ensure_fixtures(tasks, fixtures)
    agent = ProceduralRepairAgent(
        store           = store,
        embedding_dim   = store.dim,
        retrieval_top_k = 3,
        max_retries     = max_retries,
        update_enabled  = True,
        mode            = "full_memory",
    )
    return agent.repair_batch(tasks, fix, seed=seed)


# ── 2. Reset Baseline ─────────────────────────────────────────────────────────

def run_reset(
    tasks:    List[RepoTask],
    fixtures: Optional[Dict[str, RepoFixture]],
    seed:     int = 0,
) -> List[AgentTrace]:
    """
    Reset baseline: empty memory, no procedure retrieved.
    Applied steps = [] for every task.
    """
    fix = _ensure_fixtures(tasks, fixtures)
    traces = []
    for task in tasks:
        ver = verify_repair(task, [], "Unknown")
        traces.append(AgentTrace(
            task_id           = task.task_id,
            repo_name         = task.repo_name,
            family            = task.family,
            retrieved_proc_id = None,
            retrieved_family  = "Unknown",
            applied_steps     = [],
            verification      = ver,
            n_retries         = 0,
            steps_to_repair   = 0,
            procedure_updated = False,
            update_success    = False,
            mode              = "reset",
        ))
    return traces


# ── 3. Retrieval-Disabled Baseline ────────────────────────────────────────────

def run_retrieval_disabled(
    tasks:    List[RepoTask],
    fixtures: Optional[Dict[str, RepoFixture]],
    store:    ProceduralMemoryStore,
    seed:     int = 0,
) -> List[AgentTrace]:
    """
    Memory is populated but retrieval is disabled.

    Applied steps come from distractor_steps() — plausible but semantically wrong.
    Selected family is deliberately wrong (the family AFTER the correct one in the
    cycle), simulating what happens when the retrieval mechanism is completely off
    and we fall back to an arbitrary family label.

    This baseline tests: does disabling retrieval (losing the correct procedure)
    hurt performance compared to full TAC?
    """
    fix = _ensure_fixtures(tasks, fixtures)
    from .repository_task import ALL_FAMILY_NAMES
    rng = random.Random(seed)
    traces = []
    for task in tasks:
        steps = task.distractor_steps()
        # Select a wrong family: rotate index by 1 so it's always incorrect
        family_idx   = ALL_FAMILY_NAMES.index(task.family) if task.family in ALL_FAMILY_NAMES else 0
        wrong_family = ALL_FAMILY_NAMES[(family_idx + 1) % len(ALL_FAMILY_NAMES)]
        ver   = verify_repair(task, steps, wrong_family)
        traces.append(AgentTrace(
            task_id           = task.task_id,
            repo_name         = task.repo_name,
            family            = task.family,
            retrieved_proc_id = None,
            retrieved_family  = wrong_family,
            applied_steps     = steps,
            verification      = ver,
            n_retries         = 0,
            steps_to_repair   = len(steps),
            procedure_updated = False,
            update_success    = False,
            mode              = "retrieval_disabled",
        ))
    return traces


# ── 4. Random-Procedure Baseline ──────────────────────────────────────────────

def run_random_procedure(
    tasks:    List[RepoTask],
    fixtures: Optional[Dict[str, RepoFixture]],
    store:    ProceduralMemoryStore,
    seed:     int = 0,
) -> List[AgentTrace]:
    """
    Random procedure: ignores query similarity, picks a uniformly random
    stored procedure. Tests whether wrong-procedure retrieval causes harm.
    """
    fix = _ensure_fixtures(tasks, fixtures)
    rng = random.Random(seed)
    traces = []
    procs  = store._procs  # direct access for random sampling

    for task in tasks:
        if procs:
            proc = rng.choice(procs)
            steps  = [s.action for s in proc.steps]
            family = proc.problem_family
            pid    = proc.procedure_id
        else:
            steps  = []
            family = "Unknown"
            pid    = None

        ver = verify_repair(task, steps, family)
        traces.append(AgentTrace(
            task_id           = task.task_id,
            repo_name         = task.repo_name,
            family            = task.family,
            retrieved_proc_id = pid,
            retrieved_family  = family,
            applied_steps     = steps,
            verification      = ver,
            n_retries         = 0,
            steps_to_repair   = len(steps),
            procedure_updated = False,
            update_success    = False,
            mode              = "random_procedure",
        ))
    return traces


# ── 5. Structure-Memory-Only Baseline ────────────────────────────────────────

def run_structure_only(
    tasks:    List[RepoTask],
    fixtures: Optional[Dict[str, RepoFixture]],
    store:    ProceduralMemoryStore,
    seed:     int = 0,
) -> List[AgentTrace]:
    """
    Structure memory only: uses family-level embedding for retrieval but
    strips the actual step content (uses family label as the only signal).

    Simulates a system that knows the right family but has no procedure steps.
    Applied steps = [f"Apply {family} repair procedure"] — structural hint only.
    """
    fix = _ensure_fixtures(tasks, fixtures)
    traces = []
    for task in tasks:
        # Use family embedding (not task embedding) to retrieve
        fam_emb    = task.family_embedding(store.dim)
        candidates = store.retrieve(fam_emb.astype(np.float32), top_k=1)

        if candidates:
            _, best = candidates[0]
            family  = best.problem_family
            pid     = best.procedure_id
            # Strip step content: only keep family-level structural hint
            steps   = [f"Apply {family} repair procedure (structure hint)"]
        else:
            family  = "Unknown"
            pid     = None
            steps   = []

        ver = verify_repair(task, steps, family)
        traces.append(AgentTrace(
            task_id           = task.task_id,
            repo_name         = task.repo_name,
            family            = task.family,
            retrieved_proc_id = pid,
            retrieved_family  = family,
            applied_steps     = steps,
            verification      = ver,
            n_retries         = 0,
            steps_to_repair   = len(steps),
            procedure_updated = False,
            update_success    = False,
            mode              = "structure_only",
        ))
    return traces


# ── 6. Oracle-Procedure Baseline (Upper Bound) ───────────────────────────────

def run_oracle(
    tasks:    List[RepoTask],
    fixtures: Optional[Dict[str, RepoFixture]],
    seed:     int = 0,
) -> List[AgentTrace]:
    """
    Oracle upper bound: always applies the ground-truth oracle repair steps.
    Family is always correct. No retrieval needed.
    """
    fix = _ensure_fixtures(tasks, fixtures)
    traces = []
    for task in tasks:
        steps = list(task.oracle_repair_steps)
        ver   = verify_repair(task, steps, task.family)
        traces.append(AgentTrace(
            task_id           = task.task_id,
            repo_name         = task.repo_name,
            family            = task.family,
            retrieved_proc_id = None,
            retrieved_family  = task.family,
            applied_steps     = steps,
            verification      = ver,
            n_retries         = 0,
            steps_to_repair   = len(steps),
            procedure_updated = False,
            update_success    = False,
            mode              = "oracle",
        ))
    return traces


# ── 7. No-Update Baseline ────────────────────────────────────────────────────

def run_no_update(
    tasks:    List[RepoTask],
    fixtures: Optional[Dict[str, RepoFixture]],
    store:    ProceduralMemoryStore,
    seed:     int = 0,
    max_retries: int = 2,
) -> List[AgentTrace]:
    """
    No-update baseline: retrieves correctly (same as full memory) but never
    updates memory after verification. Memory is frozen after warm-up.

    This tests whether the update step provides measurable benefit.
    """
    fix = _ensure_fixtures(tasks, fixtures)
    agent = ProceduralRepairAgent(
        store           = store,
        embedding_dim   = store.dim,
        retrieval_top_k = 3,
        max_retries     = max_retries,
        update_enabled  = False,   # ← key difference
        mode            = "no_update",
    )
    return agent.repair_batch(tasks, fix, seed=seed)


# ── Dispatcher ────────────────────────────────────────────────────────────────

BASELINE_NAMES = [
    "full_memory",
    "reset",
    "retrieval_disabled",
    "random_procedure",
    "structure_only",
    "oracle",
    "no_update",
]


def run_all_baselines(
    tasks:       List[RepoTask],
    fixtures:    Optional[Dict[str, RepoFixture]],
    store_full:  ProceduralMemoryStore,   # warm store for full / no_update / struct / rand
    seed:        int = 0,
) -> Dict[str, List[AgentTrace]]:
    """
    Run all 7 baselines and return a dict of name → List[AgentTrace].

    `store_full` must already be warm (via agent.warm_up on training tasks).
    The reset and oracle baselines do not use the store.
    """
    return {
        "full_memory":        run_full_memory(tasks, fixtures, store_full, seed),
        "reset":              run_reset(tasks, fixtures, seed),
        "retrieval_disabled": run_retrieval_disabled(tasks, fixtures, store_full, seed),
        "random_procedure":   run_random_procedure(tasks, fixtures, store_full, seed),
        "structure_only":     run_structure_only(tasks, fixtures, store_full, seed),
        "oracle":             run_oracle(tasks, fixtures, seed),
        "no_update":          run_no_update(tasks, fixtures, store_full, seed),
    }
