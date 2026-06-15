"""
TAC-PSM-006B: Baseline Runner
==============================

Runs all 7 experimental variants on a fixture set and returns traces for
metrics computation.

Variants:
  1. full_memory         — TAC full procedural memory (seeded, retrieval on)
  2. reset               — memory cleared before each fixture (no reuse)
  3. retrieval_disabled  — store exists but retrieval always returns wrong family
  4. random_procedure    — randomly selects any family's procedure
  5. structure_only      — correct file targeted, wrong patch content
  6. no_update           — full memory, retrieval on, but no update after repair
  7. oracle              — always uses the correct family's procedure (upper bound)

Each variant runs all fixtures in order with the same seed.
The full_memory agent updates its store as it processes fixtures, so later
fixtures benefit from earlier repairs — this is the TAC memory reuse claim.
All other variants are ablations that isolate one component.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .fixture_schema import Fixture, FAMILY_NAMES
from .pytest_verifier import PytestVerifier
from .patch_applier import PatchApplier
from .procedural_repair_agent import (
    ProceduralRepairAgent006B,
    RepairTrace006B,
    seed_procedural_memory,
)
from .memory_store import SimpleProceduralMemoryStore


VARIANT_NAMES: List[str] = [
    "full_memory",
    "reset",
    "retrieval_disabled",
    "random_procedure",
    "structure_only",
    "no_update",
    "oracle",
]


def _make_seeded_store(rng_seed: int) -> SimpleProceduralMemoryStore:
    """Create a fresh store pre-seeded with oracle procedures for all families."""
    store = SimpleProceduralMemoryStore()
    seed_procedural_memory(store, n_records_per_family=2, rng_seed=rng_seed)
    return store


def _make_empty_store() -> SimpleProceduralMemoryStore:
    """Create an empty store (for reset baseline — cleared before each fixture)."""
    return SimpleProceduralMemoryStore()


def run_variant(
    variant:   str,
    fixtures:  List[Fixture],
    seed:      int,
    verifier:  PytestVerifier,
    applier:   PatchApplier,
) -> List[RepairTrace006B]:
    """
    Run a single variant on all fixtures and return traces.

    Parameters
    ----------
    variant  : one of VARIANT_NAMES
    fixtures : all 60 (or subset) benchmark fixtures
    seed     : RNG seed for reproducibility
    verifier : PytestVerifier instance (shared across variants)
    applier  : PatchApplier instance (shared across variants)
    """
    assert variant in VARIANT_NAMES, f"Unknown variant: {variant}"

    if variant == "reset":
        return _run_reset(fixtures, seed, verifier, applier)

    store = _make_seeded_store(seed)

    agent = ProceduralRepairAgent006B(
        store           = store,
        verifier        = verifier,
        applier         = applier,
        mode            = variant,
        retrieval_noise = 0.10,
        rng_seed        = seed,
        max_retries     = 1 if variant == "full_memory" else 0,
    )

    traces: List[RepairTrace006B] = []
    for fx in fixtures:
        t = agent.repair(fx)
        traces.append(t)
    return traces


def _run_reset(
    fixtures: List[Fixture],
    seed:     int,
    verifier: PytestVerifier,
    applier:  PatchApplier,
) -> List[RepairTrace006B]:
    """
    Reset baseline: memory is cleared before every fixture.

    This ablation tests whether accumulated memory actually helps.
    A fresh store is re-seeded before each fixture to match the seeded-state
    at the start of a run (so the agent has the oracle procedures available
    on step 1, but cannot benefit from successful repairs on previous fixtures).
    """
    traces: List[RepairTrace006B] = []
    rng_state = np.random.default_rng(seed)

    for i, fx in enumerate(fixtures):
        local_seed = int(rng_state.integers(0, 2**31))
        store = _make_seeded_store(local_seed)
        agent = ProceduralRepairAgent006B(
            store           = store,
            verifier        = verifier,
            applier         = applier,
            mode            = "reset",
            retrieval_noise = 0.10,
            rng_seed        = local_seed,
            max_retries     = 0,
        )
        traces.append(agent.repair(fx))
    return traces


def run_all_baselines(
    fixtures:         List[Fixture],
    seed:             int = 0,
    timeout_s:        float = 10.0,
    variants:         List[str] = None,
) -> Dict[str, List[RepairTrace006B]]:
    """
    Run all (or a subset of) variants on all fixtures.

    Parameters
    ----------
    fixtures  : list of Fixture objects to test
    seed      : shared RNG seed
    timeout_s : per-fixture pytest timeout in seconds
    variants  : if given, only run these variants; else run all

    Returns
    -------
    {variant_name: [RepairTrace006B, ...]}
    """
    verifier = PytestVerifier(timeout=timeout_s)
    applier  = PatchApplier()
    to_run   = variants or VARIANT_NAMES

    results: Dict[str, List[RepairTrace006B]] = {}
    for v in to_run:
        traces = run_variant(v, fixtures, seed, verifier, applier)
        results[v] = traces
    return results
