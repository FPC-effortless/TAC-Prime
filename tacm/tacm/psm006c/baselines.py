"""
TAC-PSM-006C: Baseline Runner
==============================

Runs the PSM-006C comparison set on a fixture list and returns traces.

PSM-006C comparison variants:
  1. full_memory_embedding_update  — TAC + text update + embedding update (NEW)
  2. full_memory                   — TAC text update only (PSM-006B baseline)
  3. reset                         — memory cleared before each fixture
  4. no_update                     — no update of any kind
  5. oracle                        — always uses correct family (upper bound)

The other two PSM-006B control variants (retrieval_disabled, random_procedure,
structure_only) are omitted from the core comparison set to keep the ablation
focused, but can be run via run_variant() if needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..psm006b.fixture_schema import Fixture, FAMILY_NAMES
from ..psm006b.patch_applier import PatchApplier
from ..psm006b.memory_store import SimpleProceduralMemoryStore
from ..psm006b.procedural_repair_agent import (
    seed_procedural_memory,
    RepairTrace006B,
)
from .agent import ProceduralRepairAgent006C, RepairTrace006C
from .embedding_update import OnlineEmbeddingAdapter


VARIANT_NAMES_006C: List[str] = [
    "full_memory_embedding_update",
    "full_memory",
    "reset",
    "no_update",
    "oracle",
]


# ── Store helpers ─────────────────────────────────────────────────────────

def _make_seeded_store(rng_seed: int) -> SimpleProceduralMemoryStore:
    store = SimpleProceduralMemoryStore()
    seed_procedural_memory(store, n_records_per_family=2, rng_seed=rng_seed)
    return store


# ── Variant runners ───────────────────────────────────────────────────────

def run_embedding_update_sequential(
    fixtures:  List[Fixture],
    seed:      int,
    verifier,
    applier:   PatchApplier,
) -> List[RepairTrace006C]:
    """
    full_memory_embedding_update: sequential memory accumulation + embedding updates.
    Must stay sequential — store state carries forward fixture-to-fixture.
    """
    store   = _make_seeded_store(seed)
    adapter = OnlineEmbeddingAdapter(lr_fail=0.10, lr_success=0.05)
    agent   = ProceduralRepairAgent006C(
        store           = store,
        verifier        = verifier,
        applier         = applier,
        adapter         = adapter,
        mode            = "full_memory_embedding_update",
        retrieval_noise = 0.10,
        rng_seed        = seed,
        max_retries     = 1,
    )
    return [agent.repair(fx) for fx in fixtures]


def run_full_memory_sequential(
    fixtures:  List[Fixture],
    seed:      int,
    verifier,
    applier:   PatchApplier,
) -> List[RepairTrace006C]:
    """
    full_memory: PSM-006B-style text update only.
    Sequential because memory accumulates across fixtures.
    """
    store   = _make_seeded_store(seed)
    adapter = OnlineEmbeddingAdapter()          # present but never used (text-only mode)
    agent   = ProceduralRepairAgent006C(
        store           = store,
        verifier        = verifier,
        applier         = applier,
        adapter         = adapter,
        mode            = "full_memory",
        retrieval_noise = 0.10,
        rng_seed        = seed,
        max_retries     = 1,
    )
    return [agent.repair(fx) for fx in fixtures]


def run_reset_per_fixture(
    fixtures:  List[Fixture],
    seed:      int,
    verifier,
    applier:   PatchApplier,
) -> List[RepairTrace006C]:
    """
    reset: fresh seeded store before every fixture (no memory reuse).
    Each fixture is independent.
    """
    rng    = np.random.default_rng(seed)
    traces = []
    for fx in fixtures:
        fx_seed = int(rng.integers(0, 2 ** 31))
        store   = _make_seeded_store(fx_seed)
        adapter = OnlineEmbeddingAdapter()
        agent   = ProceduralRepairAgent006C(
            store           = store,
            verifier        = verifier,
            applier         = applier,
            adapter         = adapter,
            mode            = "reset",
            retrieval_noise = 0.10,
            rng_seed        = fx_seed,
            max_retries     = 0,
        )
        traces.append(agent.repair(fx))
    return traces


def run_no_update_sequential(
    fixtures:  List[Fixture],
    seed:      int,
    verifier,
    applier:   PatchApplier,
) -> List[RepairTrace006C]:
    """no_update: retrieval on, no update of any kind."""
    store   = _make_seeded_store(seed)
    adapter = OnlineEmbeddingAdapter()
    agent   = ProceduralRepairAgent006C(
        store           = store,
        verifier        = verifier,
        applier         = applier,
        adapter         = adapter,
        mode            = "no_update",
        retrieval_noise = 0.10,
        rng_seed        = seed,
        max_retries     = 0,
    )
    return [agent.repair(fx) for fx in fixtures]


def run_oracle_sequential(
    fixtures:  List[Fixture],
    seed:      int,
    verifier,
    applier:   PatchApplier,
) -> List[RepairTrace006C]:
    """oracle: always uses the correct family procedure (upper bound)."""
    store   = _make_seeded_store(seed)
    adapter = OnlineEmbeddingAdapter()
    agent   = ProceduralRepairAgent006C(
        store           = store,
        verifier        = verifier,
        applier         = applier,
        adapter         = adapter,
        mode            = "oracle",
        retrieval_noise = 0.10,
        rng_seed        = seed,
        max_retries     = 0,
    )
    return [agent.repair(fx) for fx in fixtures]


# ── Top-level runner ──────────────────────────────────────────────────────

def run_all_baselines_006c(
    fixtures:  List[Fixture],
    seed:      int   = 0,
    timeout_s: float = 10.0,
    variants:  Optional[List[str]] = None,
    verifier   = None,
) -> Dict[str, List[RepairTrace006C]]:
    """
    Run the PSM-006C comparison set on all fixtures.

    Returns {variant_name: [RepairTrace006C]}.
    """
    from ..psm006b.pytest_verifier import PytestVerifier
    verifier = verifier or PytestVerifier(timeout=timeout_s)
    applier  = PatchApplier()
    to_run   = variants or VARIANT_NAMES_006C

    runners = {
        "full_memory_embedding_update": run_embedding_update_sequential,
        "full_memory":                  run_full_memory_sequential,
        "reset":                        run_reset_per_fixture,
        "no_update":                    run_no_update_sequential,
        "oracle":                       run_oracle_sequential,
    }

    results: Dict[str, List[RepairTrace006C]] = {}
    for v in to_run:
        if v not in runners:
            raise ValueError(f"Unknown PSM-006C variant: {v}")
        results[v] = runners[v](fixtures, seed, verifier, applier)
    return results
