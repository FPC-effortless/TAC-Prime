"""
TAC-PSM-006B: Procedural Repair Agent (Pytest-Grounded)
=========================================================

Implements the full TAC retrieve-apply-verify-update loop for real
pytest-verified repository repair.

Loop (per fixture):
  1. RETRIEVE  — query SimpleProceduralMemoryStore for best matching procedure
  2. APPLY     — generate patch from retrieved procedure family
  3. VERIFY    — run pytest in isolated temp dir; capture exit code
  4. UPDATE    — strengthen/augment procedure based on verification outcome
  5. RECORD    — emit RepairTrace006B for metrics collection

Key difference from PSM-006:
  Verification is real pytest execution (PytestVerifier) rather than a
  heuristic composite score.  Success = exit code 0.

Procedural memory embedding:
  Each fixture is embedded as a d_model-dimensional vector computed from
  its family, difficulty, and bug_report text hash.  This gives each family
  a distinct centroid that the retrieval system can distinguish.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .fixture_schema import Fixture, FAMILY_NAMES
from .pytest_verifier import PytestVerifier, PytestResult
from .patch_applier import PatchApplier, PatchResult
from .memory_store import SimpleProceduralMemoryStore, ProcedureRecord


# ── Constants ──────────────────────────────────────────────────────────────

EMBEDDING_DIM = 64

_FAMILY_CENTROID_SEED: Dict[str, int] = {
    fam: i for i, fam in enumerate(FAMILY_NAMES)
}


# ── Trace ─────────────────────────────────────────────────────────────────

@dataclass
class RepairTrace006B:
    """
    Full record of one agent repair attempt on a single PSM-006B fixture.

    Attributes
    ----------
    fixture_id           : links to Fixture
    family               : fixture's true repair family
    retrieved_family     : which family's procedure was retrieved (may differ)
    retrieved_proc_id    : ID of the retrieved procedure record
    retrieval_correct    : True if retrieved_family == family
    patch_result         : outcome of patch application (dict)
    before_result        : pytest result before patch (should fail)
    after_result         : pytest result after patch (should succeed iff correct)
    pytest_pass          : True if after_result.success
    n_retries            : number of update-and-retry cycles performed
    steps_to_repair      : total procedure steps applied
    procedure_updated    : True if memory was updated after this attempt
    update_improved      : True if a second attempt after update succeeded
    mode                 : "full_memory" | "reset" | ...
    failure_class        : one of the PSM-006B failure classes or None
    time_to_repair_s     : wall-clock seconds for the full loop
    """
    fixture_id:        str
    family:            str
    retrieved_family:  str
    retrieved_proc_id: Optional[str]
    retrieval_correct: bool
    patch_result:      dict
    before_result:     dict
    after_result:      dict
    pytest_pass:       bool
    n_retries:         int
    steps_to_repair:   int
    procedure_updated: bool
    update_improved:   bool
    mode:              str
    failure_class:     Optional[str]
    time_to_repair_s:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "fixture_id":        self.fixture_id,
            "family":            self.family,
            "retrieved_family":  self.retrieved_family,
            "retrieved_proc_id": self.retrieved_proc_id,
            "retrieval_correct": self.retrieval_correct,
            "patch_result":      self.patch_result,
            "before_result":     self.before_result,
            "after_result":      self.after_result,
            "pytest_pass":       self.pytest_pass,
            "n_retries":         self.n_retries,
            "steps_to_repair":   self.steps_to_repair,
            "procedure_updated": self.procedure_updated,
            "update_improved":   self.update_improved,
            "mode":              self.mode,
            "failure_class":     self.failure_class,
            "time_to_repair_s":  self.time_to_repair_s,
        }


# ── Embedding helpers ─────────────────────────────────────────────────────

def fixture_embedding(fixture: Fixture, rng: np.random.Generator) -> np.ndarray:
    """
    Compute a deterministic embedding for a fixture.

    The embedding has a family-specific centroid (so the same family's
    fixtures cluster together) plus jitter from the fixture_id hash (so
    different fixtures within a family are distinguishable).

    Returns a unit-normalised float32 vector of shape (EMBEDDING_DIM,).
    """
    centroid = family_centroid(fixture.family)

    id_hash = int(hashlib.sha256(fixture.fixture_id.encode()).hexdigest(), 16)
    jit_rng = np.random.default_rng(id_hash % (2 ** 31))
    jitter  = jit_rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.3

    emb   = centroid + jitter
    emb  /= np.linalg.norm(emb) + 1e-8
    return emb


def family_centroid(family: str) -> np.ndarray:
    """Return the clean centroid embedding for a family (no jitter)."""
    fam_seed = _FAMILY_CENTROID_SEED.get(family, 0)
    rng      = np.random.default_rng(fam_seed * 1000)
    c        = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    c       /= np.linalg.norm(c) + 1e-8
    return c


def oracle_procedure_dict(family: str) -> dict:
    """Return the oracle procedure dict for a given family."""
    _procedures: Dict[str, dict] = {
        "import_module_error": {
            "family": "import_module_error",
            "steps":  ["identify_import_error", "locate_missing_symbol",
                       "add_alias_or_rename_symbol", "verify_import_resolves"],
        },
        "dependency_config_conflict": {
            "family": "dependency_config_conflict",
            "steps":  ["identify_fixture_conflict", "isolate_fixture_scope_or_definition",
                       "resolve_conflicting_definition", "verify_fixtures_resolve"],
        },
        "version_api_mismatch": {
            "family": "version_api_mismatch",
            "steps":  ["identify_api_change", "locate_deprecated_call_signature",
                       "update_call_to_new_signature", "verify_api_call_succeeds"],
        },
        "path_module_resolution": {
            "family": "path_module_resolution",
            "steps":  ["identify_module_path_error", "locate_incorrect_import_path",
                       "correct_import_path_or_sys_path", "verify_module_importable"],
        },
        "configuration_failure": {
            "family": "configuration_failure",
            "steps":  ["identify_configuration_error", "locate_config_file_and_key",
                       "correct_config_value_or_structure", "verify_config_loads_correctly"],
        },
        "test_assertion_repair": {
            "family": "test_assertion_repair",
            "steps":  ["identify_assertion_failure", "locate_failing_assertion",
                       "correct_expected_value_or_logic", "verify_assertion_passes"],
        },
    }
    return _procedures.get(family, {"family": family, "steps": []})


# ── Memory seeding ────────────────────────────────────────────────────────

def seed_procedural_memory(
    store:                 SimpleProceduralMemoryStore,
    n_records_per_family:  int = 2,
    rng_seed:              int = 0,
) -> None:
    """
    Pre-populate the store with oracle procedures at each family centroid.

    n_records_per_family: multiple records per family add diversity and
    reduce single-record retrieval variance.
    """
    rng = np.random.default_rng(rng_seed)
    for fam in FAMILY_NAMES:
        proc = oracle_procedure_dict(fam)
        for rep in range(n_records_per_family):
            centroid = family_centroid(fam)
            jitter   = rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.05
            emb      = centroid + jitter
            emb     /= np.linalg.norm(emb) + 1e-8
            store.write(
                family       = fam,
                task_type    = f"pytest_fixture_{fam}",
                steps        = proc["steps"],
                embedding    = emb,
                success_rate = 0.75 + 0.1 * rep,
            )


# ── Main agent ────────────────────────────────────────────────────────────

class ProceduralRepairAgent006B:
    """
    TAC-PSM-006B procedural repair agent.

    Wraps a SimpleProceduralMemoryStore and uses PytestVerifier + PatchApplier
    to run the full retrieve-apply-verify-update loop against real pytest
    fixtures.

    Parameters
    ----------
    store           : SimpleProceduralMemoryStore shared across fixtures in a run
    verifier        : PytestVerifier instance
    applier         : PatchApplier instance
    mode            : one of:
                        "full_memory" | "reset" | "retrieval_disabled" |
                        "random_procedure" | "structure_only" |
                        "no_update" | "oracle"
    retrieval_noise : std of Gaussian noise added to query embedding (default 0.1)
    rng_seed        : random seed for reproducibility
    max_retries     : max update-and-retry cycles per fixture (default 1)
    """

    def __init__(
        self,
        store:           SimpleProceduralMemoryStore,
        verifier:        PytestVerifier,
        applier:         PatchApplier,
        mode:            str   = "full_memory",
        retrieval_noise: float = 0.1,
        rng_seed:        int   = 0,
        max_retries:     int   = 1,
    ):
        self.store           = store
        self.verifier        = verifier
        self.applier         = applier
        self.mode            = mode
        self.retrieval_noise = retrieval_noise
        self.rng             = np.random.default_rng(rng_seed)
        self.max_retries     = max_retries

    def repair(self, fixture: Fixture) -> RepairTrace006B:
        """Run the full repair loop on one fixture and return a RepairTrace006B."""
        t0      = time.time()
        all_files = fixture.all_files()
        emb       = fixture_embedding(fixture, self.rng)
        noisy_emb = self._add_noise(emb)

        # ── 1. Verify "before" (fixture should fail before patch) ──────
        before = self.verifier.run(
            all_files, fixture.verification_command,
            fixture_id=fixture.fixture_id, variant="before_patch",
        )

        # ── 2. Retrieve procedure ──────────────────────────────────────
        retrieved_family, retrieved_id, steps = self._retrieve(fixture, noisy_emb)
        retrieval_correct = (retrieved_family == fixture.family)

        # ── 3. Generate and apply patch ────────────────────────────────
        patch_result, patched_files = self._apply_patch(
            fixture, all_files, retrieved_family
        )

        # ── 4. Verify "after" ──────────────────────────────────────────
        after = self.verifier.run(
            patched_files, fixture.verification_command,
            fixture_id=fixture.fixture_id, variant="after_patch",
        )

        failure_class = _classify_failure(after.success, retrieval_correct, patch_result)

        # ── 5. Update memory ───────────────────────────────────────────
        procedure_updated = False
        update_improved   = False

        if self.mode not in ("no_update", "oracle", "retrieval_disabled", "reset"):
            if not after.success and retrieval_correct and retrieved_id:
                # Correct family retrieved but patch failed → augment with oracle
                self.store.augment(retrieved_id, fixture.oracle_repair_procedure.get("steps", []))
                procedure_updated = True

                # Retry once after update
                if self.max_retries > 0:
                    new_emb   = self._add_noise(emb)
                    rf2, ri2, _ = self._retrieve(fixture, new_emb)
                    pr2, pf2  = self._apply_patch(fixture, all_files, rf2)
                    after2    = self.verifier.run(
                        pf2, fixture.verification_command,
                        fixture_id=fixture.fixture_id, variant="after_update_retry",
                    )
                    if after2.success and not after.success:
                        update_improved = True
                        after = after2
            elif after.success and retrieved_id:
                self.store.reinforce(retrieved_id, delta=0.05)
                procedure_updated = True

        elapsed = time.time() - t0
        return RepairTrace006B(
            fixture_id        = fixture.fixture_id,
            family            = fixture.family,
            retrieved_family  = retrieved_family,
            retrieved_proc_id = retrieved_id,
            retrieval_correct = retrieval_correct,
            patch_result      = patch_result.to_dict() if patch_result else {},
            before_result     = before.to_dict(),
            after_result      = after.to_dict(),
            pytest_pass       = after.success,
            n_retries         = 1 if update_improved else 0,
            steps_to_repair   = len(steps),
            procedure_updated = procedure_updated,
            update_improved   = update_improved,
            mode              = self.mode,
            failure_class     = failure_class,
            time_to_repair_s  = elapsed,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _add_noise(self, emb: np.ndarray) -> np.ndarray:
        noise  = self.rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        noisy  = emb + noise * self.retrieval_noise
        noisy /= np.linalg.norm(noisy) + 1e-8
        return noisy

    def _retrieve(
        self,
        fixture:   Fixture,
        noisy_emb: np.ndarray,
    ) -> Tuple[str, Optional[str], List[str]]:
        """Return (retrieved_family, proc_id_or_None, steps)."""
        if self.mode == "oracle":
            proc = oracle_procedure_dict(fixture.family)
            return fixture.family, None, proc["steps"]

        if self.mode == "retrieval_disabled":
            fallback = FAMILY_NAMES[0]
            proc = oracle_procedure_dict(fallback)
            return fallback, None, proc["steps"]

        if self.mode == "random_procedure":
            rand_fam = FAMILY_NAMES[int(self.rng.integers(0, len(FAMILY_NAMES)))]
            proc     = oracle_procedure_dict(rand_fam)
            return rand_fam, None, proc["steps"]

        records = self.store.retrieve(noisy_emb, top_k=1)
        if not records:
            fallback = FAMILY_NAMES[0]
            return fallback, None, oracle_procedure_dict(fallback)["steps"]

        rec = records[0]
        return rec.family, rec.proc_id, rec.steps

    def _apply_patch(
        self,
        fixture:          Fixture,
        all_files:        Dict[str, str],
        retrieved_family: str,
    ) -> Tuple[PatchResult, Dict[str, str]]:
        """Generate and apply patch based on retrieved procedure family."""
        if retrieved_family == fixture.family:
            patch_result = self.applier.apply(all_files, fixture.expected_patch)
        elif self.mode == "structure_only":
            patch_result = self.applier.apply_structure_only_patch(
                all_files, fixture.expected_patch
            )
        else:
            patch_result = self.applier.apply_wrong_family_patch(
                all_files, retrieved_family
            )
        return patch_result, patch_result.patched_files


# ── Failure classification ────────────────────────────────────────────────

def _classify_failure(
    success:           bool,
    retrieval_correct: bool,
    patch_result:      Optional[PatchResult],
) -> Optional[str]:
    if success:
        return None
    if not retrieval_correct:
        return "wrong_procedure_retrieval"
    if patch_result and not patch_result.success:
        return patch_result.failure_class or "correct_procedure_wrong_patch"
    return "correct_procedure_wrong_patch"
