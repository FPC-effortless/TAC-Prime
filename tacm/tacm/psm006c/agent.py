"""
TAC-PSM-006C: Procedural Repair Agent with Online Embedding Adaptation
=======================================================================

Extends PSM-006B's ProceduralRepairAgent006B with one additional mechanism:

    When the retrieved family is WRONG and verification fails, the
    OnlineEmbeddingAdapter updates the retrieval embeddings (push wrong
    record away, pull correct-family records toward the task) before
    retrying retrieval and repair.

Everything else — fixtures, verifier, patch applier, memory store, metrics
schema — is identical to PSM-006B.  This is a clean ablation.

Mode
----
  "full_memory_embedding_update"   TAC + text update + embedding update (NEW)

All other modes ("full_memory", "reset", "no_update", "oracle",
"retrieval_disabled", "random_procedure") delegate to the parent agent and
return a RepairTrace006C with default embedding fields (applied=False).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..psm006b.fixture_schema import Fixture, FAMILY_NAMES
from ..psm006b.pytest_verifier import PytestVerifier
from ..psm006b.patch_applier import PatchApplier
from ..psm006b.memory_store import SimpleProceduralMemoryStore, ProcedureRecord
from ..psm006b.procedural_repair_agent import (
    ProceduralRepairAgent006B,
    RepairTrace006B,
    fixture_embedding,
    oracle_procedure_dict,
    seed_procedural_memory,
    family_centroid,
    _classify_failure,
    EMBEDDING_DIM,
)
from .embedding_update import OnlineEmbeddingAdapter, EmbeddingUpdateRecord, _unit


# ── Extended trace ─────────────────────────────────────────────────────────

@dataclass
class RepairTrace006C(RepairTrace006B):
    """
    Extends RepairTrace006B with PSM-006C-specific embedding-update fields.

    Additional fields
    -----------------
    embedding_update_applied      : True if an embedding update was triggered
    embedding_shift_norm          : L2 shift of the updated embedding
    retrieval_changed_after_update: True if top-1 record changed after update
    family_changed_after_update   : True if retrieved family changed after update
    successful_retrieval_recovery : True if retrieval went wrong→correct after update
    """
    embedding_update_applied:        bool  = False
    embedding_shift_norm:            float = 0.0
    retrieval_changed_after_update:  bool  = False
    family_changed_after_update:     bool  = False
    successful_retrieval_recovery:   bool  = False

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "embedding_update_applied":        self.embedding_update_applied,
            "embedding_shift_norm":            self.embedding_shift_norm,
            "retrieval_changed_after_update":  self.retrieval_changed_after_update,
            "family_changed_after_update":     self.family_changed_after_update,
            "successful_retrieval_recovery":   self.successful_retrieval_recovery,
        })
        return d


# ── Agent ─────────────────────────────────────────────────────────────────

class ProceduralRepairAgent006C:
    """
    PSM-006C repair agent.

    Identical to PSM-006B for all modes except
    "full_memory_embedding_update", which adds online embedding adaptation
    after wrong-family retrieval failures.

    Parameters
    ----------
    store           : SimpleProceduralMemoryStore
    verifier        : PytestVerifier
    applier         : PatchApplier
    adapter         : OnlineEmbeddingAdapter (shared across fixtures in a run)
    mode            : one of the 006B modes OR "full_memory_embedding_update"
    retrieval_noise : Gaussian noise std on query embeddings
    rng_seed        : for reproducibility
    max_retries     : max update-and-retry cycles per fixture
    """

    NEW_MODE = "full_memory_embedding_update"

    def __init__(
        self,
        store:           SimpleProceduralMemoryStore,
        verifier:        PytestVerifier,
        applier:         PatchApplier,
        adapter:         OnlineEmbeddingAdapter,
        mode:            str   = "full_memory_embedding_update",
        retrieval_noise: float = 0.10,
        rng_seed:        int   = 0,
        max_retries:     int   = 1,
    ):
        self.store           = store
        self.verifier        = verifier
        self.applier         = applier
        self.adapter         = adapter
        self.mode            = mode
        self.retrieval_noise = retrieval_noise
        self.rng             = np.random.default_rng(rng_seed)
        self.max_retries     = max_retries

    def repair(self, fixture: Fixture) -> RepairTrace006C:
        """Run the full PSM-006C repair loop on one fixture."""
        t0        = time.time()
        all_files = fixture.all_files()
        task_emb  = fixture_embedding(fixture, self.rng)
        noisy_emb = self._add_noise(task_emb)

        # ── Before ────────────────────────────────────────────────────────
        before = self.verifier.run(
            all_files, fixture.verification_command,
            fixture_id=fixture.fixture_id, variant="before_patch",
        )

        # ── Retrieve ──────────────────────────────────────────────────────
        retrieved_family, retrieved_id, steps = self._retrieve(fixture, noisy_emb)
        retrieval_correct = (retrieved_family == fixture.family)

        # ── Apply & Verify ────────────────────────────────────────────────
        patch_result, patched_files = self._apply_patch(fixture, all_files, retrieved_family)
        after = self.verifier.run(
            patched_files, fixture.verification_command,
            fixture_id=fixture.fixture_id, variant="after_patch",
        )
        failure_class = _classify_failure(after.success, retrieval_correct, patch_result)

        # ── Update (text + optional embedding) ────────────────────────────
        procedure_updated        = False
        update_improved          = False
        emb_update_applied       = False
        emb_shift_norm           = 0.0
        retrieval_changed        = False
        family_changed           = False
        successful_recovery      = False

        if self.mode == self.NEW_MODE:
            if not after.success and not retrieval_correct and retrieved_id:
                # ── WRONG family retrieved → embedding update ──────────────
                ur = self.adapter.adapt_on_failure(
                    self.store, retrieved_id, task_emb, fixture.family
                )

                # Probe: did retrieval change after the update?
                new_noisy = self._add_noise(task_emb)
                pc, fc = self.adapter.check_retrieval_change(
                    self.store, new_noisy, retrieved_id, retrieved_family
                )

                # Find what the new top-1 family would be (for annotation)
                probe_records = self.store.retrieve(new_noisy, top_k=1)
                new_family = probe_records[0].family if probe_records else retrieved_family
                self.adapter.annotate_record(ur, pc, fc, new_family, fixture.family)

                emb_update_applied  = ur.applied
                emb_shift_norm      = ur.embedding_shift_norm
                retrieval_changed   = pc
                family_changed      = fc
                successful_recovery = ur.successful_recovery
                procedure_updated   = True

                # Retry with fresh noisy embedding after the update
                if self.max_retries > 0:
                    rf2, ri2, _    = self._retrieve(fixture, new_noisy)
                    pr2, pf2       = self._apply_patch(fixture, all_files, rf2)
                    after2         = self.verifier.run(
                        pf2, fixture.verification_command,
                        fixture_id=fixture.fixture_id, variant="after_emb_update_retry",
                    )
                    if after2.success and not after.success:
                        update_improved = True
                        after = after2

            elif not after.success and retrieval_correct and retrieved_id:
                # Correct family, patch failed → text augment only (same as 006B)
                self.store.augment(
                    retrieved_id,
                    fixture.oracle_repair_procedure.get("steps", [])
                )
                procedure_updated = True

                if self.max_retries > 0:
                    rf2, ri2, _ = self._retrieve(fixture, self._add_noise(task_emb))
                    pr2, pf2    = self._apply_patch(fixture, all_files, rf2)
                    after2      = self.verifier.run(
                        pf2, fixture.verification_command,
                        fixture_id=fixture.fixture_id, variant="after_text_update_retry",
                    )
                    if after2.success and not after.success:
                        update_improved = True
                        after = after2

            elif after.success and retrieved_id:
                # Success → reinforce text + embedding
                self.store.reinforce(retrieved_id, delta=0.05)
                ur = self.adapter.adapt_on_success(
                    self.store, retrieved_id, task_emb
                )
                emb_update_applied = ur.applied
                emb_shift_norm     = ur.embedding_shift_norm
                procedure_updated  = True

        elif self.mode not in ("no_update", "oracle", "retrieval_disabled", "reset"):
            # PSM-006B-style update (text only)
            if not after.success and retrieval_correct and retrieved_id:
                self.store.augment(retrieved_id, fixture.oracle_repair_procedure.get("steps", []))
                procedure_updated = True
                if self.max_retries > 0:
                    rf2, ri2, _ = self._retrieve(fixture, self._add_noise(task_emb))
                    pr2, pf2    = self._apply_patch(fixture, all_files, rf2)
                    after2      = self.verifier.run(
                        pf2, fixture.verification_command,
                        fixture_id=fixture.fixture_id, variant="after_update_retry",
                    )
                    if after2.success and not after.success:
                        update_improved = True
                        after = after2
            elif after.success and retrieved_id:
                self.store.reinforce(retrieved_id, delta=0.05)
                procedure_updated = True

        # ── Failure class (re-evaluate after potential retry) ──────────────
        if update_improved:
            failure_class = None

        elapsed = time.time() - t0
        return RepairTrace006C(
            fixture_id                   = fixture.fixture_id,
            family                       = fixture.family,
            retrieved_family             = retrieved_family,
            retrieved_proc_id            = retrieved_id,
            retrieval_correct            = retrieval_correct,
            patch_result                 = patch_result.to_dict() if patch_result else {},
            before_result                = before.to_dict(),
            after_result                 = after.to_dict(),
            pytest_pass                  = after.success,
            n_retries                    = 1 if update_improved else 0,
            steps_to_repair              = len(steps),
            procedure_updated            = procedure_updated,
            update_improved              = update_improved,
            mode                         = self.mode,
            failure_class                = failure_class,
            time_to_repair_s             = elapsed,
            embedding_update_applied     = emb_update_applied,
            embedding_shift_norm         = emb_shift_norm,
            retrieval_changed_after_update = retrieval_changed,
            family_changed_after_update  = family_changed,
            successful_retrieval_recovery = successful_recovery,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

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
        if self.mode == "oracle":
            proc = oracle_procedure_dict(fixture.family)
            return fixture.family, None, proc["steps"]
        if self.mode == "retrieval_disabled":
            fallback = FAMILY_NAMES[0]
            return fallback, None, oracle_procedure_dict(fallback)["steps"]
        if self.mode == "random_procedure":
            rand_fam = FAMILY_NAMES[int(self.rng.integers(0, len(FAMILY_NAMES)))]
            return rand_fam, None, oracle_procedure_dict(rand_fam)["steps"]

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
    ):
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
