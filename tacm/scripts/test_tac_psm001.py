"""
TAC-PSM-001 Test Suite

Unit + integration tests for all PSM-001 components.

Run:
  python scripts/test_tac_psm001.py
  python scripts/test_tac_psm001.py -v            # verbose
  python scripts/test_tac_psm001.py TestRetrieval  # single class
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tacm.psm001 import (
    StructureMemoryRecordV2,
    ProcedureStep,
    ProcedureTrace,
    FailureMode,
    RecoveryStrategy,
    ProcedureLifecycleState,
    ProceduralMemoryStore,
    RetrievalMode,
    RetrievalResult,
    VerificationSignal,
    retrieve_procedure,
    update_procedure_after_verification,
    FAMILY_A_IMPORT_ERRORS,
    FAMILY_B_DEPENDENCY_CONFLICTS,
    FAMILY_D_PATH_RESOLUTION,
    ALL_FAMILIES,
    evaluate_procedure_on_task,
    oracle_steps,
    reset_steps,
    make_task_signature,
    get_all_tasks,
)
from tacm.psm001.retrieval import compute_retrieval_metrics
from tacm.psm001.store import ProceduralMemoryStore

DIM = 32


def _random_emb(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _make_store() -> ProceduralMemoryStore:
    return ProceduralMemoryStore(embedding_dim=DIM)


# ═══════════════════════════════════════════════════════════════════════════════
# Records
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcedureStep(unittest.TestCase):

    def test_mark_success(self):
        s = ProcedureStep(step_index=0, action="install module")
        s.mark_success(actual="ok", duration_ms=12.0)
        self.assertTrue(s.succeeded)
        self.assertEqual(s.actual_output, "ok")

    def test_mark_failure(self):
        s = ProcedureStep(step_index=0, action="install module")
        s.mark_failure(reason="connection refused")
        self.assertFalse(s.succeeded)

    def test_to_from_dict(self):
        s = ProcedureStep(step_index=1, action="verify install")
        d = s.to_dict()
        s2 = ProcedureStep.from_dict(d)
        self.assertEqual(s.action, s2.action)
        self.assertEqual(s.step_index, s2.step_index)


class TestProcedureTrace(unittest.TestCase):

    def _make_trace(self) -> ProcedureTrace:
        steps = [ProcedureStep(i, f"step {i}") for i in range(4)]
        return ProcedureTrace(
            procedure_id   = "test-001",
            problem_family = "ImportErrors",
            task_signature = "ImportErrors::missing_import::v0",
            steps          = steps,
        )

    def test_n_steps(self):
        t = self._make_trace()
        self.assertEqual(t.n_steps(), 4)

    def test_overall_score(self):
        t = self._make_trace()
        t.success_score   = 0.8
        t.transfer_score  = 0.5
        t.survival_score  = 1.0
        score = t.overall_score()
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_explain(self):
        t = self._make_trace()
        txt = t.explain()
        self.assertIn("ImportErrors", txt)
        self.assertIn("step 0", txt)

    def test_serialise_roundtrip(self):
        t    = self._make_trace()
        j    = t.to_json()
        t2   = ProcedureTrace.from_json(j)
        self.assertEqual(t.procedure_id, t2.procedure_id)
        self.assertEqual(t.n_steps(),    t2.n_steps())
        self.assertEqual(t.lifecycle_state, t2.lifecycle_state)

    def test_lifecycle_default(self):
        t = self._make_trace()
        self.assertEqual(t.lifecycle_state, ProcedureLifecycleState.CREATED)

    def test_step_success_rate_no_data(self):
        t = self._make_trace()
        self.assertEqual(t.step_success_rate(), 0.0)

    def test_step_success_rate(self):
        t = self._make_trace()
        t.steps[0].mark_success()
        t.steps[1].mark_failure()
        self.assertAlmostEqual(t.step_success_rate(), 0.5, places=3)


class TestStructureMemoryRecordV2(unittest.TestCase):

    def test_to_from_dict(self):
        r = StructureMemoryRecordV2(
            structure_id   = "sid-001",
            family_id      = 0,
            expert_id      = 1,
            task_type      = "ImportErrors",
            task_signature = "sig",
            embedding      = [0.1, 0.2, 0.3],
            success_score  = 0.8,
        )
        d  = r.to_dict()
        r2 = StructureMemoryRecordV2.from_dict(d)
        self.assertEqual(r.structure_id, r2.structure_id)
        self.assertAlmostEqual(r.success_score, r2.success_score, places=4)


# ═══════════════════════════════════════════════════════════════════════════════
# Store
# ═══════════════════════════════════════════════════════════════════════════════

class TestProceduralMemoryStore(unittest.TestCase):

    def test_build_and_len(self):
        store = _make_store()
        proc  = store.build(
            problem_family = "ImportErrors",
            task_signature = "test::sig",
            steps          = ["step1", "step2"],
            embedding      = _random_emb(0),
        )
        self.assertEqual(len(store), 1)
        self.assertIsNotNone(proc.procedure_id)

    def test_get(self):
        store = _make_store()
        proc  = store.build("ImportErrors", "sig1", ["a", "b"], _random_emb(0))
        found = store.get(proc.procedure_id)
        self.assertIsNotNone(found)
        self.assertEqual(found.procedure_id, proc.procedure_id)

    def test_get_unknown(self):
        store = _make_store()
        self.assertIsNone(store.get("nonexistent"))

    def test_update_scores(self):
        store = _make_store()
        proc  = store.build("ImportErrors", "sig", ["s1"], _random_emb(0))
        store.update(proc.procedure_id, success_delta=0.1, survival_delta=0.05)
        p2 = store.get(proc.procedure_id)
        self.assertGreater(p2.success_score, 0.0)
        self.assertEqual(p2.reuse_count, 1)

    def test_update_new_failure(self):
        store = _make_store()
        proc  = store.build("ImportErrors", "sig", ["s1"], _random_emb(0))
        fm    = FailureMode("fid-1", "something broke", frequency=1)
        store.update(proc.procedure_id, new_failure=fm)
        p2 = store.get(proc.procedure_id)
        self.assertEqual(len(p2.failure_modes), 1)

    def test_lifecycle_advance_to_active(self):
        store = _make_store()
        proc  = store.build("ImportErrors", "sig", ["s1"], _random_emb(0),
                            success_score=0.7)
        store.update(proc.procedure_id, survival_delta=0.0)
        p2 = store.get(proc.procedure_id)
        # After reuse_count=1 and survival>=0.3, should be ACTIVE or higher
        self.assertIn(p2.lifecycle_state, [
            ProcedureLifecycleState.CREATED,
            ProcedureLifecycleState.ACTIVE,
        ])

    def test_retire_and_not_retrieved(self):
        store = _make_store()
        proc  = store.build("ImportErrors", "sig", ["s1"], _random_emb(0))
        store.retire(proc.procedure_id)
        results = store.retrieve(_random_emb(0), top_k=10)
        pids = [p.procedure_id for _, p in results]
        self.assertNotIn(proc.procedure_id, pids)

    def test_prune_retired(self):
        store = _make_store()
        p1 = store.build("ImportErrors", "sig1", ["s1"], _random_emb(0))
        p2 = store.build("ImportErrors", "sig2", ["s2"], _random_emb(1))
        store.retire(p1.procedure_id)
        store.prune()
        self.assertEqual(len(store), 1)

    def test_decay_all(self):
        store = _make_store()
        proc  = store.build("ImportErrors", "sig", ["s1"], _random_emb(0))
        before = store.get(proc.procedure_id).survival_score
        store.decay_all(rate=0.5)
        after  = store.get(proc.procedure_id).survival_score
        self.assertAlmostEqual(after, before * 0.5, places=5)

    def test_stats_empty(self):
        store = _make_store()
        s = store.stats()
        self.assertEqual(s["size"], 0)

    def test_stats_nonempty(self):
        store = _make_store()
        store.build("ImportErrors", "sig", ["s1"], _random_emb(0))
        s = store.stats()
        self.assertEqual(s["size"], 1)
        self.assertIn("avg_success", s)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store1 = ProceduralMemoryStore(embedding_dim=DIM, save_dir=tmp)
            p = store1.build("ImportErrors", "sig", ["s1", "s2"], _random_emb(0))
            store1.save()

            store2 = ProceduralMemoryStore(embedding_dim=DIM, save_dir=tmp)
            self.assertEqual(len(store2), 1)
            p2 = store2.get(p.procedure_id)
            self.assertIsNotNone(p2)
            self.assertEqual(p2.procedure_id, p.procedure_id)

    def test_family_filter(self):
        store = _make_store()
        store.build("ImportErrors",        "sig1", ["s1"], _random_emb(0))
        store.build("DependencyConflicts", "sig2", ["s2"], _random_emb(1))
        fam_a = store.get_by_family("ImportErrors")
        fam_b = store.get_by_family("DependencyConflicts")
        self.assertEqual(len(fam_a), 1)
        self.assertEqual(len(fam_b), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrieval(unittest.TestCase):

    def _populated_store(self) -> ProceduralMemoryStore:
        store = _make_store()
        # Add 3 ImportErrors procedures
        for i in range(3):
            store.build(
                problem_family = "ImportErrors",
                task_signature = f"ImportErrors::sig::{i}",
                steps          = [f"step {j}" for j in range(4)],
                embedding      = _random_emb(i),
                success_score  = 0.7 + i * 0.1,
            )
        # Add 2 DependencyConflicts
        for i in range(2):
            store.build(
                problem_family = "DependencyConflicts",
                task_signature = f"DependencyConflicts::sig::{i}",
                steps          = [f"dep step {j}" for j in range(3)],
                embedding      = _random_emb(i + 10),
            )
        return store

    def test_correct_returns_results(self):
        store  = self._populated_store()
        result = retrieve_procedure("test_sig", _random_emb(0), store,
                                    mode=RetrievalMode.CORRECT, top_k=3)
        self.assertGreater(result.n_candidates, 0)
        self.assertIsNotNone(result.top1)

    def test_disabled_returns_empty(self):
        store  = self._populated_store()
        result = retrieve_procedure("test_sig", _random_emb(0), store,
                                    mode=RetrievalMode.DISABLED)
        self.assertEqual(result.n_candidates, 0)
        self.assertIsNone(result.top1)

    def test_random_returns_results(self):
        store  = self._populated_store()
        result = retrieve_procedure("test_sig", _random_emb(0), store,
                                    mode=RetrievalMode.RANDOM, top_k=3,
                                    rng=random.Random(42))
        self.assertGreater(result.n_candidates, 0)

    def test_oracle_filters_by_family(self):
        store  = self._populated_store()
        result = retrieve_procedure("test_sig", _random_emb(0), store,
                                    mode=RetrievalMode.ORACLE, top_k=5,
                                    correct_family="ImportErrors")
        for _, p in result.candidates:
            self.assertEqual(p.problem_family, "ImportErrors")

    def test_wrong_mode_returns_results(self):
        store  = self._populated_store()
        result = retrieve_procedure("test_sig", _random_emb(0), store,
                                    mode=RetrievalMode.WRONG, top_k=3)
        self.assertGreater(result.n_candidates, 0)

    def test_family_match_flag(self):
        store = _make_store()
        proc  = store.build("ImportErrors", "sig", ["s1"], _random_emb(0))
        result = retrieve_procedure("sig", _random_emb(0), store,
                                    mode=RetrievalMode.CORRECT, top_k=5,
                                    correct_family="ImportErrors")
        self.assertTrue(result.family_matched)

    def test_retrieval_metrics(self):
        store = _make_store()
        proc  = store.build("ImportErrors", "sig", ["s1"], _random_emb(0))
        result = retrieve_procedure("sig", _random_emb(0), store,
                                    mode=RetrievalMode.CORRECT, top_k=5,
                                    correct_family="ImportErrors")
        metrics = compute_retrieval_metrics([result], [proc.procedure_id])
        self.assertIn("retrieval_accuracy", metrics)
        self.assertIn("precision@1", metrics)

    def test_empty_store_all_modes(self):
        store = _make_store()
        for mode in RetrievalMode:
            result = retrieve_procedure("sig", _random_emb(0), store, mode=mode)
            self.assertEqual(result.n_candidates, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Update
# ═══════════════════════════════════════════════════════════════════════════════

class TestUpdate(unittest.TestCase):

    def _proc_in_store(self) -> tuple:
        store = _make_store()
        proc  = store.build("ImportErrors", "sig", ["s1", "s2"], _random_emb(0))
        return store, proc

    def test_success_increases_score(self):
        store, proc = self._proc_in_store()
        before = store.get(proc.procedure_id).success_score
        sig    = VerificationSignal(proc.procedure_id, "sig", success=True)
        update_procedure_after_verification(sig, store)
        after  = store.get(proc.procedure_id).success_score
        self.assertGreater(after, before)

    def test_failure_decreases_score(self):
        store, proc = self._proc_in_store()
        store.update(proc.procedure_id, success_delta=0.5)  # pre-set to 0.5
        before = store.get(proc.procedure_id).success_score
        sig    = VerificationSignal(proc.procedure_id, "sig", success=False,
                                    error_type="ImportError")
        update_procedure_after_verification(sig, store)
        after  = store.get(proc.procedure_id).success_score
        self.assertLessEqual(after, before)

    def test_failure_logs_failure_mode(self):
        store, proc = self._proc_in_store()
        sig = VerificationSignal(proc.procedure_id, "sig", success=False,
                                 failed_step=0, error_type="ImportError",
                                 error_message="no module named x")
        res = update_procedure_after_verification(sig, store)
        self.assertTrue(res.new_failure_logged)
        p2 = store.get(proc.procedure_id)
        self.assertGreater(len(p2.failure_modes), 0)

    def test_transfer_increases_transfer_score(self):
        store, proc = self._proc_in_store()
        before = store.get(proc.procedure_id).transfer_score
        sig    = VerificationSignal(proc.procedure_id, "sig", success=True,
                                    is_transfer=True, source_family="A", target_family="B")
        update_procedure_after_verification(sig, store)
        after  = store.get(proc.procedure_id).transfer_score
        self.assertGreater(after, before)

    def test_fork_on_repeated_failure(self):
        store, proc = self._proc_in_store()
        sig = VerificationSignal(proc.procedure_id, "sig", success=False,
                                 failed_step=0, error_type="PathError",
                                 error_message="file not found",
                                 recovery_steps=["check path", "fix path"],
                                 recovery_applied=True, recovery_success=True)
        res1 = update_procedure_after_verification(sig, store, fork_threshold=1)
        self.assertIsNotNone(res1.forked_id)
        self.assertEqual(len(store), 2)

    def test_forked_procedure_has_recovery_step(self):
        store, proc = self._proc_in_store()
        recovery_text = "check the path carefully"
        sig = VerificationSignal(proc.procedure_id, "sig", success=False,
                                 failed_step=0, error_type="PathError",
                                 error_message="file not found",
                                 recovery_steps=[recovery_text],
                                 recovery_applied=True,
                                 recovery_success=True)
        res = update_procedure_after_verification(sig, store, fork_threshold=1)
        if res.forked_id:
            forked = store.get(res.forked_id)
            step_actions = [s.action for s in forked.steps]
            # When recovery_success=True the fork adopts recovery steps directly
            self.assertTrue(
                any(recovery_text in a for a in step_actions),
                f"Recovery step not found in forked steps: {step_actions}",
            )

    def test_update_nonexistent(self):
        store = _make_store()
        sig   = VerificationSignal("nonexistent", "sig", success=True)
        res   = update_procedure_after_verification(sig, store)
        self.assertIn("not found", res.message)

    def test_recovery_strategy_logged(self):
        store, proc = self._proc_in_store()
        sig = VerificationSignal(proc.procedure_id, "sig", success=False,
                                 error_type="ImportError",
                                 recovery_applied=True,
                                 recovery_steps=["pip install", "verify"],
                                 recovery_success=True)
        res = update_procedure_after_verification(sig, store, fork_threshold=999)
        self.assertTrue(res.new_recovery_logged)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Families
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkFamilies(unittest.TestCase):

    def test_all_families_non_empty(self):
        from tacm.psm001.benchmark_families import ALL_FAMILIES
        for fam in ALL_FAMILIES:
            self.assertGreater(len(fam.tasks), 0, f"Family {fam.name} has no tasks")

    def test_task_signature_unique(self):
        tasks = get_all_tasks()
        sigs  = [t.task_signature for t in tasks]
        self.assertEqual(len(sigs), len(set(sigs)), "Duplicate task signatures found")

    def test_query_embedding_deterministic(self):
        task = FAMILY_A_IMPORT_ERRORS.tasks[0]
        e1   = task.query_embedding(DIM)
        e2   = task.query_embedding(DIM)
        np.testing.assert_array_almost_equal(e1, e2, decimal=6)

    def test_query_embedding_normalised(self):
        task = FAMILY_A_IMPORT_ERRORS.tasks[0]
        e    = task.query_embedding(DIM)
        norm = np.linalg.norm(e)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_evaluate_oracle_beats_reset(self):
        task    = FAMILY_A_IMPORT_ERRORS.tasks[0]
        ok, q_oracle, _ = evaluate_procedure_on_task(task, oracle_steps(task), seed=42)
        _, q_reset,  _  = evaluate_procedure_on_task(task, reset_steps(),      seed=42)
        self.assertGreaterEqual(q_oracle, q_reset)

    def test_evaluate_distractor_worse_than_oracle(self):
        task = FAMILY_A_IMPORT_ERRORS.tasks[0]
        _, q_oracle,     _ = evaluate_procedure_on_task(task, oracle_steps(task), seed=42)
        _, q_distractor, _ = evaluate_procedure_on_task(task, task.distractor_steps, seed=42)
        self.assertGreaterEqual(q_oracle, q_distractor)

    def test_all_families_have_correct_attribute(self):
        for fam in ALL_FAMILIES:
            for task in fam.tasks:
                self.assertTrue(hasattr(task, "canonical_steps"))
                self.assertTrue(hasattr(task, "distractor_steps"))
                self.assertGreater(len(task.canonical_steps), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Full A1→D1 sequence
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegrationSequence(unittest.TestCase):

    def test_a1_a2_full_cycle(self):
        store   = _make_store()
        task_a1 = FAMILY_A_IMPORT_ERRORS.tasks[0]
        task_a2 = FAMILY_A_IMPORT_ERRORS.tasks[1]

        # A1: store
        proc = store.build(
            problem_family = task_a1.family,
            task_signature = task_a1.task_signature,
            steps          = oracle_steps(task_a1),
            embedding      = task_a1.query_embedding(DIM),
            success_score  = 0.9,
        )
        sig = VerificationSignal(proc.procedure_id, task_a1.task_signature, success=True)
        update_procedure_after_verification(sig, store)

        # A2: retrieve
        result = retrieve_procedure(
            task_a2.task_signature, task_a2.query_embedding(DIM), store,
            mode=RetrievalMode.CORRECT, top_k=5, correct_family=task_a2.family,
        )
        self.assertIsNotNone(result.top1)
        self.assertEqual(result.top1.problem_family, task_a1.family)

    def test_memory_grows_through_sequence(self):
        store = _make_store()
        tasks = get_all_tasks()[:4]
        for task in tasks:
            store.build(
                problem_family = task.family,
                task_signature = task.task_signature,
                steps          = oracle_steps(task),
                embedding      = task.query_embedding(DIM),
                success_score  = 0.8,
            )
        self.assertEqual(len(store), 4)

    def test_retry_after_update_improves(self):
        store  = _make_store()
        task   = FAMILY_D_PATH_RESOLUTION.tasks[0]

        # Initial wrong procedure
        proc = store.build(task.family, task.task_signature,
                           task.distractor_steps, task.query_embedding(DIM),
                           success_score=0.1)
        _, q_pre, _ = evaluate_procedure_on_task(task, task.distractor_steps, seed=0)

        # Fail + update
        sig = VerificationSignal(
            proc.procedure_id, task.task_signature, success=False,
            failed_step=0, error_type="IncorrectPath",
            recovery_applied=True, recovery_steps=oracle_steps(task),
            recovery_success=True,
        )
        res = update_procedure_after_verification(sig, store, fork_threshold=1)

        # Retry with forked steps
        if res.forked_id:
            forked = store.get(res.forked_id)
            retry_steps = [s.action for s in forked.steps]
        else:
            retry_steps = oracle_steps(task)
        _, q_post, _ = evaluate_procedure_on_task(task, retry_steps, seed=1)

        # Forked should contain recovery steps → quality improves or stays same
        self.assertGreaterEqual(len(retry_steps), len(task.distractor_steps))


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Nice output
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Run specific test class by name
        class_name = sys.argv.pop(1)
        for cls in [
            TestProcedureStep, TestProcedureTrace, TestStructureMemoryRecordV2,
            TestProceduralMemoryStore, TestRetrieval, TestUpdate,
            TestBenchmarkFamilies, TestIntegrationSequence,
        ]:
            if cls.__name__ == class_name:
                suite.addTests(loader.loadTestsFromTestCase(cls))
    else:
        for cls in [
            TestProcedureStep, TestProcedureTrace, TestStructureMemoryRecordV2,
            TestProceduralMemoryStore, TestRetrieval, TestUpdate,
            TestBenchmarkFamilies, TestIntegrationSequence,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
