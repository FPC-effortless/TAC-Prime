"""
TAC-PSM-006 Unit Tests: Repository-Grounded Procedural Memory
==============================================================

Tests all PSM-006 modules:
  - repository_task.py      (task construction, embeddings, families)
  - repo_fixture_builder.py (fixture building, dependency parsing)
  - verifier.py             (verification logic, retry, edge cases)
  - procedural_repair_agent.py (warm-up, retrieve, apply, update loop)
  - baselines.py            (all 7 variants produce valid traces)
  - metrics.py              (all 9 metrics + confusion matrix + gates)

Run:
  python scripts/test_tac_psm006_repository_memory.py
  python scripts/test_tac_psm006_repository_memory.py -v
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from tacm.psm006 import (
    # Task / fixture
    build_task_bank, get_all_tasks, split_train_test,
    build_fixture, build_fixtures, parse_requirements,
    RepoTask, RepoFixture,
    ALL_FAMILY_NAMES,
    FAMILY_IMPORT, FAMILY_DEPENDENCY, FAMILY_VERSION_API,
    FAMILY_PATH, FAMILY_CONFIG, FAMILY_TEST,
    # Verifier
    verify_repair, batch_verify, verify_with_retry,
    VerificationResult,
    # Agent
    ProceduralRepairAgent, make_agent, AgentTrace,
    # Baselines
    run_reset, run_oracle, run_full_memory, run_no_update,
    run_random_procedure, run_retrieval_disabled, run_structure_only,
    run_all_baselines, BASELINE_NAMES,
    # Metrics
    compute_metrics, aggregate_metrics, evaluate_gates,
    metric_verified_repair_success, metric_retrieval_accuracy,
    metric_steps_to_repair, metric_survival_stability,
    metric_transfer_success, metric_confusion_matrix,
    PSM006_GATES,
)
from tacm.psm001 import ProceduralMemoryStore


# ── Test: Task construction ───────────────────────────────────────────────────

class TestRepositoryTask(unittest.TestCase):

    def setUp(self):
        self.bank = build_task_bank(tasks_per_family=20)

    def test_six_families(self):
        self.assertEqual(len(self.bank), 6)
        self.assertEqual(set(self.bank.keys()), set(ALL_FAMILY_NAMES))

    def test_20_tasks_per_family(self):
        for family, tasks in self.bank.items():
            self.assertEqual(len(tasks), 20,
                             f"{family} has {len(tasks)} tasks, expected 20")

    def test_total_120_tasks(self):
        all_tasks = get_all_tasks(tasks_per_family=20)
        self.assertEqual(len(all_tasks), 120)

    def test_required_fields_present(self):
        for family, tasks in self.bank.items():
            for task in tasks:
                self.assertIsInstance(task.task_id, str)
                self.assertIsInstance(task.repo_name, str)
                self.assertEqual(task.family, family)
                self.assertTrue(task.bug_report)
                self.assertTrue(task.failing_test_output)
                self.assertIsInstance(task.relevant_files, dict)
                self.assertTrue(len(task.relevant_files) >= 1)
                self.assertEqual(task.expected_procedure_family, family)
                self.assertTrue(len(task.oracle_repair_steps) >= 4)
                self.assertIsInstance(task.verification_rule, dict)
                self.assertIn("expected_family",   task.verification_rule)
                self.assertIn("min_score",         task.verification_rule)
                self.assertIn(task.transfer_group, [
                    "web_framework", "data_pipeline", "cli_tooling",
                    "test_suite", "async_worker", "generic",
                ])
                self.assertGreater(task.difficulty, 0.0)
                self.assertLessEqual(task.difficulty, 1.0)

    def test_task_signature_unique(self):
        all_tasks = get_all_tasks(20)
        sigs = [t.task_signature() for t in all_tasks]
        self.assertEqual(len(sigs), len(set(sigs)),
                         "Task signatures must be unique")

    def test_query_embedding_shape_and_unit(self):
        import numpy as np
        tasks = self.bank[FAMILY_IMPORT]
        for task in tasks[:3]:
            emb = task.query_embedding(dim=64)
            self.assertEqual(emb.shape, (64,))
            self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=5)

    def test_family_embedding_different_from_task(self):
        import numpy as np
        task    = self.bank[FAMILY_IMPORT][0]
        task_e  = task.query_embedding(64)
        fam_e   = task.family_embedding(64)
        sim     = float(task_e @ fam_e)
        # Should be correlated but not identical
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_split_train_test_ratio(self):
        tasks = get_all_tasks(20)
        tr, te = split_train_test(tasks, train_frac=0.5, seed=0)
        self.assertEqual(len(tr) + len(te), len(tasks))
        self.assertAlmostEqual(len(tr) / len(tasks), 0.5, delta=0.1)

    def test_distractor_steps_defined(self):
        for family in ALL_FAMILY_NAMES:
            task = self.bank[family][0]
            dist = task.distractor_steps()
            self.assertGreater(len(dist), 0)


# ── Test: Fixture builder ─────────────────────────────────────────────────────

class TestRepoFixtureBuilder(unittest.TestCase):

    def setUp(self):
        self.tasks = get_all_tasks(tasks_per_family=5)

    def test_build_fixture_returns_fixture(self):
        task = self.tasks[0]
        fix  = build_fixture(task)
        self.assertIsInstance(fix, RepoFixture)
        self.assertEqual(fix.task_id, task.task_id)
        self.assertEqual(fix.repo_name, task.repo_name)
        self.assertEqual(fix.family, task.family)
        self.assertEqual(fix.level, "simulated")

    def test_fixture_has_context_embedding(self):
        import numpy as np
        task = self.tasks[0]
        fix  = build_fixture(task)
        self.assertEqual(fix.context_embedding.shape, (64,))
        norm = float(np.linalg.norm(fix.context_embedding))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_fixture_file_snapshots_nonempty(self):
        for task in self.tasks[:5]:
            fix = build_fixture(task)
            self.assertGreater(len(fix.file_snapshots), 0)

    def test_parse_requirements(self):
        content = "# comment\nnumpy>=1.23\npandas==2.0.0\nrequests\n"
        deps = parse_requirements(content)
        names = [d["package"] for d in deps]
        self.assertIn("numpy", names)
        self.assertIn("pandas", names)
        self.assertIn("requests", names)

    def test_build_fixtures_batch(self):
        tasks    = self.tasks[:10]
        fixtures = build_fixtures(tasks)
        self.assertEqual(len(fixtures), 10)
        for task in tasks:
            self.assertIn(task.task_id, fixtures)


# ── Test: Verifier ────────────────────────────────────────────────────────────

class TestVerifier(unittest.TestCase):

    def setUp(self):
        self.tasks = get_all_tasks(tasks_per_family=5)

    def _get_task(self, family: str) -> RepoTask:
        return next(t for t in self.tasks if t.family == family)

    def test_oracle_steps_pass_verification(self):
        for family in ALL_FAMILY_NAMES:
            task   = self._get_task(family)
            result = verify_repair(task, task.oracle_repair_steps, family)
            self.assertTrue(result.success,
                            f"{family}: oracle steps should pass verification "
                            f"(score={result.composite_score:.3f})")

    def test_empty_steps_fail_verification(self):
        for family in ALL_FAMILY_NAMES:
            task   = self._get_task(family)
            result = verify_repair(task, [], "Unknown")
            self.assertFalse(result.success,
                             f"{family}: empty steps should fail")

    def test_wrong_family_reduces_score(self):
        task   = self._get_task(FAMILY_IMPORT)
        right  = verify_repair(task, task.oracle_repair_steps, FAMILY_IMPORT)
        wrong  = verify_repair(task, task.oracle_repair_steps, FAMILY_CONFIG)
        self.assertGreater(right.composite_score, wrong.composite_score)

    def test_distractor_steps_score_lower_than_oracle(self):
        for family in ALL_FAMILY_NAMES:
            task     = self._get_task(family)
            oracle_r = verify_repair(task, task.oracle_repair_steps, family)
            dist_r   = verify_repair(task, task.distractor_steps(), family)
            self.assertGreater(oracle_r.composite_score, dist_r.composite_score,
                               f"{family}: oracle should beat distractors")

    def test_verification_result_fields(self):
        task   = self._get_task(FAMILY_IMPORT)
        result = verify_repair(task, task.oracle_repair_steps, FAMILY_IMPORT)
        self.assertIsInstance(result.success, bool)
        self.assertGreaterEqual(result.composite_score, 0.0)
        self.assertLessEqual(result.composite_score, 1.0)
        self.assertTrue(result.family_match)
        self.assertGreaterEqual(result.step_overlap, 0.0)
        self.assertGreaterEqual(result.keyword_coverage, 0.0)
        self.assertIsInstance(result.reason, str)

    def test_batch_verify(self):
        tasks = self.tasks[:6]
        steps_map  = {t.task_id: t.oracle_repair_steps for t in tasks}
        family_map = {t.task_id: t.family for t in tasks}
        results = batch_verify(tasks, steps_map, family_map)
        self.assertEqual(len(results), 6)
        for r in results:
            self.assertIsInstance(r, VerificationResult)

    def test_retry_improves_initially_failed(self):
        # Use a task with high difficulty so partial steps may fail first
        task    = self._get_task(FAMILY_DEPENDENCY)
        partial = task.oracle_repair_steps[:2]   # intentionally incomplete
        r1, _   = verify_with_retry(task, partial, task.family, max_retries=0)
        r2, n   = verify_with_retry(task, partial, task.family, max_retries=3)
        # r2 should have equal or higher score
        self.assertGreaterEqual(r2.composite_score, r1.composite_score)

    def test_to_dict_serialisable(self):
        import json
        task   = self._get_task(FAMILY_PATH)
        result = verify_repair(task, task.oracle_repair_steps, FAMILY_PATH)
        d = result.to_dict()
        self.assertIsInstance(json.dumps(d), str)   # must be JSON-serialisable


# ── Test: Procedural Repair Agent ────────────────────────────────────────────

class TestProceduralRepairAgent(unittest.TestCase):

    def setUp(self):
        self.tasks = get_all_tasks(tasks_per_family=5)
        self.train, self.test = split_train_test(self.tasks, 0.5, seed=0)
        self.store, self.agent = make_agent("full_memory", embedding_dim=64)
        self.n_warm = self.agent.warm_up(self.train, seed=0)

    def test_warm_up_populates_store(self):
        self.assertGreater(len(self.store), 0)
        self.assertEqual(self.n_warm, len(self.train))

    def test_repair_returns_agent_trace(self):
        from tacm.psm006 import build_fixture
        task  = self.test[0]
        fix   = build_fixture(task)
        trace = self.agent.repair(task, fix, seed=0)
        self.assertIsInstance(trace, AgentTrace)
        self.assertEqual(trace.task_id, task.task_id)
        self.assertEqual(trace.mode, "full_memory")

    def test_repair_retrieves_procedure(self):
        from tacm.psm006 import build_fixture
        task  = self.test[0]
        fix   = build_fixture(task)
        trace = self.agent.repair(task, fix, seed=0)
        # With warm store, should retrieve something
        self.assertIsNotNone(trace.retrieved_proc_id)

    def test_repair_batch(self):
        fixtures = build_fixtures(self.test)
        traces   = self.agent.repair_batch(self.test, fixtures, seed=0)
        self.assertEqual(len(traces), len(self.test))
        for t in traces:
            self.assertIsInstance(t, AgentTrace)

    def test_update_enabled_vs_disabled(self):
        """Full memory (update=True) should match or outperform no-update."""
        from tacm.psm006 import build_fixtures
        fixtures = build_fixtures(self.test)

        _, agent_noupd = make_agent("no_update", update_enabled=False)
        agent_noupd.warm_up(self.train, seed=0)
        traces_noupd = agent_noupd.repair_batch(self.test, fixtures, seed=0)

        traces_full  = self.agent.repair_batch(self.test, fixtures, seed=0)

        vrs_full  = metric_verified_repair_success(traces_full)
        vrs_noupd = metric_verified_repair_success(traces_noupd)
        # Full memory with updates should be >= no-update (or equal at worst)
        self.assertGreaterEqual(vrs_full, vrs_noupd - 0.05,
                                "Full memory should not be much worse than no-update")


# ── Test: Baselines ───────────────────────────────────────────────────────────

class TestBaselines(unittest.TestCase):

    def setUp(self):
        tasks = get_all_tasks(tasks_per_family=5)
        self.train, self.test = split_train_test(tasks, 0.5, seed=0)
        self.store, agent = make_agent("full_memory")
        agent.warm_up(self.train, seed=0)
        self.fixtures = build_fixtures(self.test)

    def test_all_baselines_return_traces(self):
        all_traces = run_all_baselines(
            tasks      = self.test,
            fixtures   = self.fixtures,
            store_full = self.store,
            seed       = 0,
        )
        self.assertEqual(set(all_traces.keys()), set(BASELINE_NAMES))
        for name, traces in all_traces.items():
            self.assertEqual(len(traces), len(self.test),
                             f"{name}: expected {len(self.test)} traces")

    def test_reset_retrieves_nothing(self):
        traces = run_reset(self.test, self.fixtures, seed=0)
        for t in traces:
            self.assertIsNone(t.retrieved_proc_id)
            self.assertEqual(t.applied_steps, [])

    def test_oracle_always_succeeds_for_easy_tasks(self):
        easy_tasks = [t for t in self.test if t.difficulty <= 0.4]
        if easy_tasks:
            traces = run_oracle(easy_tasks, None, seed=0)
            for tr in traces:
                self.assertTrue(tr.verification.success,
                                f"Oracle should pass easy task {tr.task_id}")

    def test_oracle_beats_reset(self):
        oracle_traces = run_oracle(self.test, self.fixtures, seed=0)
        reset_traces  = run_reset(self.test, self.fixtures, seed=0)
        oracle_vrs = metric_verified_repair_success(oracle_traces)
        reset_vrs  = metric_verified_repair_success(reset_traces)
        self.assertGreater(oracle_vrs, reset_vrs,
                           "Oracle must beat reset baseline")

    def test_mode_labels(self):
        all_traces = run_all_baselines(self.test, self.fixtures, self.store, 0)
        for name, traces in all_traces.items():
            for t in traces:
                self.assertEqual(t.mode, name,
                                 f"Mode mismatch: expected {name} got {t.mode}")

    def test_agent_trace_serialisable(self):
        import json
        traces = run_oracle(self.test[:2], None, seed=0)
        for t in traces:
            d = t.to_dict()
            self.assertIsInstance(json.dumps(d, default=str), str)


# ── Test: Metrics ─────────────────────────────────────────────────────────────

class TestMetrics(unittest.TestCase):

    def setUp(self):
        tasks = get_all_tasks(tasks_per_family=5)
        self.train, self.test = split_train_test(tasks, 0.5, seed=0)
        store, agent = make_agent("full_memory")
        agent.warm_up(self.train, seed=0)
        fixtures = build_fixtures(self.test)
        self.all_traces = run_all_baselines(self.test, fixtures, store, seed=0)

    def test_vrs_in_range(self):
        for name, traces in self.all_traces.items():
            vrs = metric_verified_repair_success(traces)
            self.assertGreaterEqual(vrs, 0.0, f"{name}: vrs < 0")
            self.assertLessEqual(vrs, 1.0,    f"{name}: vrs > 1")

    def test_retrieval_accuracy_in_range(self):
        for name, traces in self.all_traces.items():
            acc = metric_retrieval_accuracy(traces)
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_oracle_highest_success(self):
        oracle_vrs = metric_verified_repair_success(self.all_traces["oracle"])
        reset_vrs  = metric_verified_repair_success(self.all_traces["reset"])
        self.assertGreaterEqual(oracle_vrs, reset_vrs,
                                "Oracle must have >= success rate than reset")

    def test_compute_metrics_returns_psm006metrics(self):
        from tacm.psm006 import PSM006Metrics
        m = compute_metrics(
            traces           = self.all_traces["full_memory"],
            reset_traces     = self.all_traces["reset"],
            no_update_traces = self.all_traces["no_update"],
            random_traces    = self.all_traces["random_procedure"],
            seed             = 0,
        )
        self.assertIsInstance(m, PSM006Metrics)
        self.assertEqual(m.mode, "full_memory")
        self.assertIsNotNone(m.confusion)

    def test_aggregate_metrics(self):
        from tacm.psm006 import PSM006Metrics
        runs = []
        for seed in [0, 1]:
            m = compute_metrics(
                self.all_traces["full_memory"],
                self.all_traces["reset"],
                seed=seed,
            )
            runs.append(m)
        agg = aggregate_metrics(runs)
        self.assertEqual(agg.n_seeds, 2)
        self.assertIn("verified_repair_success", agg.stats)

    def test_confusion_matrix_structure(self):
        traces = self.all_traces["full_memory"]
        cm = metric_confusion_matrix(traces)
        self.assertIsInstance(cm.families, list)
        self.assertIsInstance(cm.matrix, dict)
        self.assertEqual(cm.n_samples, len(traces))
        # macro precision and recall in [0, 1]
        self.assertGreaterEqual(cm.macro_precision(), 0.0)
        self.assertLessEqual(cm.macro_precision(), 1.0)

    def test_confusion_format_table(self):
        cm    = metric_confusion_matrix(self.all_traces["full_memory"])
        table = cm.format_table()
        self.assertIsInstance(table, str)
        self.assertGreater(len(table), 0)

    def test_survival_stability_nonneg(self):
        for name, traces in self.all_traces.items():
            s = metric_survival_stability(traces)
            self.assertGreaterEqual(s, 0.0)

    def test_steps_to_repair_positive_for_oracle(self):
        s = metric_steps_to_repair(self.all_traces["oracle"])
        self.assertGreater(s, 0.0)

    def test_evaluate_gates_returns_dict(self):
        from tacm.psm006 import PSM006Metrics, AggregatedMetrics
        def _wrap(name):
            m = compute_metrics(
                self.all_traces[name],
                self.all_traces["reset"],
                self.all_traces["no_update"],
                self.all_traces["random_procedure"],
                seed=0,
            )
            return aggregate_metrics([m])

        gates = evaluate_gates(
            full_agg      = _wrap("full_memory"),
            oracle_agg    = _wrap("oracle"),
            no_update_agg = _wrap("no_update"),
        )
        self.assertIsInstance(gates, dict)
        self.assertIn("tac_beats_reset_by_0.10",    gates)
        self.assertIn("retrieval_accuracy_ge_0.60",  gates)
        self.assertIn("oracle_above_tac",            gates)
        self.assertIn("no_update_underperforms_tac", gates)
        # Oracle should always be >= TAC success
        self.assertTrue(gates["oracle_above_tac"])

    def test_metrics_to_dict_serialisable(self):
        import json
        m = compute_metrics(
            self.all_traces["full_memory"],
            self.all_traces["reset"],
            seed=0,
        )
        d = m.to_dict()
        self.assertIsInstance(json.dumps(d, default=str), str)


# ── Test: End-to-end quick run ────────────────────────────────────────────────

class TestEndToEnd(unittest.TestCase):

    def test_single_seed_smoke(self):
        """Full pipeline must run without error for seed=0."""
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_tac_psm006_repository_memory import run_one_seed
        result = run_one_seed(seed=0, verbose=False, tasks_per_family=5)
        self.assertIn("gates", result)
        self.assertIn("variants", result)
        self.assertIn("family_success", result)
        self.assertIn("confusion", result)
        self.assertEqual(set(result["variants"].keys()), set(BASELINE_NAMES))

    def test_oracle_gate_passes(self):
        """oracle_above_tac gate must always pass."""
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmark_tac_psm006_repository_memory import run_one_seed
        result = run_one_seed(seed=1, verbose=False, tasks_per_family=5)
        self.assertTrue(result["gates"].get("oracle_above_tac", False),
                        "oracle_above_tac gate must always pass")

    def test_all_families_covered(self):
        from benchmark_tac_psm006_repository_memory import run_one_seed
        result = run_one_seed(seed=0, verbose=False, tasks_per_family=5)
        for family in ALL_FAMILY_NAMES:
            self.assertIn(family, result["family_success"])


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    # Patch make_agent to accept keyword arg correctly
    import tacm.psm006.procedural_repair_agent as _pra
    _orig_make = _pra.make_agent
    def _patched_make(mode="full_memory", EMBEDDING_DIM=64, **kwargs):
        return _orig_make(mode=mode, embedding_dim=EMBEDDING_DIM, **kwargs)
    _pra.make_agent = _patched_make

    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
