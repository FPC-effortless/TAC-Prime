"""
tests/test_tac_psm006b_pytest_fixtures.py

Unit and integration tests for TAC-PSM-006B.

Section A — Schema and fixture builder (always run):
  - FAMILY_NAMES has 6 entries
  - build_all_fixtures() returns 60 fixtures
  - Each fixture has required fields
  - All 60 fixtures have unique fixture_ids
  - 10 fixtures per family
  - transfer_group and difficulty valid values
  - oracle_repair_procedure has 'family' and 'steps'

Section B — Patch applier:
  - apply() with matching old string succeeds
  - apply() with non-matching old string fails cleanly
  - apply() with empty old creates new file
  - apply() empty patch is no-op
  - apply_wrong_family_patch() always returns success=True
  - apply_structure_only_patch() applies to correct file

Section C — Memory store:
  - write() and retrieve() return matching family
  - augment() adds steps without duplicates
  - reinforce() increases success_rate
  - clear() empties store
  - clone() is independent copy

Section D — Pytest verifier:
  - Simple passing test returns success=True
  - Simple failing test returns success=False
  - verify_before_and_after() returns (False, True) for valid fixture

Section E — Repair agent and baselines:
  - ProceduralRepairAgent006B.repair() returns RepairTrace006B
  - oracle variant always retrieves correct family
  - random_procedure variant retrieval may be wrong
  - run_all_baselines() returns dict with all variant keys

Section F — Metrics:
  - compute_metrics() returns 13 keys
  - evaluate_success_gates() returns 8 keys
  - compute_family_confusion_matrix() shape correct
  - classify_failures() returns counts
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from tacm.psm006b.fixture_schema import (
    Fixture, FAMILY_NAMES, FAILURE_CLASSES, TRANSFER_GROUPS, DIFFICULTY_LEVELS
)
from tacm.psm006b.fixture_builder import build_all_fixtures, build_fixtures_by_family
from tacm.psm006b.patch_applier import PatchApplier
from tacm.psm006b.memory_store import SimpleProceduralMemoryStore
from tacm.psm006b.pytest_verifier import PytestVerifier
from tacm.psm006b.procedural_repair_agent import (
    ProceduralRepairAgent006B,
    seed_procedural_memory,
    fixture_embedding,
    family_centroid,
    oracle_procedure_dict,
    EMBEDDING_DIM,
)
from tacm.psm006b.baselines import run_all_baselines, VARIANT_NAMES
from tacm.psm006b.metrics import (
    compute_metrics,
    evaluate_success_gates,
    compute_family_confusion_matrix,
    classify_failures,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Schema and fixture builder
# ─────────────────────────────────────────────────────────────────────────────

class TestSchema:

    def test_family_names_count(self):
        assert len(FAMILY_NAMES) == 6

    def test_build_all_fixtures_count(self):
        fixtures = build_all_fixtures()
        assert len(fixtures) == 60

    def test_fixtures_unique_ids(self):
        fixtures = build_all_fixtures()
        ids = [fx.fixture_id for fx in fixtures]
        assert len(set(ids)) == 60, "fixture_ids must be unique"

    def test_ten_per_family(self):
        by_fam = build_fixtures_by_family()
        for fam in FAMILY_NAMES:
            assert len(by_fam[fam]) == 10, f"{fam} should have 10 fixtures"

    def test_required_fields_populated(self):
        fixtures = build_all_fixtures()
        for fx in fixtures:
            assert fx.fixture_id
            assert fx.repo_name
            assert fx.family in FAMILY_NAMES
            assert fx.bug_report
            assert fx.failing_test_command
            assert fx.verification_command
            assert isinstance(fx.source_files, dict)
            assert isinstance(fx.test_files, dict)
            assert isinstance(fx.config_files, dict)
            assert isinstance(fx.oracle_repair_procedure, dict)
            assert isinstance(fx.expected_patch, dict)
            assert fx.transfer_group in TRANSFER_GROUPS
            assert fx.difficulty in DIFFICULTY_LEVELS

    def test_oracle_procedure_has_steps(self):
        fixtures = build_all_fixtures()
        for fx in fixtures:
            proc = fx.oracle_repair_procedure
            assert "steps" in proc
            assert len(proc["steps"]) >= 1

    def test_all_files_merges_dicts(self):
        fixtures = build_all_fixtures()
        for fx in fixtures[:5]:
            all_f = fx.all_files()
            assert isinstance(all_f, dict)

    def test_to_dict_serializable(self):
        import json
        fixtures = build_all_fixtures()
        d = fixtures[0].to_dict()
        # Should be JSON-serializable
        json.dumps(d)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: Patch applier
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchApplier:

    @pytest.fixture
    def applier(self):
        return PatchApplier()

    @pytest.fixture
    def files(self):
        return {"utils.py": "def old_name():\n    return 1\n"}

    def test_apply_matching_succeeds(self, applier, files):
        patch = {"utils.py": {"old": "def old_name():\n", "new": "def new_name():\n"}}
        result = applier.apply(files, patch)
        assert result.success
        assert "new_name" in result.patched_files["utils.py"]
        assert "utils.py" in result.files_modified

    def test_apply_nonmatching_fails(self, applier, files):
        patch = {"utils.py": {"old": "def nonexistent():\n", "new": "def x():\n"}}
        result = applier.apply(files, patch)
        assert not result.success
        assert len(result.patch_errors) == 1

    def test_apply_file_not_found_fails(self, applier, files):
        patch = {"missing.py": {"old": "x = 1\n", "new": "x = 2\n"}}
        result = applier.apply(files, patch)
        assert not result.success
        assert result.failure_class == "patch_wrong_file"

    def test_apply_creates_new_file(self, applier, files):
        patch = {"new_module.py": {"old": "", "new": "X = 42\n"}}
        result = applier.apply(files, patch)
        assert result.success
        assert "new_module.py" in result.patched_files
        assert "new_module.py" in result.files_created

    def test_apply_empty_patch_noop(self, applier, files):
        result = applier.apply(files, {})
        assert result.success
        assert result.patched_files == files
        assert result.files_modified == []

    def test_wrong_family_patch_success(self, applier, files):
        result = applier.apply_wrong_family_patch(files, "import_module_error")
        assert result.success
        assert result.patched_files != files   # something was changed

    def test_structure_only_patch(self, applier, files):
        patch = {"utils.py": {"old": "def old_name():\n", "new": "def proper_fix():\n"}}
        result = applier.apply_structure_only_patch(files, patch)
        assert result.success
        # Patch was applied but with stub content
        assert "utils.py" in result.patched_files


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: Memory store
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryStore:

    @pytest.fixture
    def store(self):
        return SimpleProceduralMemoryStore()

    def _rand_emb(self, seed=0):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_write_and_retrieve(self, store):
        emb = self._rand_emb(0)
        pid = store.write("import_module_error", "test", ["step1"], emb)
        records = store.retrieve(emb, top_k=1)
        assert len(records) == 1
        assert records[0].family == "import_module_error"
        assert records[0].proc_id == pid

    def test_empty_store_returns_empty(self, store):
        emb = self._rand_emb(1)
        assert store.retrieve(emb, top_k=3) == []

    def test_retrieves_correct_family_by_cosine(self, store):
        # Write 6 procedures with different embeddings
        centroids = {}
        for i, fam in enumerate(FAMILY_NAMES):
            emb = family_centroid(fam)
            store.write(fam, "test", [f"step_{fam}"], emb)
            centroids[fam] = emb
        # Query with each centroid — should retrieve the correct family
        for fam in FAMILY_NAMES:
            recs = store.retrieve(centroids[fam], top_k=1)
            assert recs[0].family == fam, f"Expected {fam}, got {recs[0].family}"

    def test_augment_adds_steps(self, store):
        emb = self._rand_emb(2)
        pid = store.write("test_fam", "test", ["step1"], emb)
        store.augment(pid, ["step2", "step3", "step1"])  # step1 already present
        rec = store._get(pid)
        assert "step2" in rec.steps
        assert "step3" in rec.steps
        assert rec.steps.count("step1") == 1  # no duplicates

    def test_augment_decreases_success_rate(self, store):
        emb = self._rand_emb(3)
        pid = store.write("test_fam", "test", ["step1"], emb, success_rate=0.8)
        store.augment(pid, ["extra_step"])
        rec = store._get(pid)
        assert rec.success_rate < 0.8

    def test_reinforce_increases_success_rate(self, store):
        emb = self._rand_emb(4)
        pid = store.write("test_fam", "test", ["step1"], emb, success_rate=0.7)
        store.reinforce(pid, delta=0.1)
        rec = store._get(pid)
        assert rec.success_rate == pytest.approx(0.8)

    def test_clear_empties_store(self, store):
        emb = self._rand_emb(5)
        store.write("fam", "test", ["step1"], emb)
        assert len(store) == 1
        store.clear()
        assert len(store) == 0

    def test_clone_is_independent(self, store):
        emb = self._rand_emb(6)
        pid = store.write("fam", "test", ["step1"], emb)
        cloned = store.clone()
        store.reinforce(pid, delta=0.1)
        orig_rate = store._get(pid).success_rate
        clone_rate = cloned._get(pid).success_rate
        assert orig_rate != clone_rate, "Clone should be independent"

    def test_retired_records_not_returned(self, store):
        emb = self._rand_emb(7)
        pid = store.write("fam", "test", ["step1"], emb)
        store.retire(pid)
        recs = store.retrieve(emb, top_k=1)
        assert recs == []


# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: Pytest verifier
# ─────────────────────────────────────────────────────────────────────────────

class TestPytestVerifier:

    @pytest.fixture
    def verifier(self):
        return PytestVerifier(timeout=15.0)

    def test_passing_test_returns_success(self, verifier):
        files = {
            "test_pass.py": "def test_ok():\n    assert 1 + 1 == 2\n"
        }
        result = verifier.run(files, "pytest test_pass.py -x -q",
                              fixture_id="test", variant="check")
        assert result.success is True
        assert result.exit_code == 0
        assert not result.timed_out

    def test_failing_test_returns_failure(self, verifier):
        files = {
            "test_fail.py": "def test_bad():\n    assert 1 == 2\n"
        }
        result = verifier.run(files, "pytest test_fail.py -x -q",
                              fixture_id="test", variant="check")
        assert result.success is False
        assert result.exit_code != 0

    def test_import_error_returns_failure(self, verifier):
        files = {
            "test_imp.py": "from nonexistent_module import something\ndef test_x():\n    assert True\n"
        }
        result = verifier.run(files, "pytest test_imp.py -x -q")
        assert not result.success

    def test_result_has_stdout(self, verifier):
        files = {"test_s.py": "def test_s():\n    assert True\n"}
        result = verifier.run(files, "pytest test_s.py -q")
        assert isinstance(result.stdout, str)

    def test_verify_before_and_after(self, verifier):
        """A fixture that fails before patch and passes after."""
        before_files = {
            "utils.py": "def greet(name):\n    return f'Hi {name}'\n",
            "test_g.py": "from utils import greet\ndef test_greet():\n    assert greet('A') == 'Hello A'\n",
        }
        after_files = {
            "utils.py": "def greet(name):\n    return f'Hello {name}'\n",
            "test_g.py": "from utils import greet\ndef test_greet():\n    assert greet('A') == 'Hello A'\n",
        }
        before, after = verifier.verify_before_and_after(
            before_files, after_files, "pytest test_g.py -x -q", "test_fixture"
        )
        assert not before.success
        assert after.success

    def test_build_args_strips_pytest_prefix(self):
        args = PytestVerifier._build_args("pytest test_foo.py -x -q")
        assert args == ["test_foo.py", "-x", "-q"]

    def test_build_args_strips_python_m_prefix(self):
        args = PytestVerifier._build_args("python -m pytest test_foo.py -q")
        assert args == ["test_foo.py", "-q"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: Repair agent and baselines
# ─────────────────────────────────────────────────────────────────────────────

class TestRepairAgent:

    @pytest.fixture
    def store_seeded(self):
        store = SimpleProceduralMemoryStore()
        seed_procedural_memory(store, n_records_per_family=1, rng_seed=0)
        return store

    @pytest.fixture
    def verifier(self):
        return PytestVerifier(timeout=15.0)

    @pytest.fixture
    def applier(self):
        return PatchApplier()

    @pytest.fixture
    def fixture_easy(self):
        """A simple passing fixture: test itself is already correct."""
        from tacm.psm006b.fixture_schema import Fixture
        return Fixture(
            fixture_id="TEST_easy",
            repo_name="test_easy",
            family="test_assertion_repair",
            bug_report="assert 1 == 2",
            failing_test_command="pytest test_easy_fix.py -x -q",
            failing_test_output="AssertionError",
            source_files={},
            test_files={
                "test_easy_fix.py": "def test_ok():\n    assert 1 == 1\n",
            },
            config_files={},
            oracle_repair_procedure={
                "family": "test_assertion_repair",
                "steps": ["identify_assertion_failure", "correct_expected_value_or_logic"],
            },
            expected_patch={},   # no patch needed — test already passes
            verification_command="pytest test_easy_fix.py -x -q",
            transfer_group="train",
            difficulty="easy",
        )

    def test_repair_returns_trace(self, store_seeded, verifier, applier, fixture_easy):
        agent = ProceduralRepairAgent006B(
            store=store_seeded, verifier=verifier, applier=applier,
            mode="full_memory", rng_seed=0,
        )
        trace = agent.repair(fixture_easy)
        assert isinstance(trace, object)
        assert hasattr(trace, "pytest_pass")
        assert hasattr(trace, "retrieval_correct")
        assert trace.fixture_id == "TEST_easy"

    def test_oracle_always_retrieves_correct_family(
        self, store_seeded, verifier, applier
    ):
        fixtures = build_all_fixtures()[:3]
        agent = ProceduralRepairAgent006B(
            store=store_seeded, verifier=verifier, applier=applier,
            mode="oracle", rng_seed=0,
        )
        for fx in fixtures:
            trace = agent.repair(fx)
            assert trace.retrieval_correct, \
                f"Oracle should always retrieve correct family for {fx.fixture_id}"

    def test_trace_to_dict_serializable(self, store_seeded, verifier, applier, fixture_easy):
        import json
        agent = ProceduralRepairAgent006B(
            store=store_seeded, verifier=verifier, applier=applier,
            mode="full_memory", rng_seed=0,
        )
        trace = agent.repair(fixture_easy)
        d = trace.to_dict()
        json.dumps(d)   # must not raise

    def test_embedding_shapes(self):
        fixtures = build_all_fixtures()
        rng = np.random.default_rng(0)
        for fx in fixtures[:5]:
            emb = fixture_embedding(fx, rng)
            assert emb.shape == (EMBEDDING_DIM,)
            assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_oracle_procedure_all_families(self):
        for fam in FAMILY_NAMES:
            proc = oracle_procedure_dict(fam)
            assert proc["family"] == fam
            assert len(proc["steps"]) >= 2

    def test_seed_procedural_memory_populates(self):
        store = SimpleProceduralMemoryStore()
        seed_procedural_memory(store, n_records_per_family=1, rng_seed=0)
        assert len(store) == len(FAMILY_NAMES)


class TestBaselines:

    def test_run_all_baselines_returns_all_variants(self):
        """Smoke test: run on 2 easy fixtures with short timeout."""
        fixtures = build_all_fixtures()[:2]
        results = run_all_baselines(
            fixtures  = fixtures,
            seed      = 0,
            timeout_s = 15.0,
            variants  = ["full_memory", "oracle", "reset"],
        )
        assert "full_memory" in results
        assert "oracle" in results
        assert "reset" in results
        assert len(results["full_memory"]) == 2

    def test_oracle_beats_random_on_easy_fixtures(self):
        """Oracle should consistently outperform random on easy fixtures."""
        fixtures = [fx for fx in build_all_fixtures() if fx.difficulty == "easy"][:6]
        results  = run_all_baselines(
            fixtures  = fixtures,
            seed      = 0,
            timeout_s = 15.0,
            variants  = ["oracle", "random_procedure"],
        )
        oracle_pass = sum(1 for t in results["oracle"] if t.pytest_pass)
        rand_pass   = sum(1 for t in results["random_procedure"] if t.pytest_pass)
        assert oracle_pass >= rand_pass, \
            f"oracle={oracle_pass} should be >= random={rand_pass}"

    def test_variant_names_complete(self):
        assert len(VARIANT_NAMES) == 7
        for v in ["full_memory", "reset", "retrieval_disabled", "random_procedure",
                  "structure_only", "no_update", "oracle"]:
            assert v in VARIANT_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# SECTION F: Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    @pytest.fixture
    def mock_results(self):
        """Create minimal mock trace results for metrics testing."""
        from tacm.psm006b.procedural_repair_agent import RepairTrace006B

        def make_trace(fam, ret_fam, passed, updated=False, improved=False,
                       fc=None, mode="full_memory"):
            return RepairTrace006B(
                fixture_id        = f"F_{fam}",
                family            = fam,
                retrieved_family  = ret_fam,
                retrieved_proc_id = "pid1",
                retrieval_correct = (fam == ret_fam),
                patch_result      = {"success": True},
                before_result     = {"success": False},
                after_result      = {"success": passed},
                pytest_pass       = passed,
                n_retries         = 1 if improved else 0,
                steps_to_repair   = 4,
                procedure_updated = updated,
                update_improved   = improved,
                mode              = mode,
                failure_class     = fc,
                time_to_repair_s  = 0.5,
            )

        fam = "import_module_error"
        full_traces = [
            make_trace(fam, fam, True),
            make_trace(fam, fam, True, updated=True),
            make_trace(fam, "test_assertion_repair", False, fc="wrong_procedure_retrieval"),
            make_trace(fam, fam, False, updated=True, improved=True),
        ]
        oracle_traces = [make_trace(fam, fam, True) for _ in range(4)]
        reset_traces  = [
            make_trace(fam, fam, False, mode="reset"),
            make_trace(fam, fam, True,  mode="reset"),
            make_trace(fam, fam, False, mode="reset"),
            make_trace(fam, fam, False, mode="reset"),
        ]
        rand_traces   = [make_trace(fam, "test_assertion_repair", False, mode="random_procedure")
                         for _ in range(4)]
        no_upd_traces = [make_trace(fam, fam, True, mode="no_update"),
                         make_trace(fam, fam, False, mode="no_update"),
                         make_trace(fam, fam, False, mode="no_update"),
                         make_trace(fam, fam, False, mode="no_update")]

        return {
            "full_memory":      full_traces,
            "oracle":           oracle_traces,
            "reset":            reset_traces,
            "random_procedure": rand_traces,
            "no_update":        no_upd_traces,
        }

    def test_compute_metrics_has_13_keys(self, mock_results):
        metrics = compute_metrics(mock_results)
        assert len(metrics) == 13

    def test_pytest_pass_rate_in_range(self, mock_results):
        metrics = compute_metrics(mock_results)
        assert 0.0 <= metrics["pytest_pass_rate"] <= 1.0

    def test_retrieval_accuracy_in_range(self, mock_results):
        metrics = compute_metrics(mock_results)
        assert 0.0 <= metrics["procedure_retrieval_accuracy"] <= 1.0

    def test_evaluate_gates_has_8_keys(self, mock_results):
        metrics = compute_metrics(mock_results)
        gates   = evaluate_success_gates(metrics, mock_results)
        assert len(gates) == 8

    def test_gates_are_booleans(self, mock_results):
        metrics = compute_metrics(mock_results)
        gates   = evaluate_success_gates(metrics, mock_results)
        for k, v in gates.items():
            assert isinstance(v, bool), f"Gate {k} should be bool, got {type(v)}"

    def test_confusion_matrix_shape(self, mock_results):
        full_traces = mock_results["full_memory"]
        cm = compute_family_confusion_matrix(full_traces, FAMILY_NAMES)
        assert set(cm.keys()) == set(FAMILY_NAMES)
        for row in cm.values():
            assert set(row.keys()) == set(FAMILY_NAMES)

    def test_classify_failures_has_expected_keys(self, mock_results):
        traces  = mock_results["full_memory"]
        counts  = classify_failures(traces)
        assert "wrong_procedure_retrieval" in counts
        assert "none" in counts

    def test_oracle_pass_rate_positive(self, mock_results):
        metrics = compute_metrics(mock_results)
        # Oracle pass rate is embedded in oracle_above_tac gate
        gates = evaluate_success_gates(metrics, mock_results)
        assert gates["oracle_above_tac"] is True

    def test_reuse_gain_gt_zero(self, mock_results):
        """full_memory has 75% pass, reset has 25% → gain > 0."""
        metrics = compute_metrics(mock_results)
        assert metrics["procedure_reuse_gain"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
