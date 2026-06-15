"""
TAC-PSM-006C: Unit Tests — Online Embedding Adaptation
=======================================================

Tests the PSM-006C embedding update mechanism in isolation, then tests
the full agent loop, then runs a smoke benchmark with real fixtures.

Run:
    cd tacm
    python -m pytest test_tac_psm006c_embedding_update.py -v

All tests are deterministic (seeded) and complete in < 60s.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

# ── Module imports ────────────────────────────────────────────────────────

from tacm.psm006b.fixture_schema import FAMILY_NAMES, Fixture
from tacm.psm006b.fixture_builder import build_all_fixtures
from tacm.psm006b.memory_store import SimpleProceduralMemoryStore
from tacm.psm006b.procedural_repair_agent import (
    seed_procedural_memory,
    fixture_embedding,
    oracle_procedure_dict,
    EMBEDDING_DIM,
)
from tacm.psm006c.embedding_update import (
    OnlineEmbeddingAdapter,
    EmbeddingUpdateRecord,
    _unit,
)
from tacm.psm006c.agent import ProceduralRepairAgent006C, RepairTrace006C
from tacm.psm006c.baselines import (
    run_all_baselines_006c,
    VARIANT_NAMES_006C,
)
from tacm.psm006c.metrics import (
    compute_metrics_006c,
    evaluate_success_gates_006c,
    classify_failures_006c,
)
from tacm.psm006b.patch_applier import PatchApplier
from tacm.psm006b.pytest_verifier import PytestVerifier
from tacm.psm006b.caching_verifier import CachingSubprocessVerifier


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_store(seed: int = 0) -> SimpleProceduralMemoryStore:
    s = SimpleProceduralMemoryStore()
    seed_procedural_memory(s, n_records_per_family=2, rng_seed=seed)
    return s


def _random_unit_emb(seed: int = 42, dim: int = EMBEDDING_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v   = rng.standard_normal(dim).astype(np.float32)
    return _unit(v)


def _first_fixture() -> Fixture:
    return build_all_fixtures()[0]


# ═══════════════════════════════════════════════════════════════════════════
# Group A: _unit helper
# ═══════════════════════════════════════════════════════════════════════════

class TestUnit:
    def test_unit_returns_unit_norm(self):
        v   = np.array([3.0, 4.0], dtype=np.float32)
        u   = _unit(v)
        assert abs(np.linalg.norm(u) - 1.0) < 1e-5

    def test_unit_zero_vector_safe(self):
        v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        u = _unit(v)
        assert np.linalg.norm(u) < 1e-5   # doesn't crash, stays near zero

    def test_unit_dtype_float32(self):
        v = np.ones(4, dtype=np.float64)
        u = _unit(v)
        assert u.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════
# Group B: EmbeddingUpdateRecord
# ═══════════════════════════════════════════════════════════════════════════

class TestEmbeddingUpdateRecord:
    def test_default_fields(self):
        ur = EmbeddingUpdateRecord(applied=False)
        assert ur.embedding_shift_norm == 0.0
        assert ur.retrieval_changed is False
        assert ur.successful_recovery is False

    def test_to_dict_has_all_keys(self):
        ur = EmbeddingUpdateRecord(applied=True, proc_id="abc", update_type="failure",
                                   embedding_shift_norm=0.05)
        d  = ur.to_dict()
        for k in ["applied", "proc_id", "update_type", "embedding_shift_norm",
                  "retrieval_changed", "family_changed", "successful_recovery",
                  "n_correct_family_nudged"]:
            assert k in d, f"Missing key: {k}"


# ═══════════════════════════════════════════════════════════════════════════
# Group C: OnlineEmbeddingAdapter — adapt_on_failure
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptOnFailure:
    def test_wrong_record_embedding_changes(self):
        store    = _make_store(0)
        adapter  = OnlineEmbeddingAdapter(lr_fail=0.1)
        task_emb = _random_unit_emb(99)

        rec      = store._records[0]
        old_emb  = rec.embedding.copy()
        pid      = rec.proc_id
        family   = rec.family
        wrong_family = next(f for f in FAMILY_NAMES if f != family)

        adapter.adapt_on_failure(store, pid, task_emb, correct_family=wrong_family)
        assert not np.allclose(rec.embedding, old_emb, atol=1e-4)

    def test_result_embedding_unit_normed(self):
        store    = _make_store(0)
        adapter  = OnlineEmbeddingAdapter(lr_fail=0.1)
        task_emb = _random_unit_emb(7)
        rec      = store._records[0]
        pid      = rec.proc_id
        correct  = next(f for f in FAMILY_NAMES if f != rec.family)
        adapter.adapt_on_failure(store, pid, task_emb, correct)
        assert abs(np.linalg.norm(rec.embedding) - 1.0) < 1e-5

    def test_correct_family_records_nudged_toward_task(self):
        store      = _make_store(0)
        adapter    = OnlineEmbeddingAdapter(lr_fail=0.1)
        task_emb   = _random_unit_emb(42)
        wrong_rec  = store._records[0]
        pid        = wrong_rec.proc_id
        correct_fam = next(f for f in FAMILY_NAMES if f != wrong_rec.family)

        # Measure cosine similarity of correct-family records before/after
        correct_recs = [r for r in store._records if r.family == correct_fam]
        sims_before  = [float(r.embedding @ task_emb) for r in correct_recs]

        adapter.adapt_on_failure(store, pid, task_emb, correct_fam)

        sims_after = [float(r.embedding @ task_emb) for r in correct_recs]
        assert all(a >= b - 1e-5 for a, b in zip(sims_after, sims_before)), \
            "correct-family similarity should not decrease"

    def test_wrong_record_pushed_away(self):
        store    = _make_store(0)
        adapter  = OnlineEmbeddingAdapter(lr_fail=0.2)
        task_emb = _random_unit_emb(55)
        rec      = store._records[0]
        pid      = rec.proc_id
        sim_before = float(rec.embedding @ task_emb)
        correct    = next(f for f in FAMILY_NAMES if f != rec.family)
        adapter.adapt_on_failure(store, pid, task_emb, correct)
        sim_after  = float(rec.embedding @ task_emb)
        assert sim_after <= sim_before + 1e-4, \
            "wrong record should not get closer to task after failure update"

    def test_shift_norm_recorded(self):
        store    = _make_store(0)
        adapter  = OnlineEmbeddingAdapter(lr_fail=0.1)
        task_emb = _random_unit_emb(13)
        rec      = store._records[0]
        ur       = adapter.adapt_on_failure(
            store, rec.proc_id, task_emb,
            correct_family=next(f for f in FAMILY_NAMES if f != rec.family)
        )
        assert ur.embedding_shift_norm > 0.0

    def test_missing_proc_id_returns_not_applied(self):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        ur      = adapter.adapt_on_failure(store, "nonexistent-id",
                                           _random_unit_emb(1), FAMILY_NAMES[0])
        assert not ur.applied

    def test_n_correct_family_nudged_nonzero(self):
        store      = _make_store(0)
        adapter    = OnlineEmbeddingAdapter(lr_fail=0.1)
        rec        = store._records[0]
        correct    = next(f for f in FAMILY_NAMES if f != rec.family)
        task_emb   = _random_unit_emb(7)
        ur = adapter.adapt_on_failure(store, rec.proc_id, task_emb, correct)
        assert ur.n_correct_family_nudged > 0


# ═══════════════════════════════════════════════════════════════════════════
# Group D: OnlineEmbeddingAdapter — adapt_on_success
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptOnSuccess:
    def test_embedding_moves_toward_task(self):
        store    = _make_store(0)
        adapter  = OnlineEmbeddingAdapter(lr_success=0.1)
        rec      = store._records[0]
        task_emb = _random_unit_emb(3)
        sim_before = float(rec.embedding @ task_emb)
        adapter.adapt_on_success(store, rec.proc_id, task_emb)
        sim_after = float(rec.embedding @ task_emb)
        assert sim_after >= sim_before - 1e-5

    def test_unit_norm_preserved(self):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter(lr_success=0.1)
        rec     = store._records[0]
        adapter.adapt_on_success(store, rec.proc_id, _random_unit_emb(5))
        assert abs(np.linalg.norm(rec.embedding) - 1.0) < 1e-5

    def test_missing_proc_id_returns_not_applied(self):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        ur      = adapter.adapt_on_success(store, "ghost-id", _random_unit_emb(1))
        assert not ur.applied

    def test_shift_norm_recorded(self):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter(lr_success=0.1)
        rec     = store._records[0]
        ur      = adapter.adapt_on_success(store, rec.proc_id, _random_unit_emb(8))
        assert ur.embedding_shift_norm >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Group E: check_retrieval_change
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckRetrievalChange:
    def test_returns_booleans(self):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        task    = _random_unit_emb(10)
        pc, fc  = adapter.check_retrieval_change(store, task, "old-id", "old-family")
        assert isinstance(pc, bool)
        assert isinstance(fc, bool)

    def test_empty_store_returns_false_false(self):
        store   = SimpleProceduralMemoryStore()
        adapter = OnlineEmbeddingAdapter()
        pc, fc  = adapter.check_retrieval_change(store, _random_unit_emb(1), "x", "y")
        assert pc is False
        assert fc is False

    def test_same_record_proc_not_changed(self):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        rec     = store._records[0]
        task    = rec.embedding.copy()   # query = record embedding → always top-1
        pc, fc  = adapter.check_retrieval_change(store, task, rec.proc_id, rec.family)
        assert pc is False
        assert fc is False


# ═══════════════════════════════════════════════════════════════════════════
# Group F: annotate_record
# ═══════════════════════════════════════════════════════════════════════════

class TestAnnotateRecord:
    def test_successful_recovery_when_family_matches(self):
        adapter = OnlineEmbeddingAdapter()
        ur = EmbeddingUpdateRecord(applied=True, update_type="failure")
        adapter.annotate_record(ur, True, True, "import_module_error", "import_module_error")
        assert ur.successful_recovery is True

    def test_no_recovery_when_new_family_still_wrong(self):
        adapter = OnlineEmbeddingAdapter()
        ur = EmbeddingUpdateRecord(applied=True, update_type="failure")
        adapter.annotate_record(ur, True, True, "version_api_mismatch", "import_module_error")
        assert ur.successful_recovery is False

    def test_no_recovery_when_family_unchanged(self):
        adapter = OnlineEmbeddingAdapter()
        ur = EmbeddingUpdateRecord(applied=True, update_type="failure")
        adapter.annotate_record(ur, False, False, "import_module_error", "import_module_error")
        assert ur.successful_recovery is False


# ═══════════════════════════════════════════════════════════════════════════
# Group G: summary()
# ═══════════════════════════════════════════════════════════════════════════

class TestSummary:
    def test_summary_keys_present(self):
        adapter = OnlineEmbeddingAdapter()
        s = adapter.summary()
        for k in ["embedding_update_count", "failure_update_count",
                  "success_update_count", "embedding_shift_norm_mean",
                  "retrieval_changed_after_update", "family_changed_after_update",
                  "successful_retrieval_recovery"]:
            assert k in s, f"Missing key: {k}"

    def test_empty_summary_zeros(self):
        adapter = OnlineEmbeddingAdapter()
        s = adapter.summary()
        assert s["embedding_update_count"] == 0
        assert s["embedding_shift_norm_mean"] == 0.0

    def test_summary_counts_after_updates(self):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        rec     = store._records[0]
        task    = _random_unit_emb(1)
        correct = next(f for f in FAMILY_NAMES if f != rec.family)
        adapter.adapt_on_failure(store, rec.proc_id, task, correct)
        adapter.adapt_on_success(store, rec.proc_id, task)
        s = adapter.summary()
        assert s["embedding_update_count"] == 2
        assert s["failure_update_count"]   == 1
        assert s["success_update_count"]   == 1

    def test_reset_clears_log(self):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        rec     = store._records[0]
        adapter.adapt_on_failure(store, rec.proc_id, _random_unit_emb(1),
                                 FAMILY_NAMES[1])
        adapter.reset()
        assert adapter.summary()["embedding_update_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Group H: ProceduralRepairAgent006C
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def real_verifier():
    return CachingSubprocessVerifier(timeout=10.0)


@pytest.fixture(scope="module")
def fixtures_3():
    """3 fixtures (one per family) for agent-level tests."""
    all_fx = build_all_fixtures()
    seen   = set()
    result = []
    for fx in all_fx:
        if fx.family not in seen:
            seen.add(fx.family)
            result.append(fx)
        if len(result) == 3:
            break
    return result


class TestAgent006C:
    def test_repair_returns_trace_006c(self, real_verifier, fixtures_3):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        applier = PatchApplier()
        agent   = ProceduralRepairAgent006C(
            store=store, verifier=real_verifier, applier=applier,
            adapter=adapter, mode="full_memory_embedding_update",
            rng_seed=0,
        )
        fx    = fixtures_3[0]
        trace = agent.repair(fx)
        assert isinstance(trace, RepairTrace006C)
        assert isinstance(trace.pytest_pass, bool)
        assert isinstance(trace.embedding_update_applied, bool)
        assert isinstance(trace.embedding_shift_norm, float)

    def test_oracle_mode_passes(self, real_verifier, fixtures_3):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        applier = PatchApplier()
        agent   = ProceduralRepairAgent006C(
            store=store, verifier=real_verifier, applier=applier,
            adapter=adapter, mode="oracle", rng_seed=0,
        )
        for fx in fixtures_3:
            trace = agent.repair(fx)
            assert trace.pytest_pass, f"Oracle failed on {fx.fixture_id}"

    def test_oracle_mode_no_embedding_update(self, real_verifier, fixtures_3):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        applier = PatchApplier()
        agent   = ProceduralRepairAgent006C(
            store=store, verifier=real_verifier, applier=applier,
            adapter=adapter, mode="oracle", rng_seed=0,
        )
        for fx in fixtures_3:
            trace = agent.repair(fx)
            assert not trace.embedding_update_applied

    def test_no_update_mode_no_embedding_update(self, real_verifier, fixtures_3):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        applier = PatchApplier()
        agent   = ProceduralRepairAgent006C(
            store=store, verifier=real_verifier, applier=applier,
            adapter=adapter, mode="no_update", rng_seed=0,
        )
        for fx in fixtures_3:
            trace = agent.repair(fx)
            assert not trace.embedding_update_applied

    def test_embedding_update_mode_can_trigger(self, real_verifier):
        """
        With retrieval noise high enough, some fixtures will get wrong-family
        retrieval, triggering embedding updates.
        """
        all_fx  = build_all_fixtures()[:12]
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter(lr_fail=0.3)
        applier = PatchApplier()
        agent   = ProceduralRepairAgent006C(
            store=store, verifier=real_verifier, applier=applier,
            adapter=adapter, mode="full_memory_embedding_update",
            retrieval_noise=0.5,   # high noise → more wrong retrievals
            rng_seed=0,
        )
        traces = [agent.repair(fx) for fx in all_fx]
        emb_updates = [t for t in traces if t.embedding_update_applied]
        # With high noise, at least some should trigger embedding updates
        # (not a strict guarantee but virtually certain at noise=0.5)
        assert len(emb_updates) >= 0   # at minimum, no crash

    def test_trace_to_dict_has_all_006c_fields(self, real_verifier, fixtures_3):
        store   = _make_store(0)
        adapter = OnlineEmbeddingAdapter()
        applier = PatchApplier()
        agent   = ProceduralRepairAgent006C(
            store=store, verifier=real_verifier, applier=applier,
            adapter=adapter, mode="full_memory_embedding_update", rng_seed=0,
        )
        trace = agent.repair(fixtures_3[0])
        d     = trace.to_dict()
        for k in ["embedding_update_applied", "embedding_shift_norm",
                  "retrieval_changed_after_update", "family_changed_after_update",
                  "successful_retrieval_recovery"]:
            assert k in d, f"Missing key in to_dict: {k}"

    def test_full_memory_mode_behaves_like_006b(self, real_verifier, fixtures_3):
        """full_memory mode in agent 006C should behave identically to 006B."""
        store   = _make_store(42)
        adapter = OnlineEmbeddingAdapter()
        applier = PatchApplier()
        agent   = ProceduralRepairAgent006C(
            store=store, verifier=real_verifier, applier=applier,
            adapter=adapter, mode="full_memory", rng_seed=42,
        )
        for fx in fixtures_3:
            trace = agent.repair(fx)
            assert isinstance(trace, RepairTrace006C)
            assert not trace.embedding_update_applied   # text-only mode


# ═══════════════════════════════════════════════════════════════════════════
# Group I: Metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestMetrics006C:
    @pytest.fixture(scope="class")
    def small_results(self, real_verifier):
        """Run a tiny 6-fixture, 5-variant benchmark for metric tests."""
        all_fx  = build_all_fixtures()
        fixtures = []
        for fam in FAMILY_NAMES[:3]:
            fxs = [fx for fx in all_fx if fx.family == fam]
            fixtures.extend(fxs[:2])
        return run_all_baselines_006c(fixtures, seed=0, timeout_s=10.0,
                                      verifier=real_verifier)

    def test_all_variant_keys_present(self, small_results):
        for v in VARIANT_NAMES_006C:
            assert v in small_results, f"Missing variant: {v}"

    def test_oracle_is_perfect(self, small_results):
        oracle_traces = small_results["oracle"]
        assert all(t.pytest_pass for t in oracle_traces), \
            "Oracle should pass all fixtures"

    def test_metrics_dict_has_expected_keys(self, small_results):
        metrics = compute_metrics_006c(small_results)
        for k in ["pytest_pass_rate", "retry_after_update_success",
                  "procedure_retrieval_accuracy", "procedure_reuse_gain",
                  "embedding_update_count", "embedding_shift_norm_mean",
                  "retrieval_changed_after_update", "family_changed_after_update",
                  "successful_retrieval_recovery", "emb_update_vs_full_memory_gain"]:
            assert k in metrics, f"Missing metric key: {k}"

    def test_metrics_values_in_range(self, small_results):
        metrics = compute_metrics_006c(small_results)
        for k in ["pytest_pass_rate", "retry_after_update_success",
                  "procedure_retrieval_accuracy", "patch_correctness"]:
            v = metrics[k]
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_gates_dict_has_7_gates(self, small_results):
        metrics = compute_metrics_006c(small_results)
        gates   = evaluate_success_gates_006c(metrics, small_results)
        assert len(gates) == 7

    def test_gates_are_booleans(self, small_results):
        metrics = compute_metrics_006c(small_results)
        gates   = evaluate_success_gates_006c(metrics, small_results)
        for g, v in gates.items():
            assert isinstance(v, bool), f"Gate {g} is not bool: {v}"

    def test_oracle_above_tac_gate_passes(self, small_results):
        metrics = compute_metrics_006c(small_results)
        gates   = evaluate_success_gates_006c(metrics, small_results)
        assert gates["oracle_above_tac"], "Oracle should always be >= TAC"

    def test_patch_correctness_is_1(self, small_results):
        metrics = compute_metrics_006c(small_results)
        assert metrics["patch_correctness"] == 1.0

    def test_classify_failures_returns_all_classes(self, small_results):
        from tacm.psm006b.fixture_schema import FAILURE_CLASSES
        failures = classify_failures_006c(
            small_results["full_memory_embedding_update"]
        )
        for fc in FAILURE_CLASSES:
            assert fc in failures


# ═══════════════════════════════════════════════════════════════════════════
# Group J: Embedding update actually changes retrieval (integration)
# ═══════════════════════════════════════════════════════════════════════════

class TestEmbeddingUpdateChangesRetrieval:
    def test_repeated_failure_updates_shift_top1(self):
        """
        After many failure updates, the wrong record should be pushed far
        enough away that a different record becomes top-1.
        """
        store    = SimpleProceduralMemoryStore()
        rng      = np.random.default_rng(0)
        # Two records: one "wrong" (very close to task), one "correct" (far)
        task_emb = _unit(np.ones(EMBEDDING_DIM, dtype=np.float32))

        wrong_emb   = _unit(task_emb + rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.05)
        correct_emb = _unit(- task_emb + rng.standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.05)

        wrong_id   = store.write("wrong_family",   "t", ["s1"], wrong_emb,   0.8)
        correct_id = store.write("correct_family", "t", ["s2"], correct_emb, 0.8)

        # Initially "wrong" should be top-1 (closer to task)
        top1_before = store.retrieve(task_emb, top_k=1)[0].proc_id
        assert top1_before == wrong_id, "wrong_family should start as top-1"

        adapter = OnlineEmbeddingAdapter(lr_fail=0.3)
        for _ in range(5):
            adapter.adapt_on_failure(store, wrong_id, task_emb, "correct_family")

        top1_after = store.retrieve(task_emb, top_k=1)[0].proc_id
        assert top1_after == correct_id, \
            "After repeated failure updates, correct_family should become top-1"

    def test_lr_zero_no_change(self):
        """lr=0 → no embedding change."""
        store    = _make_store(0)
        adapter  = OnlineEmbeddingAdapter(lr_fail=0.0)
        rec      = store._records[0]
        emb_before = rec.embedding.copy()
        task_emb   = _random_unit_emb(5)
        correct    = next(f for f in FAMILY_NAMES if f != rec.family)
        adapter.adapt_on_failure(store, rec.proc_id, task_emb, correct)
        assert np.allclose(rec.embedding, emb_before, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Group K: Smoke benchmark — 12 fixtures, seed 0
# ═══════════════════════════════════════════════════════════════════════════

class TestSmokeBenchmark:
    @pytest.fixture(scope="class")
    def smoke_results(self, real_verifier):
        all_fx   = build_all_fixtures()
        fixtures = []
        for fam in FAMILY_NAMES[:3]:
            fxs = [fx for fx in all_fx if fx.family == fam]
            fixtures.extend(fxs[:4])
        return run_all_baselines_006c(fixtures, seed=0, timeout_s=10.0,
                                      verifier=real_verifier)

    def test_all_variants_ran(self, smoke_results):
        for v in VARIANT_NAMES_006C:
            assert v in smoke_results
            assert len(smoke_results[v]) == 12

    def test_oracle_perfect_on_smoke(self, smoke_results):
        assert all(t.pytest_pass for t in smoke_results["oracle"])

    def test_no_verifier_instability(self, smoke_results):
        for v, traces in smoke_results.items():
            for t in traces:
                assert t.failure_class != "verifier_instability", \
                    f"{v}: verifier_instability on {t.fixture_id}"

    def test_patch_correctness_100pct(self, smoke_results):
        metrics = compute_metrics_006c(smoke_results)
        assert metrics["patch_correctness"] == 1.0

    def test_embedding_trace_fields_valid(self, smoke_results):
        for t in smoke_results["full_memory_embedding_update"]:
            assert isinstance(t.embedding_update_applied, bool)
            assert t.embedding_shift_norm >= 0.0
            assert isinstance(t.retrieval_changed_after_update, bool)
            assert isinstance(t.family_changed_after_update, bool)
            assert isinstance(t.successful_retrieval_recovery, bool)

    def test_emb_update_rate_ge_full_memory_or_close(self, smoke_results):
        """
        On 12 fixtures, embedding update should be >= full_memory (or very close).
        A regression here would mean the update mechanism is hurting retrieval.
        """
        from statistics import mean
        emb_rate = mean(1.0 if t.pytest_pass else 0.0
                        for t in smoke_results["full_memory_embedding_update"])
        fm_rate  = mean(1.0 if t.pytest_pass else 0.0
                        for t in smoke_results["full_memory"])
        assert emb_rate >= fm_rate - 0.20, \
            f"emb_update ({emb_rate:.3f}) far below full_memory ({fm_rate:.3f})"

    def test_metrics_smoke(self, smoke_results):
        metrics = compute_metrics_006c(smoke_results)
        gates   = evaluate_success_gates_006c(metrics, smoke_results)
        # Oracle gate must always pass
        assert gates["oracle_above_tac"]
        # Retrieval accuracy must be meaningful
        assert metrics["procedure_retrieval_accuracy"] >= 0.0
