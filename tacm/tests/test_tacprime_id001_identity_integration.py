"""
tests/test_tacprime_id001_identity_integration.py

Integration tests for TAC-Prime-ID001: Identity-Carried Structure Memory.

Section A — Pure NumPy simulation integration tests (always run):
  - Carried state improves structure retrieval vs memory knockout
  - Memory knockout drops retrieval accuracy to zero
  - Carried retrieval beats random chance across seeds
  - State accumulates non-trivially after carried run
  - Shuffled state condition runs without error
  - Reset condition returns valid accuracy
  - Family routes and active IDs are populated
  - Benchmark run_seed and aggregate execute correctly

Section B — PyTorch TACSM integration tests (skipped if torch unavailable):
  - TACSM forward with use_identity_field=True / False
  - TACSMOutput contains identity slots when ON, None when OFF
  - identity_state carries across sequential calls
  - aux losses present in output
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

from tacm.id001.state      import identity_state_zeros
from tacm.id001.memory     import IdentityStructureMemory, IdentityProceduralMemory, _normalize
from tacm.id001.routing    import IdentityRouter
from tacm.id001.simulation import (
    make_tasks, seed_memory,
    run_condition_carried, run_condition_reset,
    run_condition_shuffled, run_condition_memory_knockout,
)


# ── Shared helpers ──────────────────────────────────────────────────────────

N_FAM  = 4
D_SIM  = 16
TASKS  = 10


def _make_sim(seed: int = 0):
    router     = IdentityRouter(d_model=D_SIM, n_identities=N_FAM * 2, seed=seed)
    struct_mem = IdentityStructureMemory(embedding_dim=D_SIM)
    proc_mem   = IdentityProceduralMemory(embedding_dim=D_SIM)
    tasks, centroids = make_tasks(N_FAM, TASKS, D_SIM, seed=seed)
    seed_memory(struct_mem, proc_mem, centroids, N_FAM, router=router, rng_seed=seed)
    return router, struct_mem, proc_mem, tasks, centroids


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: NumPy simulation integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulationConditions:

    def test_carried_ge_memory_knockout(self):
        router, sm, pm, tasks, _ = _make_sim(seed=0)
        s_c, _, _, _, _ = run_condition_carried(tasks, router, sm, pm, n_families=N_FAM)
        s_ko = run_condition_memory_knockout(tasks[:TASKS], router, sm)
        assert s_c >= s_ko

    def test_memory_knockout_is_zero(self):
        router, sm, pm, tasks, _ = _make_sim(seed=1)
        s_ko = run_condition_memory_knockout(tasks[:TASKS], router, sm)
        assert s_ko == pytest.approx(0.0)

    def test_carried_struct_above_random(self):
        accs = []
        for seed in range(3):
            router, sm, pm, tasks, _ = _make_sim(seed=seed)
            s_c, _, _, _, _ = run_condition_carried(tasks, router, sm, pm, n_families=N_FAM)
            accs.append(s_c)
        assert sum(accs) / len(accs) > 1.0 / N_FAM

    def test_proc_retrieval_in_range(self):
        router, sm, pm, tasks, _ = _make_sim(seed=2)
        _, p_c, _, _, _ = run_condition_carried(tasks, router, sm, pm, n_families=N_FAM)
        assert 0.0 <= p_c <= 1.0

    def test_state_accumulates_after_carried(self):
        router, sm, pm, tasks, _ = _make_sim(seed=3)
        _, _, states, _, _ = run_condition_carried(tasks, router, sm, pm, n_families=N_FAM)
        assert not np.allclose(states[-1].identity_memory, 0.0)

    def test_shuffled_does_not_crash(self):
        router, sm, pm, tasks, _ = _make_sim(seed=4)
        _, _, states, _, _ = run_condition_carried(tasks, router, sm, pm, n_families=N_FAM)
        s_s = run_condition_shuffled(tasks, router, sm, states)
        assert 0.0 <= s_s <= 1.0

    def test_reset_struct_retrieval_in_range(self):
        router, sm, pm, tasks, _ = _make_sim(seed=5)
        s_r, p_r, _ = run_condition_reset(tasks, router, sm, pm, n_families=N_FAM)
        assert 0.0 <= s_r <= 1.0
        assert 0.0 <= p_r <= 1.0

    def test_family_routes_populated(self):
        router, sm, pm, tasks, _ = _make_sim(seed=6)
        _, _, _, fam_routes, _ = run_condition_carried(tasks, router, sm, pm, n_families=N_FAM)
        for fid in range(N_FAM):
            assert fid in fam_routes and len(fam_routes[fid]) > 0

    def test_active_ids_per_family(self):
        router, sm, pm, tasks, _ = _make_sim(seed=7)
        _, _, _, _, fam_actids = run_condition_carried(tasks, router, sm, pm, n_families=N_FAM)
        for fid in range(N_FAM):
            assert fid in fam_actids and len(fam_actids[fid]) > 0


class TestBenchmarkRunSeed:

    def _load_bench(self):
        import importlib.util, pathlib, sys
        bench_path = (
            pathlib.Path(__file__).parent.parent
            / "experiments"
            / "benchmark_tacprime_id001_identity_carried_structure_memory.py"
        )
        spec  = importlib.util.spec_from_file_location("_tacm_bench_id001", bench_path)
        bench = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = bench   # register so dataclass can resolve __module__
        spec.loader.exec_module(bench)
        return bench

    def test_run_seed_returns_valid_result(self):
        bench  = self._load_bench()
        result = bench.run_seed(seed=0)
        assert 0.0 <= result.carried_structure_retrieval <= 1.0
        assert 0.0 <= result.reset_structure_retrieval   <= 1.0
        assert 0.0 <= result.carried_procedure_retrieval <= 1.0
        assert 0.0 <= result.reset_procedure_retrieval   <= 1.0
        assert 0.0 <= result.identity_specialization     <= 1.0

    def test_aggregate_contains_benchmark_score(self):
        bench   = self._load_bench()
        results = [bench.run_seed(s) for s in range(2)]
        metrics = bench.aggregate(results)
        assert "benchmark_score" in metrics
        assert np.isfinite(metrics["benchmark_score"])

    def test_carried_struct_ge_reset_struct(self):
        """Core hypothesis: carried ≥ reset for structure retrieval."""
        bench   = self._load_bench()
        results = [bench.run_seed(s) for s in range(3)]
        metrics = bench.aggregate(results)
        assert metrics["carried_structure_retrieval"] >= metrics["reset_structure_retrieval"]

    def test_memory_knockout_is_zero(self):
        bench   = self._load_bench()
        results = [bench.run_seed(s) for s in range(2)]
        metrics = bench.aggregate(results)
        assert metrics["memory_knockout_retrieval"] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: PyTorch TACSM integration (skipped if no torch)
# ─────────────────────────────────────────────────────────────────────────────

VOCAB   = 128
D_MODEL = 32
T_SEQ   = 6
B_SZ    = 2
N_IDS   = 4


def _tiny_cfg(use_identity: bool = True):
    from tacm.config import TACSMConfig, tacm_30m
    cfg = tacm_30m()
    tc  = cfg.transformer
    tc.vocab_size   = VOCAB
    tc.d_model      = D_MODEL
    tc.n_layers     = 2
    tc.n_heads      = 2
    tc.n_kv_heads   = 1
    tc.ffn_dim      = D_MODEL * 2
    tc.max_seq_len  = 32
    tc.use_flash_attn = False
    tc.gradient_checkpointing = False
    cfg.concept_volume.volume_dim         = 16
    cfg.concept_volume.n_concept_families = N_IDS
    cfg.router.n_families   = N_IDS
    cfg.router.n_experts    = N_IDS
    cfg.router.top_k        = 1
    cfg.expert.n_experts    = N_IDS
    cfg.expert.expert_hidden_dim  = D_MODEL
    cfg.expert.shared_expert_dim  = D_MODEL
    cfg.memory.max_structures     = 64
    cfg.memory.embedding_dim      = D_MODEL
    cfg.memory.retrieval_top_k    = 2
    cfg.memory.write_threshold    = 0.0
    cfg.verifier.hidden_dim       = 16
    cfg.multi_token.n_future_tokens  = 2
    cfg.multi_token.n_future_actions = 1
    cfg.identity.n_identities              = N_IDS
    cfg.identity.identity_energy_budget    = 2.0
    cfg.identity.use_identity_field        = use_identity
    cfg.identity.identity_residual_scale   = 0.5
    cfg.identity.identity_router_bias_scale = 0.25
    cfg.identity.identity_memory_bias_scale = 0.25
    return cfg


@skip_no_torch
class TestTorchTACSMIntegration:

    @pytest.fixture
    def m_on(self):
        from tacm.model import TACSM
        torch.manual_seed(42)
        return TACSM(_tiny_cfg(True)).eval()

    @pytest.fixture
    def m_off(self):
        from tacm.model import TACSM
        torch.manual_seed(42)
        return TACSM(_tiny_cfg(False)).eval()

    @pytest.fixture
    def ids(self):
        torch.manual_seed(7)
        return torch.randint(0, VOCAB, (B_SZ, T_SEQ))

    def test_forward_on(self, m_on, ids):
        with torch.no_grad():
            out = m_on(ids)
        assert out.lm_logits.shape == (B_SZ, T_SEQ, VOCAB)
        assert torch.isfinite(out.lm_logits).all()

    def test_forward_off(self, m_off, ids):
        with torch.no_grad():
            out = m_off(ids)
        assert out.lm_logits.shape == (B_SZ, T_SEQ, VOCAB)
        assert torch.isfinite(out.lm_logits).all()

    def test_identity_state_present_when_on(self, m_on, ids):
        with torch.no_grad():
            out = m_on(ids)
        assert out.identity_state    is not None
        assert out.active_identity   is not None
        assert out.identity_coherence is not None

    def test_identity_state_none_when_off(self, m_off, ids):
        with torch.no_grad():
            out = m_off(ids)
        assert out.identity_state  is None
        assert out.active_identity is None

    def test_stability_shape(self, m_on, ids):
        with torch.no_grad():
            out = m_on(ids)
        assert out.identity_state.stability.shape == (B_SZ, N_IDS)

    def test_carried_differs_from_fresh(self, m_on, ids):
        with torch.no_grad():
            out1   = m_on(ids, identity_state=None)
            s1     = out1.identity_state
            torch.manual_seed(13)
            ids2   = torch.randint(0, VOCAB, (B_SZ, T_SEQ))
            out2_c = m_on(ids2, identity_state=s1)
            out2_f = m_on(ids2, identity_state=None)
        assert not torch.allclose(
            out2_c.identity_state.identity_memory,
            out2_f.identity_state.identity_memory, atol=1e-4,
        )

    def test_aux_losses_in_output(self, m_on, ids):
        with torch.no_grad():
            out = m_on(ids)
        assert out.identity_aux is not None
        assert "identity_reuse" in out.identity_aux


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
