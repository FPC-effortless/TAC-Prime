"""
tests/test_identity_field.py

Unit tests for TAC-Prime-ID001 identity components.

Section A — Pure NumPy tests (always run, no torch required):
  - IdentityStateNP shapes and updates
  - IdentityRouter routing shapes, energy budget, state persistence
  - IdentityStructureMemory identity-match bonus
  - IdentityProceduralMemory identity-match bonus
  - Route consistency and specialization metrics

Section B — PyTorch tests (marked skip when torch unavailable):
  - IdentityFieldLayer forward shapes
  - IdentityState persists across calls
  - Aux losses are finite scalars
  - Gradient flow
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


from tacm.id001.state  import IdentityStateNP, identity_state_zeros, decay_identity_state
from tacm.id001.memory import (
    IdentityStructureMemory, IdentityProceduralMemory, _normalize,
)
from tacm.id001.routing import (
    IdentityRouter, compute_route_consistency, compute_identity_specialization,
)


D   = 16
N_ID = 4


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: Pure NumPy tests (always run)
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityStateNP:

    def test_zeros_shapes(self):
        s = identity_state_zeros(N_ID, D)
        assert s.stability.shape       == (N_ID,)
        assert s.identity_memory.shape == (N_ID, D)
        assert s.route_history.shape   == (N_ID,)
        assert s.active_identity       == 0

    def test_zeros_values(self):
        s = identity_state_zeros(N_ID, D)
        assert np.allclose(s.stability, 1.0)
        assert np.allclose(s.identity_memory, 0.0)
        assert np.allclose(s.route_history,   0.0)

    def test_copy_is_independent(self):
        s  = identity_state_zeros(N_ID, D)
        s2 = s.copy()
        s2.stability[:] = 99.0
        assert not np.allclose(s.stability, 99.0), "copy should be independent"

    def test_decay_updates_memory(self):
        s  = identity_state_zeros(N_ID, D)
        rng = np.random.default_rng(0)
        h  = rng.standard_normal(D).astype(np.float32)
        w  = np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32)
        e  = w * 8
        s2 = decay_identity_state(s, h, w, e, energy_budget=4.0, decay=0.8)
        assert s2.identity_memory.shape == (N_ID, D)
        assert not np.allclose(s2.identity_memory, 0.0)

    def test_decay_stability_updates(self):
        s  = identity_state_zeros(N_ID, D)
        rng = np.random.default_rng(1)
        h  = rng.standard_normal(D).astype(np.float32)
        w  = np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float32)
        e  = w * 8
        s2 = decay_identity_state(s, h, w, e, energy_budget=4.0, decay=0.8)
        assert not np.allclose(s2.stability, 1.0)

    def test_decay_active_identity_is_argmax(self):
        s  = identity_state_zeros(N_ID, D)
        rng = np.random.default_rng(2)
        h  = rng.standard_normal(D).astype(np.float32)
        w  = np.array([0.05, 0.05, 0.80, 0.10], dtype=np.float32)
        e  = w * 8
        s2 = decay_identity_state(s, h, w, e, energy_budget=4.0, decay=0.8)
        assert s2.active_identity == 2


class TestIdentityRouter:

    def setup_method(self):
        self.router = IdentityRouter(d_model=D, n_identities=N_ID, seed=42)

    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(0)
        h   = _normalize(rng.standard_normal(D).astype(np.float32))
        _, _, weights = self.router.forward(h, state=None)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_weights_non_negative(self):
        rng = np.random.default_rng(1)
        h   = _normalize(rng.standard_normal(D).astype(np.float32))
        _, _, weights = self.router.forward(h, state=None)
        assert (weights >= 0).all()

    def test_active_identity_in_range(self):
        rng = np.random.default_rng(2)
        h   = _normalize(rng.standard_normal(D).astype(np.float32))
        _, active_id, _ = self.router.forward(h, state=None)
        assert 0 <= active_id < N_ID

    def test_state_updates_across_calls(self):
        rng   = np.random.default_rng(3)
        state = None
        for _ in range(3):
            h = _normalize(rng.standard_normal(D).astype(np.float32))
            state, _, _ = self.router.forward(h, state=state)
        assert not np.allclose(state.identity_memory, 0.0)

    def test_different_inputs_different_states(self):
        h_a    = _normalize(np.array([1.0] * D, dtype=np.float32))
        h_b    = _normalize(np.array([-1.0] * D, dtype=np.float32))
        state_a = state_b = None
        for _ in range(5):
            state_a, _, _ = self.router.forward(h_a, state=state_a)
            state_b, _, _ = self.router.forward(h_b, state=state_b)
        assert not np.allclose(state_a.identity_memory, state_b.identity_memory)

    def test_carried_active_id_stable_after_warmup(self):
        """After warm-up on same family, carried active_id converges."""
        h    = _normalize(self.router.identity_embeddings[0] + 0.1 *
                          np.random.default_rng(9).standard_normal(D).astype(np.float32))
        state = None
        ids   = []
        for _ in range(12):
            state, aid, _ = self.router.forward(h, state=state)
            ids.append(aid)
        # Dominant identity after warm-up should be stable
        most_common = max(set(ids[-4:]), key=ids[-4:].count)
        assert ids[-4:].count(most_common) >= 3


class TestIdentityStructureMemory:

    def _make_mem(self) -> IdentityStructureMemory:
        return IdentityStructureMemory(embedding_dim=D, max_structures=64)

    def test_write_and_retrieve(self):
        mem = self._make_mem()
        rng = np.random.default_rng(0)
        emb = _normalize(rng.standard_normal(D).astype(np.float32))
        mem.write(emb, family_id=2, expert_id=0, task_type="t",
                  success_score=0.9, identity_id=2)
        recs = mem.retrieve(emb, top_k=1)
        assert len(recs) == 1 and recs[0].family_id == 2

    def test_identity_bonus_boosts_matching(self):
        mem = self._make_mem()
        rng = np.random.default_rng(1)
        emb = _normalize(rng.standard_normal(D).astype(np.float32))
        mem.write(emb.copy(), family_id=0, expert_id=0, task_type="t",
                  success_score=0.8, identity_id=3)
        mem.write(emb.copy(), family_id=1, expert_id=0, task_type="t",
                  success_score=0.8, identity_id=7)
        recs3 = mem.retrieve(emb, top_k=1, active_identity_id=3,
                              identity_memory_bias_scale=0.25)
        recs7 = mem.retrieve(emb, top_k=1, active_identity_id=7,
                              identity_memory_bias_scale=0.25)
        assert recs3[0].identity_id == 3
        assert recs7[0].identity_id == 7

    def test_no_identity_graceful(self):
        mem = self._make_mem()
        rng = np.random.default_rng(2)
        emb = _normalize(rng.standard_normal(D).astype(np.float32))
        mem.write(emb.copy(), family_id=0, expert_id=0, task_type="t",
                  success_score=0.9)
        recs = mem.retrieve(emb, top_k=1, active_identity_id=5,
                             identity_memory_bias_scale=0.25)
        assert len(recs) == 1

    def test_retrieve_returns_top_k(self):
        mem = self._make_mem()
        rng = np.random.default_rng(3)
        for fid in range(6):
            emb = _normalize(rng.standard_normal(D).astype(np.float32))
            mem.write(emb, family_id=fid, expert_id=0, task_type="t",
                      success_score=0.9)
        q    = _normalize(rng.standard_normal(D).astype(np.float32))
        recs = mem.retrieve(q, top_k=3)
        assert len(recs) == 3

    def test_clear_empties_store(self):
        mem = self._make_mem()
        rng = np.random.default_rng(4)
        emb = _normalize(rng.standard_normal(D).astype(np.float32))
        mem.write(emb, family_id=0, expert_id=0, task_type="t", success_score=0.9)
        assert len(mem) == 1
        mem.clear()
        assert len(mem) == 0

    def test_empty_memory_returns_empty_list(self):
        mem  = self._make_mem()
        rng  = np.random.default_rng(5)
        q    = _normalize(rng.standard_normal(D).astype(np.float32))
        assert mem.retrieve(q, top_k=4) == []

    def test_write_threshold_respected(self):
        mem = IdentityStructureMemory(embedding_dim=D, write_threshold=0.7)
        rng = np.random.default_rng(6)
        emb = _normalize(rng.standard_normal(D).astype(np.float32))
        sid = mem.write(emb, family_id=0, expert_id=0, task_type="t",
                        success_score=0.5)
        assert sid is None and len(mem) == 0


class TestIdentityProceduralMemory:

    def _make_mem(self) -> IdentityProceduralMemory:
        return IdentityProceduralMemory(embedding_dim=D, max_procedures=64)

    def test_write_and_retrieve(self):
        pm  = self._make_mem()
        rng = np.random.default_rng(0)
        emb = _normalize(rng.standard_normal(D).astype(np.float32))
        pm.write("FamA", "TypeA", ["s1", "s2"], emb, success_rate=0.8, identity_id=1)
        recs = pm.retrieve(emb, top_k=1)
        assert len(recs) == 1 and recs[0].family == "FamA"

    def test_identity_bonus_boosts_matching(self):
        pm  = self._make_mem()
        rng = np.random.default_rng(1)
        emb = _normalize(rng.standard_normal(D).astype(np.float32))
        pm.write("FamA", "TypeA", ["s1"], emb.copy(), success_rate=0.8, identity_id=2)
        pm.write("FamB", "TypeB", ["s1"], emb.copy(), success_rate=0.8, identity_id=9)
        recs2 = pm.retrieve(emb, top_k=1, active_identity_id=2,
                             identity_memory_bias_scale=0.25)
        recs9 = pm.retrieve(emb, top_k=1, active_identity_id=9,
                             identity_memory_bias_scale=0.25)
        assert recs2[0].identity_id == 2
        assert recs9[0].identity_id == 9

    def test_no_identity_graceful(self):
        pm  = self._make_mem()
        rng = np.random.default_rng(2)
        emb = _normalize(rng.standard_normal(D).astype(np.float32))
        pm.write("FamA", "TypeA", ["s1"], emb, success_rate=0.8)
        recs = pm.retrieve(emb, top_k=1, active_identity_id=42,
                            identity_memory_bias_scale=0.25)
        assert len(recs) == 1

    def test_empty_memory_returns_empty_list(self):
        pm  = self._make_mem()
        rng = np.random.default_rng(3)
        q   = _normalize(rng.standard_normal(D).astype(np.float32))
        assert pm.retrieve(q, top_k=2) == []


class TestMetrics:

    def test_route_consistency_perfect(self):
        routes = {0: [0, 0, 0, 0], 1: [2, 2, 2, 2]}
        score  = compute_route_consistency(routes, n_families=4)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_route_consistency_uniform(self):
        routes = {0: [0, 1, 2, 3] * 4}
        score  = compute_route_consistency(routes, n_families=4)
        assert score < 0.1

    def test_specialization_perfect(self):
        ids   = {0: [3, 3, 3, 3], 1: [1, 1, 1, 1]}
        score = compute_identity_specialization(ids, n_identities=N_ID)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_specialization_uniform(self):
        ids   = {0: list(range(N_ID)) * 4}
        score = compute_identity_specialization(ids, n_identities=N_ID)
        assert score < 0.5

    def test_empty_routes(self):
        score = compute_route_consistency({}, n_families=4)
        assert score == pytest.approx(0.0)

    def test_empty_ids(self):
        score = compute_identity_specialization({}, n_identities=N_ID)
        assert score == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: PyTorch tests (skipped if torch unavailable)
# ─────────────────────────────────────────────────────────────────────────────

D_TORCH  = 32
N_TORCH  = 8
B        = 2
T        = 6


@skip_no_torch
class TestTorchIdentityFieldLayer:

    @pytest.fixture
    def layer(self):
        from tacm.identity import IdentityFieldLayer
        torch.manual_seed(0)
        return IdentityFieldLayer(
            d_model               = D_TORCH,
            n_identities          = N_TORCH,
            identity_energy_budget= 4.0,
            identity_state_decay  = 0.8,
            identity_loss_weight  = 0.05,
        )

    @pytest.fixture
    def hidden(self):
        torch.manual_seed(1)
        return torch.randn(B, T, D_TORCH)

    def test_context_shape(self, layer, hidden):
        out = layer(hidden)
        assert out.identity_context.shape == (B, T, D_TORCH)

    def test_active_identity_shape(self, layer, hidden):
        out = layer(hidden)
        assert out.active_identity.shape == (B,)

    def test_coherence_shape(self, layer, hidden):
        out = layer(hidden)
        assert out.identity_coherence.shape == (B, N_TORCH, N_TORCH)

    def test_state_shapes(self, layer, hidden):
        out = layer(hidden)
        assert out.identity_state.stability.shape       == (B, N_TORCH)
        assert out.identity_state.identity_memory.shape == (B, N_TORCH, D_TORCH)

    def test_aux_losses_keys(self, layer, hidden):
        out = layer(hidden)
        expected = {"identity_reuse", "identity_energy",
                    "identity_coherence", "identity_separation"}
        assert expected == set(out.aux_losses.keys())

    def test_aux_losses_scalars_finite(self, layer, hidden):
        out = layer(hidden)
        for k, v in out.aux_losses.items():
            assert v.shape == (), f"{k} not scalar"
            assert torch.isfinite(v), f"{k} not finite"

    def test_state_persists(self, layer, hidden):
        from tacm.identity import IdentityFieldLayer
        out1   = layer(hidden)
        state1 = out1.identity_state
        torch.manual_seed(2)
        h2 = torch.randn(B, T, D_TORCH)
        out2_c = layer(h2, state=state1)
        out2_f = layer(h2, state=None)
        assert not torch.allclose(
            out2_c.identity_state.identity_memory,
            out2_f.identity_state.identity_memory, atol=1e-4,
        )

    def test_backward_aux(self, layer, hidden):
        out   = layer(hidden)
        total = sum(out.aux_losses.values())
        total.backward()
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_backward_context(self, layer, hidden):
        out  = layer(hidden)
        loss = out.identity_context.mean()
        loss.backward()
        grads = [p.grad for p in layer.parameters() if p.grad is not None]
        assert len(grads) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
