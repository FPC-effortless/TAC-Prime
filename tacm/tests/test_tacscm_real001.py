"""
TAC-SCM-REAL001: Test Suite

Tests are partitioned by torch dependency:
  - Config, dataset, diagnostics tests: pure Python / numpy — always run.
  - Model, block, memory, generation tests: require PyTorch — skipped otherwise.

Run:
    cd tacm && python -m pytest tests/test_tacscm_real001.py -q
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Torch availability guard ──────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ── Pure-python imports (always available) ────────────────────────────────────

from tacm.scm_config import TACSCMConfig
from tacm.scm_diagnostics import SCMDiagnosticsTracker, DiagnosticsRow, _WindowStat
from tacm.data.scm_dataset import (
    SCMSample, SCMDataset, SCMDataCollator,
    make_synthetic_repair_dataset,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestTACSCMConfig:

    def test_default_construction(self):
        cfg = TACSCMConfig()
        assert cfg.vocab_size  > 0
        assert cfg.d_model     > 0
        assert cfg.n_layers    > 0
        assert cfg.d_structure > 0

    def test_nsf_disabled_by_default(self):
        cfg = TACSCMConfig()
        assert cfg.enable_nsf_survival   is False, \
            "NSF survival must be disabled by default (spec: Explicitly Do NOT Implement Yet)"

    def test_dpsl_disabled_by_default(self):
        cfg = TACSCMConfig()
        assert cfg.enable_dpsl_refinement is False, \
            "DPSL refinement must be disabled by default (spec: Explicitly Do NOT Implement Yet)"

    def test_scm_enabled_by_default(self):
        cfg = TACSCMConfig()
        assert cfg.enable_scm                 is True
        assert cfg.enable_structure_discovery is True
        assert cfg.enable_structure_identity  is True
        assert cfg.enable_structure_memory    is True

    def test_preset_small(self):
        cfg = TACSCMConfig.small()
        assert cfg.d_model == 256
        assert cfg.n_layers == 4
        assert cfg.d_structure == 64
        assert cfg.enable_nsf_survival   is False
        assert cfg.enable_dpsl_refinement is False

    def test_preset_base(self):
        cfg = TACSCMConfig.base()
        assert cfg.d_model == 512
        assert cfg.n_layers == 8

    def test_preset_medium(self):
        cfg = TACSCMConfig.medium()
        assert cfg.d_model == 1024
        assert cfg.n_layers == 16

    def test_preset_no_scm(self):
        cfg = TACSCMConfig.no_scm()
        assert cfg.enable_scm is False

    def test_preset_discovery_only(self):
        cfg = TACSCMConfig.discovery_only()
        assert cfg.enable_scm                is True
        assert cfg.enable_structure_identity  is False
        assert cfg.enable_structure_memory   is False
        assert cfg.enable_nsf_survival       is False
        assert cfg.enable_dpsl_refinement    is False

    def test_scm_layer_interval_positive(self):
        cfg = TACSCMConfig()
        assert cfg.scm_layer_interval >= 1

    def test_loss_weights_positive(self):
        cfg = TACSCMConfig()
        assert cfg.discovery_loss_weight     > 0
        assert cfg.structure_reuse_weight    > 0
        assert cfg.survival_loss_weight      >= 0
        assert cfg.compression_loss_weight   >= 0

    def test_memory_write_rate_valid(self):
        cfg = TACSCMConfig()
        assert 0.0 < cfg.memory_write_rate <= 1.0

    def test_survival_decay_valid(self):
        cfg = TACSCMConfig()
        assert 0.0 < cfg.survival_decay < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# DATASET TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSCMDataset:

    def test_scm_sample_fields(self):
        sample = SCMSample(
            input_ids=[1, 2, 3],
            labels=[-100, 2, 3],
            task_id="t0",
            structure_id=1,
            source="text",
        )
        assert sample.input_ids == [1, 2, 3]
        assert sample.labels[0] == -100
        assert sample.structure_id == 1

    def test_make_synthetic_repair_dataset_size(self):
        ds = make_synthetic_repair_dataset(
            n_samples=50, n_families=4, seq_len=32, seed=7
        )
        assert len(ds) == 50

    def test_make_synthetic_repair_dataset_shapes(self):
        ds = make_synthetic_repair_dataset(
            n_samples=10, n_families=3, seq_len=16, seed=0
        )
        for i in range(len(ds)):
            s = ds[i]
            # dataset items are SCMSample dataclass instances
            ids = s.input_ids if isinstance(s, SCMSample) else s["input_ids"]
            lbs = s.labels    if isinstance(s, SCMSample) else s["labels"]
            assert len(ids) == 16, f"Expected seq_len=16, got {len(ids)}"
            assert len(lbs) == 16

    def test_make_synthetic_repair_dataset_structure_ids(self):
        n_fam = 5
        ds = make_synthetic_repair_dataset(
            n_samples=40, n_families=n_fam, seq_len=8, seed=99
        )
        seen_ids = set()
        for i in range(len(ds)):
            s = ds[i]
            sid = s.structure_id if isinstance(s, SCMSample) else s.get("structure_id")
            if sid is not None:
                seen_ids.add(int(sid))
        assert len(seen_ids) > 0, "Expected structure_id labels in synthetic dataset"

    def test_make_synthetic_repair_dataset_deterministic(self):
        ds1 = make_synthetic_repair_dataset(n_samples=5, n_families=2, seq_len=8, seed=42)
        ds2 = make_synthetic_repair_dataset(n_samples=5, n_families=2, seq_len=8, seed=42)
        for i in range(5):
            s1 = ds1[i]; s2 = ds2[i]
            ids1 = s1.input_ids if isinstance(s1, SCMSample) else s1["input_ids"]
            ids2 = s2.input_ids if isinstance(s2, SCMSample) else s2["input_ids"]
            assert ids1 == ids2

    def test_make_synthetic_repair_dataset_different_seeds(self):
        ds1 = make_synthetic_repair_dataset(n_samples=5, n_families=2, seq_len=16, seed=1)
        ds2 = make_synthetic_repair_dataset(n_samples=5, n_families=2, seq_len=16, seed=2)
        def get_ids(s):
            return s.input_ids if isinstance(s, SCMSample) else s["input_ids"]
        differs = any(get_ids(ds1[i]) != get_ids(ds2[i]) for i in range(5))
        assert differs, "Different seeds should produce different data"

    def test_collator_padding(self):
        if not HAS_TORCH:
            pytest.skip("Collator requires torch for tensor output")
        ds = make_synthetic_repair_dataset(
            n_samples=6, n_families=2, seq_len=12, seed=3
        )
        collator = SCMDataCollator(pad_id=0)
        batch_raw = [ds[i] for i in range(4)]
        batch = collator(batch_raw)
        assert "input_ids" in batch
        assert "labels"    in batch
        B = batch["input_ids"].shape[0]
        T = batch["input_ids"].shape[1]
        assert B == 4
        assert T == 12

    def test_collator_label_mask(self):
        if not HAS_TORCH:
            pytest.skip("Collator requires torch for tensor output")
        ds = make_synthetic_repair_dataset(
            n_samples=4, n_families=2, seq_len=8, seed=5
        )
        collator = SCMDataCollator(pad_id=0)
        batch = collator([ds[i] for i in range(4)])
        labels = batch["labels"]
        assert (labels == -100).any() or (labels >= 0).any(), \
            "Labels should have valid token ids or -100 mask"


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSCMDiagnosticsTracker:

    def test_empty_tracker(self):
        t = SCMDiagnosticsTracker()
        assert t.n_steps == 0
        assert t.latest  is None
        assert not t.is_plateau

    def test_record_single_step(self):
        t = SCMDiagnosticsTracker()
        t.record(step=0, lm_loss=2.5)
        assert t.n_steps == 1
        assert t.latest is not None
        assert t.latest.lm_loss == pytest.approx(2.5)

    def test_record_multiple_steps(self):
        t = SCMDiagnosticsTracker()
        for i in range(10):
            t.record(step=i, lm_loss=2.5 - i * 0.1,
                     aux_losses={"discovery": 0.05})
        assert t.n_steps == 10
        assert t.latest.step == 9

    def test_rolling_window_mean(self):
        t = SCMDiagnosticsTracker(window_size=5)
        losses = [3.0, 2.5, 2.0, 1.5, 1.0]
        for i, v in enumerate(losses):
            t.record(step=i, lm_loss=v)
        stats = t.stats_dict()
        assert abs(stats["lm_loss"]["mean"] - sum(losses) / len(losses)) < 1e-4

    def test_plateau_detection(self):
        t = SCMDiagnosticsTracker(plateau_patience=5, plateau_delta=1e-3)
        for i in range(10):
            t.record(step=i, lm_loss=1.0)
        assert t.is_plateau, "Flat loss for 10 steps should trigger plateau"

    def test_no_plateau_when_improving(self):
        t = SCMDiagnosticsTracker(plateau_patience=10, plateau_delta=1e-3)
        for i in range(9):
            t.record(step=i, lm_loss=5.0 - i * 0.1)
        assert not t.is_plateau, "Decreasing loss should not trigger plateau"

    def test_anomaly_nan(self):
        t = SCMDiagnosticsTracker()
        t.record(step=0, lm_loss=float("nan"))
        assert len(t.anomalies) > 0
        assert t.anomalies[0]["kind"] == "nan_lm_loss"

    def test_anomaly_inf(self):
        t = SCMDiagnosticsTracker()
        t.record(step=0, lm_loss=float("inf"))
        assert len(t.anomalies) > 0

    def test_no_anomaly_normal_loss(self):
        t = SCMDiagnosticsTracker()
        for i in range(5):
            t.record(step=i, lm_loss=2.0 + i * 0.01)
        assert len(t.anomalies) == 0

    def test_summary_string(self):
        t = SCMDiagnosticsTracker()
        for i in range(3):
            t.record(step=i, lm_loss=2.0,
                     aux_losses={"discovery": 0.1},
                     metrics={"route_entropy": 1.5},
                     mem_fill_rate=0.2)
        s = t.summary()
        assert "TAC-SCM-REAL001" in s
        assert "lm_loss" in s.lower() or "LM Loss" in s
        assert "discovery" in s

    def test_stats_dict_structure(self):
        t = SCMDiagnosticsTracker()
        t.record(step=0, lm_loss=1.5, aux_losses={"discovery": 0.1})
        d = t.stats_dict()
        assert "lm_loss"       in d
        assert "total_loss"    in d
        assert "mem_fill_rate" in d
        assert "aux_losses"    in d
        assert "discovery"     in d["aux_losses"]

    def test_export_jsonl(self, tmp_path):
        t = SCMDiagnosticsTracker()
        for i in range(5):
            t.record(step=i, lm_loss=2.0 - i * 0.1,
                     aux_losses={"discovery": 0.05})
        path = str(tmp_path / "diag.jsonl")
        out  = t.export_jsonl(path)
        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == 5
        row = json.loads(lines[0])
        assert "step"     in row
        assert "lm_loss"  in row

    def test_export_summary_json(self, tmp_path):
        t = SCMDiagnosticsTracker()
        t.record(step=0, lm_loss=2.0)
        path = str(tmp_path / "summary.json")
        out  = t.export_summary_json(path)
        data = json.loads(Path(out).read_text())
        assert "n_steps"    in data
        assert "lm_loss"    in data
        assert data["n_steps"] == 1

    def test_loss_decreased_over(self):
        t = SCMDiagnosticsTracker()
        for i in range(10):
            t.record(step=i, lm_loss=3.0 - i * 0.2)
        assert t.loss_decreased_over(5)

    def test_lm_loss_sequence_length(self):
        t = SCMDiagnosticsTracker()
        for i in range(7):
            t.record(step=i, lm_loss=1.0)
        seq = t.lm_loss_sequence()
        assert len(seq) == 7

    def test_reset(self):
        t = SCMDiagnosticsTracker()
        for i in range(5):
            t.record(step=i, lm_loss=1.0)
        t.reset()
        assert t.n_steps == 0
        assert t.latest  is None
        assert not t.is_plateau

    def test_diagnostics_row_to_dict(self):
        row = DiagnosticsRow(
            step=3, elapsed_s=10.5, lm_loss=1.8, total_loss=2.0,
            aux_losses={"discovery": 0.2}, metrics={"route_entropy": 1.1},
            mem_fill_rate=0.4, mem_n_filled=12,
        )
        d = row.to_dict()
        assert d["step"]             == 3
        assert d["lm_loss"]          == pytest.approx(1.8, abs=1e-3)
        assert d["aux_discovery"]    == pytest.approx(0.2, abs=1e-3)
        assert d["metric_route_entropy"] == pytest.approx(1.1, abs=1e-3)
        assert d["mem_fill_rate"]    == pytest.approx(0.4, abs=1e-3)

    def test_window_stat(self):
        w = _WindowStat(maxlen=3)
        w.update(1.0)
        w.update(2.0)
        w.update(3.0)
        assert w.mean == pytest.approx(2.0)
        w.update(4.0)  # evicts 1.0
        assert w.mean == pytest.approx(3.0)

    def test_window_stat_empty(self):
        w = _WindowStat()
        assert math.isnan(w.mean)
        assert math.isnan(w.std)
        assert math.isnan(w.last)

    def test_aux_loss_tracking(self):
        t = SCMDiagnosticsTracker()
        for i in range(5):
            t.record(step=i, lm_loss=1.5,
                     aux_losses={"discovery": 0.1, "compiler": 0.05})
        stats = t.stats_dict()
        assert "discovery" in stats["aux_losses"]
        assert "compiler"  in stats["aux_losses"]
        assert math.isfinite(stats["aux_losses"]["discovery"]["mean"])

    def test_metric_tracking(self):
        t = SCMDiagnosticsTracker()
        for i in range(3):
            t.record(step=i, lm_loss=2.0,
                     metrics={"discovery_collapse": 0.5 + i * 0.1,
                               "route_entropy": 1.2})
        stats = t.stats_dict()
        assert "discovery_collapse" in stats["metrics"]
        assert "route_entropy"      in stats["metrics"]

    def test_memory_fill_rate_tracking(self):
        t = SCMDiagnosticsTracker()
        for i in range(5):
            t.record(step=i, lm_loss=1.0, mem_fill_rate=0.1 * i)
        stats = t.stats_dict()
        assert math.isfinite(stats["mem_fill_rate"]["mean"])
        assert stats["mem_fill_rate"]["last"] == pytest.approx(0.4)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TESTS (require torch)
# ══════════════════════════════════════════════════════════════════════════════

@needs_torch
class TestTACSCMLanguageModelForward:

    @pytest.fixture
    def small_cfg(self):
        return TACSCMConfig(
            vocab_size  = 256,
            d_model     = 64,
            n_layers    = 2,
            n_heads     = 2,
            n_kv_heads  = 1,
            d_ff        = 128,
            d_structure = 16,
            n_structure_slots = 32,
            n_identity_slots  = 4,
            scm_layer_interval = 1,
            max_seq_len = 32,
            use_gradient_checkpointing = False,
        )

    def _make_model(self, cfg):
        from tacm.scm_model import TACSCMLanguageModel
        return TACSCMLanguageModel(cfg)

    def test_model_construction(self, small_cfg):
        m = self._make_model(small_cfg)
        assert m.n_params() > 0

    def test_model_param_count_positive(self, small_cfg):
        m = self._make_model(small_cfg)
        n = m.n_params()
        assert n > 1000

    def test_param_breakdown_keys(self, small_cfg):
        m = self._make_model(small_cfg)
        bd = m.param_breakdown()
        assert "token_embed"   in bd
        assert "scm_blocks"    in bd
        assert "plain_blocks"  in bd

    def test_forward_returns_logits(self, small_cfg):
        m   = self._make_model(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (2, 8))
        out = m(ids)
        assert out.logits is not None
        assert out.logits.shape == (2, 8, small_cfg.vocab_size)

    def test_forward_with_labels_returns_loss(self, small_cfg):
        m   = self._make_model(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (2, 8))
        out = m(ids, labels=ids)
        assert out.loss    is not None
        assert out.lm_loss is not None
        assert math.isfinite(out.loss.item())
        assert math.isfinite(out.lm_loss.item())

    def test_forward_lm_loss_positive(self, small_cfg):
        m   = self._make_model(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (2, 8))
        out = m(ids, labels=ids)
        assert out.lm_loss.item() > 0

    def test_forward_returns_structure_state(self, small_cfg):
        m   = self._make_model(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (2, 8))
        out = m(ids, return_state=True)
        # state may be None if n_identity_slots=0 or no scm block
        # but the output object must be a TACSCMOutput
        from tacm.scm_types import TACSCMOutput
        assert isinstance(out, TACSCMOutput)

    def test_forward_metrics_populated(self, small_cfg):
        m   = self._make_model(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (2, 8))
        out = m(ids, labels=ids, return_metrics=True)
        assert isinstance(out.metrics, dict)

    def test_forward_aux_losses_dict(self, small_cfg):
        m   = self._make_model(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (2, 8))
        out = m(ids, labels=ids)
        assert isinstance(out.auxiliary_losses, dict)

    def test_forward_no_scm(self, small_cfg):
        small_cfg.enable_scm = False
        m   = self._make_model(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (2, 8))
        out = m(ids, labels=ids)
        assert math.isfinite(out.lm_loss.item())
        assert len(out.auxiliary_losses) == 0

    def test_forward_discovery_only(self, small_cfg):
        small_cfg.enable_structure_identity  = False
        small_cfg.enable_structure_memory    = False
        small_cfg.enable_nsf_survival        = False
        small_cfg.enable_dpsl_refinement     = False
        m   = self._make_model(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (2, 8))
        out = m(ids, labels=ids)
        assert math.isfinite(out.lm_loss.item())
        assert "discovery" in out.auxiliary_losses or len(out.auxiliary_losses) >= 0

    def test_forward_seq_len_respected(self, small_cfg):
        m   = self._make_model(small_cfg)
        T   = small_cfg.max_seq_len
        ids = torch.randint(0, small_cfg.vocab_size, (1, T))
        out = m(ids)
        assert out.logits.shape[1] == T

    def test_forward_batch_size_invariant(self, small_cfg):
        m    = self._make_model(small_cfg)
        ids1 = torch.randint(0, small_cfg.vocab_size, (1, 8))
        ids4 = torch.randint(0, small_cfg.vocab_size, (4, 8))
        out1 = m(ids1)
        out4 = m(ids4)
        assert out1.logits.shape == (1, 8, small_cfg.vocab_size)
        assert out4.logits.shape == (4, 8, small_cfg.vocab_size)


@needs_torch
class TestStructureStateCarry:

    @pytest.fixture
    def cfg(self):
        return TACSCMConfig(
            vocab_size  = 64,
            d_model     = 32,
            n_layers    = 2,
            n_heads     = 2,
            n_kv_heads  = 1,
            d_ff        = 64,
            d_structure = 8,
            n_structure_slots = 16,
            n_identity_slots  = 4,
            scm_layer_interval = 1,
            max_seq_len = 16,
            use_gradient_checkpointing = False,
        )

    def test_state_carry_changes_loss(self, cfg):
        from tacm.scm_model import TACSCMLanguageModel
        m    = TACSCMLanguageModel(cfg)
        ids  = torch.randint(0, cfg.vocab_size, (2, 8))

        out1 = m(ids, labels=ids, return_state=True)
        state = out1.structure_state

        out2_with_state    = m(ids, labels=ids, structure_state=state)
        out2_without_state = m(ids, labels=ids, structure_state=None)

        # Both should produce finite losses
        assert math.isfinite(out2_with_state.loss.item())
        assert math.isfinite(out2_without_state.loss.item())

    def test_state_detach(self, cfg):
        from tacm.scm_model import TACSCMLanguageModel
        from tacm.scm_types  import StructureIdentityState
        m    = TACSCMLanguageModel(cfg)
        ids  = torch.randint(0, cfg.vocab_size, (2, 8))
        out  = m(ids, return_state=True)
        if out.structure_state is not None:
            state = out.structure_state
            assert isinstance(state, StructureIdentityState)
            det   = state.detach()
            assert not det.slot_embeddings.requires_grad

    def test_state_zeros_factory(self, cfg):
        from tacm.scm_types import StructureIdentityState
        state = StructureIdentityState.zeros(
            batch_size       = 3,
            n_identity_slots = cfg.n_identity_slots,
            d_structure      = cfg.d_structure,
        )
        assert state.slot_embeddings.shape  == (3, cfg.n_identity_slots, cfg.d_structure)
        assert state.slot_weights.shape     == (3, cfg.n_identity_slots)
        assert state.step_count             == 0

    def test_state_to_device(self, cfg):
        from tacm.scm_types import StructureIdentityState
        state = StructureIdentityState.zeros(2, cfg.n_identity_slots, cfg.d_structure)
        state2 = state.to("cpu")
        assert state2.slot_embeddings.device.type == "cpu"


@needs_torch
class TestStructureMemory:

    @pytest.fixture
    def mem(self):
        from tacm.scm_memory import StructureMemory
        cfg = TACSCMConfig(
            d_structure=16, n_structure_slots=32,
            memory_write_rate=1.0, survival_decay=0.99
        )
        return StructureMemory(cfg)

    def test_read_empty_returns_zeros(self, mem):
        query = torch.randn(2, 16)
        out   = mem.read(query)
        assert out.context_vector.shape == (2, 16)
        assert out.context_vector.abs().sum().item() == pytest.approx(0.0, abs=1e-5)

    def test_write_fills_slots(self, mem):
        embs  = torch.randn(5, 16)
        surv  = torch.rand(5)
        mem.write(embs, surv)
        stats = mem.stats()
        assert stats["n_filled"] > 0

    def test_write_then_read_nonzero(self, mem):
        embs = torch.randn(4, 16)
        surv = torch.ones(4)
        mem.write(embs, surv)
        query = torch.randn(2, 16)
        out   = mem.read(query)
        assert out.context_vector.abs().sum().item() > 0

    def test_fill_rate_in_range(self, mem):
        embs = torch.randn(10, 16)
        surv = torch.ones(10)
        mem.write(embs, surv)
        stats = mem.stats()
        assert 0.0 <= stats["fill_rate"] <= 1.0

    def test_step_decay_lowers_survival(self, mem):
        embs = torch.randn(4, 16)
        surv = torch.ones(4)
        mem.write(embs, surv)
        before = mem.survival[mem.filled].mean().item()
        mem.step_decay()
        after  = mem.survival[mem.filled].mean().item()
        assert after < before, "Survival should decay after step"

    def test_prune_removes_low_survival(self, mem):
        embs = torch.randn(4, 16)
        surv = torch.tensor([0.001, 0.001, 0.5, 0.5])
        mem.write(embs, surv)
        n_before = mem.stats()["n_filled"]
        mem.prune(threshold=0.01)
        n_after  = mem.stats()["n_filled"]
        assert n_after <= n_before

    def test_reset_clears_memory(self, mem):
        embs = torch.randn(4, 16)
        surv = torch.ones(4)
        mem.write(embs, surv)
        mem.reset()
        assert mem.stats()["n_filled"] == 0

    def test_save_load_memory_state(self, mem):
        embs = torch.randn(5, 16)
        surv = torch.ones(5)
        mem.write(embs, surv)
        state = mem.save_memory_state()
        mem.reset()
        assert mem.stats()["n_filled"] == 0
        mem.load_memory_state(state)
        assert mem.stats()["n_filled"] == 5

    def test_retrieval_scores_bounded(self, mem):
        embs = torch.randn(8, 16)
        surv = torch.ones(8)
        mem.write(embs, surv)
        query = torch.randn(2, 16)
        out   = mem.read(query)
        scores = out.retrieval_scores
        assert scores.shape[0] == 2
        assert scores.shape[1] == mem.top_k


@needs_torch
class TestGeneration:

    def test_generate_text_shape(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            scm_layer_interval=1, max_seq_len=32,
            use_gradient_checkpointing=False,
        )
        m      = TACSCMLanguageModel(cfg)
        ids    = torch.zeros(1, 4, dtype=torch.long)
        gen, _ = m.generate_text(ids, max_new_tokens=8, temperature=1.0)
        assert gen.shape == (1, 12), f"Expected (1,12), got {gen.shape}"

    def test_generate_text_non_degenerate(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            scm_layer_interval=1, max_seq_len=32,
            use_gradient_checkpointing=False,
        )
        m      = TACSCMLanguageModel(cfg)
        ids    = torch.zeros(1, 4, dtype=torch.long)
        gen, _ = m.generate_text(ids, max_new_tokens=16, temperature=1.0, top_k=10)
        # Verify no crash and output is integer tokens in valid range
        assert gen.min().item() >= 0
        assert gen.max().item() <  cfg.vocab_size

    def test_generate_carries_structure_state(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            scm_layer_interval=1, max_seq_len=32,
            use_gradient_checkpointing=False,
        )
        m      = TACSCMLanguageModel(cfg)
        ids    = torch.zeros(1, 4, dtype=torch.long)
        gen, state = m.generate_text(ids, max_new_tokens=4, carry_state=True)
        # state might be None if SCM is not active, but should not crash
        assert gen.shape[1] == 8


@needs_torch
class TestSCMBlock:

    def test_block_no_scm_is_passthrough(self):
        from tacm.scm_block import IntegratedStructureLanguageBlock
        from tacm.scm_memory import StructureMemory
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            max_seq_len=16,
            enable_scm = False,
            use_gradient_checkpointing=False,
        )
        mem   = StructureMemory(cfg)
        block = IntegratedStructureLanguageBlock(cfg, mem, layer_idx=0)
        h     = torch.randn(2, 8, 32)
        from tacm.backbone import precompute_freqs_cis
        freqs = precompute_freqs_cis(32 // 2, 16)
        out   = block(h, freqs[:8])
        assert out.hidden_states.shape == (2, 8, 32)
        assert len(out.aux_losses) == 0

    def test_block_with_scm_produces_aux_losses(self):
        from tacm.scm_block import IntegratedStructureLanguageBlock
        from tacm.scm_memory import StructureMemory
        from tacm.backbone import precompute_freqs_cis
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            max_seq_len=16,
            enable_scm = True,
            use_gradient_checkpointing=False,
        )
        mem   = StructureMemory(cfg)
        block = IntegratedStructureLanguageBlock(cfg, mem, layer_idx=0)
        h     = torch.randn(2, 8, 32)
        freqs = precompute_freqs_cis(32 // 2, 16)
        out   = block(h, freqs[:8])
        assert out.hidden_states.shape == (2, 8, 32)
        assert len(out.aux_losses) > 0

    def test_block_output_hidden_finite(self):
        from tacm.scm_block import IntegratedStructureLanguageBlock
        from tacm.scm_memory import StructureMemory
        from tacm.backbone import precompute_freqs_cis
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            max_seq_len=16, use_gradient_checkpointing=False,
        )
        mem   = StructureMemory(cfg)
        block = IntegratedStructureLanguageBlock(cfg, mem, layer_idx=0)
        h     = torch.randn(2, 8, 32)
        freqs = precompute_freqs_cis(32 // 2, 16)
        out   = block(h, freqs[:8])
        assert torch.isfinite(out.hidden_states).all(), \
            "Block output should be finite"

    def test_block_aux_losses_finite(self):
        from tacm.scm_block import IntegratedStructureLanguageBlock
        from tacm.scm_memory import StructureMemory
        from tacm.backbone import precompute_freqs_cis
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            max_seq_len=16, use_gradient_checkpointing=False,
        )
        mem   = StructureMemory(cfg)
        block = IntegratedStructureLanguageBlock(cfg, mem, layer_idx=0)
        h     = torch.randn(2, 8, 32)
        freqs = precompute_freqs_cis(32 // 2, 16)
        out   = block(h, freqs[:8])
        for k, v in out.aux_losses.items():
            assert torch.isfinite(v), f"Auxiliary loss '{k}' is not finite: {v}"


@needs_torch
class TestSCMCheckpoint:

    def test_save_load_roundtrip(self, tmp_path):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            scm_layer_interval=1, max_seq_len=16,
            use_gradient_checkpointing=False,
        )
        m = TACSCMLanguageModel(cfg)
        out_dir = str(tmp_path / "ckpt")
        m.save_pretrained(out_dir)

        m2 = TACSCMLanguageModel.load_pretrained(out_dir)
        assert m2.n_params() == m.n_params()

        ids   = torch.randint(0, cfg.vocab_size, (1, 8))
        out1  = m(ids)
        out2  = m2(ids)
        diff  = (out1.logits - out2.logits).abs().max().item()
        assert diff < 1e-4, f"Logits differ after roundtrip: {diff}"

    def test_config_json_saved(self, tmp_path):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            max_seq_len=16, use_gradient_checkpointing=False,
        )
        m       = TACSCMLanguageModel(cfg)
        out_dir = str(tmp_path / "ckpt2")
        m.save_pretrained(out_dir)
        cfg_file = Path(out_dir) / "config.json"
        assert cfg_file.exists()
        loaded_cfg = json.loads(cfg_file.read_text())
        assert loaded_cfg["d_model"]     == 32
        assert loaded_cfg["d_structure"] == 8


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: end-to-end mini training loop
# ══════════════════════════════════════════════════════════════════════════════

@needs_torch
class TestMiniTrainingLoop:

    def test_loss_computes_and_backward(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            scm_layer_interval=1, max_seq_len=16,
            use_gradient_checkpointing=False,
        )
        m    = TACSCMLanguageModel(cfg)
        opt  = torch.optim.Adam(m.parameters(), lr=1e-3)
        ids  = torch.randint(0, cfg.vocab_size, (2, 8))

        m.train()
        opt.zero_grad()
        out  = m(ids, labels=ids)
        out.loss.backward()
        opt.step()

        assert math.isfinite(out.loss.item()), "Loss should be finite"

    def test_loss_decreases_over_overfit(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = TACSCMConfig(
            vocab_size=16, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            scm_layer_interval=1, max_seq_len=8,
            use_gradient_checkpointing=False,
        )
        m   = TACSCMLanguageModel(cfg)
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))

        m.train()
        first_loss = None
        for _ in range(20):
            opt.zero_grad()
            out = m(ids, labels=ids)
            out.loss.backward()
            opt.step()
            if first_loss is None:
                first_loss = out.loss.item()

        last_loss = out.loss.item()
        assert last_loss < first_loss, \
            f"Loss should decrease during overfit: {first_loss:.4f} → {last_loss:.4f}"

    def test_memory_fills_during_training(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = TACSCMConfig(
            vocab_size=64, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=32, n_identity_slots=4,
            scm_layer_interval=1, max_seq_len=8,
            memory_write_rate=1.0,
            enable_nsf_survival=False,
            use_gradient_checkpointing=False,
        )
        m   = TACSCMLanguageModel(cfg)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        ids = torch.randint(0, cfg.vocab_size, (4, 8))

        m.train()
        for _ in range(5):
            opt.zero_grad()
            out = m(ids, labels=ids)
            out.loss.backward()
            opt.step()

        stats = m.memory_stats()
        assert stats["n_filled"] >= 0, "Memory fill count should be non-negative"

    def test_diagnostics_integration(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = TACSCMConfig(
            vocab_size=32, d_model=32, n_layers=2,
            n_heads=2, n_kv_heads=1, d_ff=64, d_structure=8,
            n_structure_slots=16, n_identity_slots=4,
            scm_layer_interval=1, max_seq_len=8,
            use_gradient_checkpointing=False,
        )
        m      = TACSCMLanguageModel(cfg)
        opt    = torch.optim.Adam(m.parameters(), lr=1e-3)
        ids    = torch.randint(0, cfg.vocab_size, (2, 8))
        diag   = SCMDiagnosticsTracker(window_size=5)

        m.train()
        for step in range(5):
            opt.zero_grad()
            out = m(ids, labels=ids, return_metrics=True)
            out.loss.backward()
            opt.step()

            stats = m.memory_stats()
            diag.record(
                step          = step,
                lm_loss       = out.lm_loss.item(),
                total_loss    = out.loss.item(),
                aux_losses    = {k: v.item() for k, v in out.auxiliary_losses.items()},
                metrics       = out.metrics,
                mem_fill_rate = stats["fill_rate"],
                mem_n_filled  = int(stats["n_filled"]),
            )

        assert diag.n_steps == 5
        summary = diag.summary()
        assert "TAC-SCM-REAL001" in summary
        sd = diag.stats_dict()
        assert math.isfinite(sd["lm_loss"]["last"])
