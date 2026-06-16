"""
TAC-SCM-REAL001: Model Tests

Run:
    cd tacm
    python -m pytest tests_py/test_tac_scm_real001_model.py -v

Section A — Pure Python / config tests (always run):
  - TACSCMConfig initialises with defaults
  - All config presets valid
  - SCMSample and SCMDataset construction
  - Synthetic dataset generation shape
  - SCMDataCollator output shape

Section B — PyTorch model tests (skipped when torch unavailable):
  - Model constructs and has correct parameter count
  - Forward pass: logits shape correct
  - Loss is finite when labels provided
  - All auxiliary losses finite
  - No SCM forward pass works (pure transformer)
  - Structure state carries across calls (step_count increments)
  - generate_text produces correct output shape
  - Memory write and read works
  - Memory save/load roundtrip
  - Survival scores finite and gates in [0,1]
  - Refinement modifies structures
  - Discovery output shapes and collapse metric
  - Compiler output shapes
  - Identity field output shapes and state update
  - save_pretrained / load_pretrained roundtrip
"""

from __future__ import annotations

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
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ── Section A: Pure Python / Config tests ────────────────────────────────────

class TestTACSCMConfig:

    def test_config_defaults(self):
        from tacm.scm_config import TACSCMConfig
        cfg = TACSCMConfig()
        assert cfg.vocab_size > 0
        assert cfg.d_model > 0
        assert cfg.d_structure > 0
        assert cfg.n_structure_slots > 0
        assert cfg.n_identity_slots > 0

    def test_config_preset_small(self):
        from tacm.scm_config import TACSCMConfig
        cfg = TACSCMConfig.small()
        assert cfg.d_model > 0
        assert cfg.enable_scm is True

    def test_config_preset_base(self):
        from tacm.scm_config import TACSCMConfig
        cfg = TACSCMConfig.base()
        assert cfg.d_model >= 256

    def test_config_preset_no_scm(self):
        from tacm.scm_config import TACSCMConfig
        cfg = TACSCMConfig.no_scm()
        assert cfg.enable_scm is False

    def test_config_preset_discovery_only(self):
        from tacm.scm_config import TACSCMConfig
        cfg = TACSCMConfig.discovery_only()
        assert cfg.enable_scm is True
        assert cfg.enable_structure_identity is False

    def test_config_loss_weights_non_negative(self):
        from tacm.scm_config import TACSCMConfig
        cfg = TACSCMConfig()
        assert cfg.discovery_loss_weight  >= 0
        assert cfg.survival_loss_weight   >= 0
        assert cfg.refinement_loss_weight >= 0

    def test_config_scm_layer_interval_positive(self):
        from tacm.scm_config import TACSCMConfig
        cfg = TACSCMConfig()
        assert cfg.scm_layer_interval >= 1


class TestSCMDataset:

    def test_scm_sample_construct(self):
        from tacm.data.scm_dataset import SCMSample
        s = SCMSample(input_ids=[1, 2, 3], labels=[1, 2, 3])
        assert s.input_ids  == [1, 2, 3]
        assert s.labels     == [1, 2, 3]
        assert s.task_id    is None
        assert s.source     == "text"

    def test_scm_dataset_len(self):
        from tacm.data.scm_dataset import SCMSample, SCMDataset
        samples = [
            SCMSample(input_ids=[i, i+1], labels=[i, i+1])
            for i in range(5)
        ]
        ds = SCMDataset(samples, seq_len=4, pad_id=0)
        assert len(ds) == 5

    def test_scm_dataset_getitem_pads(self):
        from tacm.data.scm_dataset import SCMSample, SCMDataset
        s  = SCMSample(input_ids=[1, 2], labels=[1, 2])
        ds = SCMDataset([s], seq_len=6, pad_id=0)
        item = ds[0]
        assert len(item.input_ids) == 6
        assert item.input_ids[2]   == 0  # padded

    def test_scm_dataset_getitem_truncates(self):
        from tacm.data.scm_dataset import SCMSample, SCMDataset
        s  = SCMSample(input_ids=list(range(20)), labels=list(range(20)))
        ds = SCMDataset([s], seq_len=8, pad_id=0)
        item = ds[0]
        assert len(item.input_ids) == 8
        assert len(item.labels)    == 8

    def test_synthetic_dataset_length(self):
        from tacm.data.scm_dataset import make_synthetic_repair_dataset
        ds = make_synthetic_repair_dataset(n_samples=50, seq_len=32, seed=0)
        assert len(ds) == 50

    def test_synthetic_dataset_structure_ids_valid(self):
        from tacm.data.scm_dataset import make_synthetic_repair_dataset
        ds = make_synthetic_repair_dataset(n_samples=20, n_families=4, seq_len=32, seed=0)
        for i in range(len(ds)):
            item = ds[i]
            assert item.structure_id is not None
            assert 0 <= item.structure_id < 4

    def test_synthetic_dataset_source_field(self):
        from tacm.data.scm_dataset import make_synthetic_repair_dataset
        ds = make_synthetic_repair_dataset(n_samples=10, seq_len=32, seed=0)
        for i in range(len(ds)):
            assert ds[i].source == "repair"

    def test_collator_batches_correctly(self):
        from tacm.data.scm_dataset import SCMSample, SCMDataCollator
        samples = [
            SCMSample(input_ids=[1, 2, 3],    labels=[1, 2, 3],    structure_id=0),
            SCMSample(input_ids=[4, 5, 6, 7], labels=[4, 5, 6, 7], structure_id=1),
        ]
        collator = SCMDataCollator(pad_id=0)
        batch    = collator(samples)
        assert "input_ids"      in batch
        assert "labels"         in batch
        assert "attention_mask" in batch
        # Both padded to length 4
        assert len(batch["input_ids"][0]) == 4
        assert len(batch["input_ids"][1]) == 4

    def test_collator_mask_correct(self):
        from tacm.data.scm_dataset import SCMSample, SCMDataCollator
        samples = [
            SCMSample(input_ids=[1, 2],    labels=[1, 2]),
            SCMSample(input_ids=[3, 4, 5], labels=[3, 4, 5]),
        ]
        collator = SCMDataCollator(pad_id=0)
        batch    = collator(samples)
        # First sample has 1 padded token → last mask position is 0
        assert batch["attention_mask"][0][-1] == 0
        # Second sample fully filled → all mask positions are 1
        assert all(v == 1 for v in batch["attention_mask"][1])


# ── Section B: PyTorch model tests ───────────────────────────────────────────

def _small_cfg():
    from tacm.scm_config import TACSCMConfig
    cfg = TACSCMConfig()
    cfg.vocab_size          = 256
    cfg.d_model             = 64
    cfg.n_layers            = 2
    cfg.n_heads             = 2
    cfg.n_kv_heads          = 1
    cfg.d_ff                = 128
    cfg.d_structure         = 32
    cfg.n_structure_slots   = 16
    cfg.n_identity_slots    = 4
    cfg.n_structure_candidates = 4
    cfg.max_seq_len         = 64
    cfg.use_gradient_checkpointing = False
    return cfg


B, T = 2, 16


@skip_no_torch
class TestModelConstruction:

    def test_constructs_with_scm(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg())
        assert m.n_params() > 0

    def test_constructs_no_scm(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = _small_cfg()
        cfg.enable_scm = False
        m = TACSCMLanguageModel(cfg)
        assert m.n_params() > 0

    def test_param_breakdown_keys(self):
        from tacm.scm_model import TACSCMLanguageModel
        m  = TACSCMLanguageModel(_small_cfg())
        bd = m.param_breakdown()
        assert "scm_blocks"    in bd
        assert "plain_blocks"  in bd
        assert "token_embed"   in bd

    def test_is_scm_layer_list(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = _small_cfg()
        cfg.scm_layer_interval = 2
        m   = TACSCMLanguageModel(cfg)
        # Layer 0 should be SCM (0 % 2 == 0), layer 1 plain
        assert m.is_scm_layer[0] is True
        assert m.is_scm_layer[1] is False

    def test_scm_and_plain_layers_coexist(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = _small_cfg()
        cfg.n_layers = 4; cfg.scm_layer_interval = 2
        m   = TACSCMLanguageModel(cfg)
        n_scm   = sum(m.is_scm_layer)
        n_plain = len(m.is_scm_layer) - n_scm
        assert n_scm   == 2
        assert n_plain == 2


@skip_no_torch
class TestForwardPass:

    def test_logits_shape_no_labels(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        ids = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            out = m(ids)
        assert out.logits.shape == (B, T, 256)
        assert out.loss is None

    def test_lm_loss_finite(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        ids = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            out = m(ids, labels=ids)
        assert out.lm_loss is not None
        assert math.isfinite(out.lm_loss.item())

    def test_total_loss_finite(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        ids = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            out = m(ids, labels=ids)
        assert out.loss is not None
        assert math.isfinite(out.loss.item())

    def test_auxiliary_losses_finite(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        ids = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            out = m(ids, labels=ids)
        for k, v in out.auxiliary_losses.items():
            assert math.isfinite(v.item()), f"aux_loss[{k}]={v.item()}"

    def test_no_scm_forward(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = _small_cfg(); cfg.enable_scm = False
        m   = TACSCMLanguageModel(cfg); m.eval()
        ids = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            out = m(ids, labels=ids)
        assert math.isfinite(out.lm_loss.item())
        assert not out.auxiliary_losses

    def test_logits_not_nan(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        ids = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            out = m(ids)
        assert not torch.isnan(out.logits).any()
        assert not torch.isinf(out.logits).any()

    def test_backward_runs(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.train()
        ids = torch.randint(0, 256, (B, T))
        out = m(ids, labels=ids)
        out.loss.backward()
        # At least one param has gradient
        grad_norms = [
            p.grad.norm().item()
            for p in m.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0
        assert any(g > 0 for g in grad_norms)


@skip_no_torch
class TestStructureState:

    def test_state_carries_step_count(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        ids = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            out1 = m(ids, return_state=True)
            assert out1.structure_state is not None
            out2 = m(ids, structure_state=out1.structure_state, return_state=True)
            assert out2.structure_state.step_count > out1.structure_state.step_count

    def test_reset_state_is_blank(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        state = m.reset_structure_state(batch_size=B)
        assert state.step_count == 0
        assert state.slot_embeddings.shape[0] == B

    def test_state_is_detached(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.train()
        ids = torch.randint(0, 256, (B, T))
        out = m(ids, labels=ids, return_state=True)
        if out.structure_state is not None:
            assert not out.structure_state.slot_embeddings.requires_grad

    def test_no_state_return_when_disabled(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        ids = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            out = m(ids, return_state=False)
        assert out.structure_state is None


@skip_no_torch
class TestGeneration:

    def test_generate_text_shape(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        prompt = torch.zeros(1, 8, dtype=torch.long)
        with torch.no_grad():
            gen, _ = m.generate_text(prompt, max_new_tokens=10)
        assert gen.shape == (1, 18)

    def test_generate_text_carries_state(self):
        from tacm.scm_model import TACSCMLanguageModel
        cfg = _small_cfg()
        m   = TACSCMLanguageModel(cfg); m.eval()
        prompt = torch.zeros(1, 4, dtype=torch.long)
        with torch.no_grad():
            _, state = m.generate_text(prompt, max_new_tokens=8, carry_state=True)
        if cfg.enable_scm:
            assert state is not None

    def test_generate_text_no_nan(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        prompt = torch.zeros(1, 4, dtype=torch.long)
        with torch.no_grad():
            gen, _ = m.generate_text(prompt, max_new_tokens=8, temperature=0.8)
        assert not torch.isnan(gen.float()).any()

    def test_generate_text_temperature_one(self):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        prompt = torch.zeros(2, 4, dtype=torch.long)
        with torch.no_grad():
            gen, _ = m.generate_text(prompt, max_new_tokens=4, temperature=1.0)
        assert gen.shape == (2, 8)


@skip_no_torch
class TestStructureMemory:

    def test_write_increases_fill(self):
        from tacm.scm_memory import StructureMemory
        cfg = _small_cfg()
        mem = StructureMemory(cfg)
        mem.write_rate = 1.0
        embs = torch.randn(4, cfg.d_structure)
        mem.write(embs, torch.ones(4), torch.ones(4, dtype=torch.bool))
        assert mem.filled.sum().item() > 0

    def test_read_returns_correct_shape(self):
        from tacm.scm_memory import StructureMemory
        cfg   = _small_cfg()
        mem   = StructureMemory(cfg)
        mem.write_rate = 1.0
        embs  = torch.randn(4, cfg.d_structure)
        mem.write(embs, torch.ones(4), torch.ones(4, dtype=torch.bool))
        query = torch.randn(B, cfg.d_structure)
        out   = mem.read(query)
        assert out.context_vector.shape  == (B, cfg.d_structure)
        assert out.retrieval_scores.shape[0] == B

    def test_memory_reset(self):
        from tacm.scm_memory import StructureMemory
        cfg  = _small_cfg()
        mem  = StructureMemory(cfg)
        mem.write_rate = 1.0
        embs = torch.randn(4, cfg.d_structure)
        mem.write(embs, torch.ones(4), torch.ones(4, dtype=torch.bool))
        assert mem.filled.any()
        mem.reset()
        assert not mem.filled.any()

    def test_memory_save_load(self):
        from tacm.scm_memory import StructureMemory
        cfg  = _small_cfg()
        mem  = StructureMemory(cfg)
        mem.write_rate = 1.0
        embs = torch.randn(4, cfg.d_structure)
        mem.write(embs, torch.ones(4), torch.ones(4, dtype=torch.bool))
        n_before = mem.filled.sum().item()
        state = mem.save_memory_state()
        mem.reset()
        assert not mem.filled.any()
        mem.load_memory_state(state)
        assert mem.filled.sum().item() == n_before

    def test_memory_stats(self):
        from tacm.scm_memory import StructureMemory
        cfg   = _small_cfg()
        mem   = StructureMemory(cfg)
        stats = mem.stats()
        assert "n_filled"    in stats
        assert "fill_rate"   in stats
        assert "mean_survival" in stats


@skip_no_torch
class TestSurvivalScorer:

    def test_scores_shape(self):
        from tacm.scm_survival import NSFSurvivalScorer
        cfg    = _small_cfg()
        scorer = NSFSurvivalScorer(cfg)
        embs   = torch.randn(6, cfg.d_structure)
        out    = scorer(embs)
        assert out.survival_score.shape == (6,)

    def test_scores_finite(self):
        from tacm.scm_survival import NSFSurvivalScorer
        cfg    = _small_cfg()
        scorer = NSFSurvivalScorer(cfg)
        embs   = torch.randn(4, cfg.d_structure)
        out    = scorer(embs)
        for s in out.survival_score:
            assert math.isfinite(s.item())

    def test_gates_in_range(self):
        from tacm.scm_survival import NSFSurvivalScorer
        cfg    = _small_cfg()
        scorer = NSFSurvivalScorer(cfg)
        embs   = torch.randn(4, cfg.d_structure)
        out    = scorer(embs)
        assert (out.write_gate  >= 0).all() and (out.write_gate  <= 1).all()
        assert (out.refine_gate >= 0).all() and (out.refine_gate <= 1).all()
        assert (out.decay_gate  >= 0).all() and (out.decay_gate  <= 1).all()

    def test_loss_finite(self):
        from tacm.scm_survival import NSFSurvivalScorer
        cfg    = _small_cfg()
        scorer = NSFSurvivalScorer(cfg)
        embs   = torch.randn(4, cfg.d_structure)
        out    = scorer(embs)
        assert math.isfinite(out.loss_total.item())


@skip_no_torch
class TestDPSLRefinement:

    def test_modifies_structures(self):
        from tacm.scm_refinement import DPSLRefinementLayer
        cfg     = _small_cfg()
        refiner = DPSLRefinementLayer(cfg)
        embs    = torch.randn(4, cfg.d_structure)
        surv    = torch.ones(4) * 0.7
        feedback = torch.randn(4, cfg.d_structure)
        out     = refiner(embs, surv, feedback)
        assert out.refined_embeddings.shape == (4, cfg.d_structure)
        diff = (out.refined_embeddings - embs).abs().max().item()
        assert diff > 0

    def test_without_feedback(self):
        from tacm.scm_refinement import DPSLRefinementLayer
        cfg     = _small_cfg()
        refiner = DPSLRefinementLayer(cfg)
        embs    = torch.randn(3, cfg.d_structure)
        surv    = torch.rand(3)
        out     = refiner(embs, surv, feedback=None)
        assert math.isfinite(out.loss_total.item())

    def test_merge_mask_shape(self):
        from tacm.scm_refinement import DPSLRefinementLayer
        cfg     = _small_cfg()
        refiner = DPSLRefinementLayer(cfg)
        embs    = torch.randn(4, cfg.d_structure)
        out     = refiner(embs, torch.rand(4))
        assert out.merge_mask.shape == (4,)

    def test_empty_input(self):
        from tacm.scm_refinement import DPSLRefinementLayer
        cfg     = _small_cfg()
        refiner = DPSLRefinementLayer(cfg)
        embs    = torch.zeros(0, cfg.d_structure)
        out     = refiner(embs, torch.zeros(0))
        assert out.refined_embeddings.shape == (0, cfg.d_structure)


@skip_no_torch
class TestStructureDiscovery:

    def test_output_shapes(self):
        from tacm.scm_discovery import StructureDiscoveryLayer
        cfg  = _small_cfg()
        disc = StructureDiscoveryLayer(cfg)
        h    = torch.randn(B, T, cfg.d_model)
        out  = disc(h)
        assert out.latent_state.shape         == (B, T, cfg.d_structure)
        assert out.structure_candidates.shape == (B, cfg.n_structure_candidates, cfg.d_structure)

    def test_loss_finite(self):
        from tacm.scm_discovery import StructureDiscoveryLayer
        cfg  = _small_cfg()
        disc = StructureDiscoveryLayer(cfg)
        h    = torch.randn(B, T, cfg.d_model)
        out  = disc(h)
        assert math.isfinite(out.loss_total.item())

    def test_collapse_metric_non_negative(self):
        from tacm.scm_discovery import StructureDiscoveryLayer
        cfg  = _small_cfg()
        disc = StructureDiscoveryLayer(cfg)
        h    = torch.randn(B, T, cfg.d_model)
        out  = disc(h)
        assert out.collapse_metric.item() >= 0


@skip_no_torch
class TestStructureCompiler:

    def test_output_shapes(self):
        from tacm.scm_compiler import StructureCompiler
        cfg  = _small_cfg()
        comp = StructureCompiler(cfg)
        h    = torch.randn(B, T, cfg.d_model)
        lat  = torch.randn(B, T, cfg.d_structure)
        cand = torch.randn(B, cfg.n_structure_candidates, cfg.d_structure)
        out  = comp(h, lat, cand)
        n    = cfg.n_structure_candidates
        assert out.concept_center.shape   == (B, n, cfg.d_structure)
        assert out.structure_tokens.shape == (B, n, cfg.d_structure)
        assert out.compression_score.shape == (B, n)

    def test_loss_finite(self):
        from tacm.scm_compiler import StructureCompiler
        cfg  = _small_cfg()
        comp = StructureCompiler(cfg)
        h    = torch.randn(B, T, cfg.d_model)
        lat  = torch.randn(B, T, cfg.d_structure)
        cand = torch.randn(B, cfg.n_structure_candidates, cfg.d_structure)
        out  = comp(h, lat, cand)
        assert math.isfinite(out.loss_total.item())


@skip_no_torch
class TestIdentityField:

    def test_output_shapes(self):
        from tacm.scm_identity import StructureIdentityFieldLayer
        cfg   = _small_cfg()
        idf   = StructureIdentityFieldLayer(cfg)
        h     = torch.randn(B, T, cfg.d_model)
        state = idf.init_state(B, "cpu")
        cand  = torch.randn(B, cfg.n_structure_candidates, cfg.d_structure)
        (updated_h, new_state, route_logits,
         route_weights, readout, aux) = idf(h, cand, state)
        assert updated_h.shape     == (B, T, cfg.d_model)
        assert route_logits.shape  == (B, T, cfg.n_identity_slots)
        assert route_weights.shape == (B, T, cfg.n_identity_slots)
        assert readout.shape       == (B, T, cfg.d_model)

    def test_step_count_increments(self):
        from tacm.scm_identity import StructureIdentityFieldLayer
        cfg   = _small_cfg()
        idf   = StructureIdentityFieldLayer(cfg)
        h     = torch.randn(B, T, cfg.d_model)
        state = idf.init_state(B, "cpu")
        _, new_state, _, _, _, _ = idf(h, None, state)
        assert new_state.step_count == 1

    def test_state_detach(self):
        from tacm.scm_identity import StructureIdentityFieldLayer
        cfg   = _small_cfg()
        idf   = StructureIdentityFieldLayer(cfg)
        state = idf.init_state(B, "cpu")
        det   = idf.detach_state(state)
        assert not det.slot_embeddings.requires_grad

    def test_route_weights_sum_to_one(self):
        from tacm.scm_identity import StructureIdentityFieldLayer
        cfg   = _small_cfg()
        idf   = StructureIdentityFieldLayer(cfg)
        h     = torch.randn(B, T, cfg.d_model)
        state = idf.init_state(B, "cpu")
        _, _, _, route_weights, _, _ = idf(h, None, state)
        row_sums = route_weights.sum(dim=-1)  # (B, T)
        assert (row_sums - 1.0).abs().max().item() < 1e-4


@skip_no_torch
class TestSaveLoad:

    def test_save_load_roundtrip(self, tmp_path):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg()); m.eval()
        m.save_pretrained(str(tmp_path / "ckpt"))
        loaded = TACSCMLanguageModel.load_pretrained(str(tmp_path / "ckpt"), device="cpu")
        loaded.eval()
        ids = torch.randint(0, 256, (B, T))
        with torch.no_grad():
            out1 = m(ids)
            out2 = loaded(ids)
        diff = (out1.logits - out2.logits).abs().max().item()
        assert diff < 1e-4, f"Logit difference after reload: {diff}"

    def test_config_json_exists(self, tmp_path):
        from tacm.scm_model import TACSCMLanguageModel
        m = TACSCMLanguageModel(_small_cfg())
        m.save_pretrained(str(tmp_path / "ckpt"))
        assert (tmp_path / "ckpt" / "config.json").exists()
        assert (tmp_path / "ckpt" / "model.pt").exists()
        assert (tmp_path / "ckpt" / "memory.pt").exists()
