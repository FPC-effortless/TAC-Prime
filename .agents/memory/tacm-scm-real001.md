---
name: TAC-SCM-REAL001 architecture
description: Real trainable structure-native language model; design decisions, quirks, and invariants.
---

# TAC-SCM-REAL001: Architecture Notes

## Core invariant
`TACSCMLanguageModel` is in `tacm/scm_model.py`. It interleaves `IntegratedStructureLanguageBlock` every `scm_layer_interval` layers with plain `TransformerBlock`s. Shared `StructureMemory` is passed into every SCM block (not owned by the block).

## Critical identity distinction
- `tacm/identity.py` → `IdentityState`: carries symbolic identity IDs + stability scalars. Used in TAC-Prime-ID001.
- `tacm/scm_identity.py` → `StructureIdentityState`: carries actual structure embedding tensors (B, n_slots, d_structure). Used in SCM-REAL001.
Do NOT confuse these two.

## torch import guard
`tacm/data/scm_dataset.py` uses `try: import torch; _HAS_TORCH = True` because all tests in this project skip torch-gated code when torch is not installed. `SCMDataCollator.__call__` returns plain lists when `_HAS_TORCH=False`, tensors otherwise.

**Why:** This environment has no torch installed; existing tests (`test_identity_field.py`) use the same `HAS_TORCH = pytest.mark.skipif(...)` pattern. We must follow it.

## Memory bank architecture
- Learnable *parameters* (query/value/gate projections in `StructureMemory`) are gradient-trained.
- *Buffer bank* (keys, values, usage, age, survival registers) stores discovered structures outside of the parameter update path.
- Near-duplicate suppression: cosine sim > 0.95 → in-place update instead of new slot.

**Why:** Keeps the bank from filling with redundant structures; maintains diverse coverage.

## Discovery: no collapse
`StructureDiscoveryLayer` uses EMA target encoder (decay ≈ 0.996). The `collapse_metric` is mean std of latents. If it drops to 0, latents have collapsed; VICReg variance loss is the primary defense.

**How to apply:** Monitor `collapse_metric` during training. If < 0.01 sustained for > 500 steps, reduce EMA decay or increase `spread_loss_weight`.

## SCM block memory query
`IntegratedStructureLanguageBlock` uses `self.hidden_to_query` (d_model → d_structure) for the memory read query when no structure tokens are available. Earlier versions had a multi-branch mess at this location — it was cleaned up in the final version.

## Config presets
- `TACSCMConfig.no_scm()` — pure transformer baseline (enable_scm=False)
- `TACSCMConfig.discovery_only()` — only JEPA discovery, no memory/identity/survival/refinement
- `TACSCMConfig.small()` — d_model=256, n_layers=4, d_structure=64
- `TACSCMConfig.base()` — d_model=512, n_layers=8, d_structure=128

## Test gate counts
- 60 tests total: 16 pure-Python (always pass), 44 torch-gated (skip without torch).
- Location: `tacm/tests_py/test_tac_scm_real001_model.py`

## Training
- `experiments/train_tac_scm_real001.py` — CLI; synthetic dataset works without tokenizer.
- `experiments/benchmark_tac_scm_real001.py` — 5-condition ablation benchmark.
- Tokenizer: currently placeholder (char → int). Real training needs tiktoken or sentencepiece.
