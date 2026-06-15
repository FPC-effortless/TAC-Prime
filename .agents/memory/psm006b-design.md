---
name: PSM-006B design
description: Semi-real pytest-grounded repository repair benchmark; key calibration decisions and architecture choices.
---

# PSM-006B Design

## Core claim
TAC procedural memory reuse improves real pytest-verified repair over 6 ablation baselines.

## Key architecture decisions

### Memory store
- Use `SimpleProceduralMemoryStore` (pure numpy, `tacm/tacm/psm006b/memory_store.py`)
- Do NOT use PSM-001's `ProceduralMemoryStore` (requires FAISS, different API)
- PSM-001 `retrieve()` returns `List[Tuple[float, ProcedureTrace]]`; PSM-006B returns `List[ProcedureRecord]` directly
- PSM-006B write API: `store.write(family, task_type, steps, embedding, success_rate)` → proc_id (str)
- PSM-006B retrieve: `store.retrieve(query_emb, top_k)` → List[ProcedureRecord]
- Update: `store.augment(proc_id, extra_steps)` / `store.reinforce(proc_id, delta)`

### Embedding
- EMBEDDING_DIM = 64
- Family centroid seeded from `np.random.default_rng(family_idx * 1000)`; family_idx = position in FAMILY_NAMES
- Fixture jitter: σ = 0.30 from `sha256(fixture_id)` hash
- Retrieval noise: σ = 0.10 (moderate, realistic)
- 2 oracle records per family seeded at start for centroid stability

### Calibration
- `max_retries = 1` for `full_memory` variant only; 0 for all ablations
- `n_records_per_family = 2` in `seed_procedural_memory()`
- Retrieval blend: `score = 0.7 * cosine_sim + 0.3 * success_rate`
- Centroid seeds (family_idx × 1000) provide max separation in 64-dim space

### Fixture design
- 6 families × 10 fixtures = 60 total
- All stdlib + pytest only (no pip at test time)
- Empty `expected_patch = {}` = fixture already passes (no patch needed)
- Transfer groups: train (6), near_transfer (2), far_transfer (2)
- Difficulty: easy (4), medium (3), hard (3) per family

### Gate calibration
- `update_improves_retry` and `no_update_underperforms_tac` gates only meaningful with all 7 variants
- Quick smoke (3 variants) reliably gets 6/8 gates; full 7-variant run needed for all 8
- Easy fixtures pass at 1.000 for full_memory and oracle; harder fixtures needed for update gate

## File map
```
tacm/tacm/psm006b/
  fixture_schema.py       — Fixture, FAMILY_NAMES, FAILURE_CLASSES
  fixture_builder.py      — build_all_fixtures() (60 Fixture objects)
  pytest_verifier.py      — PytestVerifier (subprocess runner)
  patch_applier.py        — PatchApplier, PatchResult
  memory_store.py         — SimpleProceduralMemoryStore
  procedural_repair_agent.py — ProceduralRepairAgent006B, RepairTrace006B
  baselines.py            — run_all_baselines(), VARIANT_NAMES (7)
  metrics.py              — compute_metrics() (13 metrics), evaluate_success_gates() (8)

tacm/
  benchmark_tac_psm006b_pytest_fixtures.py
  run_psm006b_replication.py
  tests/test_tac_psm006b_pytest_fixtures.py  (49 tests, all pass)
  docs/tac_psm006b_pytest_fixture_report.md
  docs/tac_psm006b_failure_analysis.md
```

**Why:** PSM-001 store has FAISS dependency that may be unavailable; PSM-006B needs a clean self-contained store so the benchmark can run anywhere with just numpy+pytest.
