# TAC Research Log

Dated entries, most recent first.

---

## 2026-06-15 — TAC-PSM-006B replication benchmark completed

**Result:** PARTIALLY_VALIDATES — 4/8 gates pass on all 5 seeds. Honest run, no tuning.

Full 5-seed, 7-variant, 60-fixture benchmark executed with subprocess-based pytest
verification (real exit codes, no mocks).  Results saved to `tacm/reports/`.

### Exact aggregate metrics (5 seeds, 60 fixtures)

| Metric | Mean | Std |
|---|---|---|
| pytest_pass_rate | 0.8633 | 0.0217 |
| first_attempt_repair_success | 0.8633 | 0.0217 |
| retry_after_update_success | 0.0000 | 0.0000 |
| procedure_retrieval_accuracy | 0.8133 | 0.0321 |
| procedure_reuse_gain | 0.0000 | 0.0264 |
| cross_fixture_transfer_success | 0.8633 | 0.0217 |
| cross_family_transfer_success | 0.0500 | 0.0408 |
| wrong_procedure_harm | 0.5600 | 0.0303 |
| patch_correctness | 1.0000 | 0.0000 |
| family_confusion_rate | 0.1867 | 0.0321 |

### Variant pass rates (mean ± std)

| Variant | Mean | Std |
|---|---|---|
| oracle | 1.000 | 0.000 |
| structure_only | 0.927 | 0.022 |
| full_memory | 0.863 | 0.022 |
| reset | 0.863 | 0.022 |
| no_update | 0.860 | 0.035 |
| retrieval_disabled | 0.550 | 0.000 |
| random_procedure | 0.440 | 0.030 |

### Gate results

| Gate | Pass/Total |
|---|---|
| retrieval_accuracy_ge_0.55 | 5/5 ✓ |
| random_procedure_no_benefit | 5/5 ✓ |
| oracle_above_tac | 5/5 ✓ |
| cross_fixture_transfer_positive | 5/5 ✓ |
| tac_beats_reset_by_0.10 | 0/5 ✗ |
| update_improves_retry | 0/5 ✗ |
| no_update_underperforms_tac | 3/5 ✗ |
| reuse_gain_positive | 2/5 ✗ |

### Per-seed summary

| Seed | full_memory | oracle | reset | no_update | random | Gates | Time |
|---|---|---|---|---|---|---|---|
| 0 | 0.833 | 1.000 | 0.850 | 0.817 | 0.400 | 5/8 | 39s |
| 1 | 0.883 | 1.000 | 0.867 | 0.867 | 0.433 | 6/8 | 30s |
| 2 | 0.867 | 1.000 | 0.900 | 0.900 | 0.433 | 4/8 | 23s |
| 3 | 0.883 | 1.000 | 0.850 | 0.883 | 0.450 | 5/8 | 18s |
| 4 | 0.850 | 1.000 | 0.850 | 0.833 | 0.483 | 5/8 | 19s |

### Interpretation

**Passes (4/8 gates):**
- Retrieval accuracy 0.813 >> 0.55 threshold — cosine retrieval works well
- Oracle always beats full_memory (correct upper bound)
- Cross-fixture transfer positive — procedures transfer across same-family fixtures
- Random procedure consistently below full_memory

**Failures (4/8 gates):**
- `tac_beats_reset_by_0.10` **FAILS** (0/5): full_memory ≈ reset (both ~0.863).
  Root cause: reset also gets seeded oracle procedures before each fixture, so the
  memory advantage is masked.  Memory reuse provides no incremental lift over
  per-fixture re-seeding.
- `update_improves_retry` **FAILS** (0/5): retry_after_update = 0.000 across all seeds.
  The agent never triggers a successful retry via the update path.  Max_retries=1 for
  full_memory but update does not improve embedding quality, only procedure steps.
- `no_update_underperforms_tac` **FAILS** (3/5): no_update ≈ full_memory.
  Update step adds procedure steps but embedding and success_rate scoring are
  unchanged, so the updated store retrieves the same family on retry.
- `reuse_gain_positive` **FAILS** (2/5): procedure_reuse_gain ≈ 0 (mean 0.000).
  Confirms the reset vs. full_memory parity observed above.

### What this tells us

The benchmark reveals a structural limitation: the update mechanism strengthens
*procedure steps* (text) but not *embedding quality* (vector proximity).  When
the wrong family is retrieved, augmentation cannot correct the retrieval error
for the *current* fixture.  Future work should augment the embedding vector
directly (online metric learning) to make the update gate meaningful.

The 4/8 partial validation is honest — the fixture set is working correctly
(oracle=1.000, no verifier_instability, patch_correctness=1.000).

### Failure breakdown (full_memory, mean across seeds)

- wrong_procedure_retrieval: 8.2 ± 1.3 per seed (13.7% of fixtures)
- All other failure classes: 0.0 (no patch errors, no design errors, no instability)

### Reproduce

```bash
cd tacm
# Full benchmark (fast runner, ~4 min)
python run_psm006b_fast.py --seeds 0 1 2 3 4 --workers 8 --out reports

# Quick smoke test (12 fixtures, 1 seed, ~30s)
python run_psm006b_fast.py --seeds 0 --quick --workers 8 --out reports
```

---

## 2026-06-15 — TAC-PSM-006B implemented and unit-tested

**Result:** 49/49 unit tests pass (44.5s). Benchmark and replication scripts ready.

TAC-PSM-006B upgrades PSM-006 from simulated repository repair to semi-real
pytest-grounded repair.  Fixtures contain executable Python source/test files;
verification is real `pytest` subprocess execution (exit code 0 = pass).

### Summary

| Component | Files | Status |
|---|---|---|
| Fixture schema | `tacm/psm006b/fixture_schema.py` | ✓ 60 fixtures, 6 families |
| Fixture builder | `tacm/psm006b/fixture_builder.py` | ✓ 10 fixtures × 6 families |
| Pytest verifier | `tacm/psm006b/pytest_verifier.py` | ✓ subprocess runner |
| Patch applier | `tacm/psm006b/patch_applier.py` | ✓ exact-string replace |
| Memory store | `tacm/psm006b/memory_store.py` | ✓ numpy cosine, no FAISS |
| Repair agent | `tacm/psm006b/procedural_repair_agent.py` | ✓ retrieve-apply-verify-update |
| Baselines | `tacm/psm006b/baselines.py` | ✓ 7 variants |
| Metrics | `tacm/psm006b/metrics.py` | ✓ 13 metrics, 8 gates |
| Benchmark | `benchmark_tac_psm006b_pytest_fixtures.py` | ✓ 5-seed runner |
| Replication | `run_psm006b_replication.py` | ✓ writes `reports/` |
| Unit tests | `tests/test_tac_psm006b_pytest_fixtures.py` | ✓ 49/49 pass |
| Report | `docs/tac_psm006b_pytest_fixture_report.md` | ✓ |
| Failure analysis | `docs/tac_psm006b_failure_analysis.md` | ✓ 8 classes |

### Key design decisions

- **Ground truth**: real `pytest` subprocess per fixture (not heuristic score)
- **Memory**: `SimpleProceduralMemoryStore` (pure numpy, no FAISS dependency)
- **Embedding**: 64-dim, family-centroid seeded from `rng(family_idx × 1000)`
- **Retrieval noise**: σ = 0.10 (moderate; 2 oracle records per family seeded)
- **Baselines**: 7 variants isolate retrieval, update, and memory reuse

### Reproduce

```bash
cd tacm
# Quick smoke test (12 fixtures, 3 seeds)
python benchmark_tac_psm006b_pytest_fixtures.py --quick --seeds 0 1 2

# Full benchmark (60 fixtures, 5 seeds) — takes ~10–15 min
python run_psm006b_replication.py --seeds 0 1 2 3 4

# Unit tests only
python -m pytest tests/test_tac_psm006b_pytest_fixtures.py -q
```

### Next

Full 5-seed benchmark run with gate validation; report outcome in this log.

---

## 2026-06-15 — TAC-Prime-ID001 (Identity-Carried Structure Memory) implemented and tested

**Result:** 42 passed, 16 skipped. All routing and benchmark fixes confirmed.

TAC-Prime-ID001 integrates identity-carried structure memory into the base
TAC-SM model. The identity field propagates a persistent structural signature
across token positions, enabling family-aware routing.

### Bug fixes applied

- `tacm/tacm/id001/routing.py: compute_route_consistency` — fixed empty-dict
  crash when no routes have been observed; fixed uniform-entropy threshold
  to use `log(n_families)` normalization instead of `log(n_routes)`.
- `tacm/tests/test_tacprime_id001_identity_integration.py` — fixed benchmark
  loader to register module in `sys.modules` before calling `exec_module`,
  preventing `ModuleNotFoundError` during module initialization.

### Reproduce

```bash
cd tacm
python -m pytest tests/test_identity_field.py tests/test_tacprime_id001_identity_integration.py -q
```

---

## 2026-06-15 — TAC-PSM-006 (Repository-Grounded Procedural Memory) validated

**Result:** 8/8 gates pass on 5 seeds. Simulated repository repair.

PSM-006 is the predecessor to PSM-006B. It uses a composite heuristic score
(Jaccard similarity + patch applicability + dependency correctness) as a proxy
for pytest pass/fail. All gates pass in the synthetic setting.

### Reproduce

```bash
cd tacm && python benchmark_tac_psm006_repository_repair.py --seeds 0 1 2 3 4
```

---

## 2026-06-15 — TAC-PSM-001 through TAC-PSM-005 validated

**Result:** 27/27 gates passed across 5 studies, 5 seeds each.

TAC-PSM is a five-stage controlled benchmark progression validating the
procedural memory mechanism in TAC-SM. All five stages pass in synthetic
benchmarks.

### Summary

| Study | Title | Gates | Seeds |
|---|---|---|---|
| PSM-001 | Procedure Memory | 7/7 | 5 |
| PSM-002 | Procedure Transfer | 5/5 | 5 |
| PSM-003 | Procedure Lifecycle | 5/5 | 5 |
| PSM-004 | Procedure Survival | 5/5 | 5 |
| PSM-005 | Procedure Discovery | 5/5 | 5 |

### Key results

- **PSM-001:** Retrieval accuracy 1.0, reuse gain 1.0 over reset agent.
  Procedures are stored, retrieved, updated, and forked correctly.

- **PSM-002:** Transfer gain 1.0, A→B→C chain retention 0.89.
  Procedures learned on ImportErrors transfer to DependencyConflicts and VersionMismatch.

- **PSM-003:** Merge quality gain 0.22, strengthening monotone rate 1.0,
  retirement accuracy 1.0. Procedures evolve through their lifecycle correctly.

- **PSM-004:** Survival gap 1.0 (hi-fit procedures all alive, lo-fit all dead after 30 steps).
  Mean robustness 0.43. High-fitness procedures survive selection pressure.

- **PSM-005:** Discovery accuracy 0.49, beats no-discovery at 1.0 rate.
  Unsupervised pattern mining extracts useful procedures from raw traces.

### Framing

This is a **controlled synthetic benchmark validation**. It validates the
procedural memory mechanism, not real-world coding intelligence. No real
repository repair has been tested yet.

### Reproduce

```bash
cd tacm && python3 scripts/run_psm_progression.py --seeds 0 1 2 3 4
```

### Next

PSM-006: Repository-Grounded Procedural Memory — move from synthetic repair
families to real or semi-real repositories.

---

## 2026-06-14 — PSM-001 Procedural Memory validated

7/7 gates passed on 5 seeds. FAISS-backed ProceduralMemoryStore implemented.
50/50 unit tests pass. Replication, ablation, and report scripts complete.

---

## 2026-06-01 — TAC-SM architecture finalised

14-component architecture defined and forward-pass tested at 30M parameters.
Components: Backbone, ConceptVolume, Router, MoE, StructureMemory,
ProceduralMemory, NeuralSurvivalField, Verifier, MultiTokenPrediction,
Agent loop, Losses, Evaluation.
