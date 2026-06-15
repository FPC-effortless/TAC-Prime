# TAC Research Log

Dated entries, most recent first.

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
