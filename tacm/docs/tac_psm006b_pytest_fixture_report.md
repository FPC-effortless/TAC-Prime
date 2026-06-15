# TAC-PSM-006B: Semi-Real Pytest Repository Repair Fixtures — Research Report

## Abstract

PSM-006B upgrades TAC's procedural memory benchmark from simulated
repository repair (PSM-006) to semi-real pytest-grounded repair.  Fixtures
contain executable Python source and test files.  The verifier runs actual
`pytest` subprocesses; pass/fail is determined by exit code 0 — not a
heuristic score.  The central claim is that TAC's procedural memory reuse
improves real pytest-verified repair over six controlled baselines.

---

## 1. Motivation

PSM-006 validated TAC's ability to retrieve and apply repository repair
procedures in a simulated environment where a composite heuristic score
proxied pytest pass/fail.  PSM-006B replaces this proxy with ground truth:
every success metric is backed by a subprocess pytest invocation.  This
eliminates the risk that heuristic calibration masked retrieval failures.

---

## 2. Fixture Design

### 2.1 Structure

Each fixture (`tacm/tacm/psm006b/fixture_schema.py: Fixture`) contains:

| Field | Content |
|---|---|
| `source_files` | Buggy Python source code with the injected defect |
| `test_files`   | Pytest test file that detects the defect |
| `config_files` | Optional `conftest.py` / `pytest.ini` |
| `expected_patch` | Minimal `{file: {old, new}}` that fixes the defect |
| `oracle_repair_procedure` | `{family, steps}` — ground-truth procedure |
| `verification_command` | Exact `pytest ...` command run after patching |
| `transfer_group` | `train` \| `near_transfer` \| `far_transfer` |
| `difficulty`   | `easy` \| `medium` \| `hard` |

### 2.2 Families (6 × 10 = 60 fixtures)

| Family | Description | Example defect |
|---|---|---|
| `import_module_error` | Renamed/moved symbol | `from utils import deprecated_fn` |
| `dependency_config_conflict` | Duplicate fixture def | Two `@pytest.fixture` with same name |
| `version_api_mismatch` | Changed call signature | `open(f, 'U')` (removed in 3.11) |
| `path_module_resolution` | Wrong import path | `from lib.old import X` after refactor |
| `configuration_failure` | Bad config key/value | `pytest.ini` `[pytest] addopts = --bad` |
| `test_assertion_repair` | Wrong expected value | `assert compute(3) == 7` (should be 9) |

### 2.3 Transfer groups

- **train** (6/10 per family): seeded into memory before evaluation
- **near_transfer** (2/10): same family, slightly different defect form
- **far_transfer** (2/10): same family, structurally different defect

### 2.4 Self-containedness

All 60 fixtures use only the Python standard library and `pytest`.  No pip
installs are required at test time.  This guarantees reproducibility across
Python 3.9+ environments.

---

## 3. Agent Architecture

The TAC repair agent (`ProceduralRepairAgent006B`) implements the standard
retrieve-apply-verify-update loop:

```
for each fixture:
  1. RETRIEVE  — query SimpleProceduralMemoryStore (cosine sim + success_rate)
  2. APPLY     — generate/apply patch from retrieved family
  3. VERIFY    — subprocess pytest; capture exit code
  4. UPDATE    — reinforce on success; augment with oracle steps on failure
  5. RECORD    — emit RepairTrace006B
```

### 3.1 Memory store

`SimpleProceduralMemoryStore` is a pure-numpy store (no FAISS dependency).
Retrieval: `score = 0.7 * cosine_sim + 0.3 * success_rate`.

### 3.2 Embedding

Each fixture maps to a 64-dimensional embedding with a family-specific
centroid (from `np.random.default_rng(family_idx * 1000)`) plus
fixture-specific jitter (from `sha256(fixture_id)`, σ = 0.30).  Retrieval
noise σ = 0.10 simulates imperfect embedding.  Centroid separation makes
cosine retrieval reliable when noise is moderate.

---

## 4. Baselines (7 variants)

| Variant | Description |
|---|---|
| `full_memory` | TAC full procedure memory (seeded, retrieval on, update on) |
| `reset` | Memory cleared and re-seeded before every fixture (no reuse) |
| `retrieval_disabled` | Store exists but always returns first family (wrong) |
| `random_procedure` | Randomly selects any family's procedure |
| `structure_only` | Correct file targeted, stub content patched in |
| `no_update` | Full memory + retrieval, but no update after verification |
| `oracle` | Always uses the correct family's procedure (upper bound) |

---

## 5. Success Gates (8)

| Gate | Threshold |
|---|---|
| TAC beats reset by ≥ 0.10 | `pytest_pass_rate(full_memory) - pytest_pass_rate(reset) >= 0.10` |
| Retrieval accuracy ≥ 0.55 | `procedure_retrieval_accuracy >= 0.55` |
| Update improves retry | `retry_after_update_success > 0` |
| no_update underperforms TAC | `pass_rate(no_update) < pass_rate(full_memory)` |
| Random procedure no benefit | `pass_rate(random) <= pass_rate(full_memory)` |
| Oracle remains above TAC | `pass_rate(oracle) >= pass_rate(full_memory)` |
| Cross-fixture transfer > 0 | `cross_fixture_transfer_success > 0` |
| Reuse gain positive | `procedure_reuse_gain > 0` |

---

## 6. Metrics (13)

| Metric | Definition |
|---|---|
| `pytest_pass_rate` | Fraction fixtures where after-patch pytest exits 0 |
| `first_attempt_repair_success` | Pass on first attempt without retry |
| `retry_after_update_success` | Pass only after update-and-retry |
| `procedure_retrieval_accuracy` | Fraction with correct family retrieved |
| `procedure_reuse_gain` | `pass_rate(full_memory) - pass_rate(reset)` |
| `cross_fixture_transfer_success` | Pass rate on near+far transfer fixtures |
| `cross_family_transfer_success` | Pass rate on wrong-family retrievals that still passed |
| `wrong_procedure_harm` | `pass_rate(oracle) - pass_rate(random)` |
| `patch_correctness` | Fraction where patch was applied without error |
| `steps_to_repair` | Mean procedure step count |
| `time_to_repair_s` | Mean wall-clock seconds per fixture |
| `procedure_survival_stability` | Fraction of traces with no failure class |
| `family_confusion_rate` | Fraction of retrieval errors that are cross-family |

---

## 7. Implementation Notes

### 7.1 Calibration decisions
- Retrieval noise σ = 0.10 (lower would be unrealistically easy)
- 2 records per family seeded (adds diversity, reduces single-record variance)
- Oracle procedure steps match the minimal repair action sequence per family
- `max_retries = 1` for `full_memory` variant only

### 7.2 Known limitations
- Fixtures use stub source code, not real production repositories
- 64-dim embedding is smaller than a production model would use
- Subprocess pytest has ~0.5s overhead per fixture

### 7.3 Reproducibility
- All fixtures are deterministic (seeded from `fixture_id` hash)
- 5 seeds (0–4) run by default; gate evaluation requires majority pass
- All fixtures self-contained (stdlib + pytest only)

---

## 8. File map

```
tacm/tacm/psm006b/
  __init__.py                   — public exports
  fixture_schema.py             — Fixture dataclass, FAMILY_NAMES, FAILURE_CLASSES
  fixture_builder.py            — build_all_fixtures() → 60 Fixture objects
  pytest_verifier.py            — PytestVerifier (subprocess pytest runner)
  patch_applier.py              — PatchApplier, PatchResult
  memory_store.py               — SimpleProceduralMemoryStore (numpy, no FAISS)
  procedural_repair_agent.py    — ProceduralRepairAgent006B, RepairTrace006B
  baselines.py                  — run_all_baselines(), VARIANT_NAMES
  metrics.py                    — compute_metrics(), evaluate_success_gates()

tacm/
  benchmark_tac_psm006b_pytest_fixtures.py   — main benchmark script
  run_psm006b_replication.py                 — replication runner (writes reports/)
  tests/test_tac_psm006b_pytest_fixtures.py  — 40+ unit + integration tests
  docs/tac_psm006b_pytest_fixture_report.md  — this file
  docs/tac_psm006b_failure_analysis.md       — failure taxonomy and analysis
```
