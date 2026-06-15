# TAC-PSM-006B: Failure Analysis

This document catalogues the failure classes that can occur during
TAC-PSM-006B benchmark runs, their root causes, and the mitigations
implemented in the benchmark design.

---

## 1. Failure Taxonomy

PSM-006B defines eight failure classes (`FAILURE_CLASSES` in `fixture_schema.py`):

| Failure Class | Description |
|---|---|
| `wrong_procedure_retrieval` | Retrieved procedure is from the wrong repair family |
| `correct_procedure_wrong_patch` | Right family retrieved, but patch content does not match fixture |
| `patch_wrong_file` | Patch targets a filename not present in the fixture's file set |
| `insufficient_update` | Augmented procedure steps did not improve second-attempt success |
| `family_confusion` | Two similar-sounding families are confused (e.g. `import_module_error` vs `path_module_resolution`) |
| `transfer_failure` | Procedure retrieved correctly but fails to transfer to near/far transfer group |
| `fixture_design_error` | Fixture's expected_patch is inconsistent with its source_files (internal error) |
| `verifier_instability` | Pytest returns different exit codes on repeated runs of the same fixture |

---

## 2. Failure Analysis by Class

### 2.1 `wrong_procedure_retrieval`

**Root cause**: Fixture embedding is too close to a non-target family's centroid due
to high retrieval noise or a low-diversity fixture set.

**Conditions**: Most likely when:
- `retrieval_noise` σ > 0.30 (high noise) AND families have low centroid separation
- Two families share overlapping bug descriptions (e.g. `import_module_error` and
  `path_module_resolution` both involve Python import failures)

**Mitigation in PSM-006B**:
- Family centroids seeded from `np.random.default_rng(family_idx * 1000)` for
  maximum centroid diversity in 64-dim space
- Retrieval noise set to σ = 0.10 (moderate — realistic but not unrealistically low)
- 2 records per family stored for centroid stability

**Expected rate (5-seed mean)**: < 0.45 (retrieval accuracy >= 0.55)

---

### 2.2 `correct_procedure_wrong_patch`

**Root cause**: The correct family's procedure is retrieved but the oracle patch string
(`expected_patch[file]["old"]`) does not appear verbatim in the fixture's current file.

**Conditions**: Occurs when:
- A fixture has been modified from its original state by a previous (wrong-family) patch
- The `expected_patch` `old` string has a typo or extra whitespace

**Mitigation**:
- All 60 fixtures are independently constructed — no fixture depends on another
- `PatchApplier.apply()` uses verbatim substring match (`str.replace(..., 1)`) and
  reports `correct_procedure_wrong_patch` when the match fails

**Expected rate**: Should be 0 for correct-family retrievals given self-consistent
fixture design. Appears in baselines where the wrong-family patch mutates the file
before the oracle patch is attempted (e.g. `full_memory` with update-and-retry).

---

### 2.3 `patch_wrong_file`

**Root cause**: Patch targets a filename not present in `fixture.all_files()`.

**Conditions**: Occurs in the `structure_only` baseline when the structure-only
patch stub uses a wrong filename.

**Mitigation**:
- `apply_structure_only_patch()` reuses the same filenames as `expected_patch`
- `apply_wrong_family_patch()` only annotates existing `.py` files (never invents filenames)

---

### 2.4 `insufficient_update`

**Root cause**: Memory augmentation with oracle steps does not change the retrieval
output enough to fix the bug on retry.

**Conditions**: Occurs when `augment()` adds steps but the retrieval score for the
wrong family is still higher after the update (embedding has not changed, only steps).

**Mitigation**:
- In PSM-006B, augmentation targets procedure steps (not the embedding), so it
  helps on future fixtures in the same family but may not fix the current one
- This is by design: step augmentation is not a silver bullet, and its limited
  effectiveness is part of what the benchmark measures

---

### 2.5 `family_confusion`

**Root cause**: Two families with similar embeddings are systematically confused.

**Most likely pairs**:
- `import_module_error` ↔ `path_module_resolution` (both involve Python imports)
- `dependency_config_conflict` ↔ `configuration_failure` (both involve config)

**Detection**: Use `compute_family_confusion_matrix()` to identify confusion pairs.

**Mitigation**:
- Centroid seeds (family_idx × 1000) are chosen to maximise separation in 64-dim space
- Retrieval blends cosine similarity + success_rate; success_rate diverges as the
  store accumulates per-family repair experience

---

### 2.6 `transfer_failure`

**Root cause**: Procedure retrieved from a `train`-group record fails to generalise
to `near_transfer` or `far_transfer` fixtures.

**Expected behaviour**:
- `near_transfer`: minor variation → should transfer well with a seeded oracle procedure
- `far_transfer`: structural variation → harder; some transfer failure expected

**Monitoring**: `cross_fixture_transfer_success` metric tracks per-group pass rate.

---

### 2.7 `fixture_design_error`

**Root cause**: Internal consistency error — `expected_patch` cannot repair the
bug in `source_files` as designed.

**Prevention**: All 60 fixtures are validated at build time by `build_all_fixtures()`.
The oracle patch is always applied and the result verified at fixture construction time
via a dry-run of `PatchApplier.apply()` (structural check only — not pytest execution,
which would make fixture building slow).

If this class appears at runtime, it is a regression in `fixture_builder.py`.

---

### 2.8 `verifier_instability`

**Root cause**: Pytest behaves non-deterministically for a fixture (e.g. timing-based
assertion, OS-specific path, or test that depends on global state).

**Prevention**: All 60 fixtures use deterministic, stdlib-only operations:
- No `time.sleep()` / timing tests
- No filesystem-path-dependent assertions
- No global state mutations

**Detection**: `PytestVerifier.check_instability()` runs pytest twice and compares
exit codes. If instability is detected in CI, the fixture should be redesigned.

---

## 3. Observed Failure Rates — 2026-06-15 Replication Run

Run: 5 seeds × 60 fixtures × 7 variants. Subprocess pytest verification.
All numbers are mean ± std across 5 seeds.

### 3.1 Failure class distribution (full_memory variant)

| Failure Class | Mean/seed | Std | % of fixtures |
|---|---|---|---|
| `wrong_procedure_retrieval` | 8.2 | 1.3 | **13.7%** |
| `correct_procedure_wrong_patch` | 0.0 | 0.0 | 0.0% |
| `patch_wrong_file` | 0.0 | 0.0 | 0.0% |
| `insufficient_update` | 0.0 | 0.0 | 0.0% |
| `family_confusion` | 0.0 | 0.0 | 0.0% |
| `transfer_failure` | 0.0 | 0.0 | 0.0% |
| `fixture_design_error` | 0.0 | 0.0 | 0.0% |
| `verifier_instability` | 0.0 | 0.0 | 0.0% |
| none (success) | 51.8 | 1.3 | **86.3%** |

**Key finding**: All failures are of class `wrong_procedure_retrieval`.
Zero patch errors, zero design errors, zero verifier instability — the
fixture set is internally consistent and the patch mechanism is correct.

### 3.2 Observed pass rates by variant

| Variant | Observed mean | Observed std | Pre-run expectation |
|---|---|---|---|
| `oracle` | **1.000** | 0.000 | 0.90 – 1.00 ✓ |
| `structure_only` | **0.927** | 0.022 | 0.10 – 0.30 ✗ (much higher than expected) |
| `full_memory` | **0.863** | 0.022 | 0.65 – 0.90 ✓ |
| `reset` | **0.863** | 0.022 | 0.55 – 0.75 ✗ (higher than expected) |
| `no_update` | **0.860** | 0.035 | 0.60 – 0.85 ✓ |
| `retrieval_disabled` | **0.550** | 0.000 | 0.10 – 0.20 ✗ (much higher than expected) |
| `random_procedure` | **0.440** | 0.030 | 0.10 – 0.20 ✗ (higher than expected) |

**Analysis of expectation mismatches:**

- `structure_only` scored 0.927 (expected 0.10–0.30): The stub patch replaces
  target file content with a comment. Many tests only check imports or constants
  that are *not* in the patched file — so the stub doesn't break them. This
  reveals that some fixtures are under-constrained.

- `retrieval_disabled` scored 0.550 (expected 0.10–0.20): With retrieval
  disabled the agent always applies the first family's procedure. That procedure
  happens to produce valid-enough patches for ~55% of fixtures (the pass-regardless
  baseline). This is the floor below which no retrieval can fall.

- `random_procedure` scored 0.440 (expected 0.10–0.20): Same reason — the
  pass-regardless rate drives the floor up, and some random families happen
  to fix some fixtures.

- `reset` scored 0.863 (expected 0.55–0.75): Reset re-seeds oracle procedures
  before every fixture, so the "no memory reuse" baseline is also a "fresh oracle
  every time" baseline. This makes reset nearly as strong as full_memory.

### 3.3 Per-family retrieval accuracy (mean across seeds)

| Family | Retrieval acc | Main confusion target |
|---|---|---|
| `path_module_resolution` | 0.960 | — |
| `test_assertion_repair` | 0.840 | `dependency_config_conflict` (0.8/seed) |
| `import_module_error` | 0.800 | `path_module_resolution` (0.6/seed) |
| `version_api_mismatch` | 0.800 | `dependency_config_conflict` (0.8/seed) |
| `dependency_config_conflict` | 0.740 | `test_assertion_repair` (2.0/seed) |
| `configuration_failure` | 0.740 | `import_module_error` (1.2/seed) |

### 3.4 Pre-run expectations vs observed

| Expectation | Met? | Notes |
|---|---|---|
| `oracle` = 1.000 | ✓ | Confirmed — fixtures well-formed |
| `patch_correctness` = 1.000 | ✓ | Zero patch application failures |
| `verifier_instability` = 0 | ✓ | All fixtures deterministic |
| `fixture_design_error` = 0 | ✓ | No internal inconsistencies |
| `wrong_proc_retrieval` dominates | ✓ | 100% of failures are retrieval errors |
| `full_memory` > `reset` by ≥ 0.10 | ✗ | Both 0.863 — seeded reset is too strong |
| update_improves_retry > 0 | ✗ | Embedding unchanged by update; same retrieval on retry |

---

## 4. Failure Mitigation Hierarchy

```
Priority 1: fixture_design_error, verifier_instability
  → Critical — indicates a broken fixture; stop and fix before reporting results.

Priority 2: wrong_procedure_retrieval
  → Most common; tunable via retrieval_noise and centroid seed.

Priority 3: correct_procedure_wrong_patch
  → Indicates a patch generation gap; augment oracle_repair_procedure steps.

Priority 4: patch_wrong_file
  → Implementation error in PatchApplier or fixture_builder; rare.

Priority 5: insufficient_update, family_confusion, transfer_failure
  → Expected partial failures; documented and tracked via metrics.
```

---

## 5. Debugging Runbook

When a gate fails on a seed:

1. **Check `variant_rates`** — which variants are below expected?
2. **Check `confusion matrix`** — which family pairs are confused most?
3. **Check `failures` dict** — which failure class dominates?
4. **Inspect individual traces** — look for fixtures where `before_result.success` is True
   (fixture was already passing — design error) or `after_result.exit_code == 2`
   (pytest collection error — import issue in fixture).
5. **Run `check_instability()`** — confirm the fixture passes/fails deterministically.
6. **Raise retrieval noise threshold** only if failure is dominated by
   `wrong_procedure_retrieval` — this makes the benchmark *harder*, not easier.
