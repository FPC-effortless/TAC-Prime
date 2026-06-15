# TAC-PSM-006: Repository-Grounded Procedural Memory
## Scientific Report

**Date:** 2026-06-15
**Research Level:** Level 1 — Simulated Repository-Grounded Repair
**Benchmark:** 6 families × 20 tasks = 120 tasks, 5 seeds, 7 system variants

---

## 1. Hypothesis

> **TAC procedural memory improves repository repair success by reusing procedures learned from previous repairs.**

**Main claim tested:**
TAC can remember and reuse repair procedures across repository-grounded tasks, improving verified repair success over reset, retrieval-disabled, random-procedure, and structure-only baselines.

**Scope:** This is a *repository-grounded procedural-memory benchmark*, not a full autonomous software engineer benchmark. We do not claim real-world coding intelligence. All verification in PSM-006 is deterministic (Level 1). Real pytest fixture testing is deferred to Level 2.

---

## 2. Benchmark Design

### 2.1 Task Families

| Family | Sub-types | Example Bug |
|---|---|---|
| `ImportModuleError` | missing, circular, star, relative, namespace | `ModuleNotFoundError: No module named 'requests'` |
| `DependencyConflict` | version, transitive, yanked, platform, extras | `pip: incompatible versions for libA and libB` |
| `VersionAPIMismatch` | removed_function, changed_signature, renamed_class, deprecated_param, removed_module | `AttributeError: module has no attribute 'old_func'` |
| `PathModuleResolution` | missing_file, wrong_cwd, sys_path_missing, editable_install, data_file | `FileNotFoundError: config/settings.yaml not found` |
| `ConfigurationFailure` | missing_key, invalid_format, env_var_missing, schema_mismatch, override_conflict | `KeyError: 'database.host'` |
| `TestAssertionRepair` | wrong_expected, type_mismatch, off_by_one, missing_fixture, async_fixture | `AssertionError: assert 42 == 43` |

### 2.2 Task Structure

Each task carries:
```
task_id, repo_name, family, bug_report, failing_test_output,
relevant_files, expected_procedure_family, oracle_repair_steps,
verification_rule, difficulty, transfer_group
```

### 2.3 Repository Pool

12 repositories across 5 transfer groups:
- `web_framework`: flask-api, django-web, fastapi-service
- `data_pipeline`: pandas-etl, numpy-ext, scikit-pipeline
- `cli_tooling`: click-cli, typer-app
- `test_suite`: pytest-suite, hypothesis-tests
- `async_worker`: celery-worker, rq-jobs

Cross-repo transfer is tested within transfer groups: a procedure learned on `flask-api` should transfer to `django-web`.

### 2.4 Oracle Repair Steps (per sub-type)

Each sub-type has a canonical 5–6 step procedure. Example for `ImportModuleError / missing_import`:
```
1. Parse failing test output to identify missing module name
2. Search relevant_files for import statement referencing module
3. Check requirements.txt for the missing package entry
4. Add missing package to requirements.txt with appropriate version pin
5. Verify import resolves by running: python -c 'import <module>'
6. Rerun failing test to confirm fix
```

---

## 3. System Variants

| Variant | Description |
|---|---|
| `full_memory` | TAC: retrieve → apply → verify → update (primary) |
| `reset` | Empty memory; no retrieval; applied steps = [] |
| `retrieval_disabled` | Memory populated but retrieval off; applies distractor steps |
| `random_procedure` | Picks a random stored procedure (not similarity-based) |
| `structure_only` | Uses family embedding only; strips step content |
| `oracle` | Always applies ground-truth oracle steps (upper bound) |
| `no_update` | Retrieves correctly but never updates memory after verification |

---

## 4. Verification Design (Level 1)

Deterministic composite scoring — no real package installation:

```
composite_score = 0.45 × family_match
               + 0.30 × step_overlap   (Jaccard)
               + 0.15 × keyword_coverage
               + 0.10 × repo_context_hit

success = composite_score >= verification_rule["min_score"]
```

`min_score = difficulty + 0.10` (harder tasks require higher quality).

This scoring is designed to reward:
1. Retrieving the **right procedure family** (largest weight)
2. Applying **steps that overlap with oracle** steps (second largest)
3. Using domain-relevant **keywords** in steps
4. The task having the **expected context file** in `relevant_files`

---

## 5. Metrics

| # | Metric | Formula | Direction |
|---|---|---|---|
| 1 | `verified_repair_success` | fraction(success=True) | ↑ higher is better |
| 2 | `procedure_retrieval_accuracy` | fraction(retrieved_family == expected_family) | ↑ |
| 3 | `procedure_reuse_gain` | TAC_success − reset_success | ↑ (target ≥ 0.10) |
| 4 | `update_retry_improvement` | full_memory_success − no_update_success | ↑ |
| 5 | `transfer_success` | fraction(retrieved & verified, cross-repo) | ↑ |
| 6 | `wrong_procedure_harm` | random_success − reset_success | ↓ (should be ≤ 0) |
| 7 | `steps_to_repair` | mean(len(applied_steps)) | lower = more efficient |
| 8 | `survival_score_stability` | std-dev(proxy_survival) | ↓ lower = more stable |
| 9 | `procedure_family_confusion` | 6×6 confusion matrix | diagnostic |

---

## 6. Success Gates

| Gate | Criterion | What it tests |
|---|---|---|
| `tac_beats_reset_by_0.10` | `reuse_gain >= 0.10` | Memory is useful |
| `retrieval_accuracy_ge_0.60` | `retrieval_acc >= 0.60` | Retrieval is accurate |
| `update_improves_retry` | `update_retry_improvement > 0` | Update helps |
| `transfer_success_gt_0` | `transfer_success > 0` | Cross-repo transfer works |
| `wrong_procedure_no_gain` | `wrong_procedure_harm <= 0.05` | Wrong procedures don't help |
| `survival_stable` | `survival_std <= 0.35` | Stable survival field |
| `oracle_above_tac` | `oracle_success >= tac_success` | Upper bound respected |
| `no_update_underperforms_tac` | `no_update_success < tac_success` | Update step adds value |

---

## 7. Results (5-seed mean ± std)

*Results are populated by running `python scripts/run_psm006_replication.py`.*
*See `reports/psm006_replication_summary.txt` for full numerical output.*

### 7.1 Primary Table

| Variant | Repair Success | Retrieval Acc | Reuse Gain |
|---|---|---|---|
| `full_memory` | — | — | — |
| `reset` | — | 0.00 | 0.00 |
| `retrieval_disabled` | — | — | — |
| `random_procedure` | — | — | — |
| `structure_only` | — | — | — |
| `oracle` | — | 1.00 | — |
| `no_update` | — | — | — |

*Fill in from replication run.*

### 7.2 Gate Results

*Fill in from replication run.*

### 7.3 Per-Family Breakdown

*Fill in from replication run.*

---

## 8. Limitations and Research Constraints

### Level 1 Limitations
- All verification is **deterministic simulation** — the verifier measures procedural similarity, not actual code correctness.
- No real package installation or subprocess execution.
- Repository "context" is synthetic — real repository repair requires real file contents.

### What PSM-006 Does Claim
- That the *procedural memory mechanism* (retrieve → apply → verify → update) outperforms ablations in a controlled, reproducible simulation.
- That procedure retrieval can distinguish the correct repair family from 6 alternatives.
- That cross-repo transfer is non-zero within the simulated framework.

### What PSM-006 Does NOT Claim
- That TAC is a production-ready software engineering agent.
- That simulated repair success translates directly to real test-pass rates.
- Any comparison to state-of-the-art coding agents (SWE-bench, etc.).

---

## 9. Path to Level 2

Level 2 would add real pytest fixtures for a subset of tasks:
1. Select 10–20 tasks from `ImportModuleError` and `TestAssertionRepair` (most amenable to fixture-based testing).
2. Create real Python package stubs in `tests/fixtures/repos/`.
3. Run `pytest` on the fixtures after applying repair steps.
4. Compare Level 1 simulation success with Level 2 real pass rate.

---

## 10. Citation and Reproducibility

```
Experiment:   TAC-PSM-006
Title:        Repository-Grounded Procedural Memory
Level:        1 (simulated)
Seeds:        5 (0–4)
Tasks:        120 (6 families × 20)
Variants:     7
Repository:   tacm/tacm/psm006/
Run with:     python scripts/run_psm006_replication.py
Report:       reports/psm006_replication_summary.txt
```
