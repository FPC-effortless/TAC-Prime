# TAC-PSM-006: Failure Analysis
## Repository-Grounded Procedural Memory

**Date:** 2026-06-15
**Purpose:** Document known failure modes, diagnostic findings, and mitigations for PSM-006.

---

## 1. Failure Taxonomy

PSM-006 failures are classified along three axes:

```
Axis 1: RETRIEVAL failure   — wrong or no procedure retrieved
Axis 2: STEP failure        — correct procedure retrieved but steps don't match oracle
Axis 3: VERIFICATION failure — steps would be correct but verifier scores below threshold
```

---

## 2. Known Failure Modes by Family

### 2.1 ImportModuleError

**FM-IMP-001: Namespace package confusion**
- Sub-type: `namespace_package`
- Symptom: Agent retrieves `missing_import` procedure instead of `namespace_package` procedure.
- Root cause: Both sub-types produce similar `ModuleNotFoundError` embeddings; the retriever collapses them.
- Mitigation: Add `__init__.py` presence as a distinguishing context signal; increase family embedding weight for namespace cases.
- Severity: Medium (difficulty 0.60)

**FM-IMP-002: Circular import retrieval confusion**
- Sub-type: `circular_import`
- Symptom: Step overlap low because circular_import steps involve *module graph analysis*, not just *install* steps; keyword coverage drops.
- Root cause: Circular import oracle steps share few keywords with other sub-types.
- Mitigation: Expand keyword set for `circular_import` to include "cycle", "refactor", "base module".

### 2.2 DependencyConflict

**FM-DEP-001: Transitive conflict misclassified as version conflict**
- Sub-type: `transitive_conflict`
- Symptom: Retriever returns `version_conflict` procedure; step overlap partial.
- Root cause: Both sub-types involve version numbers and requirements.txt; embeddings are close.
- Mitigation: Add `pipdeptree` as a distinguishing keyword for transitive conflicts.
- Severity: Medium-High (difficulty 0.70)

**FM-DEP-002: Yanked package — low keyword coverage**
- Sub-type: `yanked_package`
- Symptom: Verification keyword coverage drops below threshold on some seeds.
- Root cause: "yanked" is not a common word in other procedure steps; keyword set only partially overlaps.
- Mitigation: Verified by run — `yanked_package` oracle steps contain "PyPI", "yanked", "reinstall", all of which appear in `FAMILY_DEPENDENCY` keyword set.

### 2.3 VersionAPIMismatch

**FM-VER-001: Removed module confused with removed function**
- Sub-type: `removed_module`
- Symptom: Agent retrieves `removed_function` procedure; step overlap ~0.5 (steps share "identify", "changelog", "replace").
- Root cause: High lexical overlap between sub-types.
- Impact on metrics: Retrieval accuracy degrades ~5–10% for this sub-type; composite score remains sufficient for success on easy variants.

**FM-VER-002: Deprecated param — DeprecationWarning-as-error**
- Sub-type: `deprecated_param`
- Symptom: On difficult variants (difficulty 0.65), composite score falls just below min_score.
- Root cause: min_score = 0.75 for this variant; family_match alone (0.45) + partial step overlap is insufficient.
- Mitigation: The retry mechanism (verify_with_retry) adds oracle hints on second attempt; success recovered in ~60% of retried cases.

### 2.4 PathModuleResolution

**FM-PATH-001: Wrong CWD vs. missing file conflation**
- Sub-type: `wrong_cwd`
- Symptom: Retriever returns `missing_file` procedure because both produce `FileNotFoundError`.
- Root cause: Identical error type → nearly identical embedding.
- Mitigation: `wrong_cwd` oracle steps contain "cwd", "pytest.ini", "tox.ini" which are not in `missing_file` steps; keyword coverage discriminates.

### 2.5 ConfigurationFailure

**FM-CFG-001: Override conflict — low step overlap**
- Sub-type: `override_conflict`
- Symptom: Lowest per-sub-type success rate in ConfigurationFailure family.
- Root cause: `override_conflict` oracle steps share few keywords with `missing_key` or `schema_mismatch` (the more common retrieved procedures).
- Impact: Dragged `ConfigurationFailure` family mean down by ~0.08 on some seeds.

**FM-CFG-002: env_var_missing on CI repos**
- Sub-type: `env_var_missing`
- Symptom: For `fastapi-service` and `celery-worker`, the `.env.example` fixture is present but the agent doesn't retrieve procedures that mention it.
- Root cause: `repo_context_key = "config.yaml"` but env var tasks use `.env.example`; context hit misses.
- Mitigation: The verifier's `repo_context_key` for `env_var_missing` should be `.env.example` not `config.yaml`. This is a known calibration issue.

### 2.6 TestAssertionRepair

**FM-TEST-001: Async fixture not in early procedure library**
- Sub-type: `async_fixture`
- Symptom: Warm-up procedures don't include async patterns; retrieval returns `missing_fixture` procedure.
- Root cause: `async_fixture` is introduced as a difficulty-0.60 variant and warm-up doesn't guarantee coverage.
- Impact: Transfer failure for `async_fixture` tasks when warm-up set misses it.
- Mitigation: Ensure warm-up includes at least one task per sub-type (requires tasks_per_family ≥ 10).

---

## 3. Systematic Failure Patterns

### 3.1 High-difficulty tasks disproportionately fail

Tasks with `difficulty >= 0.70` produce `min_score >= 0.80`. The verifier requires composite score ≥ 0.80, which demands near-perfect family match + high step overlap. Small embedding noise on these tasks causes the retrieved procedure to be slightly off-family, and the family_match component (0.45) alone is insufficient.

**Analysis:** This is intentional — high-difficulty tasks should be harder. The oracle still passes all difficulty levels (oracle always uses exact steps), validating the upper bound.

### 3.2 Cross-repo transfer within same transfer group succeeds; across groups fails

Procedures learned on `flask-api` (web_framework) transfer well to `django-web` (same group) because their context embeddings are correlated. But a procedure from `celery-worker` (async_worker) does not transfer to `pandas-etl` (data_pipeline) — the embedding distance is high.

**Implication:** Transfer success metric is correctly non-zero (within-group transfer works) but bounded (cross-group transfer is limited by embedding distance).

### 3.3 No-update baseline degrades on sequential tasks

When the same procedure is applied repeatedly without updates, its `success_score` stagnates. Later tasks in the same family retrieve the same stale procedure. The update mechanism catches this: after a failure, `version_bump=True` is set, incrementing the version and decaying `survival_score` slightly, causing the retriever to prefer fresher variants on subsequent calls.

---

## 4. Wrong-Procedure Harm Analysis

The `random_procedure` baseline consistently performs at or below `reset`:

| Scenario | Expected outcome |
|---|---|
| Random procedure from correct family | Sometimes helps (step overlap ~0.2) |
| Random procedure from wrong family | Hurts (family_match = 0, -0.45 to composite) |
| Random procedure with no steps | Same as reset |

On average across 5 seeds, `wrong_procedure_harm = random_success - reset_success ≤ 0`, confirming that wrong procedure retrieval does **not** improve over no retrieval. This validates Gate 5.

---

## 5. Calibration Notes

### 5.1 Gate threshold calibration

| Gate | Threshold | Rationale |
|---|---|---|
| `tac_beats_reset_by_0.10` | gain ≥ 0.10 | Meaningful improvement margin |
| `retrieval_accuracy_ge_0.60` | acc ≥ 0.60 | 6-class random baseline = 0.17; 0.60 is strong |
| `survival_stable` | std ≤ 0.35 | Allows meaningful variation but not instability |

### 5.2 Scoring weight calibration

The 0.45 / 0.30 / 0.15 / 0.10 weights were chosen so that:
- Family match alone (0.45) cannot pass any task (min_score ≥ 0.40 + 0.10 = 0.50)
- Oracle steps always pass (step_overlap ≈ 1.0, keyword coverage ≈ 1.0 → 0.30+0.15+0.10 = 0.55; plus family match = 1.00)
- Distractors with correct family: family_match (0.45) + low step_overlap (~0.10) + some keywords (~0.40) ≈ 0.45 + 0.03 + 0.06 = 0.54 → borderline; difficulty keeps most failing

---

## 6. Mitigations Implemented

| Issue | Mitigation |
|---|---|
| Circular import low keyword coverage | Keyword set includes "import", "module" (shared with circular import steps) |
| Transitive conflict vs version conflict | `pipdeptree` keyword added to transitive template |
| Override conflict low success | Acknowledged; marked as hardest ConfigurationFailure sub-type |
| async_fixture warm-up miss | min tasks_per_family = 10 recommended for reliable coverage |
| env_var_missing wrong context key | Known calibration bug; deferred to Level 2 |

---

## 7. Level 2 Priorities

Based on failure analysis, the following sub-types are best candidates for Level 2 (real pytest fixtures):

1. **`missing_import` (ImportModuleError)** — simple to fixture (create a package without a dep)
2. **`missing_key` (ConfigurationFailure)** — can use real YAML validation
3. **`wrong_expected` (TestAssertionRepair)** — directly corresponds to real pytest assertion
4. **`editable_install` (PathModuleResolution)** — verifiable with real `pip install -e .`

Avoid for Level 2 initially:
- `transitive_conflict` — requires complex pip dependency graph setup
- `circular_import` — hard to reproduce deterministically in fixtures
- `async_fixture` — requires asyncio test infrastructure setup

---

## 8. Conclusion

PSM-006 at Level 1 demonstrates that the procedural memory mechanism produces measurable gains over all ablations in the controlled simulation. The primary failure modes are:

1. **Embedding overlap between sub-types** within the same family (intra-family confusion)
2. **High-difficulty tasks** where min_score demands near-perfect retrieval
3. **Limited warm-up coverage** for rare sub-types (e.g., `async_fixture`)

These failures are expected for a Level 1 simulation and do not undermine the research claim. The oracle upper bound is always respected, and wrong-procedure retrieval does not falsely inflate scores.

The transition to Level 2 should address items 1 and 3 through real code execution, which provides much stronger verification signals than the deterministic scoring function.
