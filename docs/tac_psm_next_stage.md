# TAC-PSM-006: Repository-Grounded Procedural Memory

**Status:** Planned
**Depends on:** PSM-001 through PSM-005 (all validated)
**Date defined:** 2026-06-15

---

## Scientific Question

Can TAC's procedural memory system, validated in synthetic benchmarks, transfer to a real or semi-real repository repair setting?

---

## The Gap

PSM-001–005 validated the procedural memory *mechanism* on controlled synthetic tasks. The four benchmark families (ImportErrors, DependencyConflicts, VersionMismatch, PathResolution) use:

- Symbolic step strings, not real shell commands
- Jaccard-based success metrics, not test suite execution
- Deterministic embeddings, not learned representations

PSM-006 closes this gap by grounding the benchmark in real repositories.

---

## Core Benchmark Design

### Task Definition

```
Given:
  - Repository context (file tree + key files)
  - Failing test output
  - Bug report / error trace

Required:
  - Retrieve or discover a repair procedure
  - Apply the procedure (execute steps)
  - Verify success (test suite must pass)
  - Update memory (strengthen or fork)
```

### Task Families (6 planned)

| Family | Description | Example |
|---|---|---|
| Import Resolution | Fix Python import errors in real packages | `ModuleNotFoundError` in CI |
| Dependency Conflict | Resolve real pip/conda conflicts | `pip install` version clash |
| API Migration | Adapt code to a changed third-party API | Deprecated sklearn call |
| Test Infrastructure | Fix pytest configuration / fixture errors | Missing conftest.py |
| Environment Isolation | Fix virtualenv / Docker path issues | Wrong Python interpreter |
| Build System | Fix Makefile / setup.py / pyproject.toml issues | Missing build dependency |

### Scale

- At least **20 tasks per family** (120 tasks total)
- At least **5 random seeds** per task
- At least **3 procedures per family** in memory at evaluation time

---

## Required Baselines

| Baseline | Description |
|---|---|
| Reset agent | No memory; solves each task from scratch |
| Retrieval-disabled agent | Memory exists but retrieval is disabled |
| Random procedure agent | Retrieves a random procedure |
| Wrong procedure agent | Retrieves the worst-ranked procedure |
| Same-size Transformer | Standard decoder-only, same parameter count |
| Same-size MoE | DeepSeek-style MoE without procedural memory |
| Oracle procedure | Ground-truth repair steps (upper bound) |

---

## Success Gates

| Gate | Threshold | Rationale |
|---|---|---|
| Real repair success vs reset | ≥ 0.10 improvement | Memory must help |
| Procedure retrieval accuracy | ≥ 0.60 | Retrieval must be reliable |
| Update improves retry | > 0 | Memory must learn from failure |
| Transfer across repositories | > 0 | Procedures must generalise |
| Wrong procedure does not improve | Negative gain | Retrieval must be selective |
| Seeds | ≥ 5 | Reproducibility |
| Tasks per family | ≥ 20 | Statistical power |

---

## Evaluation Metrics

- **Repair success rate** — fraction of tasks where tests pass after applying procedure
- **Retrieval accuracy** — fraction of retrievals matching the correct family
- **Steps to success** — mean number of procedure steps executed before success
- **Memory efficiency** — quality gain per unit of memory footprint
- **Transfer rate** — fraction of tasks solved using a procedure from a different repository
- **Discovery rate** — fraction of tasks solved using a discovered (not pre-stored) procedure

---

## Repository Sources

**Semi-real approach (lower risk, faster):**
- Use real package names and real error messages but simulate the repository files.
- Step execution is simulated (not real shell commands) but steps reference real API calls.
- Success is still measured by a simulated test suite.

**Fully real approach (higher risk, stronger claim):**
- Clone real GitHub repositories.
- Run real test suites (pytest, tox).
- Execute real repair steps via a subprocess agent.
- This requires a sandboxed execution environment.

**Recommended path:** Start semi-real, then add one fully real family as a proof-of-concept.

---

## Neural Integration Requirement

PSM-006 is the first stage that requires a *trained* TACSM model (or a pretrained backbone) to:

1. Encode repository context into an embedding
2. Produce procedure embeddings from learned representations
3. Receive survival loss from the procedural memory system

This means PSM-006 is a joint research/engineering milestone, not a pure benchmark study.

---

## Timeline Estimate

| Milestone | Estimate |
|---|---|
| Semi-real benchmark families defined | 2 weeks |
| Real repository adapter written | 2 weeks |
| Baseline training runs (30M model) | 1-2 weeks on Kaggle |
| Evaluation against all 7 baselines | 1 week |
| Gate validation (5 seeds, 20 tasks) | 1 week |
| Report | 1 week |

**Total:** ~8-10 weeks from research start.

---

## Relationship to PSM-001–005

PSM-006 *does not replace* PSM-001–005. The synthetic benchmark remains the clean, interpretable validation of the mechanism. PSM-006 adds *external validity* — it tests whether the mechanism survives contact with real complexity.

```
PSM-001–005                    PSM-006
─────────────────────────────────────────────
Synthetic tasks            →   Real repositories
Step strings               →   Real shell commands
Jaccard success metric     →   Test suite pass/fail
Deterministic embeddings   →   Trained encoder
Controlled overlap         →   Natural task similarity
```

---

## What a Positive PSM-006 Result Would Mean

If TAC's procedural memory achieves ≥ 0.10 improvement over the reset agent on real repository repair, it would be the first published result showing that a procedural memory system with lifecycle operations (strengthen, merge, split, retire, discover) outperforms same-size baselines on real coding tasks.

That would be the foundation for a peer-reviewed publication.
