---
name: PSM-001 design decisions
description: Key non-obvious decisions in the TAC-PSM-001 Procedural Memory implementation.
---

## Fork procedure: recovery steps must be used directly

**Rule:** In `_fork_procedure` (update.py), when `signal.recovery_success=True`, the forked procedure's steps must be set to `list(signal.recovery_steps)` — NOT prefixed with "[recovery] ".

**Why:** The `evaluate_procedure_on_task` function computes Jaccard similarity and word-overlap between retrieved steps and canonical steps. The "[recovery] " prefix makes strings different from canonical strings, dropping Jaccard to 0 and leaving only a small word-overlap bonus (max 0.2). With task difficulty ≥ 0.4, this causes the retry to fail even with correct recovery steps. The gate `update_improves_retry` fails unless recovery steps are used verbatim.

**How to apply:** Three cases in `_fork_procedure`:
1. `recovery_steps` + `recovery_success=True` → use recovery_steps directly as new steps
2. `recovery_steps` + `recovery_success=False/None` → merge original + recovery steps
3. no recovery_steps → keep original, append diagnostic step

## Test expectations

`test_forked_procedure_has_recovery_step` checks that recovery text appears in step actions (not the "[recovery]" prefix). Keep this aligned with fork logic.

## Success gates (all 7 pass on 5 seeds)

| Gate | Threshold | Result |
|---|---|---|
| retrieval_accuracy | ≥ 0.70 | 1.00 |
| reuse_gain | ≥ 0.10 | 1.00 |
| update_improves_retry | > 0 | 1.00 |
| reset_deficit | ≥ 0.20 | 1.00 |
| random_worse_than_correct | < 0 | pass |
| transfer_gain | > 0 | 1.00 |
| survival_cv | < 0.30 | 0.00 |

## File layout

```
tacm/tacm/memory_faiss.py         FAISS-backed StructureMemory (drop-in)
tacm/tacm/psm001/__init__.py      all exports
tacm/tacm/psm001/records.py       StructureMemoryRecordV2, ProcedureTrace, ProcedureStep
tacm/tacm/psm001/store.py         ProceduralMemoryStore (FAISS + numpy fallback)
tacm/tacm/psm001/retrieval.py     retrieve_procedure(), 5 RetrievalModes
tacm/tacm/psm001/update.py        update_procedure_after_verification(), _fork_procedure
tacm/tacm/psm001/benchmark_families.py  Families A–D, TaskInstance, evaluate_procedure_on_task
tacm/scripts/benchmark_tac_psm001.py   5-seed benchmark, 5 baselines, all gates
tacm/scripts/test_tac_psm001.py        50-test suite (all pass)
tacm/scripts/replicate_psm001.py       multi-seed replication runner
tacm/scripts/run_ablations.py          5-ablation runner
tacm/reports/psm001_report.py          generates 3 Markdown reports
```
