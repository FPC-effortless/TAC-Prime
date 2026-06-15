---
name: PSM-006C results
description: TAC-PSM-006C ablation results — online embedding adaptation — 4-seed benchmark outcome and key calibration facts
---

# PSM-006C: Online Procedural Embedding Adaptation

## Result: VALIDATES

7/7 gates pass on all 4 seeds (0, 1, 2, 3).
Seed 4 not collected (CI timeout 82s prewarm + seed time > 120s limit).

## Key numbers (4 seeds × 60 fixtures)

| Variant | Mean | Std |
|---|---|---|
| full_memory_embedding_update | 0.979 | 0.008 |
| full_memory | 0.867 | 0.024 |
| reset | 0.867 | 0.024 |
| no_update | 0.867 | 0.036 |
| oracle | 1.000 | 0.000 |

| Metric | Mean | Std |
|---|---|---|
| retry_after_update_success | 0.079 | 0.029 |
| retrieval_changed_after_update | 0.100 | 0.030 |
| family_changed_after_update | 0.092 | 0.022 |
| successful_retrieval_recovery | 0.079 | 0.029 |
| embedding_update_count | 60.0 | 0.000 |
| embedding_shift_norm_mean | 0.053 | 0.001 |
| procedure_reuse_gain | 0.113 | 0.021 |
| emb_update_vs_full_memory_gain | +0.113 | 0.016 |

## PSM-006B → 006C comparison (the key change)

- retry_after_update_success: 0.000 → **0.079** (+0.079)
- procedure_reuse_gain: 0.000 → **0.113** (+0.113)
- full_memory pass rate: 0.863 → 0.867 (≈ same, correct)
- emb_update beats reset by +0.112 on every seed

## Why embedding_update_count = 60

The adapter fires on BOTH failure (push wrong away, pull correct toward task) AND success
(correct-family reinforcement). Since every fixture results in an update event, count = fixtures.
This is correct behavior per design.

## Architecture: OnlineEmbeddingAdapter

- `tacm/tacm/psm006c/embedding_update.py`
- lr=0.10 (default)
- On wrong retrieval: push wrong embedding away from task embedding; pull correct-family centroid toward task
- On success: reinforce correct-family centroid toward task
- Centroid embeddings are maintained per procedure family (64-dim, same as PSM-006B)

## Timing facts (CachingSubprocessVerifier, 8 workers)

- Prewarm (180 tasks = 60 fixtures × 3 states): ~82s
- Per-seed after prewarm: 10–22s
- Single-seed with plain PytestVerifier: too slow (180+ sequential subprocess calls)
- Total for seeds 0+1: ~115s; seeds 2+3: ~125s
- Seed 4 would need a separate prewarm which pushes total over 120s limit

## Report locations

- `tacm/reports/psm006c_results.json` — full aggregate JSON (all 4 seeds)
- `tacm/reports/psm006c_summary.txt` — human-readable summary
- `tacm/reports/psm006c_per_family_rates.txt` — per-family retrieval accuracy
- `tacm/reports/psm006c_confusion_matrix.txt` — confusion matrix
- `tacm/reports/psm006c_failure_analysis.txt` — failure classes
- `tacm/docs/tac_psm006c_report.md` — full scientific report (Sections 8, 9 populated)
- `tacm/docs/tac_psm006c_failure_analysis.md` — failure analysis (Section 8 populated)
- `tacm/research.md` — top entry added with full tables

## Reproduce

```bash
cd tacm
python run_psm006c_replication.py --seeds 0 1 2 3 4 --workers 8 --out reports
```
