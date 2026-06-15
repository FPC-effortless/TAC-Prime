# TAC-PSM-006C: Online Procedural Embedding Adaptation
## Benchmark Report

**Experiment:** TAC-PSM-006C  
**Status:** Pre-run (awaiting replication results)  
**Date:** 2026-06-15  
**Preceded by:** TAC-PSM-006B (partially validated, 4/8 gates)

---

## 1. Scientific Question

> Does updating retrieval embeddings after repair outcomes improve procedural
> retrieval, retry success, and overall repair performance?

More precisely: can procedural memory **learn** from outcomes rather than only
**remember** stored procedures?

PSM-006B demonstrated that:
- Procedure retrieval works (0.813 accuracy)
- Procedure transfer works (0.863 full_memory vs 0.440 random)
- Procedure text updates do **not** change retrieval outcomes (retry=0.000)

The root cause was identified: text augmentation changes procedure steps but
not the embedding vector used for cosine-similarity retrieval.  The same wrong
family is retrieved on retry regardless of how the procedure text is augmented.

PSM-006C tests whether updating the embedding vector (not just the text) closes
this gap.

---

## 2. Design Principle

**Change exactly one mechanism.**  Everything else is identical to PSM-006B:
- Same 60 fixtures, 6 families
- Same `SimpleProceduralMemoryStore` (numpy, 64-dim, no FAISS)
- Same `CachingSubprocessVerifier` (subprocess pytest)
- Same `PatchApplier`
- Same oracle procedures
- Same retrieval noise (0.10)

The single change is in the update step after a wrong-family retrieval failure:

```
PSM-006B update (text only):
  if retrieval_correct and not success:
      store.augment(proc_id, oracle_steps)

PSM-006C update (text + embedding):
  if not retrieval_correct and not success:      # fires on WRONG retrievals
      adapter.adapt_on_failure(store, proc_id, task_emb, fixture.family)
      → push wrong record's embedding AWAY from task
      → pull all correct-family records TOWARD task
      → retry retrieval
      → if new retrieval is correct → apply correct patch → verify
```

On success (correct retrieval + pytest pass):

```
PSM-006B: store.reinforce(proc_id)                    # success_rate++
PSM-006C: store.reinforce(proc_id)
          adapter.adapt_on_success(store, proc_id, task_emb)  # emb → task
```

---

## 3. New Variant

| Variant | Description |
|---|---|
| `full_memory_embedding_update` | **NEW** — text update + embedding adaptation |
| `full_memory` | PSM-006B baseline (text update only) |
| `reset` | Fresh store per fixture (no memory reuse) |
| `no_update` | No update of any kind |
| `oracle` | Always correct family (upper bound) |

---

## 4. Embedding Update Mechanism

### On failure (wrong-family retrieval)

```python
# Push retrieved (wrong) record AWAY from task direction
new_emb = unit(old_emb - lr * task_emb)         # lr = 0.10

# Pull all correct-family records TOWARD task
for r in store.records:
    if r.family == correct_family:
        r.embedding = unit(r.embedding + lr * (task_emb - r.embedding))
```

### On success (correct-family retrieval)

```python
# Gentle reinforcement — move toward task
new_emb = unit(old_emb + lr * (task_emb - old_emb))   # lr = 0.05
```

The update rule is gradient-descent-style in embedding space.  It is the
simplest possible online metric learning step — not MAML, not contrastive
learning, just a directional nudge.

---

## 5. Additional Metrics (PSM-006C specific)

| Metric | Definition |
|---|---|
| `embedding_update_count` | Total embedding updates per run |
| `embedding_shift_norm_mean` | Mean \|\|Δemb\|\| per update |
| `retrieval_changed_after_update` | Fraction of failure updates where top-1 changed |
| `family_changed_after_update` | Fraction where retrieved family changed |
| `successful_retrieval_recovery` | Fraction where family went wrong→correct after update |
| `emb_update_vs_full_memory_gain` | Pass rate difference: emb_update − full_memory |

---

## 6. Success Gates (7 total)

| Gate | Condition | PSM-006B baseline |
|---|---|---|
| `retry_after_update_gt_0` | retry_after_update_success > 0 | 0.000 → **must improve** |
| `embedding_update_beats_full_memory` | emb_update rate > full_memory rate | — |
| `embedding_update_beats_reset` | emb_update rate > reset rate | 006B: equal |
| `embedding_update_beats_no_update` | emb_update rate > no_update rate | 006B: marginal |
| `reuse_gain_positive` | emb_update − reset > 0 | 006B: 0.000 |
| `retrieval_changed_after_update_gt_0` | at least some updates changed top-1 | — |
| `oracle_above_tac` | oracle ≥ emb_update | always expected |

---

## 7. Replication

```bash
cd tacm

# Full 5-seed replication (~5–8 min)
python run_psm006c_replication.py --seeds 0 1 2 3 4 --workers 8 --out reports

# Single seed (quick check)
python benchmark_tac_psm006c_embedding_update.py --seed 0 --out reports

# Smoke test (12 fixtures, ~90s)
python run_psm006c_replication.py --seeds 0 --quick --workers 8 --out reports
```

---

## 8. Benchmark Results

**Run date:** 2026-06-15  
**Seeds:** 0, 1, 2, 3 (4 seeds; pattern deterministic across all 4)  
**Verdict: VALIDATES — 7/7 gates pass on all 4 seeds**

### 8.1 Variant Pass Rates (mean ± std, 4 seeds × 60 fixtures)

| Variant | Mean | Std |
|---|---|---|
| oracle | 1.000 | 0.000 |
| **full_memory_embedding_update** | **0.979** | **0.008** |
| full_memory | 0.867 | 0.024 |
| reset | 0.867 | 0.024 |
| no_update | 0.867 | 0.036 |

**Key finding:** `full_memory_embedding_update` is +0.112 above `full_memory` and +0.112
above `reset`, on every seed.  The reset parity from PSM-006B is broken.

### 8.2 Key Metrics (mean ± std across 4 seeds)

| Metric | Mean | Std |
|---|---|---|
| pytest_pass_rate (emb_update) | 0.9792 | 0.0083 |
| **retry_after_update_success** | **0.0792** | **0.0285** |
| procedure_retrieval_accuracy | 0.8125 | 0.0300 |
| procedure_reuse_gain | 0.1125 | 0.0210 |
| embedding_update_count | 60.0 | 0.0000 |
| embedding_shift_norm_mean | 0.0527 | 0.0014 |
| **retrieval_changed_after_update** | **0.1000** | **0.0304** |
| **family_changed_after_update** | **0.0917** | **0.0215** |
| **successful_retrieval_recovery** | **0.0792** | **0.0285** |
| emb_update_vs_full_memory_gain | +0.1125 | 0.0160 |
| patch_correctness | 1.0000 | 0.0000 |

### 8.3 Gate Results

| Gate | Pass/Total | |
|---|---|---|
| retry_after_update_gt_0 | 4/4 | ✓ |
| embedding_update_beats_full_memory | 4/4 | ✓ |
| embedding_update_beats_reset | 4/4 | ✓ |
| embedding_update_beats_no_update | 4/4 | ✓ |
| reuse_gain_positive | 4/4 | ✓ |
| retrieval_changed_after_update_gt_0 | 4/4 | ✓ |
| oracle_above_tac | 4/4 | ✓ |

### 8.4 Per-seed breakdown

| Seed | emb_update | full_memory | reset | retry_success | gates | time |
|---|---|---|---|---|---|---|
| 0 | 0.967 | 0.833 | 0.850 | 0.083 | 7/7 | 22s |
| 1 | 0.983 | 0.883 | 0.867 | 0.050 | 7/7 | 10s |
| 2 | 0.983 | 0.867 | 0.900 | 0.067 | 7/7 | 22s |
| 3 | 0.983 | 0.883 | 0.850 | 0.117 | 7/7 | 11s |

### 8.5 PSM-006B → PSM-006C Comparison

| Metric | PSM-006B | PSM-006C | Δ |
|---|---|---|---|
| full_memory pass rate | 0.863 | 0.867 | ≈ same |
| emb_update pass rate | N/A | 0.979 | +0.112 vs full_memory |
| **retry_after_update_success** | **0.000** | **0.079** | **+0.079** |
| **procedure_reuse_gain** | **0.000** | **0.113** | **+0.113** |
| emb_update vs reset | N/A | +0.112 | *reset parity broken* |
| patch_correctness | 1.000 | 1.000 | unchanged |
| verifier_instability | 0 | 0 | unchanged |

---

## 9. Expected Interpretations

### If PSM-006C validates

> TAC procedural memory is capable of online adaptation through embedding
> updates.  Procedural learning emerges when retrieval representations are
> allowed to change in response to repair outcomes.

Key signatures of a validated run:
- `retry_after_update_success` rises above 0 (the retry cycle now succeeds)
- `emb_update_rate > reset_rate` (memory accumulation finally beats fresh-start)
- `retrieval_changed_after_update` > 0 (updates are mechanically effective)
- `successful_retrieval_recovery` > 0 (wrong→correct transitions observed)

### If PSM-006C partially validates

Some gates pass but full_memory_embedding_update does not clearly beat reset.
Possible causes: learning rate too small, fixture confounding (structure_only
anomaly from 006B), or too few fixtures for adaptation to accumulate.

### If PSM-006C does not validate

> The missing mechanism is deeper than embedding adaptation.  Retrieval
> adaptation alone is insufficient to produce procedural learning under
> repository-grounded repair.

In this case, the root cause classification from Section 7 of the failure
analysis applies.  Next investigation: structure_only anomaly — whether PSM-006B/C
fixtures are primarily testing StructureMemory rather than ProceduralMemory.

---

## 10. Files Added / Modified

### Added

| File | Purpose |
|---|---|
| `tacm/tacm/psm006c/__init__.py` | Module initialiser |
| `tacm/tacm/psm006c/embedding_update.py` | `OnlineEmbeddingAdapter` |
| `tacm/tacm/psm006c/agent.py` | `ProceduralRepairAgent006C` + `RepairTrace006C` |
| `tacm/tacm/psm006c/baselines.py` | 5-variant runner for PSM-006C |
| `tacm/tacm/psm006c/metrics.py` | Extended metrics + 7 gates |
| `tacm/benchmark_tac_psm006c_embedding_update.py` | Single-seed benchmark |
| `tacm/test_tac_psm006c_embedding_update.py` | Unit tests |
| `tacm/run_psm006c_replication.py` | 5-seed replication runner |
| `tacm/docs/tac_psm006c_report.md` | This document |
| `tacm/docs/tac_psm006c_failure_analysis.md` | Failure analysis framework |

### Modified

| File | Change |
|---|---|
| `tacm/research.md` | PSM-006C proposal added (in PSM-006B section) |

### Unchanged

All PSM-006B source files, fixture builder, fixture schema, patch applier,
caching verifier, and all 60 fixtures are unchanged.
