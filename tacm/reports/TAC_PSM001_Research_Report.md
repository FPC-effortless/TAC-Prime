# TAC-PSM-001: Procedural Memory Build / Retrieve / Update
**Research Report**  |  Generated: 2026-06-15 00:16 UTC

---

## 1. Experiment Overview

**Hypothesis:** TAC can learn, store, retrieve, update, and reuse procedures,
producing measurable gains over reset, retrieval-disabled, and incorrect-procedure baselines.

**Null hypothesis:** Procedural memory provides no measurable advantage over
reset systems, retrieval-disabled systems, random procedure retrieval, or structure-only memory.

**Central claim:**
> "TAC can remember, reuse, adapt, and improve procedures, producing measurable
> gains over memory-disabled and reset baselines."

---

## 2. Experimental Design

| Parameter | Value |
|---|---|
| Seeds | 5 |
| Task families | 4 (ImportErrors, DependencyConflicts, VersionMismatch, PathResolution) |
| Baselines | 5 (Reset, Disabled, Random, Structure-only, Oracle) |
| Ablations | 5 (failure modes, recovery, update, transfer, survival) |
| Embedding dim | 64 (benchmark), 512 (production model) |
| Evaluation sequence | A1 → A2 → B1 → C1 → D1 |

### Evaluation Sequence

| Step | Description | Measurement |
|---|---|---|
| A1 | Solve initial import error; store procedure | Baseline success rate |
| A2 | Retrieve + reuse procedure on similar import error | Retrieval accuracy, reuse gain |
| B1 | Transfer A→B: dependency conflict | Transfer accuracy, transfer gain |
| C1 | Transfer A→C: version mismatch | Broader transfer quality |
| D1 | Force failure → update → retry | Retry improvement, failure recovery |

---

## 3. Results

### 3.1 Primary Metrics

| Metric | Result |
|---|---|
| A1 — Initial success | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| A2 — Procedure reuse success | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| B1 — Transfer (A→B) | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| C1 — Transfer (A→C) | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| D1 — Retry after update | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| Retrieval accuracy | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| Family match rate | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| Procedure reuse gain | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| Retry improvement | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| Transfer gain | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| Reset deficit | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| Memory survival (final) | 0.9850 ± 0.0000  (95% CI ±0.0000) |

### 3.2 Baseline Comparison

| Condition | Reuse Success Rate |
|---|---|
| TAC-PSM (correct retrieval) | 1.0000 ± 0.0000  (95% CI ±0.0000) |
| Memory disabled | 0.0000 ± 0.0000  (95% CI ±0.0000) |
| Random retrieval | 0.2000 ± 0.4472  (95% CI ±0.3920) |
| Oracle retrieval | 1.0000 ± 0.0000  (95% CI ±0.0000) |

### 3.3 Success Gates

| Gate | Threshold | Result |
|---|---|---|
| retrieval_accuracy_ge_0.70 | ≥ 0.70 | ✓ PASS |
| reuse_gain_ge_0.10 | ≥ 0.10 | ✓ PASS |
| update_improves_retry | > 0.0 | ✓ PASS |
| reset_deficit_ge_0.20 | ≥ 0.20 | ✓ PASS |
| random_worse_than_correct | < 0.0 (random worse) | ✓ PASS |
| transfer_gain_gt_0 | > 0.0 | ✓ PASS |
| survival_stable_across_seeds | CV < 0.30 | ✓ PASS |

**Overall gate verdict: ALL PASS ✓**

---

## 4. Ablation Study

Ablation study measures performance degradation when each component is removed.
Positive degradation = component contributes positively to performance.

| Ablation | Reuse Δ | Transfer Δ | Retry Δ | Retrieval Δ |
|---|---|---|---|---|
| A: No failure modes | 0.0 | 0.0 | 0.0 | 0.0 |
| B: No recovery strategies | 0.0 | 0.0 | 0.0 | 0.0 |
| C: No update mechanism | 0.0 | 0.0 | 0.0 | 0.0 |
| D: No transfer metadata | 0.0 | 0.0 | 0.0 | 0.0 |
| E: No survival scoring | 0.0 | 0.0 | 0.0 | 0.0 |

---

## 5. Replication

| Parameter | Value |
|---|---|
| Seeds | 5 |
| Verdict | REPLICATED |
| Gates passed | 7/7 |

---

## 6. Discussion

### What this experiment tests

TAC-PSM-001 validates whether a procedural memory system — storing ordered action
sequences rather than raw text or embeddings — provides measurable improvements
over memory-disabled baselines.

### What counts as success

- **Retrieval accuracy ≥ 0.70:** The system must retrieve the correct procedure
  for at least 70% of tasks — otherwise memory is not useful.

- **Reuse gain ≥ 0.10:** Using retrieved procedures must improve success rate
  by at least 10 percentage points over not using memory.

- **Retry improvement > 0:** After a failure and update, the retry must succeed
  more often than the original attempt.

- **Transfer gain > 0:** Procedures from family A must help on families B/C.

### Limitations

- The benchmark uses synthetic task families. Transfer to real codebases
  requires integration with actual code execution (stage 3, agent loop).

- The embedding dimension (64) used in benchmarks is smaller than the
  production model (512). Performance may differ at production scale.

- Procedure quality is measured by step-set Jaccard similarity to canonical
  procedures, not by actual code execution success.

---

## 7. Next Experiments

If PSM-001 validates successfully:

| Experiment | Hypothesis |
|---|---|
| TAC-PSM-002 | Procedural transfer improves with chain length A→B→C |
| TAC-PSM-003 | Procedure lifecycle transitions improve long-run performance |
| TAC-PSM-004 | Survival fields correctly identify high-value procedures |
| TAC-PSM-005 | Autonomous procedure discovery from unlabelled traces |

---

*Report generated by `tacm/reports/psm001_report.py`*