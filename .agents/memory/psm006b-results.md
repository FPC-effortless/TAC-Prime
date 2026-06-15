---
name: PSM-006B replication results
description: Actual benchmark results from the 5-seed 60-fixture PSM-006B run; structural diagnosis of 4 failing gates.
---

## PSM-006B Replication — 2026-06-15

**Verdict**: PARTIALLY_VALIDATES — 4/8 gates pass on all 5 seeds.

### Gate summary
- PASS (5/5): retrieval_accuracy_ge_0.55, random_procedure_no_benefit, oracle_above_tac, cross_fixture_transfer_positive
- FAIL (0/5): tac_beats_reset_by_0.10, update_improves_retry
- FAIL (3/5): no_update_underperforms_tac
- FAIL (2/5): reuse_gain_positive

### Key numbers
- full_memory: 0.863 ± 0.022
- oracle: 1.000 ± 0.000 (correct upper bound)
- reset: 0.863 ± 0.022 (ties full_memory — structural issue)
- retrieval accuracy: 0.813 ± 0.032
- retry_after_update_success: 0.000 across all seeds
- procedure_reuse_gain: 0.000 ± 0.026
- patch_correctness: 1.000 (no patch failures)
- verifier_instability: 0 (all fixtures deterministic)

### Structural diagnosis of failing gates

**Why reset ≈ full_memory**: The reset baseline re-seeds oracle procedures before EACH
fixture. It is "no memory reuse" but also "fresh oracle every time". This masks the
memory advantage. Fix for PSM-007: make reset start from an EMPTY store.

**Why update_improves_retry = 0**: The update step augments procedure STEPS (text),
not the embedding VECTOR. Wrong-family retrieval is driven by cosine similarity on
embeddings; step augmentation doesn't change embedding proximity. The same wrong
family is retrieved on retry. Fix: online metric learning to update embeddings.

**Why structure_only = 0.927 (unexpectedly high)**: Stub patches leave many test
checks unaffected. Some fixtures are under-constrained.

**Why retrieval_disabled = 0.550 / random = 0.440 (unexpectedly high)**: The
pass-regardless floor (fixtures that pass with any patch) is ~0.44–0.55.

### PSM-006C follow-up: VALIDATES (7/7 gates, 4 seeds)
- full_memory_embedding_update = 0.979 ± 0.008 (+0.112 over full_memory and reset)
- retry_after_update_success went from 0.000 → 0.079 ± 0.029
- reuse_gain went from 0.000 → 0.113 ± 0.021
- reset parity broken: emb_update beats reset by +0.112 on every seed
- The missing mechanism was confirmed: push wrong embedding away, pull correct toward task
- See psm006c-results.md for full details

### Runner
- `tacm/run_psm006b_fast.py` with `CachingSubprocessVerifier` (subprocess-based, thread-safe cache)
- pytest.main() in forked processes has correctness issues (oracle < full_memory impossible)
- Use subprocess.run([python, -m, pytest, ...]) for correct results; ThreadPoolExecutor for parallelism
- Prewarm: 240 tasks / 8 workers ≈ 97s wall time; seeds 1–4 are near-instant (cache hits)
- Total benchmark time: 233.7s

### Report files
- tacm/reports/psm006b_results.json (full data)
- tacm/reports/psm006b_summary.txt
- tacm/reports/psm006b_per_family_rates.txt
- tacm/reports/psm006b_confusion_matrix.txt
- tacm/reports/psm006b_failure_analysis.txt
