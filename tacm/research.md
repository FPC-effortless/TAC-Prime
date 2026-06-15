# TAC Research Log

Dated entries, most recent first.

---

## 2026-06-15 — TAC-PSM-001 through TAC-PSM-005 validated

**Result:** 27/27 gates passed across 5 studies, 5 seeds each.

TAC-PSM is a five-stage controlled benchmark progression validating the
procedural memory mechanism in TAC-SM. All five stages pass in synthetic
benchmarks.

### Summary

| Study | Title | Gates | Seeds |
|---|---|---|---|
| PSM-001 | Procedure Memory | 7/7 | 5 |
| PSM-002 | Procedure Transfer | 5/5 | 5 |
| PSM-003 | Procedure Lifecycle | 5/5 | 5 |
| PSM-004 | Procedure Survival | 5/5 | 5 |
| PSM-005 | Procedure Discovery | 5/5 | 5 |

### Key results

- **PSM-001:** Retrieval accuracy 1.0, reuse gain 1.0 over reset agent.
  Procedures are stored, retrieved, updated, and forked correctly.

- **PSM-002:** Transfer gain 1.0, A→B→C chain retention 0.89.
  Procedures learned on ImportErrors transfer to DependencyConflicts and VersionMismatch.

- **PSM-003:** Merge quality gain 0.22, strengthening monotone rate 1.0,
  retirement accuracy 1.0. Procedures evolve through their lifecycle correctly.

- **PSM-004:** Survival gap 1.0 (hi-fit procedures all alive, lo-fit all dead after 30 steps).
  Mean robustness 0.43. High-fitness procedures survive selection pressure.

- **PSM-005:** Discovery accuracy 0.49, beats no-discovery at 1.0 rate.
  Unsupervised pattern mining extracts useful procedures from raw traces.

### Framing

This is a **controlled synthetic benchmark validation**. It validates the
procedural memory mechanism, not real-world coding intelligence. No real
repository repair has been tested yet.

### Reproduce

```bash
cd tacm && python3 scripts/run_psm_progression.py --seeds 0 1 2 3 4
```

### Next

PSM-006: Repository-Grounded Procedural Memory — move from synthetic repair
families to real or semi-real repositories.

---

## 2026-06-14 — PSM-001 Procedural Memory validated

7/7 gates passed on 5 seeds. FAISS-backed ProceduralMemoryStore implemented.
50/50 unit tests pass. Replication, ablation, and report scripts complete.

---

## 2026-06-01 — TAC-SM architecture finalised

14-component architecture defined and forward-pass tested at 30M parameters.
Components: Backbone, ConceptVolume, Router, MoE, StructureMemory,
ProceduralMemory, NeuralSurvivalField, Verifier, MultiTokenPrediction,
Agent loop, Losses, Evaluation.
