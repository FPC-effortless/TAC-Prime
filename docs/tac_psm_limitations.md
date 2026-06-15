# TAC-PSM Limitations

**Version:** PSM-001 through PSM-005
**Date:** 2026-06-15

This document is a direct companion to the progression report. Every limitation listed here is real and should be stated in any presentation or publication of these results.

---

## 1. Synthetic Benchmark

All five studies operate on a **synthetic procedural benchmark**, not on real code repositories.

- Tasks are constructed from four hand-designed families (ImportErrors, DependencyConflicts, VersionMismatch, PathResolution).
- Steps are strings like `"Inspect dependency"`, not real shell commands or AST operations.
- Success is measured by Jaccard similarity between step strings and canonical step strings, not by whether a test suite passes.

**What this means:** The benchmark validates the *mechanism* of procedural memory, not the ability to repair real code.

---

## 2. Hand-Designed Task Families

The four task families (A–D) were designed by the researchers, not drawn from a real corpus.

- Family step overlap is controlled: A and B share structural vocabulary on purpose.
- Transfer gains are partly an artefact of this controlled overlap.
- In real repositories, task families are not cleanly separable.

---

## 3. Controlled Procedure Representation

Procedures are lists of action strings (`ProcedureStep`). They are not:

- Abstract syntax trees
- Shell command sequences
- API call graphs
- Learned neural representations (yet)

The current representation is symbolic and interpretable, which is useful for validation but is not what a production agent would use.

---

## 4. No Real Repository Repair

None of the five studies involve:

- Cloning a real repository
- Running a test suite
- Executing shell commands
- Observing a real failure trace
- Verifying a real patch

The "repair" in PSM-001 through PSM-005 is entirely simulated.

---

## 5. No Learned Neural Policy

The PSM-001–005 experiments do not train the TACSM neural network. The procedural memory system runs as a standalone module.

- Embeddings are deterministic (derived from task signatures), not produced by a trained encoder.
- There is no gradient signal flowing from the memory into the backbone.
- The Neural Survival Field (NSF) module exists in `tacm/survival.py` but is not yet connected to the PSM experiments.

---

## 6. No External Benchmark Comparison

PSM-001–005 do not compare against:

- SWE-bench
- HumanEval
- CodeContests
- Any published agent baseline

The baselines used (reset agent, random retrieval, wrong procedure, fresh learning) are internal and purpose-built.

---

## 7. No Large-Scale Model Training

The TACSM model at 30M–150M parameters has not been trained on PSM-grounded data. The model architecture is defined and forward-pass tested, but no training run using the PSM benchmark has been completed.

---

## Summary Table

| Limitation | PSM-001–005 Status |
|---|---|
| Synthetic tasks | Yes — all synthetic |
| Hand-designed families | Yes — 4 families |
| Controlled procedure strings | Yes — symbolic |
| Real repository repair | No |
| Learned neural policy | No |
| External benchmark comparison | No |
| Large-scale training | No |

---

## What the Results *Do* Establish

Despite the above limitations, PSM-001–005 establish:

1. The procedural memory *mechanism* works correctly on controlled tasks.
2. Procedures transfer across task families with measurable gain over baselines.
3. Lifecycle operations (merge, split, specialize, retire) behave correctly.
4. A fitness-based survival field correctly selects high-fitness procedures.
5. Unsupervised pattern mining can extract useful procedures from raw traces.

These are **necessary preconditions** for a real coding agent, not sufficient ones.
The next stage (PSM-006) moves toward real or semi-real repository grounding.
