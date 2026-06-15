# TAC Procedural Memory — Investor Summary

**Date:** 2026-06-15
**Stage:** Research validation (PSM-001 through PSM-005)

---

## The One-Sentence Claim

TAC has moved from *remembering structures* to *remembering reusable procedures* — the beginnings of a system that accumulates operational knowledge the way an experienced engineer does.

---

## What Is a Procedure?

An experienced engineer solving a `ModuleNotFoundError` does not re-read the Python import documentation. They remember:

1. Check whether the package is installed.
2. Check the environment it is installed in.
3. Verify the version.
4. Patch the configuration.
5. Run the tests.

This is a **procedure** — an ordered sequence of steps that has worked before.

TAC now stores, retrieves, adapts, and discovers these procedures automatically.

---

## What TAC Can Now Do

### It remembers how to solve classes of problems

Given a `DependencyConflict` it has seen before, TAC retrieves the procedure that worked, applies it, and succeeds — without re-learning from scratch.

Measured result: **retrieval accuracy 1.0, reuse gain 1.0 over a reset agent** (PSM-001).

### It reuses successful procedures across problem types

A procedure learned on `ImportErrors` transfers to `DependencyConflicts` and `VersionMismatch` with measurable gain over fresh learning.

Measured result: **transfer gain 1.0, A→B→C chain retention 0.89** (PSM-002).

### It improves procedures after failure

When a procedure fails, TAC forks a recovery variant. Over repeated use, procedures strengthen. Redundant procedures merge. Procedures that serve two different sub-tasks split into specialised children. Weak procedures retire automatically.

Measured result: **merge quality gain 0.22, retirement accuracy 1.0, strengthening monotone rate 1.0** (PSM-003).

### It retires weak procedures

Not all procedures are equally useful. TAC computes a fitness score (reuse frequency, transfer success, robustness, recovery ability, verification score) and applies selection pressure. High-fitness procedures survive. Low-fitness procedures decay and are removed.

Measured result: **survival gap 1.0 — all high-fitness procedures alive after 30 steps, all low-fitness procedures dead** (PSM-004).

### It can discover new procedures from successful traces

Given 12 successful repair traces, TAC mines the common step patterns and extracts a canonical procedure — without any label, template, or human instruction. The discovered procedure beats the no-discovery baseline across all 5 seeds.

Measured result: **discovery accuracy 0.49 (49% of oracle quality), beats no-discovery at 1.0 rate** (PSM-005).

---

## Unified Result

| Study | Gates Passed | Claim |
|---|---|---|
| PSM-001: Memory | 7/7 | Procedures are stored and retrieved |
| PSM-002: Transfer | 5/5 | Procedures transfer across families |
| PSM-003: Lifecycle | 5/5 | Procedures are living assets |
| PSM-004: Survival | 5/5 | Useful procedures persist |
| PSM-005: Discovery | 5/5 | Procedures can be invented |
| **Total** | **27/27** | **Validated** |

5 seeds each. Benchmarks run deterministically and are fully reproducible.

---

## What TAC Is Not

- **Not a GPT replacement.** TAC is a research model at 30M–150M parameters.
- **Not a production coding agent.** The current benchmark is synthetic.
- **Not AGI.** These are narrow, controlled experiments on procedural repair tasks.
- **Not trained at scale.** No large GPU training run has been completed.

---

## What TAC Is

A research foundation for **coding agents, repair agents, and long-horizon AI systems** that need to accumulate and reuse operational knowledge over time.

Most AI systems today can memorize, retrieve, and imitate.
Very few can demonstrate all five stages of procedural memory:
memory → transfer → evolution → survival → discovery.

TAC has demonstrated all five in controlled benchmarks.

---

## Near-Term Roadmap

**PSM-006 (next):** Move from synthetic repair tasks to real or semi-real repositories. Given a real failing test + bug report, TAC retrieves or discovers a procedure, applies it, verifies with the test suite, and updates memory.

**PSM-007 (planned):** Full neural integration — train the TACSM model with PSM-grounded loss signals.

**Target benchmark:** SWE-bench-lite or a custom repository repair benchmark.

---

## Reproducibility

All experiments are fully reproducible:

```bash
cd tacm && python3 scripts/run_psm_progression.py --seeds 0 1 2 3 4
```

Expected output: `PROGRESSION VALIDATED — 5/5 studies, 27/27 gates`.
