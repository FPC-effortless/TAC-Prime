# PNDS Master Benchmark Specification

**Benchmark:** PNDS-MSB v0.1  
**Date:** 2026-09-24  
**Status:** Research control specification

## 1. Purpose

This is the central benchmark for testing the PNDS commercial thesis:

> As irrelevant accumulated history grows, useful computation should depend primarily on the relevant structural subset rather than on total history, while preserving task capability.

This benchmark is a falsification target. A system that merely retrieves relevant information but does not reduce executed computation does not satisfy the full claim.

## 2. Canonical comparison

For the same task family, data, hardware, and evaluation conditions, compare:

| Arm | Description | Canonical cost |
|---|---|---|
| FC | Full-context Transformer | C(N) |
| PS | Persistent state only / TAC | C(S) |
| PR | Persistent + relevance routing / TAC + CDL | C(R(S,Q)) |
| PNDS | Persistent + routed structural execution / full substrate | C(G(R(S,Q))) |

Where:
- N = accumulated history/context;
- S = persistent state representation;
- Q = current query/task;
- R(S,Q) = selected relevant subset;
- G(R(S,Q)) = executable structural graph induced by the selected subset.

A future verifier-enabled arm may be added as PNDS-V, but verification must not be conflated with the routing/execution scaling result.

## 3. Primary scaling hypothesis

Test whether, over increasing irrelevant history:

[
C_{PNDS}(N) \approx f(|R_t|)
]

while:

[
C_{FC}(N) \text{ grows materially with } N.
]

Do not assume constant-time behavior. Estimate empirical scaling from matched measurements.

The primary result is the joint capability/cost curve, not a single speedup number.

## 4. History-scaling protocol

Hold constant:
- task distribution;
- relevant information;
- model capacity where the comparison permits;
- hardware;
- batch size;
- precision;
- decoding/evaluation procedure;
- seed policy;
- measurement methodology.

Vary only irrelevant accumulated history, using predefined levels such as:
- 1k
- 4k
- 16k
- 32k
- 64k
- 100k tokens

The exact levels may be adjusted to hardware limits, but the final levels must be frozen before the confirmatory run.

For every level record:
- task accuracy / exact match;
- wall-clock latency;
- peak memory;
- tokens processed;
- compute proxy or FLOPs where measurable;
- active state/edges/candidates;
- routing candidate count;
- selected structural fraction;
- seed-level results.

## 5. Accuracy parity gate

Cost reduction alone is insufficient.

A scaling result can support the PNDS thesis only if capability remains within a predefined parity margin against the full-context baseline, or exceeds it.

The margin must be declared before the confirmatory run and must be task-specific.

Do not choose the parity margin after seeing results.

## 6. Relevant-subset test

The benchmark must contain tasks where:
- relevant information remains approximately fixed;
- irrelevant history grows substantially;
- the correct decision cannot be recovered from a forbidden gold metadata channel.

The router may use only permitted query, observation, state, and candidate properties.

## 7. Anti-leakage boundary

Routing inputs must exclude:
- target;
- answer;
- outcome;
- reward;
- true/gold structure identifiers;
- oracle masks;
- correct action;
- metadata uniquely encoding the correct action.

Evaluation may use hidden gold structure to score routing, but the model and router must not receive it.

## 8. Causal execution gate

Routing accuracy is not sufficient evidence of execution.

For selected structure z_i, perform a controlled intervention forcing an alternative z_j while holding all other relevant variables fixed.

Record:

[
\Delta_{causal} =
P(success\mid do(z_i)) -
P(success\mid do(z_j)).
]

If the intervention effect is approximately zero under the predefined practical threshold, the selected structure is not established as causally necessary for the claimed computation.

Intervention thresholds must be frozen before confirmatory runs.

## 9. State-integrity falsification

Persistence must be attacked, not assumed.

Required controls where state is claimed:
1. reset state;
2. shuffle state across episodes/users/tasks;
3. replace state with a wrong-context state;
4. corrupt selected state fields/edges;
5. probe whether claimed context can be recovered from state;
6. test whether the state is functionally used for prediction/decision;
7. evaluate on a held-out regime.

A persistence claim is weakened or falsified when the claimed advantage survives destructive state controls without a documented alternative explanation.

## 10. Verified-only commit gate

When verification and persistent learning are under test, compare:
- unconditional update: S_{t+1}=U(S_t,O_t)
- verified update: commit only after V_t accepts the proposed update.

Evaluate:
- held-out task performance;
- false commits;
- stale-state acceptance;
- poisoning/regression rate;
- recovery/repair rate.

If unconditional and verified updates are statistically indistinguishable under the predefined held-out integrity tests, the verifier has not demonstrated functional value.

## 11. Four commercial adapter environments

The same PNDS substrate must eventually be evaluated without changing the core research claims.

### E1 — Code Repair
State: repository structure.  
Graph: dependency/test graph.  
Action: patch.  
Outcome: tests/build/static checks.  
Metrics: affected-node localization, patch sparsity, regression rate, multi-turn state reuse.

### E2 — Enterprise Memory
State: facts/procedures/relations.  
Graph: knowledge/workflow graph.  
Action: retrieve/reason/execute.  
Outcome: task result.  
Metrics: retrieval under irrelevant context, long-session retention, compression ratio, provenance preservation.

### E3 — Action-Conditioned World Model
State: environment state.  
Graph: entity/causal/spatial structure.  
Action: intervention.  
Outcome: environment transition.  
Metrics: state prediction error, intervention sensitivity, long-horizon drift, action-conditioned planning validity.

### E4 — Verified Decision Graph
State: decision/evidence state.  
Graph: rules/dependencies.  
Action: decision.  
Outcome: downstream observation.  
Metrics: routing exactness, stale-state rejection, execution correctness, repair rate.

These environments are adapters around one substrate, not four independent architectures.

## 12. Research hygiene invariants

The following are mandatory for every PNDS experiment:
- no answer/target/outcome/gold-structure leakage into routing;
- causal intervention whenever execution causality is claimed;
- destructive state controls whenever persistence is claimed;
- verified-vs-unconditional update comparison whenever persistent learning is claimed;
- matched baselines and explicit denominators for efficiency claims;
- immutable experiment IDs and provenance;
- append-only result ledgers;
- negative results preserved;
- protocol changes versioned;
- no integration of a mechanism before its standalone primitive has passed its causal gate.

## 13. Promotion gates

Do not promote the architecture to a scaling or commercial claim until:
1. the benchmark interface passes leakage audit;
2. baseline arms are operational;
3. the relevant-subset task is validated;
4. routing has a held-out success signal;
5. causal execution intervention is positive;
6. state-integrity controls behave as predicted;
7. verified-only update shows functional value when verification is claimed;
8. confirmatory multi-seed runs reproduce the result.

Failure at any gate blocks the corresponding higher-level claim.

## 14. Interpretation rule

Separate:
- observed measurements;
- statistical/causal inference;
- architectural interpretation;
- commercial hypothesis.

A benchmark result may establish an empirical scaling relationship without establishing a universal complexity class or commercial advantage.

## 15. Current research consequence

The immediate research priority is not a large fused model.

First build and validate the benchmark substrate and its controls. Then complete Stage 1 baselines. Only after the benchmark has a nontrivial observable signal should Stage 2 outcome-trained routing proceed.

The Stage 2 router must learn from observed action outcomes rather than hidden structure IDs:

[
(Q_t,S_t) \rightarrow R_t \rightarrow A_t \rightarrow O_t.
]

## 16. Provenance

Every benchmark result must identify:

repository → branch → commit → benchmark version → experiment ID → configuration → seed → artifact → metric.

## 17. Controlled adaptation and registration extension

The master benchmark must distinguish three state layers when persistent learning is evaluated:

| Layer | Meaning |
|---|---|
| S_P | registered persistent state available to future episodes |
| S_E | experimental/provisional state available only within the validation protocol |
| Q | quarantined or rejected state unavailable to ordinary persistent execution |

The minimum persistent-learning comparison is:

1. immediate registration;
2. experimental state followed by repeated validation and registration.

A single successful episode must not be sufficient evidence for permanent registration.

For each candidate update record:

- utility delta on the discovery cohort;
- utility delta on held-out cohorts;
- validation/cohort count;
- persistent complexity delta;
- inference/resource cost delta;
- regression/poisoning rate;
- stale-state acceptance;
- quarantine/rejection rate;
- recovery/rollback success.

The benchmark should report both capability and state growth.

## 18. Adaptation regularization gate

When self-modification is evaluated, compare regularized and unregularized adaptation under the same discovery budget and evaluation protocol.

Regularization may constrain:

- number of edits;
- edit scope;
- repeated failed hypotheses;
- unexplored-component search;
- candidate cost increases;
- changes whose gain is within evaluation noise.

The primary question is:

> Does controlled accumulation improve held-out utility and/or integrity per unit of persistent complexity and inference cost?

Do not assume a particular mathematical penalty or coefficient before the experiment.

## 19. Intermediate verification gate

Where computation consists of a sequence of transitions z_1,...,z_n, compare final-only verification against path-aware verification:

V_final(y)

versus

V_path(z_1,...,z_n,y).

Measure invalid-transition detection, false acceptance, false rejection, repair/backtracking success, final task performance, calibration, and verification cost.

Final correctness alone does not establish transition validity.

## 20. State-to-computation coupling gate

A persistent state is not credited as an executable computational substrate merely because it correlates with outcomes.

Separate:

- predictive state use;
- functional state use;
- causal state dependence.

Use interventions that hold routing fixed while changing state, and interventions that hold state fixed while changing routing, where the environment permits.

The relevant measurement is a predefined intervention effect rather than observational correlation.

## 21. Typed authority routing gate

For multi-capability or multi-specialist environments, distinguish relevance routing from authority routing.

The router may select:

- retrieval;
- execution;
- prediction;
- inspection;
- verification;
- repair;
- external tools;
- specialist computation.

Compare sparse typed routing against an appropriate dense/unrestricted baseline and an authority-shuffled control.

Do not treat natural-language consensus as evidence of correct computation. Measure specialist selection, deference/override behavior, unnecessary activation, intervention effect, and coordination cost.

## 22. Promotion rule

No adaptation, registration, authority, or intermediate-verification mechanism may be credited as a full PNDS capability until its standalone controlled experiment passes the relevant causal/integrity gate.

The Context-Scaling Proof remains downstream of these primitive gates.

## Mathematical specification linkage — 2026-09-24

The authoritative mathematical treatment of the benchmark variables, routing/index cost, soft TopK training, leakage bound, SCM interventions, state-integrity controls, verification, registration, and adaptation is now:

`docs/pnds/PNDS_MATHEMATICAL_SPEC_v0.2.md`

The master benchmark should be interpreted together with that specification. In particular:

- candidate generation/indexing is part of PNDS routing cost;
- a full scan of all persistent objects is not a sublinear routing result;
- hard TopK requires an explicit train-time relaxation;
- causal execution requires a defined SCM intervention;
- state controls should use matched/on-manifold alternatives where possible;
- verification probabilities must be conditional on prior path steps and normalized or otherwise length-accounted when comparing path lengths;
- registration uses discovery/validation data and a locked final test;
- false admission and registered-state contamination are distinct metrics.

The context-scaling proof remains a hypothesis until measured under matched conditions.
