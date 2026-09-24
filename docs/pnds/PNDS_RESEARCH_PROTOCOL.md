# PNDS Universal Research Protocol

**Protocol:** PNDS-URP v0.1  
**Status:** Active research control document  
**Purpose:** Provide one uniform research, experiment, evidence, and handoff protocol across the TAC Transformer, TAC-Prime, and CDL Attention research lines.

---

## 1. Research program

The repositories are separate experimental laboratories for one converging research program:

> **Persistent Structural Computation:** useful computation should be reusable across time through persistent state, selectively addressable through learned relevance routing, executable through structural pathways, and subject to verification and repair.

The intended integration target is the **Persistent Neural Decision Substrate (PNDS)**.

The repositories are not required to share implementation. They share:

- hypotheses;
- experiment identifiers;
- evidence standards;
- control requirements;
- metric definitions;
- result-record format;
- provenance rules;
- integration criteria.

No result from one repository becomes a result of the unified PNDS architecture until the required cross-repository evidence exists.

---

## 2. Repository roles

### TAC-transformer

Primary evidence for:

- persistent state;
- temporal carry;
- state reuse;
- action-conditioned prediction;
- intervention sensitivity;
- long-horizon behavior;
- reset/context-gap behavior;
- world-model computation.

### TAC-Prime

Primary laboratory for:

- architecture exploration;
- new PNDS substrate implementations;
- integration experiments;
- hypotheses that require a clean architectural branch before entering established TAC experiments.

### CDL Attention

Primary evidence for:

- relevance routing;
- selective retrieval;
- content-addressed lookup;
- sparse activation;
- context compression;
- attention alternatives;
- routing stability and retrieval fidelity.

### CASM-related work

When CASM experiments are hosted in CDL or another repository, treat CASM as the structural execution layer connecting relevance to computation:

**relevance → structural gate → targeted execution → outcome → verification**.

---

## 3. Canonical PNDS loop

All experiments should be expressible, where applicable, using:

\[
S_t \rightarrow R_t \rightarrow C_t \rightarrow A_t \rightarrow O_t \rightarrow V_t \rightarrow S_{t+1}
\]

where:

- **S_t:** persistent state;
- **R_t:** relevance/routing decision;
- **C_t:** selected computational structure;
- **A_t:** action, computation, retrieval, or intervention;
- **O_t:** observed outcome;
- **V_t:** verification/validation;
- **S_{t+1}:** updated persistent state.

Not every experiment needs every component. Missing components must be explicitly marked **N/A**, not silently omitted.

---

## 4. Four required capability claims

PNDS research is organized around four separable claims.

### C1 — Persistence

Useful information can survive beyond the immediate input/context and remain usable later.

### C2 — Addressability

The system can selectively retrieve task-relevant persistent information instead of processing all accumulated information.

### C3 — Executability

Retrieved structure can change which computation is actually executed, rather than merely changing generated output.

### C4 — Verification

The system can evaluate an outcome, reject unsupported results, and/or perform bounded repair before committing state.

A fused PNDS result should identify exactly which claims were tested.

---

## 5. Experiment identity

Every experiment receives a stable identifier:

**PNDS-[REPO]-[CAPABILITY]-[NUMBER]**

Examples:

- PNDS-TAC-PERSIST-001
- PNDS-TAC-ACTION-002
- PNDS-CDL-RETRIEVAL-001
- PNDS-CDL-ROUTING-003
- PNDS-FUSION-001

If an experiment is a continuation, preserve the parent identifier and append a revision:

- PNDS-CDL-RETRIEVAL-001-r2
- PNDS-FUSION-001-r3

Never reuse an identifier for a materially different experiment.

---

## 6. Required experiment record

Every non-trivial experiment must record:

### Identity

- Experiment ID
- repository
- branch
- commit SHA
- date/time
- researcher/agent
- parent experiment, if any

### Hypothesis

State one falsifiable claim.

Bad:
> The model is better.

Good:
> At matched task accuracy, persistent-state routing requires less inference memory than full-context attention as irrelevant history increases.

### Variables

Record:

- independent variables;
- dependent variables;
- controlled variables;
- dataset/task generation;
- model configuration;
- seed(s);
- optimizer;
- learning rate;
- training steps;
- evaluation protocol.

### Controls

At minimum identify:

- baseline;
- ablation;
- positive control, when available;
- leakage control;
- reset/context control, when relevant;
- matched-compute or matched-parameter control, when relevant.

### Results

Record:

- raw result location;
- summary metrics;
- uncertainty/error bars where applicable;
- seed-level results;
- failures;
- anomalies;
- stopped runs;
- negative results.

### Interpretation

Separate:

1. **Observed**
2. **Inferred**
3. **Speculative**

Do not convert an inference into an observed result.

### Decision

Exactly one:

- **SUPPORTED**
- **PARTIALLY_SUPPORTED**
- **INCONCLUSIVE**
- **FALSIFIED**
- **BLOCKED**
- **NOT_RUN**

Explain the decision in one or more sentences.

---

## 7. Evidence hierarchy

Use the following evidence states:

**E0 — Idea**  
Unimplemented hypothesis.

**E1 — Implemented**  
Code exists; no meaningful empirical evidence.

**E2 — Smoke-tested**  
Pipeline executes; correctness only.

**E3 — Controlled result**  
Predefined experiment and controls completed.

**E4 — Reproduced**  
Result reproduced across independent seeds/runs.

**E5 — Cross-condition**  
Result survives materially different conditions.

**E6 — Cross-domain**  
Result transfers to another relevant task/domain.

Commercial or architectural claims must not be written as established facts when the evidence is below the level required by the claim.

---

## 8. Mandatory negative-result policy

Negative results are first-class research outputs.

Never delete, overwrite, or hide:

- failed runs;
- failed controls;
- unstable seeds;
- routing collapse;
- leakage discoveries;
- optimizer failures;
- regressions;
- unexplained anomalies.

A failed experiment receives an identifier and remains in the ledger.

If a result is invalidated by a discovered methodological flaw, mark it **INVALIDATED** and preserve:

- original result;
- flaw;
- corrected protocol;
- rerun identifier.

---

## 9. No result without provenance

Every reported metric must be traceable to:

**repository → branch → commit → experiment ID → configuration → run artifact → metric**

If a metric cannot be traced, label it:

> **UNVERIFIED / PROVENANCE MISSING**

Do not use it as evidence for a PNDS claim.

---

## 10. Cross-repository result transfer

A result discovered in one repository can be referenced elsewhere using:

**Source Experiment ID + source commit SHA + evidence level**

Example:

> CDL result: PNDS-CDL-RETRIEVAL-003 @ abc1234, E4.

Do not copy the conclusion into another repository as though it were independently reproduced.

The receiving repository should classify it as:

**EXTERNAL_EVIDENCE**

until independently reproduced.

---

## 11. Unified metric vocabulary

When applicable, report:

### Capability

- task accuracy;
- exact match;
- prediction error;
- intervention sensitivity;
- long-horizon degradation;
- retrieval accuracy;
- routing accuracy;
- verification precision/recall.

### Efficiency

- wall-clock latency;
- GPU/CPU time;
- peak memory/VRAM;
- tokens processed;
- active tokens/state fraction;
- active edges/computational fraction;
- parameter count;
- FLOPs or a clearly defined compute proxy.

### Stability

- seed variance;
- training variance;
- routing entropy;
- gate occupancy;
- state drift;
- reset sensitivity.

### Reliability

- failure rate;
- false retrieval;
- stale-state acceptance;
- verification rejection;
- repair success;
- regression rate.

Never report an efficiency ratio without specifying the denominator and matched conditions.

---

## 12. Core scaling experiment

The common cross-repository benchmark should eventually compare:

1. full-context baseline;
2. persistent state without learned routing;
3. learned routing without persistent state;
4. persistent state + learned routing;
5. persistent state + routing + structural execution;
6. full PNDS loop when verification is implemented.

Increase irrelevant history/context systematically.

Measure:

\[
C_{persistent}/C_{full-context}
\]

alongside capability retention.

The primary research question is:

> Does computational cost depend more strongly on relevant structural state than on total accumulated history while preserving task performance?

Do not assume O(1), O(N/20), 10×, 20×, 80%, 90%, or similar efficiency claims until measured under an explicitly defined workload and baseline.

---

## 13. Leakage and causal controls

Every experiment involving routing, memory, state, actions, or interventions must ask:

- Can the target be inferred from an unintended feature?
- Does the environment expose information that should be inaccessible?
- Does the mask encode the answer?
- Does the training/evaluation split leak structure?
- Does reset actually remove the claimed information?
- Does changing an action change the prediction for causal reasons?
- Is the model using the intended state or a shortcut?

For action-conditioned experiments, distinguish:

**observation**, **action**, **intervention**, and **outcome**.

For routing experiments, distinguish:

**relevance selection** from **answer supervision**.

---

## 14. Branch discipline

Research branches should use:

- one hypothesis or tightly coupled hypothesis family;
- immutable experiment IDs;
- explicit baseline/control commits;
- small, reviewable commits;
- no silent protocol changes.

Branch naming:

- `pnds` — integration/research substrate;
- `research/<experiment>` — experiment;
- `diagnostic/<question>` — diagnostic;
- `control/<baseline>` — control;
- `fix/<methodology>` — methodology correction.

A branch that changes the experimental protocol must state the change explicitly.

Do not retroactively rewrite old experiment records to match later findings.

---

## 15. Integration gate

A mechanism may enter the unified PNDS substrate only when:

1. the mechanism has a controlled experiment;
2. its failure modes are documented;
3. leakage controls pass;
4. its primary metric is defined;
5. at least one baseline exists;
6. the result is reproducible when the claim requires reproducibility;
7. the exact commit and artifacts are recorded.

Integration does **not** mean the mechanism is proven. It means it is sufficiently characterized to participate in the next controlled experiment.

---

## 16. Commercial model mapping

The unified substrate can later be evaluated in four domains:

### Persistent Software Engineering

State = repository structure  
Graph = dependency/test graph  
Action = patch  
Outcome = test/build result  
Verification = regression/security checks

### Enterprise Persistent Memory

State = facts/procedures/relations  
Graph = knowledge/workflow graph  
Action = retrieve/reason/execute  
Outcome = task result  
Verification = provenance/consistency

### Action-Conditioned World Model

State = environment state  
Graph = entity/causal/spatial structure  
Action = intervention  
Outcome = environment transition  
Verification = prediction/intervention consistency

### Verified Decision System

State = decision/evidence state  
Graph = rules/dependencies  
Action = decision  
Outcome = downstream observation  
Verification = executable rule/evidence checks

These are application hypotheses, not evidence of commercial performance.

---

## 17. Required result ledger

Each repository should maintain or create:

`docs/pnds/RESULTS.md`

Each entry should contain:

| Field | Required |
|---|---|
| Experiment ID | Yes |
| Hypothesis | Yes |
| Branch | Yes |
| Commit | Yes |
| Baseline | Yes |
| Controls | Yes |
| Seeds | Yes |
| Main metrics | Yes |
| Result | Yes |
| Evidence level | Yes |
| Decision | Yes |
| Artifact path | Yes |
| Known limitations | Yes |
| Next experiment | Yes |

The ledger is append-oriented. Do not erase historical entries.

---

## 18. Required decision record

After every completed experiment, write:

**Finding:**  
What happened.

**Evidence:**  
Which measurements establish it.

**Failure/limitation:**  
What the experiment cannot establish.

**Decision:**  
SUPPORTED / PARTIALLY_SUPPORTED / INCONCLUSIVE / FALSIFIED / BLOCKED / NOT_RUN.

**Next experiment:**  
The smallest experiment that resolves the remaining uncertainty.

This prevents research from becoming a sequence of unconnected implementation steps.

---

## 19. Unified research questions

The long-term PNDS program should answer, in order:

1. Can useful state persist?
2. Can persistent state be queried selectively?
3. Does relevance routing reduce unnecessary computation?
4. Can routing select executable structure?
5. Does selected structure improve intervention/action response?
6. Can outcomes update persistent state without catastrophic drift?
7. Can verification reject or repair incorrect computation?
8. Does the combined system scale better than full-context baselines?
9. Does the mechanism transfer across domains?
10. Does the efficiency/capability tradeoff support a commercially useful runtime?

Do not skip an earlier unresolved question merely because a later architecture appears promising.

---

## 20. Current integration hypothesis

The working architectural hypothesis is:

\[
S_t
\xrightarrow{\text{CDL}}
R_t
\xrightarrow{\text{CASM}}
C_t
\xrightarrow{\text{TAC}}
O_t
\xrightarrow{\text{Verifier}}
S_{t+1}
\]

This is a hypothesis to be tested, not a settled architecture.

The repositories should therefore remain experimentally separable while sharing this protocol.

---

## 21. Handoff requirements

Every research handoff must state:

- work ID;
- repository;
- branch;
- commit;
- experiment IDs touched;
- tests run;
- results;
- failed tests;
- artifacts;
- open questions;
- exact next action.

A handoff is incomplete if another researcher cannot reconstruct what was done from the repository.

---

## 22. Change-control rule

Changes to this protocol require:

- version increment;
- dated changelog entry;
- reason for change;
- affected experiment classes.

Existing experiment records retain the protocol version under which they were conducted.

---

## 23. Changelog

### v0.1 — 2026-09-24

Initial universal PNDS research protocol established across the TAC Transformer, TAC-Prime, and CDL Attention research lines.

Primary goals:

- unify experiment identity;
- unify evidence levels;
- preserve negative results;
- establish provenance;
- separate observations from interpretations;
- standardize controls and metrics;
- create a common path from TAC + CDL + CASM toward PNDS;
- prevent commercial claims from outrunning experimental evidence.
