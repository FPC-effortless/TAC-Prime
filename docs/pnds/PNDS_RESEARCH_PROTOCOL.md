# PNDS Universal Research Protocol

**Protocol:** PNDS-URP v0.2  
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


---

## 24. Master benchmark and macro-level research control

The authoritative cross-program benchmark is:

`docs/pnds/PNDS_MASTER_BENCHMARK.md`

It defines the Context-Scaling Proof and the four canonical comparison arms:

1. **FC:** full-context Transformer, (C(N))
2. **PS:** persistent state only / TAC, (C(S))
3. **PR:** persistent + routed / TAC + CDL, (C(R(S,Q)))
4. **PNDS:** persistent + routed structural execution, (C(G(R(S,Q))))

The central falsifiable hypothesis is:

[
C_{PNDS}(N) approx f(|R_t|)
]

over increasing irrelevant history, while preserving predefined task-capability parity against the full-context baseline.

This is an empirical scaling hypothesis, not an assumed complexity class. No O(1), O(N/20), 10×, 20×, or similar claim is valid until measured under matched conditions.

---

## 25. Mandatory benchmark invariants

Every confirmatory PNDS benchmark must:

- freeze the task definition and evaluation protocol before measurement;
- vary irrelevant accumulated history while holding relevant information approximately fixed;
- use matched hardware, precision, batch/evaluation procedure, and explicitly defined compute accounting;
- report capability and cost together;
- declare the accuracy-parity margin before confirmatory runs;
- report seed-level results and uncertainty where appropriate;
- preserve raw artifacts and exact provenance;
- keep the routing anti-leakage boundary intact;
- distinguish routing success from causal execution;
- distinguish persistence from mere recoverability/correlation.

The benchmark is a falsification instrument, not a demonstration script.

---

## 26. Anti-leakage boundary is an invariant

For routing and structure selection, the router must not receive:

- target or answer;
- outcome or reward;
- true/gold structure ID;
- oracle mask;
- correct action;
- metadata uniquely encoding the correct action.

Permitted inputs must be explicitly enumerated for each experiment.

If a forbidden field enters the router, the affected result is invalid until corrected and rerun.

Evaluation code may retain hidden gold information for scoring, but that information must remain inaccessible to the model and router.

---

## 27. Causal necessity is required for execution claims

A structure being predictive of success is not sufficient to establish that the structure drives the computation.

Whenever an experiment claims executable structural selection, include an intervention that forces an alternative structure/action while controlling other variables.

Record a predefined intervention effect such as:

[
Delta_{causal}
=
P(successmid do(z_i))
-
P(successmid do(z_j)).
]

If the intervention effect is approximately zero under the preregistered threshold, the experiment does not establish causal execution even if routing accuracy is high.

---

## 28. Persistence requires destructive controls

Whenever a capability is attributed to persistent state, test at minimum, where applicable:

1. reset;
2. state shuffle across episodes/tasks;
3. wrong-context replacement;
4. controlled state corruption;
5. state recoverability probe;
6. functional state-use probe;
7. held-out regime.

A persistence result must explain why the claimed advantage disappears or changes under destructive controls. If it survives all controls, investigate alternative information channels before attributing the result to persistence.

---

## 29. Verified-only commit falsification

When the system updates persistent state from observed outcomes, compare:

[
S_{t+1}=U(S_t,O_t)
]

against:

[
S_{t+1}=U(S_t,O_t)quad	ext{only after }V_t=mathrm{accept}.
]

The comparison must include held-out performance, false commits, stale-state acceptance, poisoning/regression rate, and recovery/repair where applicable.

If unconditional and verified-only updates are indistinguishable under the preregistered integrity tests, the verifier has not demonstrated functional value and must not be credited as a useful PNDS primitive.

---

## 30. Repository isolation and promotion rule

TAC-transformer, TAC-Prime, and CDL Attention remain independent evidence-producing laboratories.

The `pnds` branch is an integration/research gate, not a substitute for standalone evidence.

Do not merge additional architectural complexity into the PNDS substrate merely because it is available. In particular, recurrence, RL, contrastive routing, dual masks, or verifier complexity must not be introduced to rescue an unresolved lower-level primitive.

A primitive enters the integrated substrate only after:

1. a standalone controlled experiment;
2. explicit leakage audit;
3. defined primary metric;
4. baseline/control;
5. documented failure modes;
6. reproduction where required;
7. exact provenance.

Integration means "eligible for the next controlled experiment", not "proven".

---

## 31. Research ladder and stopping rule

The macro research order is:

**persistence → addressability → relevance-efficient routing → executable structure → causal intervention → verified update → scaling → cross-domain transfer.**

Do not jump to a later layer to compensate for failure at an earlier layer.

The immediate PNDS benchmark sequence is therefore:

1. validate the benchmark interface and leakage controls;
2. complete non-semantic Stage 1 baselines;
3. construct a nontrivial observable-signal environment;
4. train Stage 2 routing from observed action outcomes, not hidden structure IDs;
5. test causal intervention;
6. test verifier and verified-only update;
7. run confirmatory history-scaling;
8. replicate across the four commercial adapter environments.

The smallest experiment that resolves the current uncertainty is preferred over a monolithic fusion run.

---

## 32. Protocol changelog

### v0.2 — 2026-09-24

Added the macro-level research controls:

- authoritative Context-Scaling Proof;
- explicit FC/PS/PR/PNDS benchmark arms;
- accuracy/cost joint gate;
- mandatory history-scaling and matched-cost methodology;
- immutable routing anti-leakage boundary;
- causal necessity requirement for execution;
- destructive persistence controls;
- verified-only commit falsification;
- repository isolation and promotion rules;
- staged research ladder and stopping rule.

Existing v0.1 experiment records remain governed by v0.1 and are not retroactively rewritten.
