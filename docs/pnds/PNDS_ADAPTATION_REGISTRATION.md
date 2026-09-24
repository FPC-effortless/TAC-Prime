# PNDS Controlled Adaptation and Registration Primitives

**Specification:** PNDS-CA v0.1  
**Date:** 2026-09-24  
**Status:** Research control specification  
**Scope:** PNDS/CASM integration across TAC Transformer, TAC-Prime, and CDL Attention.

## 1. Purpose

This specification adds five experimentally separable primitives to PNDS:

1. Adaptation regularization — constrain how persistent change accumulates rather than freezing what may change.
2. Epistemic quarantine and registration — separate discovery from permanent adoption.
3. Typed authority routing — route computation by capability/type and explicitly model authority instead of relying on unrestricted multi-agent consensus.
4. Intermediate verification — verify computational transitions, not only final outputs.
5. State-to-computation coupling — experimentally establish that changing persistent state changes the computation/outcome through the claimed pathway.

These primitives are architectural hypotheses and controls. Their inclusion does not establish that the corresponding mechanisms improve performance.

## 2. Revised PNDS state machine

The canonical loop is extended from:

S_t -> R_t -> C_t -> A_t -> O_t -> V_t -> S_{t+1}

to:

S_P
  -> R_t
  -> C_t
  -> A_t
  -> O_t
  -> V_t
  -> S_E
  -> Q/R
  -> S_P'

where:

- S_P = registered persistent state;
- R_t = relevance/authority routing;
- C_t = selected executable structure;
- A_t = action/computation;
- O_t = observed outcome;
- V_t = verification/audit;
- S_E = experimental state generated from the episode;
- Q = quarantine;
- R = registration/adoption;
- S_P' = next registered persistent state.

Critical invariant:

S_E does not become S_P merely because one episode succeeded.

Registration requires a separately defined adoption test.

## 3. Persistent-object status

A persistent object must support at least:

- experimental
- provisional
- registered
- quarantined
- retired

Each object should carry, where applicable:

- content/state payload;
- type;
- provenance;
- scope;
- dependencies;
- confidence;
- evidence count;
- validation/cohort count;
- observed utility delta;
- cost delta;
- failure/regression count;
- last validation;
- registration status;
- rollback/recovery information.

The schema may be implemented differently by each repository. The semantic fields are the experimental contract.

## 4. Primitive A — Adaptation regularization

### Hypothesis

Persistent self-modification can overfit finite evaluation feedback. Constraining the accumulation and selection of changes may improve held-out utility and reduce unnecessary persistent complexity.

Do not interpret this as requiring textbook L0/L1/L2 optimization. The experimental primitive is regularized evolutionary/adaptive search.

### Proposal-side controls

Candidate generation should optionally control:

- edit/change budget;
- number of independently changed components;
- repeated proposals that revisit falsified hypotheses;
- exploration of untested components when progress stalls.

### Selection-side controls

Candidate adoption should optionally reject:

- gains within the predefined evaluation-noise band;
- cost increases not justified by measured utility;
- components whose measured contribution has decayed;
- regressions on held-out integrity tests.

### Metrics

- held-out utility delta;
- adaptation count;
- persistent-state size/complexity delta;
- inference-cost delta;
- regression rate;
- repeated-failure rate;
- recovery rate;
- retained utility after subsequent cohorts.

Primary comparison:

unregularized adaptation vs regularized adaptation.

No universal complexity penalty or fixed coefficient is assumed.

## 5. Primitive B — Epistemic quarantine and registration

### Hypothesis

Separating discovery from permanent adoption reduces accumulation of unstable capabilities or state.

The minimum comparison is:

A. immediate registration  
B. experimental -> repeated validation -> registration

A candidate may remain usable experimentally without becoming globally persistent.

### Registration evidence

The adoption test should be based on multiple predefined evaluation cohorts/episodes, not a single success.

The exact cohort count, utility margin, stability criterion, and cost allowance must be frozen before the confirmatory experiment.

### Required measurements

- candidate discovery rate;
- registration rate;
- rejection/quarantine rate;
- retained registered capabilities;
- held-out utility;
- stale-state acceptance;
- poisoning/regression;
- recovery/rollback;
- persistent-state growth.

The claim is not that delayed registration is always better. The experiment determines whether it improves the defined integrity/utility objective.

## 6. Primitive C — Typed authority routing

### Hypothesis

When multiple computations or specialists are available, unrestricted consensus can dilute useful specialist information. Explicit routing and authority should instead determine which computation is responsible for each subproblem.

The router should select from typed computational capabilities such as:

- retrieve;
- execute;
- predict;
- inspect;
- verify;
- repair;
- external tool;
- specialist/model.

Authority is an explicit routing variable, not an emergent assumption.

### Controls

Compare at minimum, when applicable:

- unrestricted/dense routing;
- sparse typed routing;
- sparse typed routing with explicit authority;
- shuffled authority/control.

Do not encode the correct specialist as an oracle feature.

### Metrics

- routing accuracy;
- expert selection/deference rate;
- unnecessary specialist activation;
- conflict resolution accuracy;
- intervention effect of selected specialist;
- coordination/token cost;
- failure under increasing specialist count.

Natural-language multi-agent discussion is an optional baseline, not the PNDS default.

## 7. Primitive D — Intermediate verification

### Hypothesis

Final-output verification can miss invalid intermediate transitions. Verifying the computational path can provide earlier rejection/backtracking/repair.

Compare:

V_final(y)

against:

V_path(z_1, z_2, ..., z_n, y)

where z_i are intermediate transitions or executable structural steps.

### Verification modes

- local transition validity;
- structural consistency;
- tool/action precondition validity;
- empirical outcome verification;
- final-output verification.

### Metrics

- invalid-transition detection;
- false rejection;
- false acceptance;
- calibration;
- repair/backtracking success;
- final task accuracy;
- verification cost;
- recovery after corrupted intermediate state.

A verifier must not receive hidden gold structure unless that is explicitly the object of a separate oracle control.

## 8. Primitive E — State-to-computation coupling

### Hypothesis

A persistent latent/state representation is useful only if intervention on the claimed state changes the downstream computation or outcome through the claimed pathway.

Separate three claims:

1. state is predictive;
2. state is functionally used;
3. state is causally necessary/sufficient for the claimed computation.

Only the third establishes causal coupling.

### Required controls

- state reset;
- state shuffle;
- wrong-context replacement;
- targeted state corruption;
- state intervention with route held fixed;
- route intervention with state held fixed;
- output/outcome intervention where applicable.

Measure:

Delta_state = P(success | do(state_i)) - P(success | do(state_j))

and distinguish it from observational correlation.

## 9. Unified adaptation admission rule

For a candidate update Delta S, define an empirical admission objective:

L_adapt = -U(Delta S)
           + lambda_c C(Delta S)
           + lambda_r Risk(Delta S)
           + lambda_b Bandwidth(Delta S)
           + lambda_k Complexity(Delta S)

where:

- U = measured held-out utility;
- C = added computation/resource cost;
- Risk = integrity/regression risk;
- Bandwidth = state/update transfer cost;
- Complexity = added persistent structure/behavior.

The expression is a bookkeeping objective, not a requirement for differentiable optimization.

A candidate is eligible for registration only if it passes the preregistered utility, stability, integrity, and cost gates.

The principle is:

measured persistent utility must justify persistent complexity and risk.

## 10. New PNDS research ladder

The existing ladder becomes:

1. persistence;
2. addressability;
3. relevance-efficient routing;
4. executable structure;
5. causal intervention;
6. intermediate verification;
7. experimental-state update;
8. quarantine/registration;
9. adaptation regularization;
10. state-to-computation coupling;
11. typed authority routing;
12. context-scaling proof;
13. cross-domain transfer.

Ordering is a research control, not a claim that every implementation must be sequential. A later mechanism cannot be used to rescue an unresolved earlier primitive.

## 11. Integration into current GATE-001

The current Stage 2 outcome-trained router remains unchanged until its held-out evidence is verified.

After Stage 2 passes its predefined routing gate:

### Stage 3 — Causal execution
Add action/structure intervention.

### Stage 4 — Intermediate verification
Add path-level verification and final-only control.

### Stage 5 — Experimental state
Allow proposed updates to enter S_E without automatic registration.

### Stage 6 — Quarantine/registration
Compare immediate registration with repeated validation before registration.

### Stage 7 — Adaptation regularization
Compare unregularized versus regularized candidate evolution while holding the registration protocol fixed.

### Stage 8 — State-to-computation coupling
Perform state and route interventions separately.

### Stage 9 — Typed authority routing
Test specialist/compute selection and authority controls.

### Stage 10 — Context-scaling proof
Only after the primitive chain survives its causal/integrity gates.

No recurrence, RL, contrastive routing, dual masks, or additional verifier complexity should be introduced merely to rescue a failed stage.

## 12. Result classification

Each primitive receives its own experiment ID and evidence state.

A successful primitive does not automatically prove the full PNDS architecture.

Use:

- SUPPORTED — predefined primary hypothesis supported;
- PARTIALLY_SUPPORTED — some preregistered conditions pass;
- INCONCLUSIVE — evidence insufficient;
- FALSIFIED — predefined falsification condition met;
- BLOCKED — methodology or infrastructure prevents valid measurement.

## 13. Cross-repository implementation rule

TAC Transformer, TAC-Prime, and CDL Attention remain independent evidence-producing laboratories.

The shared object is the protocol and benchmark contract, not a requirement to merge implementations.

A primitive can be promoted to the integrated PNDS branch only as:

characterized mechanism -> eligible integration primitive

not:

interesting result -> proven architecture.

## 14. Immediate consequence

Do not replace the current Stage 2 experiment with the full adaptation/registration architecture.

The next valid action remains:

1. verify the Stage 2 run and artifact;
2. if valid, perform causal intervention;
3. then introduce experimental-state updates;
4. then test quarantine/registration;
5. then regularize adaptation;
6. then test state-to-computation coupling and typed authority routing;
7. only then run the macro Context-Scaling Proof.

This preserves the current falsification ladder while adding the new primitives at the correct architectural layer.
