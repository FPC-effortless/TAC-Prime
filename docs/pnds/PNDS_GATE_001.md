# PNDS-GATE-001 — Causal Routed Execution, Verification, and Persistent Update

**Protocol:** PNDS-URP v0.1  
**Status:** PRE-REGISTERED / NOT YET RUN  
**Purpose:** Directly test the unresolved PNDS gap without introducing recurrence, answer leakage, gold structural labels, or a learned verifier before the routing substrate is measurable.

## Research question

Can a cheap relevance router select the correct persistent structure, execute it causally, verify its outcome, and update persistent state without receiving the answer, true edge set, or other gold structural metadata?

## Canonical loop

`S_t -> R_t -> C_t -> A_t -> O_t -> V_t -> S_{t+1}`

- **S_t:** persistent store of candidate structures and provenance.
- **R_t:** cheap query-conditioned router.
- **C_t:** selected executable structure.
- **A_t:** intervention/action emitted by the selected structure.
- **O_t:** environment outcome, exposed only after the action.
- **V_t:** verifier comparing predicted/expected outcome with observed outcome.
- **S_{t+1}:** state update performed only under the pre-registered commit rule.

## Core causal requirement

The environment must contain hidden structure that determines which action succeeds. The model receives an observation and query, but the router must not receive:

- true edge sets;
- target answer;
- target structure ID;
- oracle masks;
- post-action outcome;
- metadata that uniquely encodes the correct action.

The environment interface must support `do(action)` so action changes can be measured independently of observation correlation.

## Experimental arms

| Arm | Persistent state | Router | Structure | Verifier | Update |
|---|---|---|---|---|---|
| A Oracle | yes | oracle | gold | yes | yes |
| B Static retrieval | yes | fixed similarity | selected | yes | yes |
| C Cheap learned router | yes | Q/K or bilinear | selected | yes | yes |
| D Teacher-distilled router | yes | distilled CDL signal | selected | yes | yes |
| E Full PNDS | yes | learned | selected | learned/thresholded | verified only |
| F Shuffled control | yes | same | shuffled state | same | same |

A is a capacity ceiling, not evidence for learned routing.

## Required controls

1. **Query shuffle:** hold state fixed, replace query.
2. **State shuffle:** hold query fixed, permute candidate structures.
3. **Structure corruption:** alter the selected structure while preserving surface statistics.
4. **Action intervention:** replace the chosen action with a matched alternative.
5. **Outcome holdout:** verifier sees outcomes only after action.
6. **Reset:** remove persistent state.
7. **Cold-start:** query before any useful structure has been stored.
8. **No-answer channel:** assert programmatically that target/outcome fields never enter router inputs.
9. **Gold-metadata audit:** router input tensors must be schema-validated against an allowlist.
10. **Update ablation:** compare verified update against unconditional update.

## Primary metrics

### Routing
- top-1 correct structure
- MRR
- selected-vs-random action success
- router latency / FLOPs proxy
- candidate count

### Causal execution
- `P(success | selected)`
- `P(success | intervened action)`
- action intervention effect
- structure corruption effect

### Verification
- verifier precision
- verifier recall
- false commit rate
- false reject rate
- calibration / Brier score where probabilistic

### Persistence
- carry success after reset boundary
- update gain
- verified-update vs unconditional-update delta
- stale-state failure rate

### Efficiency
- router compute
- full-context compute
- candidate count
- active structure fraction

## Success gate

A run can be promoted only if all are true:

1. Learned router beats random and fixed-similarity controls on held-out structures.
2. Selected actions show a positive causal intervention effect.
3. Shuffling persistent state removes or materially reduces the advantage.
4. Router inputs contain no target/outcome/gold-structure fields.
5. Verified-only update outperforms unconditional update on held-out episodes.
6. False commits remain below the pre-registered threshold.
7. Results reproduce across at least 3 seeds.
8. No component receives the hidden true structure during inference.

A single positive task-accuracy result is insufficient.

## Falsification conditions

The PNDS hypothesis is weakened if:

- router accuracy improves but action intervention has no effect;
- state shuffling does not materially affect performance;
- unconditional updates match verified updates;
- the verifier can only succeed when given gold structural information;
- distilled routing quality collapses under paraphrase or structure-preserving distractors;
- compute cost approaches full candidate evaluation.

## Stage order

**Stage 0:** interface and leakage audit.  
**Stage 1:** static/oracle/random routing baselines.  
**Stage 2:** cheap learned router.  
**Stage 3:** causal action intervention.  
**Stage 4:** verifier.  
**Stage 5:** verified persistent update.  
**Stage 6:** 3-seed replication and candidate-count scaling.

Do not add recurrence, reinforcement learning, contrastive routing loss, or dual masks until Stage 2 passes its held-out routing gate.

## Expected interpretation

- **Routing succeeds, execution fails:** relevance is not equivalent to executable structure.
- **Execution succeeds, verification fails:** action correctness is not reliably identifiable from outcomes.
- **Verification succeeds, persistence fails:** state representation/update is the bottleneck.
- **All stages pass:** first controlled evidence for the complete PNDS loop.
- **Any stage fails:** retain the failure as a boundary condition and isolate the failing component.

## External evidence policy

Related literature establishes that explicit non-parametric memory and learned retrieval can improve knowledge-intensive prediction, but does not establish this complete causal loop. RAG, differentiable memory, and neural-module approaches therefore remain comparative design references rather than PNDS evidence.

## Artifact requirements

Every run must record:

`repo -> branch -> commit -> PNDS ID -> seed -> config -> artifact -> metrics -> interpretation -> decision`

No result is promoted without this provenance chain.
