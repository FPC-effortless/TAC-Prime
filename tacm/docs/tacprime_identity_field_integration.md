# TAC-Prime-ID001: Identity-Carried Structure Memory

## Motivation

TAC-Prime stores reusable structures and procedures in StructureMemory and
ProceduralMemory.  These are retrieved by cosine similarity of a task
embedding produced from the current MoE output.

**Problem:** Retrieval is stateless across tasks.  Two structurally similar
tasks processed in different sessions produce independent task embeddings with
no mechanism to recognise that they share an underlying computational identity.
As a result, the router makes fresh decisions on each call and memory retrieval
cannot favour records accumulated during prior work on the same kind of problem.

**Hypothesis (TAC-Prime-ID001):** TAC-Prime will reuse, route, and transfer
structures better when reusable structures/procedures are **carried by
persistent computational identities** rather than only stored as detached memory
records.  An IdentityFieldLayer maintains a small set of learned identity
embeddings and a persistent IdentityState that accumulates evidence about which
identity is active across tasks.  This state then biases the router and the
memory retrieval so that the same problems consistently trigger the same experts
and the same memory records.

---

## Architecture Overview

```
tokens (B, T)
  │
  ▼
┌─────────────────────┐
│  TransformerBackbone │   hidden: (B, T, d_model)
└─────────────────────┘
  │
  ▼
┌──────────────────────────────────────────┐
│  IdentityFieldLayer  [NEW — ID001]        │
│                                          │
│  • token→identity affinity logits        │
│  • energy-budgeted soft routing          │
│  • identity_context = Σ w_i * id_mem_i   │
│  • update IdentityState (EMA)            │
│  • emit active_identity, coherence, aux  │
└──────────────────────────────────────────┘
  │  hidden += identity_residual_scale * identity_context
  ▼
┌─────────────────────┐
│    ConceptVolume     │   center: (B, T, volume_dim)
└─────────────────────┘
  │
  ▼
┌──────────────────────────────────────────┐
│  StructureRouter  [biased — ID001]        │
│                                          │
│  hidden_router += id_router_bias_scale   │
│                   * proj(identity_ctx)   │
└──────────────────────────────────────────┘
  │
  ▼
┌─────────────────────┐
│      MoELayer        │
└─────────────────────┘
  │
  ▼
┌──────────────────────────────────────────┐
│  StructureMemory  [biased — ID001]        │
│                                          │
│  score = 0.7*sim + 0.3*survival          │
│        + id_mem_bias_scale               │  ← identity_match_bonus
│          (if record.identity_id matches) │
└──────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────┐
│  ProceduralMemory [biased — ID001]        │
│                                          │
│  score = 0.6*sim + 0.4*overall_score     │
│        + id_mem_bias_scale               │  ← identity_match_bonus
└──────────────────────────────────────────┘
  │
  ▼
┌─────────────────────┐
│    SurvivalField     │
└─────────────────────┘
  │
  ▼
┌─────────────────────┐
│     VerifierHead     │
└─────────────────────┘
```

---

## IdentityField vs ConceptVolume — Key Distinction

| Aspect               | ConceptVolume                           | IdentityFieldLayer                        |
|----------------------|-----------------------------------------|-------------------------------------------|
| **Purpose**          | Models conceptual/structural *regions* of the input space | Carries *who is doing the computation* — a persistent actor identity |
| **Representation**   | Gaussian volume: center + variance       | Learned embeddings + EMA-accumulated memory |
| **State**            | Stateless per call (EMA for training)    | Stateful — IdentityState persists across calls |
| **Output**           | center, variance, family logits          | identity_context residual, active_identity, coherence |
| **Memory role**      | Determines routing family via FamilyRouter | Biases routing AND memory retrieval scores |
| **Replaces other?**  | No — sits after IdentityField            | No — sits before ConceptVolume             |

ConceptVolume answers: *"what kind of structure is this token near?"*
IdentityField answers:  *"which persistent computational agent is working on this?"*

---

## IdentityState: How It Carries Structures and Procedures

`IdentityState` has four tensors, all shaped per `(batch, n_identities, ...)`:

```
stability       : (B, n_identities)            EMA of how often each identity activates
identity_memory : (B, n_identities, d_model)   EMA of hidden states accumulated per identity
route_history   : (B, n_identities)            EMA of routing weights
active_identity : (B,)  int64                  Current dominant identity index
```

**Update rule (each forward call):**

```python
energy[b, i] = Σ_t weight[b, t, i]            # how much of the sequence used identity i
new_stability[b, i] = decay * stability + (1-decay) * (energy / budget)
delta_mem[b, i]     = weight[b, i] * hidden_mean[b]
new_id_mem[b, i]    = decay * id_mem[b, i] + (1-decay) * delta_mem[b, i]
```

After several tasks in the same "session" (state carried across calls):
- Identity `i` that dominated task 1 accumulates context from task 1's hidden states
- When task 2 arrives with the same underlying structure, the affinity logits
  activate the same identity → retrieval bonus for memory records tagged with that identity

**Memory tags:** `StructureRecord.identity_id` and `ProcedureRecord.identity_id` are set
when a structure is written during training.  During retrieval:

```
score = cosine_sim + 0.3 * survival_score + identity_match_bonus
identity_match_bonus = identity_memory_bias_scale  if record.identity_id == active_identity
                     = 0.0                         otherwise
```

This is a **soft preference**, not a hard filter.  Semantically dissimilar records
with matching identity will not outrank highly similar records without identity.

---

## Benchmark Protocol (TAC-Prime-ID001)

### Synthetic Task Design

- `N_FAMILIES` latent structure families, each with a canonical embedding (centroid)
- Each task = a surface-noised version of a canonical embedding
- `TASKS_PER_FAM` tasks per family, shuffled randomly
- Memory pre-seeded with 3 structure records + 2 procedure records per family,
  tagged with `identity_id = family_id`

### Conditions

| Condition         | IdentityState                                      |
|-------------------|----------------------------------------------------|
| **identity_carried** | Persists across all tasks in order              |
| **identity_reset**   | Re-zeroed before every task                    |
| **identity_shuffled**| Taken from the *next* task's carried state (mismatched) |
| **memory_knockout**  | StructureMemory cleared; retrieval always empty |

### Metrics

| Metric                       | Description                                       |
|------------------------------|---------------------------------------------------|
| `route_consistency`          | 1 − normalised entropy of family routing per task-family |
| `structure_retrieval_acc`    | Fraction of top-1 struct retrievals matching task family |
| `procedure_retrieval_acc`    | Same for procedural memory                        |
| `carried_vs_reset_gain`      | Δ struct retrieval: carried − reset               |
| `carried_vs_shuffled_gain`   | Δ struct retrieval: carried − shuffled            |
| `memory_knockout_drop`       | Δ struct retrieval: carried − no-memory           |
| `identity_specialization`    | Max-prob concentration of active identities within each family |
| `benchmark_score`            | Weighted aggregate (threshold ≥ 0.60)             |

### Validation Gates (6 gates)

1. `carried_route_consistency > reset_route_consistency`
2. `carried_structure_retrieval > reset_structure_retrieval`
3. `carried_procedure_retrieval > reset_procedure_retrieval`
4. `carried_vs_shuffled_gain > 0.0`
5. `memory_knockout_drop > 0.0`
6. `benchmark_score ≥ 0.60`

---

## Current Limitations

1. **Identity bias is small by design** — `identity_memory_bias_scale=0.25` adds at most
   +0.25 to retrieval scores.  For newly initialised models this may not consistently
   overcome similarity differences.  The bias grows more effective after training
   because identity_memory accumulates meaningful hidden-state patterns.

2. **identity_bias_proj starts at zeros** — The router's identity bias projection is
   zero-initialised so that at init time the router is identical to the unmodified
   TAC-SM router.  This ensures training stability; the bias is learned from scratch.

3. **Single active identity per batch item** — `active_identity` is argmax of the mean
   routing weights.  Multi-identity tasks (e.g. code review + planning) are
   represented by a mixture but only the dominant one biases retrieval.

4. **CPU benchmark uses tiny config** — `D_MODEL=64`, `N_LAYERS=2`, `N_EXPERTS=4`.
   Results on the full 30M–150M scale will differ.

5. **No training loop integration yet** — The auxiliary losses (`identity_reuse`,
   `identity_energy`, `identity_coherence`, `identity_separation`) are included in
   `loss_dict` but the `TotalLoss` aggregator does not currently weight them.
   Adding `w_identity: float = 0.05` to `TrainingConfig` would activate them.

---

## Files Added / Modified

### New Files
- `tacm/tacm/identity.py` — IdentityState, IdentityFieldOutput, IdentityFieldLayer
- `tacm/experiments/benchmark_tacprime_id001_identity_carried_structure_memory.py`
- `tacm/tests/test_identity_field.py`
- `tacm/tests/test_tacprime_id001_identity_integration.py`
- `tacm/docs/tacprime_identity_field_integration.md` (this file)

### Modified Files
- `tacm/tacm/config.py` — added `IdentityFieldConfig`, added `identity` field to `TACSMConfig`
- `tacm/tacm/model.py` — wired `IdentityFieldLayer` after backbone; extended `TACSMOutput` with 4 new slots; identity-biased struct_memory retrieval and memory write
- `tacm/tacm/router.py` — `StructureRouter.forward()` accepts `identity_context` + `identity_router_bias_scale`; added `identity_bias_proj` parameter
- `tacm/tacm/memory.py` — `StructureRecord` gains `identity_id`, `identity_embedding`; `retrieve()` / `retrieve_batch()` accept identity params
- `tacm/tacm/procedural_memory.py` — `ProcedureRecord` gains `identity_id`, `identity_embedding`; `retrieve()` / `write()` accept identity params
