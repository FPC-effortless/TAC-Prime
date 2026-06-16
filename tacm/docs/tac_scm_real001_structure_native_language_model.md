# TAC-SCM-REAL001: Structure-Native Language Model

**Research Name:** TAC-SCM-REAL001  
**Status:** Real model implementation — trainable, ablatable, saveable, loadable  
**Date:** 2026-06-15

---

## 1. Core Thesis

> Intelligence is structure acquisition and structure use.

TAC-SCM-REAL001 is a real model architecture that learns language while also:
- **discovering** reusable computational structures in its hidden representations
- **carrying** those structures across forward passes (structure state)
- **storing** high-fitness structures in an external memory bank
- **routing** tokens to active structures during inference
- **reusing** retrieved structures when processing new sequences
- **refining** structures based on feedback and survival pressure

This is **not** an AGI claim.  The correct claim is:

> "TAC-SCM-REAL001 is a real structure-native language model for testing whether
> reusable computational structures can be discovered, carried, preserved, routed,
> reused, and refined during language/task learning."

---

## 2. Architecture

```
Tokens (B, T)
    ↓
Token Embedding  (vocab_size × d_model)
    ↓
[For each layer i in 0..n_layers-1]:
    if i % scm_layer_interval == 0:  → IntegratedStructureLanguageBlock
    else:                            → TransformerBlock (plain)
    ↓
Final RMSNorm
    ↓
LM Head  (d_model → vocab_size)   [optionally tied to embedding]
    ↓
Logits (B, T, vocab_size)
```

### IntegratedStructureLanguageBlock

```
hidden_states
    ↓  TransformerBlock (GQA attention + SwiGLU FFN + RoPE)
    ↓  StructureDiscoveryLayer      → latent_state, structure_candidates
    ↓  StructureCompiler            → typed structure slots (concept/procedure/causal/trigger/transform)
    ↓  StructureIdentityFieldLayer  → route tokens to slots, read from slots, update state
    ↓  StructureMemory.read()       → retrieve long-term structures
    ↓  NSFSurvivalScorer            → score candidates (survival, write, refine gates)
    ↓  StructureMemory.write()      → persist survivors (training only)
    ↓  DPSLRefinementLayer          → refine embeddings (merge, specialize, gate)
    ↓  Language–Structure Fusion    → project (hidden || struct_summary) → hidden
    ↓  Memory context injection     → add memory context to hidden
    ↓
updated hidden_states
```

---

## 3. Module Descriptions

### 3.1 StructureDiscoveryLayer (`scm_discovery.py`)

**Purpose:** Discover structure in hidden representations without labels.

**Method:** JEPA (Joint Embedding Predictive Architecture) with VICReg regularization.

- **Online encoder:** `d_model → d_structure` (trainable, gradients flow through)
- **Target encoder:** EMA copy of online encoder (stop-gradient, EMA decay ≈ 0.996)
- **Predictor MLP:** predicts target latent at `t+offset` from online latent at `t`
- **Candidate extraction:** soft k-means over T latents → `n_structure_candidates` cluster centers

**Losses:**
- `L_prediction`: MSE between predictor(online(x_t)) and stop_grad(target(x_{t+offset}))
- `L_temporal`: MSE between consecutive latents (slow variation pressure)
- `L_variance`: VICReg spread loss — std of latents should be > 1
- `L_covariance`: VICReg decorrelation loss — off-diagonal covariance near 0

**Diagnostic:** `collapse_metric` = mean std of latents; drops to 0 if representations collapse.

### 3.2 StructureCompiler (`scm_compiler.py`)

**Purpose:** Transform raw latent structure candidates into typed structure objects.

**Typed projections per candidate:**

| Name | Purpose |
|---|---|
| `concept_center` | Concept-volume center (what the structure represents) |
| `concept_log_width` | Log uncertainty / spread of the concept volume |
| `procedure_emb` | How to execute this structure |
| `causal_emb` | Cause–effect pattern captured by the structure |
| `trigger_emb` | Activation condition for this structure |
| `transform_emb` | How this structure transforms state |
| `compression_score` | How compactly the structure is represented ∈ [0,1] |
| `structure_token` | Final compiled embedding (fusion of all above) |

### 3.3 StructureIdentityState (`scm_types.py`)

**Purpose:** Carry active computational structures across forward passes.

**Fields:**
- `slot_embeddings` (B, n_slots, d_structure) — current structure content per slot
- `slot_weights` (B, n_slots) — activation strength of each slot
- `route_history` (B, n_slots) — EMA of how often each slot was routed to
- `stability_scores` (B, n_slots) — cosine similarity between consecutive slot states
- `decision_memory` (B, n_slots, d_structure) — accumulated context per slot
- `step_count` — number of forward passes carried

**API:** `.to(device)`, `.detach()`, `.reset()`, `.zeros(batch, slots, d_str, device)`

**Distinction from `identity.py` (TAC-Prime-ID001):**
- Old `IdentityState`: carries identity IDs + stability scalars (symbolic)
- `StructureIdentityState`: carries actual structure embedding tensors (computational)

### 3.4 StructureIdentityFieldLayer (`scm_identity.py`)

**Purpose:** Route tokens to structure slots; read from and update active slots.

**Architecture:**
1. **Token→slot routing:** `token_query(hidden)` vs learnable `slot_key` → attention weights
2. **Read:** weighted sum over slot embeddings → structure readout → project to d_model
3. **Update:** slots attend over incoming structure candidates → gated EMA update
4. **Fusion:** `[hidden || structure_readout] → d_model`
5. **Stability tracking:** cosine similarity between consecutive slot states

**Auxiliary losses:**
- `identity_route_entropy`: maximize routing entropy (distributed routing)
- `identity_slot_stability`: minimize slot drift (slow slot evolution)

### 3.5 StructureMemory (`scm_memory.py`)

**Purpose:** Long-term external structure store.

**Key design:**
- Parameters (query/value/gate projections) learn *how to use* memory
- Buffer bank (keys, values, usage, age, survival) *stores* discovered structures
- Bank survives across training steps; parameters are trained
- Near-duplicate detection: cosine sim > 0.95 → in-place update instead of new slot

**Operations:**
- `read(query: (B, D))` → StructureMemoryOutput (top-k cosine retrieval)
- `write(embs, survival_scores, write_mask)` → survival-weighted slot replacement
- `step_decay()` → survival decay per training step
- `prune(threshold)` → evict weak structures
- `save_memory_state() / load_memory_state()` → persistence

### 3.6 NSFSurvivalScorer (`scm_survival.py`)

**Purpose:** Score structure embeddings for survival, memory write, and refinement.

**Survival formula:**
```
survival = 0.30 * reuse + 0.25 * transfer + 0.20 * robustness
         + 0.15 * compression - 0.05 * cost - 0.05 * interference
```
(50% heuristic, 50% learned — blended for stability)

**Gate outputs:**
- `write_gate`: ∝ survival (high survival → write to memory)
- `decay_gate`: ∝ 1 - survival (low survival → decay faster)
- `refine_gate`: ∝ survival (high survival → worth refining)
- `keep_mask`: bool, threshold at 0.4

**Loss:** NeuralSurvivalField (from `neural_survival_field.py`) for differentiable survival signal.

### 3.7 DPSLRefinementLayer (`scm_refinement.py`)

**Purpose:** Dynamically refine structure embeddings based on feedback and survival.

**Operations:**
1. **Gated update:** `gate = σ(survival_scale × gate_net([emb || feedback]))`
   `refined = emb + gate × δ`
2. **Merge:** pairs with cosine sim > 0.90 are merged (weighted by survival)
3. **Specialize:** diversity loss pushes distinct structures apart (via loss, not hard split)

**Losses:**
- `loss_div`: hinge on off-diagonal cosine similarity (push apart)
- `loss_drift`: MSE to prevent refinement from straying too far from original

### 3.8 TACSCMLanguageModel (`scm_model.py`)

**Purpose:** The top-level trainable model class.

**Forward signature:**
```python
forward(
    input_ids,
    labels=None,
    attention_mask=None,
    structure_state=None,     # StructureIdentityState
    memory_state=None,        # dict snapshot (for persistent memory)
    feedback=None,
    return_state=True,
    return_metrics=True,
) → TACSCMOutput
```

**Total loss:**
```
loss = lm_loss + Σ_k w_k × auxiliary_loss_k
```

**Generation:** `generate_text(input_ids, max_new_tokens, temperature, top_k, carry_state)` — autoregressive with structure state carry across steps.

**Persistence:** `save_pretrained(dir)` / `load_pretrained(dir)` — saves weights + memory bank + config.

---

## 4. Important Conceptual Distinctions

| Component | What it stores | Why |
|---|---|---|
| **Parameters** | General learned abilities | Gradient-trained, persistent across all batches |
| **StructureIdentityState** | Active computational structures for this batch | Carried across calls, detached between batches by default |
| **StructureMemory** | Discovered reusable structures from training | Persistent across training steps; survival-gated |
| **Attention** | Which tokens are relevant to each other | Standard transformer mechanism |
| **NSF** | Which structures are worth keeping | Decides survival of discovered structures |
| **DPSL** | How to improve stored structures | Refines structures based on feedback/survival |

---

## 5. Training

### Quick start (synthetic data)

```bash
cd tacm
python experiments/train_tac_scm_real001.py \
    --dataset synthetic \
    --out_dir ./checkpoints/scm_small \
    --steps 2000 \
    --batch_size 4 \
    --seq_len 128 \
    --d_model 256 \
    --n_layers 4
```

### Ablation: pure transformer baseline

```bash
python experiments/train_tac_scm_real001.py \
    --enable_scm false \
    --out_dir ./checkpoints/base_transformer
```

### Ablation: discovery only

```bash
python experiments/train_tac_scm_real001.py \
    --enable_scm true \
    --enable_structure_identity false \
    --enable_memory_write false \
    --enable_survival false \
    --enable_refinement false
```

### Full config (all modules)

```bash
python experiments/train_tac_scm_real001.py \
    --enable_scm true \
    --enable_discovery true \
    --enable_survival true \
    --enable_refinement true \
    --enable_memory_write true \
    --d_model 512 --n_layers 8 --d_structure 128
```

---

## 6. Generation

```python
from tacm.scm_model import TACSCMLanguageModel

model = TACSCMLanguageModel.load_pretrained("./checkpoints/scm_small/final")
model.eval()

import torch
prompt = torch.tensor([[1, 2, 3, 4, 5]])
gen_ids, final_state = model.generate_text(
    prompt, max_new_tokens=64, temperature=0.8, carry_state=True
)
# Continue from same state:
gen_ids2, _ = model.generate_text(
    gen_ids, max_new_tokens=32, structure_state=final_state
)
```

---

## 7. Benchmark

```bash
cd tacm
python experiments/benchmark_tac_scm_real001.py \
    --n_samples 200 \
    --seq_len 128 \
    --seed 42 \
    --out reports/benchmark_tac_scm_real001.json
```

Evaluates five conditions:
- `base`: pure transformer (enable_scm=False)
- `discovery_only`: SCM with only discovery
- `scm_full`: full SCM pipeline
- `scm_no_mem`: SCM without memory write
- `scm_reset`: SCM full but state reset between batches

---

## 8. Tests

```bash
cd tacm
pytest tests_py/test_tac_scm_real001_model.py -v
```

51 tests covering all modules, forward pass shapes, loss finiteness, state carry, memory I/O, save/load, and generation.

---

## 9. Validation Gates

### Minimum Real Implementation Gate

| Gate | Description |
|---|---|
| trains_without_crash | model.forward() with labels returns finite loss |
| outputs_language | generate_text() produces non-degenerate token sequences |
| structure_state_carries | step_count increments across calls |
| memory_write_read | write() increases fill_rate; read() returns context |
| scm_auxiliary_losses_finite | all auxiliary losses are finite scalars |
| discovery_no_collapse | collapse_metric > 0 (latents don't degenerate) |
| save_load_roundtrip | logits identical after save/load |

### Research Validation Gate

| Gate | Description |
|---|---|
| reset_drop_nonzero | resetting state changes LM loss (state matters) |
| memory_shuffle_drop_nonzero | shuffling memory changes LM loss (memory matters) |
| scm_beats_base | SCM full LM loss ≤ base LM loss after training |
| survival_reuse_correlation | structures with high survival are retrieved more often |
| discovery_latents_structured | structure probe accuracy > chance (latents encode task structure) |

### Strong Validation Gate (after real training)

| Gate | Description |
|---|---|
| scm_transfer_gt_baseline | SCM transfer accuracy > pure trace baseline |
| reset_hurts_transfer | resetting state lowers transfer accuracy |
| shuffled_mem_hurts_transfer | shuffling memory lowers transfer accuracy |
| survival_predicts_reuse | correlation(survival, future_reuse) > 0.5 |
| discovery_recovers_structure_id | structure probe accuracy > 1/n_families + 0.1 |

---

## 10. What Is Real vs. Still Experimental

### Real (implemented, trainable)
- Full model forward pass with gradient flow
- JEPA-inspired structure discovery with VICReg regularization
- EMA target encoder with configurable decay
- Typed structure compiler (6 aspects per structure)
- Stateful identity field with token routing and EMA slot updates
- Cosine-similarity memory bank with survival-weighted replacement
- NSF survival scorer with learnable + heuristic blend
- DPSL refinement with merge + diversity loss
- LM head with tied embedding option
- Autoregressive generation with state carry
- save_pretrained / load_pretrained with memory bank persistence

### Experimental / Stub-Level
- **Specialization in DPSL**: currently a diversity loss (not hard structure splitting)
- **Cross-batch memory write**: memory survives but structure_state is detached between batches
- **Transfer task evaluation**: requires external task datasets; smoke test only in benchmark
- **Structure labels**: structure_probe accuracy is a proxy (unsupervised latent quality)
- **Feedback conditioning**: works but feedback source is a proxy (identity readout mean)

---

## 11. Known Limitations

1. **No tokenizer**: The training script uses a placeholder char→int tokenizer for smoke tests. Production use requires a real tokenizer (e.g. BPE via tiktoken or sentencepiece).
2. **Memory bank size**: Fixed at `n_structure_slots`. No dynamic growth. Oldest/weakest slots are evicted.
3. **Gradient checkpointing**: Enabled by default for SCM blocks; may slow training slightly.
4. **Single device**: No multi-GPU / FSDP support in v1.
5. **Discovery target not strictly stop-gradient during compile**: The target encoder's parameters are stopped from gradients, but the targets themselves flow through the compiler objective. This is intentional (similar to VICReg) but differs from strict JEPA.

---

## 12. File Index

| File | Purpose |
|---|---|
| `tacm/scm_types.py` | All dataclasses (StructureObject, StructureBatch, StructureIdentityState, TACSCMOutput, …) |
| `tacm/scm_config.py` | TACSCMConfig with ablation presets |
| `tacm/scm_discovery.py` | JEPA StructureDiscoveryLayer |
| `tacm/scm_compiler.py` | StructureCompiler (6 typed heads + fusion) |
| `tacm/scm_identity.py` | StructureIdentityState + StructureIdentityFieldLayer |
| `tacm/scm_memory.py` | StructureMemory (read/write/persist) |
| `tacm/scm_survival.py` | NSFSurvivalScorer |
| `tacm/scm_refinement.py` | DPSLRefinementLayer |
| `tacm/scm_block.py` | IntegratedStructureLanguageBlock |
| `tacm/scm_model.py` | TACSCMLanguageModel (top-level) |
| `tacm/data/scm_dataset.py` | SCMDataset, SCMSample, SCMDataCollator, synthetic repair |
| `experiments/train_tac_scm_real001.py` | Training script with CLI |
| `experiments/benchmark_tac_scm_real001.py` | Five-condition research benchmark |
| `tests_py/test_tac_scm_real001_model.py` | Full test suite |
| `docs/tac_scm_real001_structure_native_language_model.md` | This document |
