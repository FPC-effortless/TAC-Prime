# TAC-SM: Token–Algorithm–Coherence with Structure Memory

Research-grade model that learns **reusable computational structures** and transfers them across tasks. Trainable at 30M–150M parameters on Kaggle-scale hardware.

---

## Core Hypothesis

Standard transformers optimise token prediction. TAC-SM additionally learns:

| What it learns | What it remembers |
|---|---|
| Reusable structures | What worked |
| Structure families | Where it worked |
| Executable specialists | Why it worked |
| Repair strategies | Whether it survived perturbation |
| Procedural patterns | Whether it transferred to new tasks |

**Central claim:** TAC-SM learns reusable computational structures that transfer across repositories and tasks better than a same-size Transformer or MoE baseline.

---

## Architecture

```
Input (tokens / repo context / bug reports / tool outputs / memory retrievals)
  ↓
TransformerBackbone        — decoder-only, RoPE, GQA, Flash Attention, bf16
  ↓
AdaptiveConceptVolume      — volume representations (center, variance, confidence)
  ↓
StructureRouter            — 2-level: ConceptVolume → Family → Specialist
  ↓
SharedExpert + MoE         — DeepSeek-style; 1 shared + N specialist experts
  ↓
StructureMemory            — stores/retrieves structures (not text)
  ↓
Verifier / Reward Head     — predicts success probability + failure class
  ↓
MultiTokenPrediction       — next-token + 4-8 future tokens + actions
  ↓
Patch / Plan / Answer / Action
```

---

## Component Summary

| Component | File | Purpose |
|---|---|---|
| Transformer Backbone | `tacm/backbone.py` | RoPE + GQA + Flash Attn + gradient ckpt |
| Adaptive Concept Volume | `tacm/concept_volume.py` | Volume representations with 4 losses |
| Two-Level Router | `tacm/router.py` | Family → Expert routing |
| MoE Experts | `tacm/experts.py` | Shared + specialist experts, utilisation tracking |
| Structure Memory | `tacm/memory.py` | READ / WRITE / UPDATE / PRUNE operations |
| Procedural Memory | `tacm/procedural_memory.py` | Ordered procedure storage + retrieval |
| Neural Survival Field | `tacm/survival.py` | Survival scores + lifecycle tracking |
| Verifier Head | `tacm/verifier.py` | Success prediction + failure classification |
| Multi-Token Prediction | `tacm/multi_token.py` | LM + multi-token + action heads |
| Full Model | `tacm/model.py` | Wires all components |
| Agent Loop | `tacm/agent.py` | Repository repair agent |
| Losses | `tacm/losses.py` | 9-component total loss |
| Evaluation | `tacm/evaluation.py` | All benchmark metrics + baseline comparison |

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Check model size and run forward pass
python3 -c "
from tacm import TACSM, tacm_30m
import torch
cfg   = tacm_30m()
model = TACSM(cfg)
print(f'Parameters: {model.n_params()/1e6:.1f}M')
ids   = torch.randint(1, 32000, (1, 128))
out   = model(ids)
print('loss:', out.loss.item())
"

# Stage 1: Train 30M model (synthetic data)
python train.py --config tacm-30m

# Stage 1: Train on your own data
python train.py --config tacm-30m --data_dir ./data/repair_corpus

# Stage 2: Train 100M model
python train.py --config tacm-100m

# Evaluate vs. baselines
python evaluate.py --config tacm-30m --checkpoint checkpoints/tacm-30m/step_50000.pt --baselines

# Full benchmark
python scripts/benchmark.py --config tacm-30m --checkpoint ...

# Prepare data from source files
python scripts/prepare_data.py --input_dir ./raw_repos --output_dir ./data/corpus
# Or generate synthetic data for testing:
python scripts/prepare_data.py --synthetic --n 10000 --output_dir ./data/synthetic
```

---

## TAC-PSM Progression: Procedural Memory Validation

TAC-PSM is a five-stage controlled benchmark that validates the procedural memory mechanism independently of the neural backbone. All five studies pass on synthetic repair tasks.

| Study | Question | Gates | Result |
|---|---|---|---|
| PSM-001: Memory | Can TAC remember a reusable procedure? | 7/7 | ✓ |
| PSM-002: Transfer | Can a procedure cross family boundaries? | 5/5 | ✓ |
| PSM-003: Lifecycle | Can procedures evolve over time? | 5/5 | ✓ |
| PSM-004: Survival | Why do some procedures survive? | 5/5 | ✓ |
| PSM-005: Discovery | Can TAC invent procedures? | 5/5 | ✓ |

**Total: 27/27 gates, 5 seeds each.**

```bash
# Run the full progression
cd tacm && python3 scripts/run_psm_progression.py --seeds 0 1 2 3 4
# Or as a module
cd tacm && python -m tacm.scripts.run_psm_progression
```

Expected output: `PROGRESSION VALIDATED — 5/5 studies, 27/27 gates`

> **Framing:** This is a controlled synthetic benchmark. It validates the procedural memory *mechanism*, not full production coding ability. See `docs/tac_psm_limitations.md`.

### PSM Files

```
tacm/psm001/   store.py, retrieval.py, update.py, records.py, benchmark_families.py
tacm/psm002/   transfer.py, metrics.py
tacm/psm003/   lifecycle.py, operations.py
tacm/psm004/   survival.py, perturbation.py
tacm/psm005/   discovery.py, verification.py
tacm/neural_survival_field.py   — differentiable NSF loss for TACSM training
scripts/       benchmark_tac_psm001..005.py, run_psm_progression.py
reports/       TAC_PSM_Progression_Report.md, psm_progression_summary.json
docs/          tac_psm_progression_report.md, tac_psm_limitations.md,
               tac_psm_next_stage.md, tac_psm_investor_summary.md
```

---

## Parameter Counts

The backbone + auxiliary components control model size. The LM head dominates at large vocab sizes — reduce `n_future_tokens` or use a smaller vocab for tighter budgets.

| Preset | d_model | Layers | Heads | KV-heads | FFN | Experts | Backbone params |
|---|---|---|---|---|---|---|---|
| `tacm-30m`  | 512  | 8  | 8  | 2  | 2048 | 8  | ~46M |
| `tacm-100m` | 768  | 12 | 12 | 4  | 3072 | 16 | ~86M |
| `tacm-150m` | 1024 | 16 | 16 | 4  | 4096 | 32 | ~150M |

> **Tip:** For strict 30M total parameters, set `vocab_size=4096` and `n_future_tokens=2`.

---

## Training Stages

### Stage 1 — 30M — Kaggle T4
Goal: routing works, memory updates, toy repair tasks.
```bash
python train.py --config tacm-30m
```
Verify: loss decreases, expert entropy > 1.0, memory grows.

### Stage 2 — 100M — Kaggle A100
Goal: repository repair accuracy, transfer benchmark, beat baseline.
```bash
python train.py --config tacm-100m --data_dir ./data/repair_corpus
```
Verify: repair_accuracy > Vanilla-Transformer; transfer_accuracy > 0.4.

### Stage 3 — Agent Loop
Goal: autonomous repair, structure reuse, memory growth.
```python
from tacm import TACSM, tacm_150m, RepositoryRepairAgent, BugReport

model = TACSM(tacm_150m())
agent = RepositoryRepairAgent(model, read_file=..., write_file=..., run_tests=..., ...)
trace = agent.repair(BugReport(
    repo_path="./my_repo",
    description="ImportError: No module named 'requests'",
    task_type="PythonImportError",
    family_hint="CodeRepair",
    affected_files=["src/main.py"],
))
print(f"Success: {trace.success}")
print(f"Structures written: {trace.structure_ids}")
```

---

## Training Objectives

```
L_total =
    1.0 * L_next_token          # standard LM loss
  + 0.5 * L_multi_token         # 4-8 future token prediction
  + 0.2 * L_volume              # concept volume consistency/separation/hierarchy/temporal
  + 0.3 * L_family_route        # family routing cross-entropy + entropy reg
  + 0.3 * L_expert_route        # expert routing + load balance
  + 0.5 * L_structure_memory    # memory triplet (anchor/positive/negative)
  + 0.4 * L_transfer            # transfer similarity
  + 0.2 * L_survival            # survival robustness under noise
  + 0.5 * L_verifier            # success prediction + failure classification
```
All weights configurable in `configs/tacm_30m.yaml`.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Repair Accuracy | Verifier success_prob vs. actual test pass labels |
| Transfer Accuracy | Source-solved tasks solved via transferred structures |
| Structure Reuse Rate | Fraction of retrievals contributing to repair success |
| Memory Retention | Fraction of written structures surviving after N steps |
| Attack Recovery | Structures above survival threshold after embedding perturbation |
| Expert Entropy | Routing distribution entropy (higher = more balanced) |
| Verifier Accuracy | Binary accuracy of success prediction head |

---

## Structure Memory Record

```python
{
  structure_id:   str,
  family_id:      int,    # CodeRepair, MathProcedure, Verification, ...
  expert_id:      int,
  task_type:      str,
  embedding:      Tensor, # (embedding_dim,)
  success_score:  float,  # what worked
  transfer_score: float,  # whether it transferred
  survival_score: float,  # whether it survived perturbation
  usage_count:    int,    # reuse frequency
  timestamp:      float,
}
```

Structure lifecycle: `NEW → ACTIVE → SPECIALIZED → TRANSFERRED → MERGED → DECAYING → REMOVED`

---

## Structure Families

| ID | Name | Specialists |
|---|---|---|
| 0 | CodeRepair | Python, JS, Go, Java repair |
| 1 | MathProcedure | Algebra, calculus, proof |
| 2 | Verification | Test, assertion, proof check |
| 3 | Planning | Task decomposition, roadmap |
| 4 | Retrieval | Context lookup, doc search |
| 5 | MemoryUpdate | Write/update/prune memory |
| 6 | Abstraction | Generalise pattern → family |
| 7 | Transfer | Cross-task, cross-repo transfer |

---

## Files

```
tacm/
├── tacm/
│   ├── __init__.py            # Package exports
│   ├── config.py              # All hyperparameters + 30M/100M/150M presets
│   ├── backbone.py            # Transformer: RoPE, GQA, Flash Attn, SwiGLU
│   ├── concept_volume.py      # Adaptive Concept Volume Layer + 4 losses
│   ├── router.py              # Two-level router + routing losses
│   ├── experts.py             # Shared expert + MoE + utilisation tracking
│   ├── memory.py              # Structure Memory (READ/WRITE/UPDATE/PRUNE)
│   ├── procedural_memory.py   # Procedural Memory Extension (TAC-S200)
│   ├── survival.py            # Neural Survival Field + Lifecycle Engine
│   ├── verifier.py            # Verifier head + reward bridge
│   ├── multi_token.py         # LM + multi-token + action prediction
│   ├── losses.py              # StructureMemory, Transfer, Survival losses
│   ├── model.py               # Full TACSM model
│   ├── agent.py               # Repository Repair Agent Loop
│   └── evaluation.py          # All benchmark metrics + baseline comparisons
├── train.py                   # Training script (all 3 stages)
├── evaluate.py                # Evaluation script + baseline comparison
├── scripts/
│   ├── prepare_data.py        # Data preparation from repos / bug reports
│   └── benchmark.py           # Full head-to-head benchmark table
├── configs/
│   ├── tacm_30m.yaml          # Stage 1 config
│   └── tacm_100m.yaml         # Stage 2 config
├── requirements.txt
└── pyproject.toml
```
