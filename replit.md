# TAC-SM: Token–Algorithm–Coherence with Structure Memory

A research-grade PyTorch model that learns reusable computational structures and transfers them across tasks. Trainable at 30M–150M parameters on Kaggle-scale hardware.

## Run & Operate

```bash
# Quick forward-pass test
cd tacm && python3 -c "from tacm import TACSM, tacm_30m; import torch; m=TACSM(tacm_30m()); print(m.n_params()/1e6,'M')"

# Stage 1: Train 30M on synthetic data
cd tacm && python3 train.py --config tacm-30m

# Stage 2: Train 100M
cd tacm && python3 train.py --config tacm-100m --data_dir ./data/corpus

# Evaluate vs. baselines
cd tacm && python3 evaluate.py --config tacm-30m --checkpoint checkpoints/.../step_N.pt --baselines

# Full benchmark table
cd tacm && python3 scripts/benchmark.py --config tacm-30m

# Prepare data from source repos
cd tacm && python3 scripts/prepare_data.py --input_dir ./raw --output_dir ./data/corpus
# Synthetic data (no repos needed):
cd tacm && python3 scripts/prepare_data.py --synthetic --n 10000 --output_dir ./data/synthetic
```

## Stack

- Python 3.11, PyTorch 2.x
- Decoder-only transformer with RoPE + GQA + Flash Attention (SDPA)
- DeepSeek-style MoE (Shared Expert + N Specialists)
- Structure Memory (vector store with READ/WRITE/UPDATE/PRUNE)
- Procedural Memory (ordered step sequences)
- Neural Survival Field + Lifecycle Engine
- Verifier / Reward Head
- Multi-token prediction (LM + future tokens + actions)

## Where things live

- `tacm/tacm/` — all model components (one file per component)
- `tacm/tacm/config.py` — all hyperparameters + preset configs (30M/100M/150M)
- `tacm/tacm/model.py` — full TACSM model wiring everything together
- `tacm/tacm/memory.py` — Structure Memory (primary innovation)
- `tacm/tacm/agent.py` — Repository Repair Agent Loop
- `tacm/train.py` — training script (3-stage Kaggle training)
- `tacm/evaluate.py` — evaluation + baseline comparison
- `tacm/scripts/benchmark.py` — head-to-head benchmark table
- `tacm/configs/` — YAML configs for 30M and 100M presets
- `tacm/README.md` — full architecture docs

## Architecture decisions

- **Component separation**: Each of the 14 components is its own file so they can be trained, tested, and swapped independently.
- **Two-level routing**: Tokens route through Family → Expert (never directly), preventing expert collapse and enabling structured specialisation.
- **Structure Memory stores embeddings, not text**: Structures carry success/transfer/survival scores and are pruned by the Lifecycle Engine.
- **Verifier as reward bridge**: The verifier head produces a reward signal that writes to Structure Memory after each repair attempt without needing external RL infrastructure.
- **Multi-token prediction**: 4-8 future tokens + action prediction trained jointly with LM loss, encouraging the model to plan ahead.
- **Vocab-size gotcha**: LM heads dominate parameter count at large vocab sizes. For strict 30M total, use `vocab_size=4096` and `n_future_tokens=2`.

## Product

TAC-SM is a research artefact, not a product. The goal is to validate the claim: *"TAC-SM learns reusable computational structures that transfer across repositories and tasks better than a same-size Transformer or MoE baseline."*

## User preferences

_Populate as you build._

## Gotchas

- `python train.py` uses synthetic data by default if `--data_dir` is not provided. Safe to run immediately.
- `n_future_tokens=8` adds 8× vocab_size parameters to multi_token heads — reduce this for tight parameter budgets.
- Structure Memory writes only when `success_score >= write_threshold` (default 0.6) — set verifier labels during training to populate memory.
- Always run from the `tacm/` directory so relative imports resolve correctly.

## Pointers

- See `tacm/README.md` for the full architecture diagram, component table, and benchmark instructions.
- See the `pnpm-workspace` skill for the JS/TS monorepo structure around this package.
