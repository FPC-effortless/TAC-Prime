---
name: TAC-SM architecture
description: Research ML model in tacm/; PyTorch 2.x; key design decisions and gotchas.
---

TAC-SM lives entirely in `tacm/` (not an artifact — it's a Python package, not a web app).

**Why:**
User requested a research-grade ML model trainable on Kaggle hardware, not a web product.

**How to apply:**
All future ML/research work for this project goes in `tacm/`. The JS/TS monorepo (`artifacts/`, `lib/`) is unrelated.

## Key non-obvious decisions

- LM heads (multi_token) dominate param count at large vocab. For strict 30M total params: `vocab_size=4096`, `n_future_tokens=2`.
- Structure Memory writes only when `success_score >= write_threshold` (default 0.6). Need real verifier labels during training to populate memory; otherwise it stays empty.
- Two-level routing (Family → Expert) prevents expert collapse. Never add direct expert routing.
- Survival Field's `RobustnessProbe` runs no-grad — it's a monitoring heuristic, not a trainable component.
- Always run scripts from `tacm/` directory so `sys.path.insert(0, '.')` resolves imports correctly.

## Confirmed working (live test)

- Python 3.11 installed
- PyTorch 2.12.0+cpu installed
- All 17 Python files pass syntax check
- Forward pass (B=2, T=64) runs end-to-end
- Memory read/write/retrieval works
- Greedy generation works
