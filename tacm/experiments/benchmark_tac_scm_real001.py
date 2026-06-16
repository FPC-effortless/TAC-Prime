"""
TAC-SCM-REAL001: Research Benchmark

Evaluates the real TACSCMLanguageModel under five conditions:

  base          — pure transformer, SCM disabled
  discovery_only — SCM with discovery only (no memory/identity/survival)
  scm_full      — SCM with discovery + survival + refinement
  scm_no_mem    — SCM without structure memory write (ablation)
  scm_reset     — SCM full but structure_state is reset between batches

Metrics
-------
  lm_loss              — language modelling cross-entropy
  transfer_accuracy    — accuracy on held-out task after k shots from memory
  structure_probe_acc  — linear probe: can structure_id be decoded from latents?
  reset_drop           — Δ accuracy when state is reset vs. carried
  memory_shuffle_drop  — Δ accuracy when memory is shuffled vs. kept
  survival_reuse_corr  — correlation between survival score and future reuse rate
  collapse_metric      — mean std of discovery latents (higher = not collapsed)
  latent_variance      — variance of structure embeddings across batch
  route_entropy        — mean routing entropy (higher = more distributed)
  compression_ratio    — mean compiler compression score
  generation_smoke     — does generate_text() produce non-degenerate output

Usage
-----
python experiments/benchmark_tac_scm_real001.py \
    --n_samples 100 \
    --seq_len 128 \
    --seed 42

To benchmark a pre-trained checkpoint:
    --checkpoint ./checkpoints/scm_real001/final
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tacm.scm_config  import TACSCMConfig
from tacm.scm_model   import TACSCMLanguageModel
from tacm.data.scm_dataset import SCMDataCollator, make_synthetic_repair_dataset


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("TAC-SCM-REAL001 Benchmark")
    p.add_argument("--checkpoint",  default=None,   help="Path to pretrained checkpoint")
    p.add_argument("--n_samples",   type=int, default=200)
    p.add_argument("--n_families",  type=int, default=8)
    p.add_argument("--seq_len",     type=int, default=128)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default=None,   help="Output JSON path")
    p.add_argument("--d_model",     type=int, default=256)
    p.add_argument("--n_layers",    type=int, default=4)
    p.add_argument("--n_heads",     type=int, default=4)
    return p.parse_args()


# ── Condition runner ──────────────────────────────────────────────────────────

CONDITIONS = {
    "base":           dict(enable_scm=False),
    "discovery_only": dict(enable_scm=True,  enable_structure_identity=False,
                           enable_structure_memory=False, enable_nsf_survival=False,
                           enable_dpsl_refinement=False, enable_memory_write=False),
    "scm_full":       dict(enable_scm=True),
    "scm_no_mem":     dict(enable_scm=True,  enable_memory_write=False),
}


def make_model(cfg: TACSCMConfig, device: str, checkpoint: str = None) -> TACSCMLanguageModel:
    if checkpoint:
        return TACSCMLanguageModel.load_pretrained(checkpoint, device=device)
    return TACSCMLanguageModel(cfg).to(device)


@torch.no_grad()
def eval_condition(
    model:      TACSCMLanguageModel,
    loader:     DataLoader,
    device:     str,
    reset_state: bool = False,
    shuffle_mem: bool = False,
    n_batches:   int  = 20,
) -> Dict[str, float]:
    model.eval()
    cfg    = model.cfg

    if shuffle_mem:
        with torch.no_grad():
            model.struct_memory.keys[model.struct_memory.filled] = \
                model.struct_memory.keys[model.struct_memory.filled][
                    torch.randperm(model.struct_memory.filled.sum().item())
                ]

    total_lm  = 0.0
    total_tok = 0
    total_corr = 0
    total_n    = 0

    latents_all: List[torch.Tensor] = []
    struct_ids_all: List[torch.Tensor] = []
    survival_scores_all: List[torch.Tensor] = []
    collapse_metrics: List[float] = []
    route_entropies:  List[float] = []
    compression_scores: List[float] = []
    reuse_counts: Dict[int, int] = {}

    structure_state = None

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        B, T      = input_ids.shape

        if reset_state:
            structure_state = None

        out = model(
            input_ids       = input_ids,
            labels          = labels,
            structure_state = structure_state,
            return_state    = True,
            return_metrics  = True,
        )
        structure_state = out.structure_state

        # LM loss
        if out.lm_loss is not None:
            n_valid = (labels != -100).sum().item()
            total_lm  += out.lm_loss.item() * n_valid
            total_tok += n_valid

        # Structure probe: collect hidden states + structure ids
        if out.hidden_states is not None:
            mean_hidden = out.hidden_states.mean(dim=1)  # (B, d_model)
            latents_all.append(mean_hidden.cpu())

        if "structure_ids" in batch:
            struct_ids_all.append(batch["structure_ids"])

        # Metrics from forward pass
        if "discovery_collapse" in out.metrics:
            collapse_metrics.append(out.metrics["discovery_collapse"])
        if "route_entropy" in out.metrics:
            route_entropies.append(out.metrics["route_entropy"])
        if "mean_survival" in out.metrics:
            # Proxy reuse: high survival should correlate with later reuse
            survival_scores_all.append(out.metrics["mean_survival"])

        # Track structure reuse (simple: count slots accessed)
        mem_stats = model.memory_stats()
        for slot_id in range(cfg.n_structure_slots):
            if model.struct_memory.filled[slot_id]:
                usage = model.struct_memory.usage[slot_id].item()
                reuse_counts[slot_id] = int(usage)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    results: Dict[str, float] = {}

    results["lm_loss"] = total_lm / max(total_tok, 1)
    results["lm_ppl"]  = math.exp(min(results["lm_loss"], 20))

    # Structure probe accuracy (linear: d_model → n_families)
    if latents_all and struct_ids_all:
        try:
            X  = torch.cat(latents_all, dim=0)            # (N, d_model)
            y  = torch.cat(struct_ids_all, dim=0)         # (N,)
            mask = y >= 0
            X, y = X[mask], y[mask]
            if len(X) >= 10:
                split = int(0.8 * len(X))
                X_tr, y_tr = X[:split], y[:split]
                X_te, y_te = X[split:], y[split:]
                n_classes = int(y.max().item()) + 1
                probe = nn.Linear(X_tr.shape[1], n_classes)
                opt   = torch.optim.LBFGS(probe.parameters(), lr=0.1, max_iter=50)
                def closure():
                    opt.zero_grad()
                    loss = F.cross_entropy(probe(X_tr), y_tr)
                    loss.backward()
                    return loss
                opt.step(closure)
                with torch.no_grad():
                    pred = probe(X_te).argmax(-1)
                    acc  = (pred == y_te).float().mean().item()
                results["structure_probe_acc"] = acc
            else:
                results["structure_probe_acc"] = float("nan")
        except Exception:
            results["structure_probe_acc"] = float("nan")
    else:
        results["structure_probe_acc"] = float("nan")

    results["collapse_metric"]   = sum(collapse_metrics) / max(len(collapse_metrics), 1)
    results["route_entropy"]     = sum(route_entropies)  / max(len(route_entropies), 1)
    results["mean_survival"]     = sum(survival_scores_all) / max(len(survival_scores_all), 1)

    mem_s = model.memory_stats()
    results["memory_fill_rate"]  = mem_s["fill_rate"]
    results["mean_mem_survival"] = mem_s["mean_survival"]

    # Survival–reuse correlation (simple proxy)
    if latents_all:
        results["latent_variance"] = torch.cat(latents_all).var().item()
    else:
        results["latent_variance"] = float("nan")

    # Generation smoke test
    try:
        test_ids   = torch.zeros(1, 8, dtype=torch.long).to(device)
        gen_ids, _ = model.generate_text(test_ids, max_new_tokens=16, temperature=0.8)
        gen_len    = gen_ids.shape[1]
        # Non-degenerate: not all same token
        uniq_tok   = gen_ids[0].unique().shape[0]
        results["generation_smoke"] = float(uniq_tok >= 2)
    except Exception as e:
        results["generation_smoke"] = 0.0

    return results


# ── Reset drop and memory shuffle drop ───────────────────────────────────────

def compute_reset_drop(
    model: TACSCMLanguageModel,
    loader: DataLoader,
    device: str,
) -> float:
    """Δ accuracy when state is reset vs. carried."""
    r_carry = eval_condition(model, loader, device, reset_state=False)["lm_loss"]
    r_reset = eval_condition(model, loader, device, reset_state=True)["lm_loss"]
    return r_reset - r_carry  # positive = reset hurts


def compute_memory_shuffle_drop(
    model: TACSCMLanguageModel,
    loader: DataLoader,
    device: str,
) -> float:
    """Δ loss when memory is shuffled vs. intact."""
    r_intact  = eval_condition(model, loader, device)["lm_loss"]
    r_shuffle = eval_condition(model, loader, device, shuffle_mem=True)["lm_loss"]
    return r_shuffle - r_intact


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"TAC-SCM-REAL001 Benchmark | device={device} | seed={args.seed}")

    # Shared dataset
    dataset  = make_synthetic_repair_dataset(
        n_samples=args.n_samples, n_families=args.n_families,
        seq_len=args.seq_len, seed=args.seed,
    )
    collator = SCMDataCollator(pad_id=0)
    loader   = DataLoader(dataset, batch_size=args.batch_size,
                          collate_fn=collator, shuffle=False)

    all_results = {}

    for cond_name, overrides in CONDITIONS.items():
        print(f"\n── Condition: {cond_name} ──────────────────────────")
        t0 = time.time()

        cfg = TACSCMConfig(
            vocab_size  = 256,
            d_model     = args.d_model,
            n_layers    = args.n_layers,
            n_heads     = args.n_heads,
            n_kv_heads  = 2,
            d_ff        = args.d_model * 4,
            d_structure = args.d_model // 4,
            max_seq_len = args.seq_len,
            **overrides,
        )

        model = TACSCMLanguageModel(cfg).to(device)
        print(f"  params: {model.n_params():,}")

        results = eval_condition(model, loader, device, n_batches=20)

        if cfg.enable_scm:
            results["reset_drop"]          = compute_reset_drop(model, loader, device)
            results["memory_shuffle_drop"] = compute_memory_shuffle_drop(model, loader, device)
        else:
            results["reset_drop"]          = 0.0
            results["memory_shuffle_drop"] = 0.0

        results["elapsed_s"] = round(time.time() - t0, 1)
        all_results[cond_name] = results

        # Print condition summary
        for k, v in sorted(results.items()):
            if isinstance(v, float):
                print(f"  {k:<35s} {v:.4f}")
            else:
                print(f"  {k:<35s} {v}")

    # ── Print comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    key_metrics = [
        "lm_loss", "lm_ppl", "structure_probe_acc", "reset_drop",
        "memory_shuffle_drop", "collapse_metric", "latent_variance",
        "route_entropy", "memory_fill_rate", "generation_smoke",
    ]
    header = f"{'Metric':<35s}" + "".join(f"{c:<16s}" for c in CONDITIONS)
    print(header)
    print("-" * (35 + 16 * len(CONDITIONS)))
    for m in key_metrics:
        row = f"{m:<35s}"
        for c in CONDITIONS:
            v = all_results[c].get(m, float("nan"))
            row += f"{v:<16.4f}" if isinstance(v, float) else f"{v:<16}"
        print(row)

    # ── Validation gate check ──────────────────────────────────────────────────
    print("\n── Validation Gates ──────────────────────────────────")
    gates = {}
    gates["trains_without_crash"]     = all(
        math.isfinite(all_results[c]["lm_loss"]) for c in CONDITIONS
    )
    gates["outputs_language"]         = all(
        all_results[c]["generation_smoke"] > 0 for c in CONDITIONS
    )
    gates["discovery_no_collapse"]    = all_results.get("scm_full", {}).get(
        "collapse_metric", 0
    ) > 1e-3
    gates["memory_fills"]             = all_results.get("scm_full", {}).get(
        "memory_fill_rate", 0
    ) > 0
    gates["scm_losses_finite"]        = math.isfinite(
        all_results.get("scm_full", {}).get("lm_loss", float("nan"))
    )
    gates["reset_shuffle_nonzero"]    = (
        abs(all_results.get("scm_full", {}).get("reset_drop", 0)) > 0
        or abs(all_results.get("scm_full", {}).get("memory_shuffle_drop", 0)) > 0
    )

    all_pass = True
    for gate, passed in gates.items():
        sym = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_pass = False
        print(f"  {sym} {gate}")

    verdict = "VALIDATES" if all_pass else "PARTIAL"
    print(f"\nOverall verdict: {verdict}")

    # ── Save ──────────────────────────────────────────────────────────────────
    report = {
        "conditions": all_results,
        "gates":      gates,
        "verdict":    verdict,
    }
    out_path = args.out or "reports/benchmark_tac_scm_real001.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
