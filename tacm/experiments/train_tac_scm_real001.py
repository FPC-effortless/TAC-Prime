"""
TAC-SCM-REAL001: Training Script

Usage
-----
python experiments/train_tac_scm_real001.py \
    --dataset text \
    --out_dir ./checkpoints/scm_base \
    --steps 10000 \
    --batch_size 4 \
    --seq_len 512 \
    --lr 3e-4 \
    --d_model 512 \
    --n_layers 8

To train with SCM fully disabled (pure transformer baseline):
    --enable_scm false

To train with only discovery (ablation):
    --enable_scm true --enable_structure_memory false \
    --enable_structure_identity false --enable_nsf_survival false \
    --enable_dpsl_refinement false
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tacm.scm_config import TACSCMConfig
from tacm.scm_model  import TACSCMLanguageModel
from tacm.data.scm_dataset import (
    SCMDataset, SCMDataCollator, make_synthetic_repair_dataset
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TAC-SCM-REAL001")

    # Data
    p.add_argument("--dataset",     default="synthetic",
                   choices=["synthetic", "text", "jsonl", "repair"],
                   help="Data source type")
    p.add_argument("--data_path",   default=None,  help="Path to data file/dir")
    p.add_argument("--out_dir",     default="./checkpoints/scm_real001")
    p.add_argument("--resume",      default=None,  help="Path to checkpoint to resume from")

    # Training
    p.add_argument("--steps",       type=int,   default=5000)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--seq_len",     type=int,   default=256)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--min_lr",      type=float, default=3e-5)
    p.add_argument("--warmup_steps",type=int,   default=200)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--weight_decay",type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--eval_interval",  type=int, default=500)
    p.add_argument("--save_interval",  type=int, default=1000)
    p.add_argument("--log_interval",   type=int, default=50)

    # Model architecture
    p.add_argument("--d_model",          type=int, default=256)
    p.add_argument("--n_layers",         type=int, default=4)
    p.add_argument("--n_heads",          type=int, default=4)
    p.add_argument("--n_kv_heads",       type=int, default=2)
    p.add_argument("--d_ff",             type=int, default=1024)
    p.add_argument("--d_structure",      type=int, default=64)
    p.add_argument("--n_structure_slots",type=int, default=128)
    p.add_argument("--n_identity_slots", type=int, default=8)
    p.add_argument("--scm_layer_interval", type=int, default=2)
    p.add_argument("--vocab_size",       type=int, default=256)

    # SCM toggles
    p.add_argument("--enable_scm",          default="true")
    p.add_argument("--enable_discovery",    default="true")
    p.add_argument("--enable_survival",     default="true")
    p.add_argument("--enable_refinement",   default="true")
    p.add_argument("--enable_memory_write", default="true")

    return p.parse_args()


def _bool(s: str) -> bool:
    return s.lower() in ("true", "1", "yes")


# ── Scheduler ─────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    import math
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = TACSCMConfig(
        vocab_size             = args.vocab_size,
        d_model                = args.d_model,
        n_layers               = args.n_layers,
        n_heads                = args.n_heads,
        n_kv_heads             = args.n_kv_heads,
        d_ff                   = args.d_ff,
        d_structure            = args.d_structure,
        n_structure_slots      = args.n_structure_slots,
        n_identity_slots       = args.n_identity_slots,
        scm_layer_interval     = args.scm_layer_interval,
        max_seq_len            = args.seq_len,
        enable_scm             = _bool(args.enable_scm),
        enable_structure_discovery = _bool(args.enable_discovery),
        enable_nsf_survival    = _bool(args.enable_survival),
        enable_dpsl_refinement = _bool(args.enable_refinement),
        enable_memory_write    = _bool(args.enable_memory_write),
    )
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = TACSCMLanguageModel.load_pretrained(args.resume, device=device)
    else:
        model = TACSCMLanguageModel(cfg).to(device)

    n_params = model.n_params()
    print(f"Parameters: {n_params:,}  ({n_params/1e6:.1f}M)")
    print(f"Breakdown: {model.param_breakdown()}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dataset == "synthetic":
        dataset = make_synthetic_repair_dataset(
            n_samples  = max(args.steps * args.batch_size, 1000),
            n_families = 8,
            seq_len    = args.seq_len,
            seed       = args.seed,
        )
    elif args.dataset == "text" and args.data_path:
        class _FakeTokenizer:
            pad_token_id = 0
            def encode(self, text, **kw): return [ord(c) % 256 for c in text]
        dataset = SCMDataset.from_text_file(
            args.data_path, _FakeTokenizer(), seq_len=args.seq_len
        )
    elif args.dataset == "jsonl" and args.data_path:
        class _FakeTokenizer:
            pad_token_id = 0
            def encode(self, text, **kw): return [ord(c) % 256 for c in text]
        dataset = SCMDataset.from_jsonl(
            args.data_path, _FakeTokenizer(), seq_len=args.seq_len
        )
    else:
        print(f"No data_path provided for {args.dataset}; using synthetic data.")
        dataset = make_synthetic_repair_dataset(
            n_samples=max(args.steps * args.batch_size, 1000),
            seq_len=args.seq_len, seed=args.seed,
        )

    collator = SCMDataCollator(pad_id=0)
    loader   = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, drop_last=True,
    )
    loader_iter = iter(loader)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    metrics_history = []
    step = 0
    t0   = time.time()

    print(f"\nTraining TAC-SCM-REAL001 for {args.steps} steps...")

    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        # LR schedule
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()

        out = model(
            input_ids       = input_ids,
            labels          = labels,
            return_state    = False,
            return_metrics  = (step % args.log_interval == 0),
        )

        if out.loss is None or not out.loss.requires_grad:
            step += 1
            continue

        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            aux_sum = sum(v.item() for v in out.auxiliary_losses.values())
            lm_val  = out.lm_loss.item() if out.lm_loss is not None else float("nan")
            total_val = out.loss.item()
            mem_stats = model.memory_stats()

            row = {
                "step":       step,
                "lm_loss":    round(lm_val, 4),
                "aux_loss":   round(aux_sum, 4),
                "total_loss": round(total_val, 4),
                "lr":         round(lr, 6),
                "mem_filled": mem_stats["n_filled"],
                "elapsed_s":  round(elapsed, 1),
            }
            row.update({f"metric_{k}": round(v, 4) for k, v in out.metrics.items()})
            metrics_history.append(row)

            print(
                f"step={step:5d}  lm={lm_val:.4f}  aux={aux_sum:.4f}"
                f"  total={total_val:.4f}  lr={lr:.2e}"
                f"  mem={mem_stats['n_filled']}/{cfg.n_structure_slots}"
                f"  t={elapsed:.0f}s"
            )

        if step % args.save_interval == 0 and step > 0:
            ckpt = out_dir / f"step_{step:06d}"
            model.save_pretrained(str(ckpt))
            print(f"  → Checkpoint saved: {ckpt}")

        step += 1

    # ── Final save ────────────────────────────────────────────────────────────
    model.save_pretrained(str(out_dir / "final"))
    print(f"\nFinal checkpoint saved: {out_dir}/final")

    # Save metrics
    metrics_path = out_dir / "metrics.jsonl"
    with metrics_path.open("w") as f:
        for row in metrics_history:
            f.write(json.dumps(row) + "\n")
    print(f"Metrics saved: {metrics_path}")

    final_lm = metrics_history[-1]["lm_loss"] if metrics_history else float("nan")
    print(f"\nDone. Final LM loss: {final_lm:.4f}")


if __name__ == "__main__":
    main()
