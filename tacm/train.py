"""
TAC-SM Training Script

Usage:
  python train.py --config tacm-30m
  python train.py --config tacm-100m --resume checkpoints/tacm-100m/step_5000.pt
  python train.py --config tacm-30m --data_dir ./data/repair_corpus

Stages:
  Stage 1 — 30M — routing works, memory updates, toy repair tasks
  Stage 2 — 100M — repository repair, transfer benchmark, baselines
  Stage 3 — 150M — agent loop, autonomous repair, structure reuse
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add tacm package to path
sys.path.insert(0, str(Path(__file__).parent))

from tacm import TACSM, CONFIGS, TACSMConfig, tacm_30m


# ── Data ─────────────────────────────────────────────────────────────────────

class TokenDataset:
    """
    Simple token-level dataset.
    Expects a directory of .pt files, each containing a (T,) LongTensor.
    Falls back to a synthetic dataset if no data directory is given.
    """

    def __init__(
        self,
        data_dir:   Optional[str],
        seq_len:    int,
        vocab_size: int,
        n_synthetic: int = 10000,
    ):
        self.seq_len    = seq_len
        self.vocab_size = vocab_size

        if data_dir and Path(data_dir).exists():
            self.files = sorted(Path(data_dir).glob("*.pt"))
            self.synthetic = False
            print(f"Found {len(self.files)} data files in {data_dir}")
        else:
            self.files     = []
            self.synthetic = True
            self.n_synthetic = n_synthetic
            print(f"No data dir found — using synthetic random tokens (n={n_synthetic})")

    def __len__(self) -> int:
        return self.n_synthetic if self.synthetic else len(self.files) * 10

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.synthetic:
            ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        else:
            file_idx = idx % len(self.files)
            tokens   = torch.load(self.files[file_idx])
            start    = torch.randint(0, max(1, tokens.shape[0] - self.seq_len - 1), (1,)).item()
            ids      = tokens[start : start + self.seq_len + 1]
            if ids.shape[0] < self.seq_len + 1:
                ids = torch.cat([ids, torch.zeros(self.seq_len + 1 - ids.shape[0], dtype=torch.long)])

        input_ids = ids[:-1]
        labels    = ids[1:].clone()
        labels[labels == 0] = -100   # pad masking
        return {"input_ids": input_ids, "labels": labels}


def batch_iter(
    dataset: TokenDataset,
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
) -> Iterator[Dict[str, torch.Tensor]]:
    n       = len(dataset)
    indices = torch.randperm(n).tolist() if shuffle else list(range(n))

    batch_input = []
    batch_label = []
    for idx in indices:
        sample = dataset[idx]
        batch_input.append(sample["input_ids"])
        batch_label.append(sample["labels"])
        if len(batch_input) == batch_size:
            yield {
                "input_ids": torch.stack(batch_input).to(device),
                "labels":    torch.stack(batch_label).to(device),
            }
            batch_input, batch_label = [], []


# ── LR Scheduler ─────────────────────────────────────────────────────────────

def get_lr(step: int, cfg) -> float:
    """Cosine decay with linear warmup."""
    tc = cfg.training
    if step < tc.warmup_steps:
        return tc.lr * step / max(tc.warmup_steps, 1)
    ratio = (step - tc.warmup_steps) / max(tc.max_steps - tc.warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * ratio))
    return tc.min_lr + (tc.lr - tc.min_lr) * cosine


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: TACSM,
    optimizer: AdamW,
    step: int,
    loss: float,
    cfg: TACSMConfig,
    out_dir: str,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(out_dir) / f"step_{step:07d}.pt"
    torch.save({
        "step":            step,
        "loss":            loss,
        "config_name":     cfg.name,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "memory_stats":    model.struct_memory.stats(),
        "lifecycle":       model.lifecycle.summary(),
    }, ckpt_path)
    # Keep only last 3 checkpoints
    existing = sorted(Path(out_dir).glob("step_*.pt"))
    for old in existing[:-3]:
        old.unlink()
    print(f"  Saved checkpoint → {ckpt_path}")


def load_checkpoint(path: str, model: TACSM, optimizer: Optional[AdamW] = None) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    step = ckpt.get("step", 0)
    print(f"  Resumed from step {step}, loss={ckpt.get('loss', '?'):.4f}")
    return step


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(args):
    # ── Config ───────────────────────────────────────────────────────────────
    if args.config in CONFIGS:
        cfg = CONFIGS[args.config]()
    else:
        cfg = tacm_30m()
    print(f"\nTAC-SM Training: {cfg.name}")
    print(f"  Max steps:     {cfg.training.max_steps}")
    print(f"  Batch size:    {cfg.training.batch_size} × {cfg.training.grad_accum_steps} (accum)")
    print(f"  LR:            {cfg.training.lr}")
    print(f"  Dtype:         {cfg.training.dtype}")

    # Override from args
    if args.max_steps:
        cfg.training.max_steps = args.max_steps
    if args.output_dir:
        cfg.training.output_dir = args.output_dir

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("  Device: CPU")

    # ── dtype ─────────────────────────────────────────────────────────────────
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype     = dtype_map.get(cfg.training.dtype, torch.float32)
    use_amp   = dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"

    # ── Seed ─────────────────────────────────────────────────────────────────
    torch.manual_seed(cfg.training.seed)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TACSM(cfg).to(device)
    total = model.n_params()
    print(f"\nModel parameters: {total:,} ({total/1e6:.1f}M)")
    for name, n in model.param_breakdown().items():
        print(f"  {name:<20}: {n:>10,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # No weight decay on biases and norm layers
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.ndim >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.ndim < 2]
    optimizer = AdamW([
        {"params": decay_params,   "weight_decay": cfg.training.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=cfg.training.lr, betas=(0.9, 0.95), fused=(device.type == "cuda"))

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = TokenDataset(
        data_dir    = args.data_dir,
        seq_len     = cfg.transformer.max_seq_len,
        vocab_size  = cfg.transformer.vocab_size,
        n_synthetic = 50000,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    out_dir = cfg.training.output_dir + f"/{cfg.name}"
    log_path = Path(out_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path / "train_log.jsonl", "a")

    step          = start_step
    grad_step     = 0
    running_loss  = 0.0
    t0            = time.time()
    opt_lr        = cfg.training.lr

    data_gen = batch_iter(dataset, cfg.training.batch_size, device)

    print(f"\nStarting training from step {step}...")

    try:
        while step < cfg.training.max_steps:
            try:
                batch = next(data_gen)
            except StopIteration:
                data_gen = batch_iter(dataset, cfg.training.batch_size, device)
                batch    = next(data_gen)

            # Set LR
            opt_lr = get_lr(step, cfg)
            for g in optimizer.param_groups:
                g["lr"] = opt_lr

            # Forward
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                out = model(
                    input_ids = batch["input_ids"],
                    labels    = batch["labels"],
                )
                loss = out.loss / cfg.training.grad_accum_steps

            scaler.scale(loss).backward()
            grad_step += 1

            if grad_step % cfg.training.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                running_loss += (loss.item() * cfg.training.grad_accum_steps)

                # ── Logging ───────────────────────────────────────────────
                if step % cfg.training.log_every == 0:
                    avg_loss = running_loss / cfg.training.log_every
                    running_loss = 0.0
                    elapsed  = time.time() - t0
                    tok_sec  = (cfg.training.batch_size * cfg.training.log_every
                                * cfg.transformer.max_seq_len) / max(elapsed, 1)
                    mem_size = len(model.struct_memory)
                    ent      = model.moe.expert_entropy()
                    lc       = model.lifecycle.summary()

                    print(
                        f"step {step:>7} | loss {avg_loss:.4f} | lr {opt_lr:.2e} | "
                        f"tok/s {tok_sec:,.0f} | mem {mem_size} | "
                        f"ent {ent:.2f} | {lc}"
                    )

                    log_entry = {
                        "step": step, "loss": avg_loss, "lr": opt_lr,
                        "mem_size": mem_size, "expert_entropy": ent,
                        "lifecycle": lc,
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()
                    t0 = time.time()

                # ── Checkpoint ────────────────────────────────────────────
                if step % cfg.training.save_every == 0:
                    save_checkpoint(model, optimizer, step, avg_loss if 'avg_loss' in dir() else 0.0, cfg, out_dir)
                    model.moe.reset_stats()   # reset utilisation counters

    except KeyboardInterrupt:
        print("\nInterrupted — saving final checkpoint...")
        save_checkpoint(model, optimizer, step, 0.0, cfg, out_dir)

    log_file.close()
    print(f"\nTraining complete. Steps: {step}")
    mem_stats = model.struct_memory.stats()
    print(f"Structure Memory: {mem_stats}")
    print(f"Expert utilisation:\n{model.moe.utilisation_report()}")
    return model


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TAC-SM Training")
    parser.add_argument("--config",     type=str, default="tacm-30m",
                        choices=list(CONFIGS.keys()),
                        help="Model config preset")
    parser.add_argument("--data_dir",   type=str, default=None,
                        help="Directory of .pt token files (optional)")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps",  type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override checkpoint output directory")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
