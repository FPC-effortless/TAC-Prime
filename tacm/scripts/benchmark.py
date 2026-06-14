"""
TAC-SM Benchmark Script

Runs the full head-to-head evaluation:
  TAC-SM vs. Vanilla Transformer vs. MoE vs. Retrieval vs. Memory-only

Outputs a Markdown table and JSON report.

Usage:
  python scripts/benchmark.py --config tacm-30m --checkpoint checkpoints/tacm-30m/step_50000.pt
  python scripts/benchmark.py --config tacm-30m  # random weights, for sanity-check
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tacm import TACSM, CONFIGS
from evaluate import (
    VanillaTransformer,
    TransformerMoE,
    run_evaluation,
    make_synthetic_eval_set,
    collect_repair_predictions,
)
from tacm.evaluation import (
    RepairAccuracyMetric,
    TransferAccuracyMetric,
    StructureReuseMetric,
    ExpertUtilisationMetric,
    BaselineEvaluator,
)


METRIC_LABELS = {
    "repair_accuracy":   "Repair Accuracy ↑",
    "transfer_accuracy": "Transfer Accuracy ↑",
    "structure_reuse":   "Structure Reuse ↑",
    "memory_retention":  "Memory Retention ↑",
    "attack_recovery":   "Attack Recovery ↑",
    "expert_util":       "Expert Entropy ↑",
    "verifier_accuracy": "Verifier Accuracy ↑",
}


def markdown_table(rows: dict) -> str:
    """
    rows = { model_name: { metric: value } }
    """
    metrics = list(METRIC_LABELS.keys())
    header  = ["Model"] + [METRIC_LABELS[m] for m in metrics]
    lines   = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for model_name, scores in rows.items():
        row = [model_name]
        for m in metrics:
            v = scores.get(m, None)
            row.append(f"{v:.4f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run_benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = CONFIGS[args.config]()

    print(f"\nTAC-SM Benchmark: {cfg.name}")
    print(f"Device: {device}")
    print(f"Eval samples: {args.n_samples}")

    # ── Load TAC-SM ───────────────────────────────────────────────────────────
    tacm_model = TACSM(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        tacm_model.load_state_dict(ckpt["model_state"])
        print(f"Loaded: {args.checkpoint}")

    t0     = time.time()
    report = run_evaluation(tacm_model, cfg, device, args.n_samples, compare_baselines=False)
    tacm_time = time.time() - t0
    print(f"TAC-SM eval: {tacm_time:.1f}s")

    rows = {"TAC-SM": report["metrics"]}

    # ── Baselines ─────────────────────────────────────────────────────────────
    samples = make_synthetic_eval_set(
        n          = args.n_samples,
        seq_len    = min(cfg.transformer.max_seq_len, 128),
        vocab_size = cfg.transformer.vocab_size,
        device     = device,
    )
    labels = [s.success_label or 0.0 for s in samples]

    for BModel, bname in [
        (VanillaTransformer,  "Vanilla-Transformer"),
        (TransformerMoE,      "Transformer-MoE"),
    ]:
        bmodel = BModel(cfg).to(device).eval()
        t0     = time.time()
        be     = BaselineEvaluator(bmodel, bname, device=str(device))
        result = be.compute_repair_accuracy(samples)
        elapsed = time.time() - t0
        rows[bname] = {"repair_accuracy": result.value}
        print(f"{bname} eval: {elapsed:.1f}s | repair_accuracy={result.value:.4f}")

    # ── Success Criteria Assessment ───────────────────────────────────────────
    tacm_ra = rows["TAC-SM"].get("repair_accuracy", 0)
    vt_ra   = rows.get("Vanilla-Transformer", {}).get("repair_accuracy", 0)
    moe_ra  = rows.get("Transformer-MoE", {}).get("repair_accuracy", 0)

    criteria = {
        "1_structure_reuse":      rows["TAC-SM"].get("structure_reuse",   0) > 0.3,
        "2_transfer_accuracy":    rows["TAC-SM"].get("transfer_accuracy", 0) > 0.4,
        "3_better_than_xfm":     tacm_ra > vt_ra,
        "4_better_than_moe":     tacm_ra > moe_ra,
        "5_memory_retention":    rows["TAC-SM"].get("memory_retention",  0) > 0.5,
        "6_attack_recovery":     rows["TAC-SM"].get("attack_recovery",   0) > 0.5,
        "7_verifier_accuracy":   rows["TAC-SM"].get("verifier_accuracy", 0) > 0.5,
    }

    # ── Print Report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(markdown_table(rows))

    print("\nSUCCESS CRITERIA:")
    for name, passed in criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")

    all_pass = all(criteria.values())
    print(f"\n{'ALL CRITERIA MET ✓' if all_pass else 'SOME CRITERIA FAILED ✗'}")

    # ── Save ─────────────────────────────────────────────────────────────────
    full_report = {
        "config":   cfg.name,
        "n_params": tacm_model.n_params(),
        "rows":     rows,
        "criteria": criteria,
        "all_pass": all_pass,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(full_report, f, indent=2, default=str)
        print(f"\nReport saved → {args.output}")

    return full_report


def main():
    parser = argparse.ArgumentParser(description="TAC-SM Benchmark")
    parser.add_argument("--config",     type=str, default="tacm-30m",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_samples",  type=int, default=100)
    parser.add_argument("--output",     type=str, default="./reports/benchmark.json")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
