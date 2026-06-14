"""
TAC-SM Evaluation Benchmarks

Metrics:
  - Repository Repair Accuracy
  - Transfer Accuracy
  - Structure Reuse Rate
  - Structure Survival Rate
  - Memory Retention
  - Attack Recovery
  - Distribution Shift Retention
  - Transfer Chain Success
  - Expert Utilisation
  - Verifier Accuracy
  - Patch Success Rate

Baselines:
  1. Vanilla Transformer
  2. Transformer + MoE
  3. Transformer + Retrieval
  4. Transformer + Memory (no structure)
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EvalSample:
    """Single evaluation sample."""
    sample_id:     str
    input_ids:     torch.Tensor        # (T,)
    labels:        Optional[torch.Tensor] = None
    task_type:     str = "unknown"
    family:        str = "unknown"
    success_label: Optional[float] = None
    source_repo:   str = ""
    target_repo:   str = ""    # for transfer evaluation


@dataclass
class EvalResult:
    """Result for a single metric pass."""
    metric:   str
    value:    float
    count:    int
    details:  Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"{self.metric}: {self.value:.4f} (n={self.count})"


class RepairAccuracyMetric:
    """
    Computes repair success rate.
    Input: verifier success_prob predictions vs. actual test pass labels.
    """
    name = "repair_accuracy"

    def compute(
        self,
        predictions: List[float],   # verifier success_prob
        labels:      List[float],   # actual {0, 1}
        threshold:   float = 0.5,
    ) -> EvalResult:
        assert len(predictions) == len(labels), "Length mismatch"
        n_correct = sum(
            1 for p, l in zip(predictions, labels)
            if (p >= threshold) == (l >= threshold)
        )
        return EvalResult(
            metric  = self.name,
            value   = n_correct / max(len(predictions), 1),
            count   = len(predictions),
            details = {"threshold": threshold},
        )


class TransferAccuracyMetric:
    """
    Measures whether structures retrieved from source repo
    help solve tasks in target repo.
    """
    name = "transfer_accuracy"

    def compute(
        self,
        source_successes: List[bool],
        transfer_successes: List[bool],
    ) -> EvalResult:
        n = len(source_successes)
        assert n == len(transfer_successes)
        n_source   = sum(source_successes)
        n_transfer = sum(transfer_successes)
        # Transfer accuracy: fraction of source-solved tasks also solved via transfer
        transferable = [s for s, t in zip(source_successes, transfer_successes) if s]
        transfer_acc = (
            sum(t for s, t in zip(source_successes, transfer_successes) if s)
            / max(len(transferable), 1)
        )
        return EvalResult(
            metric  = self.name,
            value   = transfer_acc,
            count   = n,
            details = {"source_solved": n_source, "transfer_solved": n_transfer},
        )


class StructureReuseMetric:
    """
    Tracks how often memory retrievals contribute to successful repairs.
    """
    name = "structure_reuse_rate"

    def compute(
        self,
        retrieval_used:    List[bool],   # was any retrieved structure used?
        repair_succeeded:  List[bool],
    ) -> EvalResult:
        n = len(retrieval_used)
        reuse_count   = sum(retrieval_used)
        reuse_success = sum(
            u and s for u, s in zip(retrieval_used, repair_succeeded)
        )
        reuse_rate = reuse_success / max(reuse_count, 1)
        return EvalResult(
            metric  = self.name,
            value   = reuse_rate,
            count   = n,
            details = {"reuse_count": reuse_count, "reuse_success": reuse_success},
        )


class MemoryRetentionMetric:
    """
    Evaluates whether the memory retains structures over training.
    Measured as fraction of written structures still in memory after N steps.
    """
    name = "memory_retention"

    def compute(
        self,
        written_ids:  List[str],
        memory_ids:   List[str],
    ) -> EvalResult:
        written_set = set(written_ids)
        memory_set  = set(memory_ids)
        retained    = written_set & memory_set
        retention   = len(retained) / max(len(written_set), 1)
        return EvalResult(
            metric  = self.name,
            value   = retention,
            count   = len(written_set),
            details = {"retained": len(retained), "pruned": len(written_set) - len(retained)},
        )


class AttackRecoveryMetric:
    """
    Measures robustness of structure embeddings to adversarial perturbations.
    Computes fraction of structures whose survival score stays above threshold
    after embedding corruption.
    """
    name = "attack_recovery"

    def compute(
        self,
        original_scores:  List[float],
        perturbed_scores: List[float],
        threshold:        float = 0.3,
    ) -> EvalResult:
        n = len(original_scores)
        survived = sum(
            1 for p in perturbed_scores if p >= threshold
        )
        return EvalResult(
            metric  = self.name,
            value   = survived / max(n, 1),
            count   = n,
            details = {"threshold": threshold},
        )


class ExpertUtilisationMetric:
    """
    Measures expert routing distribution.
    Ideal: uniform utilisation (max entropy).
    """
    name = "expert_utilisation"

    def compute(self, model) -> EvalResult:
        util    = model.moe.expert_utilisation()
        entropy = model.moe.expert_entropy()
        return EvalResult(
            metric  = self.name,
            value   = entropy,
            count   = util.shape[0],
            details = {
                "per_expert": {i: u.item() for i, u in enumerate(util)},
                "max_entropy": torch.log(torch.tensor(float(util.shape[0]))).item(),
            },
        )


class VerifierAccuracyMetric:
    """
    Binary accuracy of verifier head success prediction.
    """
    name = "verifier_accuracy"

    def compute(
        self,
        model,
        eval_samples:  List[EvalSample],
        device:        str = "cpu",
    ) -> EvalResult:
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for s in eval_samples:
                if s.success_label is None:
                    continue
                ids = s.input_ids.unsqueeze(0).to(device)
                out = model(ids)
                pred = (out.verifier_out.success_prob[0].item() >= 0.5)
                true = (s.success_label >= 0.5)
                correct += int(pred == true)
                total   += 1
        return EvalResult(
            metric = self.name,
            value  = correct / max(total, 1),
            count  = total,
        )


class Evaluator:
    """
    Runs full evaluation suite and returns a report dict.
    """

    def __init__(self, model, device: str = "cpu"):
        self.model  = model
        self.device = device

        self.metrics = {
            "repair_accuracy":   RepairAccuracyMetric(),
            "transfer_accuracy": TransferAccuracyMetric(),
            "structure_reuse":   StructureReuseMetric(),
            "memory_retention":  MemoryRetentionMetric(),
            "attack_recovery":   AttackRecoveryMetric(),
            "expert_util":       ExpertUtilisationMetric(),
            "verifier_accuracy": VerifierAccuracyMetric(),
        }

    def evaluate_all(
        self,
        eval_samples:         List[EvalSample],
        repair_preds:         List[float],
        repair_labels:        List[float],
        transfer_source:      List[bool],
        transfer_target:      List[bool],
        retrieval_used:       List[bool],
        repair_succeeded:     List[bool],
        written_ids:          List[str],
        original_surv:        List[float],
        perturbed_surv:       List[float],
    ) -> Dict[str, EvalResult]:
        results = {}

        results["repair_accuracy"] = self.metrics["repair_accuracy"].compute(
            repair_preds, repair_labels
        )
        results["transfer_accuracy"] = self.metrics["transfer_accuracy"].compute(
            transfer_source, transfer_target
        )
        results["structure_reuse"] = self.metrics["structure_reuse"].compute(
            retrieval_used, repair_succeeded
        )
        results["memory_retention"] = self.metrics["memory_retention"].compute(
            written_ids, list(self.model.struct_memory._store.keys())
        )
        results["attack_recovery"] = self.metrics["attack_recovery"].compute(
            original_surv, perturbed_surv
        )
        results["expert_util"] = self.metrics["expert_util"].compute(self.model)
        results["verifier_accuracy"] = self.metrics["verifier_accuracy"].compute(
            self.model, eval_samples, self.device
        )

        return results

    def report(self, results: Dict[str, EvalResult]) -> str:
        lines = ["\n" + "=" * 55, "TAC-SM Evaluation Report", "=" * 55]
        for name, result in results.items():
            lines.append(f"  {result.metric:<30} {result.value:.4f}  (n={result.count})")
        lines.append("=" * 55)
        return "\n".join(lines)


class BaselineEvaluator:
    """
    Wraps a baseline model (Transformer, MoE, etc.) in the same evaluation interface.
    """

    def __init__(self, baseline_model, name: str, device: str = "cpu"):
        self.model  = baseline_model
        self.name   = name
        self.device = device

    def compute_repair_accuracy(
        self,
        eval_samples: List[EvalSample],
        threshold: float = 0.5,
    ) -> EvalResult:
        """Run baseline on eval samples, return predicted vs. true success."""
        preds  = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for s in eval_samples:
                if s.success_label is None:
                    continue
                ids = s.input_ids.unsqueeze(0).to(self.device)
                # Baseline models just return logits
                out = self.model(ids)
                if hasattr(out, "verifier_out"):
                    sp = out.verifier_out.success_prob[0].item()
                elif hasattr(out, "loss"):
                    sp = float(not out.loss.isnan())
                else:
                    sp = 0.5
                preds.append(sp)
                labels.append(s.success_label)

        metric = RepairAccuracyMetric()
        result = metric.compute(preds, labels, threshold)
        result.details["model"] = self.name
        return result
