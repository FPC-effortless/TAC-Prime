"""
TAC-PSM-002: Transfer Metrics

TransferMetrics dataclass + compute_transfer_metrics() over a list of TransferResults.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import List

from .transfer import TransferMode, TransferResult, TransferChainResult


@dataclass
class TransferMetrics:
    """Aggregate transfer metrics across conditions and seeds."""
    # Core
    transfer_success:    float   # mean success rate across transfer conditions
    transfer_gain:       float   # mean(transfer) - mean(reset)
    transfer_retention:  float   # quality at end of chain / start
    adaptation_cost:     float   # mean adaptation cost
    transfer_efficiency: float   # transfer_success / adaptation_cost

    # Per-mode breakdown
    mode_success:        dict    # mode → mean success rate
    mode_quality:        dict    # mode → mean quality

    # vs baselines
    gain_vs_fresh:       float   # transfer - fresh learning
    gain_vs_random:      float   # transfer - random retrieval
    gain_vs_reset:       float   # transfer - no memory

    # Std dev (across seeds)
    transfer_success_std: float = 0.0
    transfer_gain_std:    float = 0.0

    def all_gates_pass(self, threshold_gain: float = 0.05) -> dict:
        return {
            "transfer_gain_gt_0":          self.transfer_gain > 0,
            "transfer_outperforms_fresh":  self.gain_vs_fresh > 0,
            "transfer_outperforms_random": self.gain_vs_random > 0,
            "transfer_outperforms_reset":  self.gain_vs_reset > threshold_gain,
        }

    def to_dict(self) -> dict:
        return {
            "transfer_success":    self.transfer_success,
            "transfer_gain":       self.transfer_gain,
            "transfer_retention":  self.transfer_retention,
            "adaptation_cost":     self.adaptation_cost,
            "transfer_efficiency": self.transfer_efficiency,
            "mode_success":        self.mode_success,
            "mode_quality":        self.mode_quality,
            "gain_vs_fresh":       self.gain_vs_fresh,
            "gain_vs_random":      self.gain_vs_random,
            "gain_vs_reset":       self.gain_vs_reset,
            "transfer_success_std": self.transfer_success_std,
        }


def compute_transfer_metrics(
    adapted_results: List[TransferResult],    # ADAPTED mode results
    control_results: dict,                    # mode → List[TransferResult]
    chain_results:   List[TransferChainResult] = None,
) -> TransferMetrics:
    """
    Compute all transfer metrics from experiment results.

    adapted_results: results for the ADAPTED transfer mode (main condition)
    control_results: dict mapping TransferMode → list of results for that mode
    chain_results:   multi-hop chain results (optional)
    """
    def _mean_success(results: List[TransferResult]) -> float:
        return mean(float(r.success) for r in results) if results else 0.0

    def _mean_quality(results: List[TransferResult]) -> float:
        return mean(r.quality for r in results) if results else 0.0

    adapted_success = _mean_success(adapted_results)
    adapted_quality = _mean_quality(adapted_results)
    adapted_cost    = mean(r.adaptation_cost for r in adapted_results) if adapted_results else 0.0

    # Per-mode breakdown
    mode_success = {"adapted": adapted_success}
    mode_quality = {"adapted": adapted_quality}
    for mode, results in control_results.items():
        k = mode.value if hasattr(mode, "value") else str(mode)
        mode_success[k] = _mean_success(results)
        mode_quality[k] = _mean_quality(results)

    fresh_success  = _mean_success(control_results.get(TransferMode.FRESH,  []))
    random_success = _mean_success(control_results.get(TransferMode.RANDOM, []))
    reset_success  = _mean_success(control_results.get(TransferMode.RESET,  []))

    # Chain metrics
    if chain_results:
        retention  = mean(c.retention   for c in chain_results)
        efficiency = mean(c.efficiency  for c in chain_results)
    else:
        retention  = 1.0
        efficiency = adapted_success / max(adapted_cost, 1e-9)

    # Std dev across seeds for adapted mode
    adapted_success_vals = [float(r.success) for r in adapted_results]
    success_std = stdev(adapted_success_vals) if len(adapted_success_vals) > 1 else 0.0

    return TransferMetrics(
        transfer_success     = adapted_success,
        transfer_gain        = adapted_success - reset_success,
        transfer_retention   = retention,
        adaptation_cost      = adapted_cost,
        transfer_efficiency  = efficiency,
        mode_success         = mode_success,
        mode_quality         = mode_quality,
        gain_vs_fresh        = adapted_success - fresh_success,
        gain_vs_random       = adapted_success - random_success,
        gain_vs_reset        = adapted_success - reset_success,
        transfer_success_std = success_std,
    )
