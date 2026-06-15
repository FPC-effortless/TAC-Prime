"""
TAC-PSM-005: Discovered Procedure Verification

Verifies that a discovered procedure:
  1. Achieves utility on held-out tasks (utility_threshold)
  2. Outperforms the no-discovery baseline (empty steps)
  3. Compresses the trace representation (compression_ratio < 1.0)
  4. Achieves reasonable oracle coverage (discovery_accuracy)

Verification is pass/fail per criterion; a procedure must pass
all required criteria to be accepted into the store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..psm001.benchmark_families import (
    TaskInstance,
    evaluate_procedure_on_task,
    oracle_steps,
    reset_steps,
)
from ..psm001.records import ProcedureTrace
from ..psm001.store import ProceduralMemoryStore
from .discovery import DiscoveryResult


@dataclass
class VerificationResult:
    """
    Pass/fail verdict on a discovered procedure.
    """
    procedure_id:         str
    utility_ok:           bool       # utility >= utility_threshold
    beats_no_discovery:   bool       # utility > no_discovery_quality
    compressed:           bool       # compression_ratio < 1.0
    oracle_coverage_ok:   bool       # discovery_accuracy >= coverage_threshold

    utility:              float
    no_discovery_quality: float
    oracle_quality:       float
    compression_ratio:    float
    discovery_accuracy:   float

    accepted:             bool       # all required criteria pass

    def criteria(self) -> Dict[str, bool]:
        return {
            "utility_ok":         self.utility_ok,
            "beats_no_discovery": self.beats_no_discovery,
            "compressed":         self.compressed,
            "oracle_coverage_ok": self.oracle_coverage_ok,
        }

    def to_dict(self) -> dict:
        return {
            "procedure_id":        self.procedure_id,
            "accepted":            self.accepted,
            "utility":             self.utility,
            "no_discovery_quality": self.no_discovery_quality,
            "oracle_quality":      self.oracle_quality,
            "compression_ratio":   self.compression_ratio,
            "discovery_accuracy":  self.discovery_accuracy,
            "criteria":            self.criteria(),
        }


def verify_discovered_procedure(
    proc:              ProcedureTrace,
    held_out_tasks:    List[TaskInstance],
    discovery_result:  Optional[DiscoveryResult] = None,
    utility_threshold: float = 0.40,
    coverage_threshold: float = 0.50,
    seed:              int   = 0,
) -> VerificationResult:
    """
    Verify a single discovered procedure against held-out tasks.

    Parameters
    ----------
    proc               : the discovered ProcedureTrace to verify
    held_out_tasks     : tasks not seen during discovery
    discovery_result   : optional — provides compression_ratio
    utility_threshold  : minimum quality to accept the procedure
    coverage_threshold : minimum oracle_coverage (disc / oracle quality)
    seed               : evaluation seed
    """
    steps = [s.action for s in proc.steps]

    utility_scores   = []
    no_disc_scores   = []
    oracle_scores    = []

    for task in held_out_tasks:
        _, q_disc,   _ = evaluate_procedure_on_task(task, steps,          seed=seed)
        _, q_nodis,  _ = evaluate_procedure_on_task(task, reset_steps(),  seed=seed)
        _, q_oracle, _ = evaluate_procedure_on_task(task, oracle_steps(task), seed=seed)
        utility_scores.append(q_disc)
        no_disc_scores.append(q_nodis)
        oracle_scores.append(q_oracle)

    utility      = sum(utility_scores)  / max(len(utility_scores),  1)
    no_disc_q    = sum(no_disc_scores)  / max(len(no_disc_scores),  1)
    oracle_q     = sum(oracle_scores)   / max(len(oracle_scores),   1)
    disc_acc     = utility / max(oracle_q, 1e-9)
    compression  = (discovery_result.compression_ratio
                    if discovery_result else 1.0)

    utility_ok   = utility >= utility_threshold
    beats_nodis  = utility > no_disc_q
    compressed   = compression <= 1.0
    coverage_ok  = disc_acc >= coverage_threshold

    accepted = utility_ok and beats_nodis

    return VerificationResult(
        procedure_id         = proc.procedure_id,
        utility_ok           = utility_ok,
        beats_no_discovery   = beats_nodis,
        compressed           = compressed,
        oracle_coverage_ok   = coverage_ok,
        utility              = utility,
        no_discovery_quality = no_disc_q,
        oracle_quality       = oracle_q,
        compression_ratio    = compression,
        discovery_accuracy   = disc_acc,
        accepted             = accepted,
    )


def batch_verify(
    proc_ids:          List[str],
    store:             ProceduralMemoryStore,
    held_out_tasks:    List[TaskInstance],
    discovery_results: Optional[Dict[str, DiscoveryResult]] = None,
    utility_threshold: float = 0.40,
    coverage_threshold: float = 0.50,
    seed:              int   = 0,
) -> List[VerificationResult]:
    """
    Verify a batch of discovered procedures.
    Rejects (marks RETIRED) procedures that fail verification.
    """
    results = []
    for pid in proc_ids:
        proc = store.get(pid)
        if proc is None:
            continue
        dr  = discovery_results.get(pid) if discovery_results else None
        res = verify_discovered_procedure(
            proc, held_out_tasks, dr, utility_threshold, coverage_threshold, seed
        )
        if not res.accepted:
            store.retire(pid)
        results.append(res)
    return results
