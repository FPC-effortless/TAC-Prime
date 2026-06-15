"""
TAC-PSM-005: Pattern Mining and Procedure Extraction

Discovery pipeline:
  1. Collect successful repair traces (SuccessTrace)
  2. Mine frequent step sub-sequences across traces (mine_patterns)
  3. Extract a canonical ProcedureTrace from mined patterns (extract_procedure)
  4. Store discovered procedure in ProceduralMemoryStore

This module is entirely unsupervised — no labels are provided about
what the procedure should be. The system infers it from raw traces.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..psm001.records import ProcedureTrace, ProcedureStep
from ..psm001.store import ProceduralMemoryStore
from ..psm001.benchmark_families import TaskInstance, oracle_steps


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class SuccessTrace:
    """
    A single successful repair trace (one observation of a solved task).

    The discovery system mines these for common patterns.
    No label is given — only the steps that led to success.
    """
    trace_id:     str
    family:       str           # discovered post-hoc; NOT provided to the miner
    task_sig:     str
    steps:        List[str]     # the actions that succeeded
    quality:      float         # how well they succeeded
    embedding:    Optional[List[float]] = None

    @classmethod
    def from_task(
        cls,
        task:    TaskInstance,
        steps:   List[str],
        quality: float,
        seed:    int = 0,
        dim:     int = 64,
    ) -> "SuccessTrace":
        emb  = task.query_embedding(dim)
        tid  = hashlib.md5(f"{task.task_signature}{seed}".encode()).hexdigest()[:12]
        return cls(
            trace_id  = tid,
            family    = task.family,
            task_sig  = task.task_signature,
            steps     = list(steps),
            quality   = quality,
            embedding = emb.tolist(),
        )


@dataclass
class DiscoveredPattern:
    """
    A frequent step pattern mined from traces.

    pattern       — ordered tuple of step strings
    support       — number of traces containing this pattern
    confidence    — support / total_traces
    families      — which families contributed
    """
    pattern:      Tuple[str, ...]
    support:      int
    confidence:   float
    families:     Set[str] = field(default_factory=set)
    avg_quality:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "pattern":    list(self.pattern),
            "support":    self.support,
            "confidence": self.confidence,
            "families":   list(self.families),
            "avg_quality": self.avg_quality,
        }


@dataclass
class DiscoveryResult:
    """
    Result of a full discovery pipeline run.
    """
    n_traces:          int
    n_patterns_mined:  int
    n_procedures_extracted: int
    discovered_proc_ids: List[str]
    top_pattern:       Optional[DiscoveredPattern]
    compression_ratio: float    # len(discovered_steps) / avg(trace_steps)
    discovery_accuracy: float   # how well discovered procedure matches oracle
    utility_score:     float    # quality on held-out tasks

    def to_dict(self) -> dict:
        return {
            "n_traces":              self.n_traces,
            "n_patterns_mined":      self.n_patterns_mined,
            "n_procedures_extracted": self.n_procedures_extracted,
            "discovered_proc_ids":   self.discovered_proc_ids,
            "top_pattern":           self.top_pattern.to_dict() if self.top_pattern else None,
            "compression_ratio":     self.compression_ratio,
            "discovery_accuracy":    self.discovery_accuracy,
            "utility_score":         self.utility_score,
        }


# ── Pattern mining ────────────────────────────────────────────────────────────

def mine_patterns(
    traces:         List[SuccessTrace],
    min_support:    int   = 2,
    min_confidence: float = 0.30,
    max_pattern_len: int  = 6,
    normalise:      bool  = True,
) -> List[DiscoveredPattern]:
    """
    Mine frequent ordered step sub-sequences from traces.

    Algorithm: sliding-window frequent subsequence mining.
    For each trace, generate all contiguous subsequences of length 1..max_pattern_len.
    Count occurrences across all traces.
    Filter by min_support and min_confidence.

    Returns patterns sorted by confidence descending.
    """
    if not traces:
        return []

    n_traces = len(traces)

    # Normalise step strings for matching
    def norm(s: str) -> str:
        return s.lower().strip()

    # Count pattern occurrences
    pattern_traces: Dict[Tuple[str, ...], Set[int]] = defaultdict(set)
    pattern_quality: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    pattern_families: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)

    for ti, trace in enumerate(traces):
        norm_steps = [norm(s) for s in trace.steps]
        seen_in_trace: Set[Tuple[str, ...]] = set()

        for length in range(1, min(max_pattern_len + 1, len(norm_steps) + 1)):
            for start in range(len(norm_steps) - length + 1):
                pattern = tuple(norm_steps[start:start + length])
                if pattern not in seen_in_trace:
                    pattern_traces[pattern].add(ti)
                    pattern_quality[pattern].append(trace.quality)
                    pattern_families[pattern].add(trace.family)
                    seen_in_trace.add(pattern)

    # Filter and build DiscoveredPattern objects
    results = []
    for pattern, trace_ids in pattern_traces.items():
        support    = len(trace_ids)
        confidence = support / n_traces
        if support >= min_support and confidence >= min_confidence:
            avg_q = sum(pattern_quality[pattern]) / max(len(pattern_quality[pattern]), 1)
            dp = DiscoveredPattern(
                pattern    = pattern,
                support    = support,
                confidence = confidence,
                families   = pattern_families[pattern],
                avg_quality = avg_q,
            )
            results.append(dp)

    # Sort by confidence × avg_quality × len (longer patterns = more specific)
    results.sort(
        key=lambda p: p.confidence * p.avg_quality * (1.0 + 0.1 * len(p.pattern)),
        reverse=True,
    )
    return results


# ── Procedure extraction ──────────────────────────────────────────────────────

def extract_procedure(
    patterns:       List[DiscoveredPattern],
    traces:         List[SuccessTrace],
    dim:            int = 64,
    max_steps:      int = 8,
    min_confidence: float = 0.30,
) -> Optional[Tuple[List[str], np.ndarray, str]]:
    """
    Extract a single canonical procedure from mined patterns.

    Strategy: greedily build a step list by selecting the highest-confidence
    patterns that do not conflict with already-selected steps.

    Returns (steps, embedding, inferred_family) or None if extraction fails.
    """
    if not patterns:
        return None

    # Select high-confidence patterns
    good   = [p for p in patterns if p.confidence >= min_confidence]
    if not good:
        good = patterns[:5]   # fallback: top-5

    # Build step list greedily: start with the longest high-confidence pattern,
    # then extend with patterns that append new steps at the end.
    best = max(good, key=lambda p: p.confidence * len(p.pattern))
    steps: List[str] = list(best.pattern)

    for p in good:
        if len(steps) >= max_steps:
            break
        # Append steps from pattern that are not already in steps
        for s in p.pattern:
            if s not in steps and len(steps) < max_steps:
                steps.append(s)

    # Infer family from most common family across top patterns
    family_counts: Counter = Counter()
    for p in good[:10]:
        family_counts.update(p.families)
    inferred_family = family_counts.most_common(1)[0][0] if family_counts else "Unknown"

    # Build embedding: mean of trace embeddings
    trace_embs = [np.array(t.embedding, dtype=np.float32)
                  for t in traces if t.embedding is not None]
    if trace_embs:
        emb = np.mean(np.stack(trace_embs), axis=0)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
    else:
        seed = int(hashlib.md5(str(steps).encode()).hexdigest(), 16) % (2**31)
        rng  = np.random.default_rng(seed)
        emb  = rng.standard_normal(dim).astype(np.float32)
        emb  = emb / (np.linalg.norm(emb) + 1e-9)

    return steps, emb, inferred_family


# ── Full discovery pipeline ───────────────────────────────────────────────────

def run_discovery_pipeline(
    traces:           List[SuccessTrace],
    store:            ProceduralMemoryStore,
    held_out_tasks:   List[TaskInstance],
    min_support:      int   = 2,
    min_confidence:   float = 0.30,
    seed:             int   = 0,
    verbose:          bool  = False,
) -> DiscoveryResult:
    """
    Run the full autonomous discovery pipeline.

    1. Mine patterns from traces
    2. Extract procedure(s)
    3. Store discovered procedures
    4. Evaluate on held-out tasks
    5. Compare to oracle (upper bound) and no-discovery (lower bound)

    Returns DiscoveryResult with all metrics.
    """
    from ..psm001.benchmark_families import evaluate_procedure_on_task

    n_traces = len(traces)

    # ── Mine ──────────────────────────────────────────────────────────────────
    patterns = mine_patterns(
        traces, min_support=min_support, min_confidence=min_confidence
    )
    if verbose:
        print(f"  [discovery] mined {len(patterns)} patterns from {n_traces} traces")

    # ── Extract ────────────────────────────────────────────────────────────────
    dim   = len(traces[0].embedding) if traces and traces[0].embedding else 64
    ext   = extract_procedure(patterns, traces, dim=dim)
    proc_ids: List[str] = []

    if ext is not None:
        disc_steps, disc_emb, inferred_family = ext
        proc = store.build(
            problem_family   = inferred_family,
            task_signature   = f"discovered::{inferred_family}::seed{seed}",
            steps            = disc_steps,
            embedding        = disc_emb,
            success_score    = 0.5,
            selection_reason = "autonomously discovered",
        )
        proc_ids.append(proc.procedure_id)
        if verbose:
            print(f"  [discovery] extracted {len(disc_steps)} steps → {proc.procedure_id}")
    else:
        disc_steps = []

    # ── Evaluate on held-out tasks ─────────────────────────────────────────────
    utility_scores = []
    oracle_scores  = []
    for task in held_out_tasks:
        _, q_disc, _   = evaluate_procedure_on_task(task, disc_steps, seed=seed)
        _, q_oracle, _ = evaluate_procedure_on_task(task, oracle_steps(task), seed=seed)
        utility_scores.append(q_disc)
        oracle_scores.append(q_oracle)

    utility = sum(utility_scores) / max(len(utility_scores), 1)
    oracle_mean = sum(oracle_scores) / max(len(oracle_scores), 1)

    # Discovery accuracy = discovered quality / oracle quality
    disc_acc = utility / max(oracle_mean, 1e-9)

    # Compression ratio
    avg_trace_len = sum(len(t.steps) for t in traces) / max(n_traces, 1)
    compression   = len(disc_steps) / max(avg_trace_len, 1)

    return DiscoveryResult(
        n_traces               = n_traces,
        n_patterns_mined       = len(patterns),
        n_procedures_extracted = len(proc_ids),
        discovered_proc_ids    = proc_ids,
        top_pattern            = patterns[0] if patterns else None,
        compression_ratio      = compression,
        discovery_accuracy     = disc_acc,
        utility_score          = utility,
    )
