"""
TAC-PSM-006: Deterministic Repair Verifier
===========================================

Implements Level 1 deterministic verification: no real package installation.

Each task carries a verification_rule dict:
  {
    "expected_family":   str    — the correct procedure family
    "keyword_match":     list   — keywords that must appear in applied steps
    "min_step_overlap":  float  — Jaccard threshold between applied and oracle
    "min_score":         float  — composite score threshold for success
    "repo_context_key":  str    — a file that must appear in relevant_files
  }

Verification score components:
  S1  family_match      (0 or 1)   weight 0.45
  S2  step_overlap      [0, 1]     weight 0.30   Jaccard(applied, oracle)
  S3  keyword_coverage  [0, 1]     weight 0.15   fraction of required keywords hit
  S4  repo_context_hit  (0 or 1)   weight 0.10   context file present

success = composite_score > rule["min_score"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .repository_task import RepoTask


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """
    Outcome of verifying an applied procedure against a task's rule.

    Attributes
    ----------
    task_id          : links to the RepoTask
    success          : True if composite_score >= min_score
    composite_score  : weighted sum of components [0, 1]
    family_match     : whether retrieved family matched expected family
    step_overlap     : Jaccard similarity of applied vs oracle steps
    keyword_coverage : fraction of rule keywords found in applied steps
    repo_context_hit : whether context file was present in relevant_files
    selected_family  : the family of the procedure that was applied
    expected_family  : the family the task requires
    reason           : human-readable explanation
    """
    task_id:          str
    success:          bool
    composite_score:  float
    family_match:     bool
    step_overlap:     float
    keyword_coverage: float
    repo_context_hit: bool
    selected_family:  str
    expected_family:  str
    reason:           str

    def to_dict(self) -> dict:
        return {
            "task_id":         self.task_id,
            "success":         self.success,
            "composite_score": round(self.composite_score, 4),
            "family_match":    self.family_match,
            "step_overlap":    round(self.step_overlap, 4),
            "keyword_coverage": round(self.keyword_coverage, 4),
            "repo_context_hit": self.repo_context_hit,
            "selected_family": self.selected_family,
            "expected_family": self.expected_family,
            "reason":          self.reason,
        }


# ── Core verification logic ────────────────────────────────────────────────────

def verify_repair(
    task:             RepoTask,
    applied_steps:    List[str],
    selected_family:  str,
) -> VerificationResult:
    """
    Deterministic Level-1 verification.

    Parameters
    ----------
    task            : the RepoTask being repaired
    applied_steps   : ordered steps produced by the repair agent
    selected_family : the procedure family that was selected

    Returns
    -------
    VerificationResult with full component breakdown.
    """
    rule = task.verification_rule

    # ── Component 1: family match ─────────────────────────────────────────────
    expected_family = rule["expected_family"]
    family_match    = selected_family == expected_family

    # ── Component 2: step overlap (Jaccard) ──────────────────────────────────
    oracle_set   = _normalise_set(task.oracle_repair_steps)
    applied_set  = _normalise_set(applied_steps)
    step_overlap = _jaccard(oracle_set, applied_set)

    # ── Component 3: keyword coverage ─────────────────────────────────────────
    required_kws = {kw.lower() for kw in rule.get("keyword_match", [])}
    applied_text = " ".join(applied_steps).lower()
    if required_kws:
        hits     = sum(1 for kw in required_kws if kw in applied_text)
        kw_cover = hits / len(required_kws)
    else:
        kw_cover = 1.0

    # ── Component 4: repo context file present ────────────────────────────────
    ctx_key     = rule.get("repo_context_key", "")
    ctx_hit     = ctx_key in task.relevant_files if ctx_key else True

    # ── Composite score ───────────────────────────────────────────────────────
    composite = (
        0.45 * float(family_match)
        + 0.30 * step_overlap
        + 0.15 * kw_cover
        + 0.10 * float(ctx_hit)
    )

    min_score = rule.get("min_score", 0.50)
    success   = composite >= min_score

    reason = (
        f"family={'✓' if family_match else '✗'}({selected_family}→{expected_family})  "
        f"step_overlap={step_overlap:.3f}  "
        f"kw_coverage={kw_cover:.3f}  "
        f"ctx_hit={'✓' if ctx_hit else '✗'}  "
        f"composite={composite:.3f}  min={min_score:.3f}  "
        f"{'PASS' if success else 'FAIL'}"
    )

    return VerificationResult(
        task_id          = task.task_id,
        success          = success,
        composite_score  = composite,
        family_match     = family_match,
        step_overlap     = step_overlap,
        keyword_coverage = kw_cover,
        repo_context_hit = ctx_hit,
        selected_family  = selected_family,
        expected_family  = expected_family,
        reason           = reason,
    )


def batch_verify(
    tasks:            List[RepoTask],
    applied_steps_map: Dict[str, List[str]],    # task_id → applied steps
    selected_family_map: Dict[str, str],         # task_id → selected family
) -> List[VerificationResult]:
    """Verify a batch of tasks. Returns one VerificationResult per task."""
    results = []
    for task in tasks:
        steps  = applied_steps_map.get(task.task_id, [])
        family = selected_family_map.get(task.task_id, "Unknown")
        results.append(verify_repair(task, steps, family))
    return results


# ── Retry verification (after update) ─────────────────────────────────────────

def verify_with_retry(
    task:            RepoTask,
    applied_steps:   List[str],
    selected_family: str,
    max_retries:     int = 2,
    step_augment:    Optional[List[str]] = None,
) -> Tuple[VerificationResult, int]:
    """
    Attempt verification; if it fails, augment steps with oracle hints and retry.

    This simulates the agent's update-then-retry behaviour.
    Returns (final_result, n_retries_used).
    """
    result = verify_repair(task, applied_steps, selected_family)
    if result.success:
        return result, 0

    for attempt in range(1, max_retries + 1):
        # Augment: add one oracle step hint per retry
        augmented = list(applied_steps)
        if step_augment:
            augmented.extend(step_augment[:attempt])
        else:
            # Partial oracle hint: add first N oracle steps not already present
            applied_norm = _normalise_set(applied_steps)
            hints_added  = 0
            for step in task.oracle_repair_steps:
                if step.lower().strip() not in applied_norm and hints_added < attempt:
                    augmented.append(step)
                    hints_added += 1

        retry_result = verify_repair(task, augmented, selected_family)
        if retry_result.success:
            return retry_result, attempt

    return result, max_retries


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_set(steps: List[str]) -> Set[str]:
    return {s.lower().strip() for s in steps if s.strip()}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    # Also add partial word-overlap bonus
    bonus = _word_overlap_bonus(a, b)
    return min(1.0, len(inter) / max(len(union), 1) + bonus)


def _word_overlap_bonus(a: Set[str], b: Set[str], cap: float = 0.15) -> float:
    """Small bonus for shared words even when full step strings don't match."""
    words_a = {w for s in a for w in s.split()}
    words_b = {w for s in b for w in s.split()}
    if not words_a or not words_b:
        return 0.0
    inter = words_a & words_b
    union = words_a | words_b
    return min(cap, (len(inter) / max(len(union), 1)) * 0.3)
