"""
TAC-PSM-001: Core Data Structures

StructureMemoryRecordV2  — enhanced structure record with versioning + failure tracking
ProcedureStep            — single inspectable step in a procedure
ProcedureTrace           — ordered sequence of steps with metadata
FailureMode              — documented failure pattern
RecoveryStrategy         — documented recovery approach
ProcedureLifecycleState  — lifecycle state machine for procedures
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ── Lifecycle ─────────────────────────────────────────────────────────────────

class ProcedureLifecycleState(Enum):
    CREATED     = "created"
    ACTIVE      = "active"
    SPECIALISED = "specialised"
    TRANSFERRED = "transferred"
    UPDATED     = "updated"      # post-failure update
    RETIRING    = "retiring"
    RETIRED     = "retired"


# ── Failure and Recovery ──────────────────────────────────────────────────────

@dataclass
class FailureMode:
    """Documented failure pattern for a procedure."""
    failure_id:   str
    description:  str
    frequency:    int   = 0      # how often this failure occurred
    family:       str   = ""     # task family where failure occurred
    step_index:   int   = -1     # which step failed (-1 = unknown)
    error_type:   str   = ""     # e.g. "ImportError", "VersionMismatch"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FailureMode":
        return cls(**d)


@dataclass
class RecoveryStrategy:
    """Documented recovery approach for a failure mode."""
    strategy_id:      str
    failure_id:       str          # links to FailureMode
    description:      str
    recovery_steps:   List[str]    # ordered recovery actions
    success_rate:     float = 0.0
    application_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RecoveryStrategy":
        return cls(**d)


# ── Procedure Step ─────────────────────────────────────────────────────────────

@dataclass
class ProcedureStep:
    """
    Single inspectable step in a procedure.
    Each step is independently logged, timed, and scored.
    """
    step_index:     int
    action:         str            # what to do (human-readable)
    tool:           str = ""       # tool or module invoked (optional)
    expected_output: str = ""      # what success looks like
    actual_output:  str = ""       # filled in after execution
    succeeded:      Optional[bool] = None
    duration_ms:    float = 0.0
    notes:          str = ""

    def mark_success(self, actual: str = "", duration_ms: float = 0.0):
        self.succeeded    = True
        self.actual_output = actual
        self.duration_ms   = duration_ms

    def mark_failure(self, reason: str = "", duration_ms: float = 0.0):
        self.succeeded    = False
        self.actual_output = reason
        self.duration_ms   = duration_ms

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProcedureStep":
        return cls(**d)

    def __repr__(self):
        status = "✓" if self.succeeded else ("✗" if self.succeeded is False else "·")
        return f"  {self.step_index + 1}. [{status}] {self.action}"


# ── Procedure Trace ────────────────────────────────────────────────────────────

@dataclass
class ProcedureTrace:
    """
    Reusable procedure: ordered sequence of actions with full provenance.

    Serialisable to JSON. Explains:
      - why it was selected (selection_reason)
      - what previous tasks used it (used_by_tasks)
      - how often it succeeded / failed (success_score, failure_modes)
      - how often it transferred (transfer_score)
    """
    procedure_id:       str
    problem_family:     str            # e.g. "ImportErrors"
    task_signature:     str            # canonical task fingerprint
    steps:              List[ProcedureStep]
    failure_modes:      List[FailureMode]       = field(default_factory=list)
    recovery_strategies: List[RecoveryStrategy] = field(default_factory=list)

    # Scoring
    success_score:      float = 0.0
    reuse_count:        int   = 0
    last_used:          float = field(default_factory=time.time)
    survival_score:     float = 1.0
    transfer_score:     float = 0.0

    # Versioning
    version:            int   = 1
    parent_id:          Optional[str] = None    # which procedure this was forked from

    # Provenance
    used_by_tasks:      List[str] = field(default_factory=list)  # task_signatures
    selection_reason:   str = ""
    creation_timestamp: float = field(default_factory=time.time)

    # Lifecycle
    lifecycle_state:    ProcedureLifecycleState = ProcedureLifecycleState.CREATED

    # Embedding (stored as list for JSON serialisation)
    embedding:          Optional[List[float]] = None

    def overall_score(self) -> float:
        reuse_norm = min(self.reuse_count / 20.0, 1.0)
        return (
            0.40 * self.success_score
            + 0.25 * self.transfer_score
            + 0.20 * self.survival_score
            + 0.15 * reuse_norm
        )

    def n_steps(self) -> int:
        return len(self.steps)

    def step_success_rate(self) -> float:
        done = [s for s in self.steps if s.succeeded is not None]
        if not done:
            return 0.0
        return sum(1 for s in done if s.succeeded) / len(done)

    def explain(self) -> str:
        lines = [
            f"Procedure {self.procedure_id}  v{self.version}",
            f"  Family:        {self.problem_family}",
            f"  Signature:     {self.task_signature}",
            f"  Success score: {self.success_score:.3f}",
            f"  Transfer score:{self.transfer_score:.3f}",
            f"  Reuse count:   {self.reuse_count}",
            f"  Lifecycle:     {self.lifecycle_state.value}",
            f"  Selected for:  {self.selection_reason or 'N/A'}",
            f"  Used by:       {', '.join(self.used_by_tasks[-5:]) or 'none'}",
            "  Steps:",
        ]
        for s in self.steps:
            lines.append(repr(s))
        if self.failure_modes:
            lines.append("  Failure modes:")
            for fm in self.failure_modes:
                lines.append(f"    [{fm.failure_id}] {fm.description} (×{fm.frequency})")
        if self.recovery_strategies:
            lines.append("  Recovery strategies:")
            for rs in self.recovery_strategies:
                lines.append(f"    [{rs.strategy_id}] {rs.description} ({rs.success_rate:.2f})")
        return "\n".join(lines)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = {
            "procedure_id":        self.procedure_id,
            "problem_family":      self.problem_family,
            "task_signature":      self.task_signature,
            "steps":               [s.to_dict() for s in self.steps],
            "failure_modes":       [f.to_dict() for f in self.failure_modes],
            "recovery_strategies": [r.to_dict() for r in self.recovery_strategies],
            "success_score":       self.success_score,
            "reuse_count":         self.reuse_count,
            "last_used":           self.last_used,
            "survival_score":      self.survival_score,
            "transfer_score":      self.transfer_score,
            "version":             self.version,
            "parent_id":           self.parent_id,
            "used_by_tasks":       self.used_by_tasks,
            "selection_reason":    self.selection_reason,
            "creation_timestamp":  self.creation_timestamp,
            "lifecycle_state":     self.lifecycle_state.value,
            "embedding":           self.embedding,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProcedureTrace":
        d = dict(d)
        d["steps"]               = [ProcedureStep.from_dict(s) for s in d["steps"]]
        d["failure_modes"]       = [FailureMode.from_dict(f)   for f in d.get("failure_modes", [])]
        d["recovery_strategies"] = [RecoveryStrategy.from_dict(r) for r in d.get("recovery_strategies", [])]
        d["lifecycle_state"]     = ProcedureLifecycleState(d.get("lifecycle_state", "created"))
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ProcedureTrace":
        return cls.from_dict(json.loads(s))


# ── StructureMemoryRecordV2 ────────────────────────────────────────────────────

@dataclass
class StructureMemoryRecordV2:
    """
    Enhanced structure memory record with:
      - full procedure trace reference
      - failure mode tracking
      - recovery strategy links
      - versioning
      - family transfer metadata
    """
    structure_id:      str
    family_id:         int
    expert_id:         int
    task_type:         str
    task_signature:    str             # canonical fingerprint
    embedding:         List[float]     # stored as list (JSON-safe)

    # Scoring
    success_score:     float = 0.0
    transfer_score:    float = 0.0
    survival_score:    float = 1.0
    usage_count:       int   = 0
    timestamp:         float = field(default_factory=time.time)

    # Procedure linkage
    procedure_id:      Optional[str]          = None    # linked ProcedureTrace
    failure_modes:     List[str]              = field(default_factory=list)   # FailureMode ids
    recovery_strategy: Optional[str]          = None    # RecoveryStrategy id

    # Versioning
    version:           int              = 1
    parent_id:         Optional[str]   = None

    # Transfer metadata
    source_family:     Optional[str]   = None
    transferred_to:    List[str]       = field(default_factory=list)

    def overall_score(self) -> float:
        return (
            0.40 * self.success_score
            + 0.30 * self.transfer_score
            + 0.30 * self.survival_score
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StructureMemoryRecordV2":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "StructureMemoryRecordV2":
        return cls.from_dict(json.loads(s))
