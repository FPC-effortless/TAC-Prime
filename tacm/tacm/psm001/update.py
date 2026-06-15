"""
TAC-PSM-001: Procedure Update After Verification

update_procedure_after_verification() is the primary interface.

It receives a VerificationSignal (success / failure + details) and:
  - adjusts scores
  - logs failure modes and recovery strategies
  - optionally forks a new versioned procedure on failure
  - drives lifecycle transitions
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from .records import (
    ProcedureTrace,
    ProcedureStep,
    FailureMode,
    RecoveryStrategy,
    ProcedureLifecycleState,
)
from .store import ProceduralMemoryStore


@dataclass
class VerificationSignal:
    """
    Output of the verifier/test runner, passed to update logic.
    """
    procedure_id:     str
    task_signature:   str
    success:          bool

    # Failure details (populated on failure)
    failed_step:      Optional[int]  = None    # index of the step that failed
    error_type:       str            = ""      # e.g. "ImportError"
    error_message:    str            = ""
    failure_family:   str            = ""      # task family where failure occurred

    # Recovery attempt
    recovery_applied: bool           = False
    recovery_steps:   List[str]      = field(default_factory=list)
    recovery_success: bool           = False

    # Transfer context
    is_transfer:      bool           = False   # was this a cross-family task?
    source_family:    str            = ""
    target_family:    str            = ""

    # Timing
    duration_ms:      float          = 0.0


@dataclass
class UpdateResult:
    """Result of update_procedure_after_verification()."""
    procedure_id:       str
    forked_id:          Optional[str]    # new procedure_id if forked
    success_delta:      float
    transfer_delta:     float
    survival_delta:     float
    version_bumped:     bool
    new_failure_logged: bool
    new_recovery_logged: bool
    lifecycle_before:   str
    lifecycle_after:    str
    message:            str

    def to_dict(self) -> dict:
        return {
            "procedure_id":       self.procedure_id,
            "forked_id":          self.forked_id,
            "success_delta":      self.success_delta,
            "transfer_delta":     self.transfer_delta,
            "survival_delta":     self.survival_delta,
            "version_bumped":     self.version_bumped,
            "new_failure_logged": self.new_failure_logged,
            "new_recovery_logged": self.new_recovery_logged,
            "lifecycle_before":   self.lifecycle_before,
            "lifecycle_after":    self.lifecycle_after,
            "message":            self.message,
        }


# ── Main function ─────────────────────────────────────────────────────────────

def update_procedure_after_verification(
    signal:     VerificationSignal,
    store:      ProceduralMemoryStore,
    fork_on_failure: bool = True,
    fork_threshold:  int  = 2,       # fork after this many failures of same type
) -> UpdateResult:
    """
    Primary update function.

    Logic:
      SUCCESS:
        - increase success_score (+0.05)
        - increase survival_score (+0.02)
        - if is_transfer: increase transfer_score (+0.05)
        - advance lifecycle toward ACTIVE / SPECIALISED / TRANSFERRED

      FAILURE:
        - decrease success_score (-0.03)
        - decrease survival_score (-0.02)
        - log new FailureMode (or increment existing)
        - if recovery attempted and succeeded: log RecoveryStrategy
        - if failure count for this error_type >= fork_threshold: fork procedure
        - bump version on fork

    Returns UpdateResult with all deltas and fork info.
    """
    proc = store.get(signal.procedure_id)
    if proc is None:
        return UpdateResult(
            procedure_id       = signal.procedure_id,
            forked_id          = None,
            success_delta      = 0.0,
            transfer_delta     = 0.0,
            survival_delta     = 0.0,
            version_bumped     = False,
            new_failure_logged = False,
            new_recovery_logged = False,
            lifecycle_before   = "unknown",
            lifecycle_after    = "unknown",
            message            = f"Procedure {signal.procedure_id} not found in store.",
        )

    lifecycle_before = proc.lifecycle_state.value

    # ── Deltas ────────────────────────────────────────────────────────────────
    success_delta  = +0.05 if signal.success else -0.03
    survival_delta = +0.02 if signal.success else -0.02
    transfer_delta = +0.05 if (signal.success and signal.is_transfer) else 0.0

    new_failure_logged  = False
    new_recovery_logged = False
    version_bumped      = False
    forked_id: Optional[str] = None

    # ── Failure mode logging ───────────────────────────────────────────────────
    new_failure:    Optional[FailureMode]      = None
    new_recovery:   Optional[RecoveryStrategy] = None

    if not signal.success and signal.error_type:
        fid        = f"fail_{signal.error_type}_{signal.failed_step}"
        existing   = {f.failure_id: f for f in proc.failure_modes}

        if fid in existing:
            # Increment existing failure mode (handled inside store.update)
            new_failure = existing[fid]
            new_failure.frequency += 1
        else:
            new_failure = FailureMode(
                failure_id  = fid,
                description = (
                    f"{signal.error_type} at step {signal.failed_step}: "
                    f"{signal.error_message[:120]}"
                ),
                frequency   = 1,
                family      = signal.failure_family,
                step_index  = signal.failed_step if signal.failed_step is not None else -1,
                error_type  = signal.error_type,
            )
        new_failure_logged = True

        # Fork if this failure type has now crossed the threshold
        total_freq = sum(
            f.frequency for f in proc.failure_modes
            if f.error_type == signal.error_type
        ) + 1   # +1 for the new one we just logged

        if fork_on_failure and total_freq >= fork_threshold:
            forked_id = _fork_procedure(proc, store, signal)
            version_bumped = True

    # ── Recovery strategy logging ─────────────────────────────────────────────
    if signal.recovery_applied and signal.recovery_steps:
        rs_id = f"recovery_{signal.error_type}_{signal.procedure_id[:8]}"
        new_recovery = RecoveryStrategy(
            strategy_id       = rs_id,
            failure_id        = (new_failure.failure_id
                                 if new_failure else "unknown"),
            description       = f"Recovery for {signal.error_type}",
            recovery_steps    = signal.recovery_steps,
            success_rate      = 1.0 if signal.recovery_success else 0.0,
            application_count = 1,
        )
        new_recovery_logged = True

    # ── Apply update ──────────────────────────────────────────────────────────
    updated = store.update(
        procedure_id    = signal.procedure_id,
        success_delta   = success_delta,
        transfer_delta  = transfer_delta,
        survival_delta  = survival_delta,
        new_failure     = new_failure,
        new_recovery    = new_recovery,
        task_signature  = signal.task_signature,
        version_bump    = version_bumped and forked_id is None,  # bump original if no fork
    )

    lifecycle_after = updated.lifecycle_state.value if updated else "unknown"

    msg_parts = [
        "SUCCESS" if signal.success else "FAILURE",
        f"Δsuccess={success_delta:+.3f}",
        f"Δsurvival={survival_delta:+.3f}",
    ]
    if transfer_delta:
        msg_parts.append(f"Δtransfer={transfer_delta:+.3f}")
    if forked_id:
        msg_parts.append(f"forked→{forked_id}")

    return UpdateResult(
        procedure_id        = signal.procedure_id,
        forked_id           = forked_id,
        success_delta       = success_delta,
        transfer_delta      = transfer_delta,
        survival_delta      = survival_delta,
        version_bumped      = version_bumped,
        new_failure_logged  = new_failure_logged,
        new_recovery_logged = new_recovery_logged,
        lifecycle_before    = lifecycle_before,
        lifecycle_after     = lifecycle_after,
        message             = "  ".join(msg_parts),
    )


def _fork_procedure(
    original: ProcedureTrace,
    store:    ProceduralMemoryStore,
    signal:   VerificationSignal,
) -> str:
    """
    Create a forked/versioned copy of the procedure with recovery-aware steps.

    If a successful recovery was applied, the fork adopts the recovery steps
    directly as its procedure — they are the validated correct steps.
    If no recovery was available, the original steps are kept and a fallback
    step is appended.

    The fork gets a fresh success_score but inherits transfer_score and survival.
    """
    if signal.recovery_steps and signal.recovery_success:
        # Recovery succeeded: fork IS the recovery procedure
        new_steps = list(signal.recovery_steps)
    elif signal.recovery_steps:
        # Recovery attempted but outcome unknown: merge original + recovery
        new_steps = [s.action for s in original.steps] + list(signal.recovery_steps)
    else:
        # No recovery info: keep original, add diagnostic step
        new_steps = [s.action for s in original.steps]
        new_steps.append(f"[handle] {signal.error_type}: {signal.error_message[:80]}")

    import numpy as np
    forked = store.build(
        problem_family    = original.problem_family,
        task_signature    = signal.task_signature,
        steps             = new_steps,
        embedding         = (np.array(original.embedding, dtype=np.float32)
                             if original.embedding else None),
        success_score     = max(0.0, original.success_score - 0.1),
        parent_id         = original.procedure_id,
        selection_reason  = f"Forked from {original.procedure_id} after {signal.error_type}",
    )
    forked.version = original.version + 1
    return forked.procedure_id


# ── Batch update ───────────────────────────────────────────────────────────────

def batch_update(
    signals: List[VerificationSignal],
    store:   ProceduralMemoryStore,
    **kwargs,
) -> List[UpdateResult]:
    return [update_procedure_after_verification(s, store, **kwargs) for s in signals]
