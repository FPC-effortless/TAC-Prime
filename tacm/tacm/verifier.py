"""
TAC-SM Verifier / Reward Head

Outputs:
  { success_probability, confidence, failure_reason }

Used for:
  - code repair verification
  - test pass/fail prediction
  - math correctness
  - plan feasibility
  - tool use success

Verifier rewards update Structure Memory scores.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VerifierConfig


FAILURE_CLASSES = [
    "SyntaxError",
    "LogicError",
    "MissingImport",
    "TypeMismatch",
    "IndexError",
    "PlanIncomplete",
    "ToolFailure",
    "Unknown",
]


class VerifierHead(nn.Module):
    """
    Predicts:
      - success_prob  : P(output is correct) in [0, 1]
      - confidence    : model's confidence in its prediction in [0, 1]
      - failure_logits: distribution over failure classes
    """

    def __init__(self, d_model: int, cfg: VerifierConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_dim

        self.pool = nn.Sequential(
            nn.Linear(d_model, H, bias=True),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )

        self.success_head  = nn.Linear(H, 1, bias=True)
        self.confidence_head = nn.Linear(H, 1, bias=True)
        self.failure_head  = nn.Linear(H, cfg.n_failure_classes, bias=True)

        self._init()

    def _init(self):
        nn.init.normal_(self.pool[0].weight, std=0.02)
        nn.init.zeros_(self.pool[0].bias)
        for head in [self.success_head, self.confidence_head, self.failure_head]:
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(self, hidden: torch.Tensor) -> "VerifierOutput":
        """
        hidden : (B, T, d_model)
        Pool over T dimension (mean) → (B, d_model), then predict.
        """
        pooled = hidden.mean(dim=1)      # (B, d_model)
        h      = self.pool(pooled)       # (B, H)

        success_prob    = torch.sigmoid(self.success_head(h)).squeeze(-1)    # (B,)
        confidence      = torch.sigmoid(self.confidence_head(h)).squeeze(-1) # (B,)
        failure_logits  = self.failure_head(h)                                # (B, n_failures)

        return VerifierOutput(
            success_prob   = success_prob,
            confidence     = confidence,
            failure_logits = failure_logits,
        )


class VerifierOutput:
    __slots__ = ["success_prob", "confidence", "failure_logits"]

    def __init__(
        self,
        success_prob:   torch.Tensor,
        confidence:     torch.Tensor,
        failure_logits: torch.Tensor,
    ):
        self.success_prob   = success_prob      # (B,)
        self.confidence     = confidence        # (B,)
        self.failure_logits = failure_logits    # (B, n_failures)

    @property
    def failure_probs(self) -> torch.Tensor:
        return F.softmax(self.failure_logits, dim=-1)

    @property
    def predicted_failure(self) -> torch.Tensor:
        return self.failure_logits.argmax(-1)   # (B,)

    def top_failure_name(self, batch_idx: int = 0) -> str:
        idx = self.predicted_failure[batch_idx].item()
        return FAILURE_CLASSES[idx] if idx < len(FAILURE_CLASSES) else "Unknown"

    def to_dict(self, batch_idx: int = 0) -> dict:
        return {
            "success_probability": self.success_prob[batch_idx].item(),
            "confidence":          self.confidence[batch_idx].item(),
            "failure_reason":      self.top_failure_name(batch_idx),
        }


class VerifierLoss(nn.Module):
    """
    Binary cross-entropy on success_prob + cross-entropy on failure class.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        verifier_out:   VerifierOutput,
        success_labels: Optional[torch.Tensor] = None,   # (B,) float in [0, 1]
        failure_labels: Optional[torch.Tensor] = None,   # (B,) long
    ) -> dict:
        losses = {}
        device = verifier_out.success_prob.device

        # Success prediction
        if success_labels is not None:
            losses["success"] = F.binary_cross_entropy(
                verifier_out.success_prob,
                success_labels.to(device).float(),
            )
        else:
            losses["success"] = torch.tensor(0.0, device=device)

        # Failure classification
        if failure_labels is not None:
            valid = failure_labels >= 0
            if valid.any():
                losses["failure_cls"] = F.cross_entropy(
                    verifier_out.failure_logits[valid],
                    failure_labels[valid].to(device),
                )
            else:
                losses["failure_cls"] = torch.tensor(0.0, device=device)
        else:
            losses["failure_cls"] = torch.tensor(0.0, device=device)

        losses["total"] = losses["success"] + 0.3 * losses["failure_cls"]
        return losses


class RewardBridge(nn.Module):
    """
    Converts verifier output into a reward signal that updates Structure Memory.
    No gradient — this is a pure information bridge.
    """

    def __init__(self, success_threshold: float = 0.7):
        super().__init__()
        self.threshold = success_threshold

    @torch.no_grad()
    def compute_memory_reward(
        self,
        verifier_out:   VerifierOutput,
        batch_idx:      int = 0,
    ) -> Tuple[float, float, float]:
        """
        Returns (success_delta, transfer_delta, survival_delta)
        for updating a StructureRecord.
        """
        sp = verifier_out.success_prob[batch_idx].item()
        c  = verifier_out.confidence[batch_idx].item()

        if sp >= self.threshold:
            # Reward
            success_delta  = 0.05 * sp * c
            transfer_delta = 0.02 * sp
            survival_delta = 0.03 * sp
        else:
            # Penalty
            success_delta  = -0.02 * (1 - sp)
            transfer_delta = 0.0
            survival_delta = -0.01 * (1 - sp)

        return success_delta, transfer_delta, survival_delta
