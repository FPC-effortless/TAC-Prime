"""
TAC-SM Multi-Token Procedure Prediction

Predicts:
  - next tokens (standard LM head)
  - next 4-8 future tokens (multi-token prediction)
  - next action in repair/plan sequence
  - next tool call
  - next repair decision / patch region

Loss: weighted combination of
  - next_token_loss
  - multi_token_loss
  - procedure_prediction_loss
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultiTokenConfig, TransformerConfig


class LMHead(nn.Module):
    """Standard next-token prediction head."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, T, d_model) → logits: (B, T, vocab_size)"""
        return self.proj(hidden)


class MultiTokenHead(nn.Module):
    """
    Predicts the next N future tokens from each position.
    Uses independent projection heads for each future offset k ∈ [1, n_future].

    Implementation: one linear per offset, tied to the main vocab size.
    Targets are shifted input_ids by k positions.
    """

    def __init__(self, d_model: int, vocab_size: int, n_future: int):
        super().__init__()
        self.n_future = n_future
        # Separate head per offset
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_future)
        ])
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, T, d_model)
        Returns stacked logits: (B, T, n_future, vocab_size)
        """
        return torch.stack([h(hidden) for h in self.heads], dim=2)

    def loss(
        self,
        hidden:    torch.Tensor,
        input_ids: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        hidden    : (B, T, d_model)
        input_ids : (B, T_full) — original token ids (unshifted)
        """
        B, T, D   = hidden.shape
        all_logits = self.forward(hidden)  # (B, T, n_future, V)

        total_loss = torch.tensor(0.0, device=hidden.device)
        count      = 0

        for k in range(self.n_future):
            offset = k + 1
            # Tokens we can predict: positions 0..T-offset-1
            # Targets: positions offset..T-1 from input_ids
            if T - offset <= 0:
                break
            logits_k = all_logits[:, : T - offset, k, :]  # (B, T-offset, V)
            targets_k = input_ids[:, offset : offset + (T - offset)]  # (B, T-offset)

            if targets_k.shape[1] == 0:
                break

            loss_k = F.cross_entropy(
                logits_k.reshape(-1, logits_k.shape[-1]),
                targets_k.reshape(-1),
                ignore_index=-100,
                reduction=reduction,
            )
            total_loss = total_loss + loss_k
            count      += 1

        return total_loss / max(count, 1)


class ActionHead(nn.Module):
    """
    Predicts the next action token in a repair/plan/tool-use sequence.
    Action vocabulary is smaller than token vocabulary.
    """

    def __init__(self, d_model: int, action_vocab_size: int, n_future_actions: int = 4):
        super().__init__()
        self.n_future = n_future_actions
        self.heads    = nn.ModuleList([
            nn.Linear(d_model, action_vocab_size, bias=False)
            for _ in range(n_future_actions)
        ])
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, T, d_model) → (B, T, n_future_actions, action_vocab)"""
        return torch.stack([h(hidden) for h in self.heads], dim=2)

    def loss(
        self,
        hidden:         torch.Tensor,
        action_labels:  torch.Tensor,  # (B, T, n_future_actions) — -100 for ignore
    ) -> torch.Tensor:
        """Compute cross-entropy loss for each future action slot."""
        B, T, D = hidden.shape
        logits  = self.forward(hidden)   # (B, T, n_actions, A)
        total   = torch.tensor(0.0, device=hidden.device)
        count   = 0
        for k in range(self.n_future):
            l   = logits[:, :, k, :]             # (B, T, A)
            tgt = action_labels[:, :, k]          # (B, T)
            valid = tgt >= 0
            if not valid.any():
                continue
            total = total + F.cross_entropy(
                l.reshape(-1, l.shape[-1]),
                tgt.reshape(-1).clamp(min=0),
                ignore_index=0,
                reduction="mean",
            )
            count += 1
        return total / max(count, 1)


class MultiTokenPredictionModule(nn.Module):
    """
    Full multi-token prediction module combining:
      - LM head (next-token)
      - Multi-token head (n_future tokens)
      - Action head (next n_future_actions)
    """

    def __init__(self, d_model: int, tc: TransformerConfig, cfg: MultiTokenConfig):
        super().__init__()
        self.cfg           = cfg
        self.lm_head       = LMHead(d_model, tc.vocab_size)
        self.mt_head       = MultiTokenHead(d_model, tc.vocab_size, cfg.n_future_tokens)
        self.action_head   = ActionHead(d_model, cfg.action_vocab_size, cfg.n_future_actions)

    def forward(self, hidden: torch.Tensor) -> "MultiTokenOutput":
        lm_logits     = self.lm_head(hidden)
        mt_logits     = self.mt_head(hidden)
        action_logits = self.action_head(hidden)
        return MultiTokenOutput(
            lm_logits=lm_logits,
            mt_logits=mt_logits,
            action_logits=action_logits,
        )

    def compute_loss(
        self,
        hidden:         torch.Tensor,
        input_ids:      torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
        action_labels:  Optional[torch.Tensor] = None,
    ) -> dict:
        """
        input_ids     : (B, T) — full unshifted ids (for multi-token targets)
        labels        : (B, T) — shifted -100 masked targets for LM head
        action_labels : (B, T, n_future_actions) optional

        Returns dict of loss tensors.
        """
        losses = {}
        device = hidden.device

        # Next-token loss
        if labels is not None:
            lm_logits = self.lm_head(hidden)
            losses["next_token"] = F.cross_entropy(
                lm_logits.reshape(-1, lm_logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            losses["next_token"] = torch.tensor(0.0, device=device)

        # Multi-token loss
        losses["multi_token"] = self.mt_head.loss(hidden, input_ids)

        # Action loss
        if action_labels is not None:
            losses["procedure"] = self.action_head.loss(hidden, action_labels)
        else:
            losses["procedure"] = torch.tensor(0.0, device=device)

        cfg = self.cfg
        losses["total"] = (
            cfg.w_next_token  * losses["next_token"]
            + cfg.w_multi_token * losses["multi_token"]
            + cfg.w_procedure   * losses["procedure"]
        )
        return losses


class MultiTokenOutput:
    __slots__ = ["lm_logits", "mt_logits", "action_logits"]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def next_token_ids(self) -> torch.Tensor:
        return self.lm_logits.argmax(-1)

    def next_action_ids(self) -> torch.Tensor:
        return self.action_logits[:, :, 0, :].argmax(-1)
