"""
TAC-SCM-REAL001: Diagnostics Tracker

Lightweight, torch-free diagnostics accumulator for TAC-SCM training runs.

Usage
-----
tracker = SCMDiagnosticsTracker(cfg)
tracker.record(step=0, lm_loss=2.3, aux_losses={"discovery": 0.1}, metrics={...})
print(tracker.summary())
tracker.export_jsonl("reports/diag.jsonl")

The tracker records per-step rows and computes running statistics:
  - loss traces (lm, auxiliary, total)
  - memory fill rate over time
  - structure collapse metric
  - route entropy
  - per-condition counts
  - convergence detection (loss plateau heuristic)
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ── Per-step row ──────────────────────────────────────────────────────────────

@dataclass
class DiagnosticsRow:
    step:          int
    elapsed_s:     float
    lm_loss:       float
    total_loss:    float
    aux_losses:    Dict[str, float] = field(default_factory=dict)
    metrics:       Dict[str, float] = field(default_factory=dict)
    mem_fill_rate: float = 0.0
    mem_n_filled:  int   = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step":          self.step,
            "elapsed_s":     round(self.elapsed_s, 2),
            "lm_loss":       round(self.lm_loss,    4),
            "total_loss":    round(self.total_loss,  4),
            "mem_fill_rate": round(self.mem_fill_rate, 4),
            "mem_n_filled":  self.mem_n_filled,
        }
        for k, v in self.aux_losses.items():
            d[f"aux_{k}"] = round(v, 4)
        for k, v in self.metrics.items():
            d[f"metric_{k}"] = round(v, 4)
        return d


# ── Moving-window statistics ──────────────────────────────────────────────────

class _WindowStat:
    """Maintains a sliding-window mean and std for a scalar."""

    def __init__(self, maxlen: int = 100):
        self._buf: deque = deque(maxlen=maxlen)

    def update(self, v: float):
        if math.isfinite(v):
            self._buf.append(v)

    @property
    def mean(self) -> float:
        if not self._buf:
            return float("nan")
        return sum(self._buf) / len(self._buf)

    @property
    def std(self) -> float:
        if len(self._buf) < 2:
            return float("nan")
        mu = self.mean
        return math.sqrt(sum((x - mu) ** 2 for x in self._buf) / len(self._buf))

    @property
    def last(self) -> float:
        return self._buf[-1] if self._buf else float("nan")

    @property
    def min(self) -> float:
        return min(self._buf) if self._buf else float("nan")

    @property
    def max(self) -> float:
        return max(self._buf) if self._buf else float("nan")

    def __len__(self) -> int:
        return len(self._buf)


# ── Main tracker ──────────────────────────────────────────────────────────────

class SCMDiagnosticsTracker:
    """
    Accumulates per-step training diagnostics for TAC-SCM-REAL001.

    Parameters
    ----------
    window_size : int
        Number of recent steps to use for rolling statistics (default 100).
    plateau_patience : int
        Steps of flat loss before flagging convergence (default 200).
    plateau_delta : float
        Minimum relative improvement to reset plateau counter (default 1e-3).
    """

    def __init__(
        self,
        window_size:      int   = 100,
        plateau_patience: int   = 200,
        plateau_delta:    float = 1e-3,
    ):
        self.window_size      = window_size
        self.plateau_patience = plateau_patience
        self.plateau_delta    = plateau_delta

        self._rows:     List[DiagnosticsRow] = []
        self._t0:       float                = time.time()
        self._step0:    int                  = 0

        # Rolling windows
        self._lm_stat     = _WindowStat(window_size)
        self._total_stat  = _WindowStat(window_size)
        self._aux_stats:   Dict[str, _WindowStat] = defaultdict(lambda: _WindowStat(window_size))
        self._metric_stats: Dict[str, _WindowStat] = defaultdict(lambda: _WindowStat(window_size))
        self._mem_stat    = _WindowStat(window_size)

        # Plateau detection
        self._best_lm:     float = float("inf")
        self._plateau_ctr: int   = 0
        self._is_plateau:  bool  = False

        # Anomaly log
        self._anomalies: List[Dict[str, Any]] = []

    # ── Record ────────────────────────────────────────────────────────────────

    def record(
        self,
        step:          int,
        lm_loss:       float,
        aux_losses:    Optional[Dict[str, float]] = None,
        metrics:       Optional[Dict[str, float]] = None,
        mem_fill_rate: float                      = 0.0,
        mem_n_filled:  int                        = 0,
        total_loss:    Optional[float]            = None,
    ) -> "SCMDiagnosticsTracker":
        """Record one training step."""
        elapsed = time.time() - self._t0
        aux  = aux_losses or {}
        mets = metrics   or {}

        if total_loss is None:
            total_loss = lm_loss + sum(aux.values())

        row = DiagnosticsRow(
            step          = step,
            elapsed_s     = elapsed,
            lm_loss       = lm_loss,
            total_loss    = total_loss,
            aux_losses    = aux,
            metrics       = mets,
            mem_fill_rate = mem_fill_rate,
            mem_n_filled  = mem_n_filled,
        )
        self._rows.append(row)

        # Update rolling windows
        self._lm_stat.update(lm_loss)
        self._total_stat.update(total_loss)
        self._mem_stat.update(mem_fill_rate)
        for k, v in aux.items():
            self._aux_stats[k].update(v)
        for k, v in mets.items():
            self._metric_stats[k].update(v)

        # Anomaly detection
        self._check_anomalies(row)

        # Plateau detection
        if lm_loss < self._best_lm * (1 - self.plateau_delta):
            self._best_lm    = lm_loss
            self._plateau_ctr = 0
        else:
            self._plateau_ctr += 1
        self._is_plateau = (self._plateau_ctr >= self.plateau_patience)

        return self

    def _check_anomalies(self, row: DiagnosticsRow):
        """Detect NaN/Inf losses and sudden spikes."""
        if not math.isfinite(row.lm_loss):
            self._anomalies.append({
                "step": row.step,
                "kind": "nan_lm_loss",
                "value": row.lm_loss,
            })
        if (self._lm_stat.std > 0
                and len(self._lm_stat) >= 10
                and abs(row.lm_loss - self._lm_stat.mean) > 5 * self._lm_stat.std):
            self._anomalies.append({
                "step":  row.step,
                "kind":  "spike",
                "value": row.lm_loss,
                "mean":  self._lm_stat.mean,
                "std":   self._lm_stat.std,
            })

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_steps(self) -> int:
        return len(self._rows)

    @property
    def is_plateau(self) -> bool:
        return self._is_plateau

    @property
    def latest(self) -> Optional[DiagnosticsRow]:
        return self._rows[-1] if self._rows else None

    @property
    def anomalies(self) -> List[Dict[str, Any]]:
        return list(self._anomalies)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a multi-line human-readable summary of training so far."""
        if not self._rows:
            return "SCMDiagnosticsTracker: no data recorded yet."

        first = self._rows[0]
        last  = self._rows[-1]
        lines = [
            "═" * 65,
            "  TAC-SCM-REAL001 Training Diagnostics",
            "═" * 65,
            f"  Steps recorded    : {self.n_steps}  (step {first.step} → {last.step})",
            f"  Elapsed           : {last.elapsed_s:.1f}s",
            "",
            "  LM Loss (rolling window)",
            f"    last            : {self._lm_stat.last:.4f}",
            f"    mean            : {self._lm_stat.mean:.4f}",
            f"    std             : {self._lm_stat.std:.4f}",
            f"    min             : {self._lm_stat.min:.4f}",
            "",
            "  Total Loss",
            f"    last            : {self._total_stat.last:.4f}",
            f"    mean            : {self._total_stat.mean:.4f}",
        ]

        if self._aux_stats:
            lines.append("")
            lines.append("  Auxiliary Losses (last value)")
            for k, stat in sorted(self._aux_stats.items()):
                lines.append(f"    {k:<28s}: {stat.last:.4f}  (mean {stat.mean:.4f})")

        if self._metric_stats:
            lines.append("")
            lines.append("  Structure Metrics (last value)")
            for k, stat in sorted(self._metric_stats.items()):
                lines.append(f"    {k:<28s}: {stat.last:.4f}")

        lines.append("")
        lines.append("  Memory")
        lines.append(f"    fill rate (last): {self._mem_stat.last:.4f}")
        lines.append(f"    fill rate (mean): {self._mem_stat.mean:.4f}")

        lines.append("")
        lines.append(f"  Plateau detected  : {'YES' if self._is_plateau else 'no'}")
        lines.append(f"  Best LM loss      : {self._best_lm:.4f}")
        lines.append(f"  Anomalies         : {len(self._anomalies)}")

        if self._anomalies:
            for a in self._anomalies[-5:]:
                lines.append(f"    step={a['step']}  kind={a['kind']}  value={a.get('value', '?'):.4f}")

        lines.append("═" * 65)
        return "\n".join(lines)

    def stats_dict(self) -> Dict[str, Any]:
        """Return a structured dict of current statistics."""
        d: Dict[str, Any] = {
            "n_steps":      self.n_steps,
            "is_plateau":   self._is_plateau,
            "best_lm_loss": self._best_lm,
            "n_anomalies":  len(self._anomalies),
            "lm_loss": {
                "last": self._lm_stat.last,
                "mean": self._lm_stat.mean,
                "std":  self._lm_stat.std,
                "min":  self._lm_stat.min,
                "max":  self._lm_stat.max,
            },
            "total_loss": {
                "last": self._total_stat.last,
                "mean": self._total_stat.mean,
            },
            "mem_fill_rate": {
                "last": self._mem_stat.last,
                "mean": self._mem_stat.mean,
            },
            "aux_losses": {
                k: {"last": s.last, "mean": s.mean}
                for k, s in self._aux_stats.items()
            },
            "metrics": {
                k: {"last": s.last, "mean": s.mean}
                for k, s in self._metric_stats.items()
            },
        }
        return d

    # ── Export ────────────────────────────────────────────────────────────────

    def export_jsonl(self, path: str) -> str:
        """Write all rows as JSONL. Returns path written."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for row in self._rows:
                f.write(json.dumps(row.to_dict()) + "\n")
        return str(out)

    def export_summary_json(self, path: str) -> str:
        """Write stats_dict() as JSON. Returns path written."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.stats_dict(), indent=2))
        return str(out)

    # ── Convergence helper ────────────────────────────────────────────────────

    def lm_loss_sequence(self) -> List[float]:
        """Return the full LM loss trace."""
        return [r.lm_loss for r in self._rows]

    def loss_decreased_over(self, n_steps: int) -> bool:
        """True if lm_loss at end is lower than at n_steps ago."""
        if len(self._rows) < n_steps + 1:
            return False
        start = self._rows[-n_steps - 1].lm_loss
        end   = self._rows[-1].lm_loss
        return end < start

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Clear all accumulated state."""
        self._rows.clear()
        self._t0             = time.time()
        self._lm_stat        = _WindowStat(self.window_size)
        self._total_stat     = _WindowStat(self.window_size)
        self._aux_stats      = defaultdict(lambda: _WindowStat(self.window_size))
        self._metric_stats   = defaultdict(lambda: _WindowStat(self.window_size))
        self._mem_stat       = _WindowStat(self.window_size)
        self._best_lm        = float("inf")
        self._plateau_ctr    = 0
        self._is_plateau     = False
        self._anomalies.clear()
