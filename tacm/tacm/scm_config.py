"""
TAC-SCM-REAL001: Configuration

TACSCMConfig is flat (not nested like TACSMConfig) to keep training CLI simple.
All SCM subsystems can be independently toggled for ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TACSCMConfig:
    # ── Language model backbone ───────────────────────────────────────────────
    vocab_size:       int   = 32000
    d_model:          int   = 512
    n_layers:         int   = 8
    n_heads:          int   = 8
    n_kv_heads:       int   = 2        # grouped-query attention (< n_heads)
    d_ff:             int   = 2048     # feed-forward inner dim
    max_seq_len:      int   = 2048
    dropout:          float = 0.1
    rope_base:        float = 10000.0
    norm_eps:         float = 1e-5

    # ── SCM structure dimensions ──────────────────────────────────────────────
    d_structure:      int   = 128      # dimensionality of structure embeddings
    n_structure_slots: int  = 256      # slots in StructureMemory bank
    n_identity_slots:  int  = 16       # active slots in StructureIdentityState

    # ── SCM block insertion ───────────────────────────────────────────────────
    scm_layer_interval: int = 2        # insert SCM block every N transformer layers

    # ── SCM subsystem toggles (for ablation) ─────────────────────────────────
    enable_scm:                     bool = True
    enable_structure_discovery:     bool = True
    enable_structure_compiler:      bool = True
    enable_structure_identity:      bool = True
    enable_structure_memory:        bool = True
    enable_nsf_survival:            bool = True
    enable_dpsl_refinement:         bool = True
    enable_language_structure_fusion: bool = True
    enable_state_carry:             bool = True
    enable_memory_write:            bool = True

    # ── Loss weights ──────────────────────────────────────────────────────────
    discovery_loss_weight:        float = 0.10   # JEPA prediction + spread + covariance
    jepa_prediction_weight:       float = 0.50   # within discovery
    temporal_consistency_weight:  float = 0.20   # within discovery
    spread_loss_weight:           float = 0.15   # VICReg-style variance
    covariance_loss_weight:       float = 0.15   # VICReg-style covariance
    structure_reuse_weight:       float = 0.05
    survival_loss_weight:         float = 0.05
    compression_loss_weight:      float = 0.02
    transfer_loss_weight:         float = 0.05
    route_entropy_weight:         float = 0.01
    refinement_loss_weight:       float = 0.02

    # ── SCM runtime hyperparameters ───────────────────────────────────────────
    memory_write_rate:      float = 0.3      # fraction of time to write to memory
    structure_dropout:      float = 0.1
    identity_dropout:       float = 0.1
    survival_decay:         float = 0.99     # per-step survival score decay
    max_active_structures:  int   = 32       # max slots populated per call
    n_structure_candidates: int   = 8        # candidates extracted by discovery

    # ── Discovery-specific ────────────────────────────────────────────────────
    stop_gradient_target:   bool  = True     # stop-gradient on JEPA target encoder
    future_offset:          int   = 4        # tokens ahead for future prediction
    target_ema_decay:       float = 0.996    # EMA decay for target encoder

    # ── Compiler-specific ─────────────────────────────────────────────────────
    spread_regularizer_type: str  = "vicreg" # "vicreg" or "std"

    # ── Identity-specific ─────────────────────────────────────────────────────
    identity_state_decay:    float = 0.9
    identity_residual_scale: float = 0.5

    # ── Training ──────────────────────────────────────────────────────────────
    use_gradient_checkpointing: bool = True
    tie_lm_head:                bool = True

    # ── Presets ───────────────────────────────────────────────────────────────

    @classmethod
    def small(cls) -> "TACSCMConfig":
        """~30M parameter model for fast iteration."""
        return cls(
            vocab_size=32000, d_model=256, n_layers=4, n_heads=4,
            n_kv_heads=2, d_ff=1024, d_structure=64, n_structure_slots=128,
            n_identity_slots=8, scm_layer_interval=2,
        )

    @classmethod
    def base(cls) -> "TACSCMConfig":
        """~100M parameter model — standard research config."""
        return cls(
            vocab_size=32000, d_model=512, n_layers=8, n_heads=8,
            n_kv_heads=2, d_ff=2048, d_structure=128, n_structure_slots=256,
            n_identity_slots=16, scm_layer_interval=2,
        )

    @classmethod
    def medium(cls) -> "TACSCMConfig":
        """~350M parameter model — extended experiments."""
        return cls(
            vocab_size=32000, d_model=1024, n_layers=16, n_heads=16,
            n_kv_heads=4, d_ff=4096, d_structure=256, n_structure_slots=512,
            n_identity_slots=32, scm_layer_interval=2,
        )

    @classmethod
    def no_scm(cls) -> "TACSCMConfig":
        """Pure transformer baseline (SCM fully disabled)."""
        cfg = cls.base()
        cfg.enable_scm = False
        return cfg

    @classmethod
    def discovery_only(cls) -> "TACSCMConfig":
        """SCM with only discovery — no memory/identity/survival."""
        cfg = cls.base()
        cfg.enable_structure_identity = False
        cfg.enable_structure_memory   = False
        cfg.enable_nsf_survival       = False
        cfg.enable_dpsl_refinement    = False
        cfg.enable_memory_write       = False
        return cfg
