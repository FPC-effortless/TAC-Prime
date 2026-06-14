"""
TAC-SM Configuration
Defines all hyperparameters for the full model.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2          # < n_heads → grouped-query attention
    ffn_dim: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.1
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    use_flash_attn: bool = True
    gradient_checkpointing: bool = True


@dataclass
class ConceptVolumeConfig:
    n_concept_families: int = 16
    volume_dim: int = 64          # dimensionality of concept center
    min_variance: float = 1e-4
    max_variance: float = 10.0
    lambda_consistency: float = 1.0
    lambda_separation: float = 0.5
    lambda_hierarchy: float = 0.3
    lambda_temporal: float = 0.2
    margin: float = 1.0


@dataclass
class RouterConfig:
    n_families: int = 8
    n_experts: int = 16
    top_k: int = 2
    lambda_entropy: float = 0.01
    lambda_load_balance: float = 0.01
    temperature: float = 1.0


@dataclass
class ExpertConfig:
    n_experts: int = 16
    expert_hidden_dim: int = 512
    shared_expert_dim: int = 512
    top_k: int = 2
    dropout: float = 0.1


@dataclass
class MemoryConfig:
    max_structures: int = 4096
    embedding_dim: int = 512
    retrieval_top_k: int = 8
    write_threshold: float = 0.6    # min success_score to write
    prune_threshold: float = 0.1    # min survival_score to keep
    decay_rate: float = 0.99
    similarity_metric: str = "cosine"


@dataclass
class SurvivalConfig:
    w_retention: float = 0.25
    w_transfer: float = 0.25
    w_robustness: float = 0.25
    w_reuse: float = 0.25
    decay_steps: int = 1000
    prune_every: int = 500


@dataclass
class VerifierConfig:
    hidden_dim: int = 256
    n_failure_classes: int = 8
    dropout: float = 0.1


@dataclass
class MultiTokenConfig:
    n_future_tokens: int = 8
    n_future_actions: int = 4
    action_vocab_size: int = 64
    w_next_token: float = 1.0
    w_multi_token: float = 0.5
    w_procedure: float = 0.3


@dataclass
class TrainingConfig:
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    max_steps: int = 100000
    weight_decay: float = 0.1
    clip_grad_norm: float = 1.0
    dtype: str = "bfloat16"
    seed: int = 42
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 50
    output_dir: str = "./checkpoints"

    # Loss weights
    w_next_token: float = 1.0
    w_multi_token: float = 0.5
    w_volume: float = 0.2
    w_family_route: float = 0.3
    w_expert_route: float = 0.3
    w_structure_memory: float = 0.5
    w_transfer: float = 0.4
    w_survival: float = 0.2
    w_verifier: float = 0.5


@dataclass
class TACSMConfig:
    name: str = "tacm-30m"
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    concept_volume: ConceptVolumeConfig = field(default_factory=ConceptVolumeConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    survival: SurvivalConfig = field(default_factory=SurvivalConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    multi_token: MultiTokenConfig = field(default_factory=MultiTokenConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ── Preset Configs ────────────────────────────────────────────────────────────

def tacm_30m() -> TACSMConfig:
    """Stage 1: 30M parameters, Kaggle-scale training."""
    cfg = TACSMConfig(name="tacm-30m")
    cfg.transformer.d_model = 512
    cfg.transformer.n_layers = 8
    cfg.transformer.n_heads = 8
    cfg.transformer.n_kv_heads = 2
    cfg.transformer.ffn_dim = 2048
    cfg.concept_volume.volume_dim = 64
    cfg.expert.n_experts = 8
    cfg.expert.expert_hidden_dim = 256
    cfg.expert.shared_expert_dim = 256
    cfg.router.n_experts = 8
    cfg.memory.max_structures = 2048
    return cfg


def tacm_100m() -> TACSMConfig:
    """Stage 2: 100M parameters, repository repair benchmark."""
    cfg = TACSMConfig(name="tacm-100m")
    cfg.transformer.d_model = 768
    cfg.transformer.n_layers = 12
    cfg.transformer.n_heads = 12
    cfg.transformer.n_kv_heads = 4
    cfg.transformer.ffn_dim = 3072
    cfg.transformer.max_seq_len = 4096
    cfg.concept_volume.volume_dim = 96
    cfg.expert.n_experts = 16
    cfg.expert.expert_hidden_dim = 512
    cfg.expert.shared_expert_dim = 512
    cfg.router.n_experts = 16
    cfg.memory.max_structures = 8192
    cfg.training.batch_size = 4
    cfg.training.grad_accum_steps = 8
    return cfg


def tacm_150m() -> TACSMConfig:
    """Stage 3: 150M parameters, agent loop."""
    cfg = TACSMConfig(name="tacm-150m")
    cfg.transformer.d_model = 1024
    cfg.transformer.n_layers = 16
    cfg.transformer.n_heads = 16
    cfg.transformer.n_kv_heads = 4
    cfg.transformer.ffn_dim = 4096
    cfg.transformer.max_seq_len = 8192
    cfg.concept_volume.volume_dim = 128
    cfg.expert.n_experts = 32
    cfg.expert.expert_hidden_dim = 768
    cfg.expert.shared_expert_dim = 768
    cfg.router.n_experts = 32
    cfg.memory.max_structures = 16384
    cfg.training.batch_size = 2
    cfg.training.grad_accum_steps = 16
    return cfg


CONFIGS = {
    "tacm-30m": tacm_30m,
    "tacm-100m": tacm_100m,
    "tacm-150m": tacm_150m,
}
