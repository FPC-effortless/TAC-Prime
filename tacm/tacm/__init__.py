"""
TAC-SM — Token–Algorithm–Coherence with Structure Memory
Research-grade model for learning and transferring reusable computational structures.
"""

try:
    from .config import TACSMConfig, tacm_30m, tacm_100m, tacm_150m, CONFIGS
    from .model  import TACSM, TACSMOutput
    from .backbone import TransformerBackbone
    from .concept_volume import ConceptVolume, ConceptVolumeOutput
    from .router import StructureRouter, StructureRoutingOutput
    from .experts import MoELayer
    from .memory import StructureMemory, StructureRecord
    from .procedural_memory import ProceduralMemory, ProcedureRecord
    from .survival import SurvivalField, StructureLifecycleTracker, LifecycleState
    from .verifier import VerifierHead, VerifierOutput
    from .multi_token import MultiTokenPredictionModule
    from .evaluation import Evaluator, EvalSample, EvalResult
    from .agent import RepositoryRepairAgent, BugReport, AgentTrace, RepairPlan, Patch
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    # Config
    "TACSMConfig", "tacm_30m", "tacm_100m", "tacm_150m", "CONFIGS",
    # Model
    "TACSM", "TACSMOutput",
    # Components
    "TransformerBackbone",
    "ConceptVolume", "ConceptVolumeOutput",
    "StructureRouter", "StructureRoutingOutput",
    "MoELayer",
    "StructureMemory", "StructureRecord",
    "ProceduralMemory", "ProcedureRecord",
    "SurvivalField", "StructureLifecycleTracker", "LifecycleState",
    "VerifierHead", "VerifierOutput",
    "MultiTokenPredictionModule",
    # Evaluation
    "Evaluator", "EvalSample", "EvalResult",
    # Agent
    "RepositoryRepairAgent", "BugReport", "AgentTrace", "RepairPlan", "Patch",
]
