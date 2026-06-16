"""
TAC-SM — Token–Algorithm–Coherence with Structure Memory
Research-grade model for learning and transferring reusable computational structures.

Includes:
  - Legacy TAC-SM architecture (TACSM)
  - TAC-SCM-REAL001: real trainable Structure-Native Language Model (TACSCMLanguageModel)
"""

# ── Legacy TAC-SM (torch-dependent) ───────────────────────────────────────────
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

# ── TAC-SCM-REAL001: always-importable (config, types, diagnostics, dataset) ──
from .scm_config      import TACSCMConfig
from .scm_diagnostics import SCMDiagnosticsTracker, DiagnosticsRow

from .data.scm_dataset import (
    SCMSample, SCMDataset, SCMDataCollator,
    make_synthetic_repair_dataset,
)

# ── TAC-SCM-REAL001: torch-dependent components ───────────────────────────────
try:
    from .scm_types  import (
        StructureObject, StructureBatch,
        StructureDiscoveryOutput, StructureCompilerOutput,
        StructureIdentityState, StructureMemoryOutput,
        SurvivalOutput, DPSLRefinementOutput,
        TACSCMOutput,
    )
    from .scm_discovery  import StructureDiscoveryLayer
    from .scm_compiler   import StructureCompiler
    from .scm_identity   import StructureIdentityFieldLayer
    from .scm_memory     import StructureMemory as SCMStructureMemory
    from .scm_block      import IntegratedStructureLanguageBlock, SCMBlockOutput
    from .scm_model      import TACSCMLanguageModel
    _SCM_TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _SCM_TORCH_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    # ── Legacy TAC-SM ──────────────────────────────────────────────────────────
    "TACSMConfig", "tacm_30m", "tacm_100m", "tacm_150m", "CONFIGS",
    "TACSM", "TACSMOutput",
    "TransformerBackbone",
    "ConceptVolume", "ConceptVolumeOutput",
    "StructureRouter", "StructureRoutingOutput",
    "MoELayer",
    "StructureMemory", "StructureRecord",
    "ProceduralMemory", "ProcedureRecord",
    "SurvivalField", "StructureLifecycleTracker", "LifecycleState",
    "VerifierHead", "VerifierOutput",
    "MultiTokenPredictionModule",
    "Evaluator", "EvalSample", "EvalResult",
    "RepositoryRepairAgent", "BugReport", "AgentTrace", "RepairPlan", "Patch",
    # ── TAC-SCM-REAL001: always available ─────────────────────────────────────
    "TACSCMConfig",
    "SCMDiagnosticsTracker", "DiagnosticsRow",
    "SCMSample", "SCMDataset", "SCMDataCollator", "make_synthetic_repair_dataset",
    # ── TAC-SCM-REAL001: torch-dependent ──────────────────────────────────────
    "StructureObject", "StructureBatch",
    "StructureDiscoveryOutput", "StructureCompilerOutput",
    "StructureIdentityState", "StructureMemoryOutput",
    "SurvivalOutput", "DPSLRefinementOutput",
    "TACSCMOutput",
    "StructureDiscoveryLayer",
    "StructureCompiler",
    "StructureIdentityFieldLayer",
    "SCMStructureMemory",
    "IntegratedStructureLanguageBlock", "SCMBlockOutput",
    "TACSCMLanguageModel",
]
