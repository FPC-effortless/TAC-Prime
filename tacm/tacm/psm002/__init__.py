"""
TAC-PSM-002: Procedural Transfer

Scientific Question: Can a learned procedure be adapted to solve a different
but related task family?

Core Hypothesis: A procedure learned in one family contains reusable structure
that can transfer to another family.

Chains evaluated: A→B, A→C, A→B→C
"""

from .transfer import (
    TransferMode,
    TransferResult,
    TransferChainResult,
    adapt_procedure_to_family,
    run_transfer,
    run_transfer_chain,
)
from .metrics import (
    compute_transfer_metrics,
    TransferMetrics,
)

__all__ = [
    "TransferMode",
    "TransferResult",
    "TransferChainResult",
    "adapt_procedure_to_family",
    "run_transfer",
    "run_transfer_chain",
    "compute_transfer_metrics",
    "TransferMetrics",
]
