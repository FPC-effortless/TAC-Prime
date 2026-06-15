"""
TAC-PSM-006B: Fixture Schema
==============================

Defines the canonical Fixture dataclass for semi-real pytest repository
repair benchmarks.  Each fixture contains executable Python source/test
files so that PytestVerifier can run them under a subprocess and return a
real exit code rather than a heuristic score.

Failure classes (tracked in RepairTrace006B):
  wrong_procedure_retrieval   — retrieved wrong family's procedure
  correct_procedure_wrong_patch — right procedure, bad patch generation
  patch_wrong_file             — patch targeted a file not in the fixture
  insufficient_update          — update step did not improve retrieval
  family_confusion             — confused two similar families
  transfer_failure             — cross-fixture/cross-family transfer failed
  fixture_design_error         — fixture itself is self-contradictory
  verifier_instability         — pytest returned different results on retry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


FAMILY_NAMES: List[str] = [
    "import_module_error",
    "dependency_config_conflict",
    "version_api_mismatch",
    "path_module_resolution",
    "configuration_failure",
    "test_assertion_repair",
]

FAILURE_CLASSES: List[str] = [
    "wrong_procedure_retrieval",
    "correct_procedure_wrong_patch",
    "patch_wrong_file",
    "insufficient_update",
    "family_confusion",
    "transfer_failure",
    "fixture_design_error",
    "verifier_instability",
]

TRANSFER_GROUPS: List[str] = ["train", "near_transfer", "far_transfer"]
DIFFICULTY_LEVELS: List[str] = ["easy", "medium", "hard"]


@dataclass
class Fixture:
    """
    A single executable pytest fixture for PSM-006B repair benchmarking.

    Parameters
    ----------
    fixture_id           : unique identifier e.g. "F001_import_easy_01"
    repo_name            : short human-readable name e.g. "calc_util"
    family               : one of FAMILY_NAMES
    bug_report           : natural-language description of the failure
    failing_test_command : shell command that fails before patch e.g. "pytest test_calc.py -x -q"
    failing_test_output  : representative pytest error output before patch
    source_files         : {filename: file_content} — buggy source code
    test_files           : {filename: file_content} — test code (usually unchanged)
    config_files         : {filename: file_content} — e.g. conftest.py / pytest.ini
    oracle_repair_procedure : {"family": str, "steps": [str], "description": str}
    expected_patch       : {filename: {"old": str, "new": str}} — minimal code change
    verification_command : command to run after patch; success = exit 0
    transfer_group       : "train" | "near_transfer" | "far_transfer"
    difficulty           : "easy" | "medium" | "hard"
    """
    fixture_id:             str
    repo_name:              str
    family:                 str
    bug_report:             str
    failing_test_command:   str
    failing_test_output:    str
    source_files:           Dict[str, str]
    test_files:             Dict[str, str]
    config_files:           Dict[str, str]
    oracle_repair_procedure: Dict
    expected_patch:         Dict[str, Dict[str, str]]
    verification_command:   str
    transfer_group:         str
    difficulty:             str

    def all_files(self) -> Dict[str, str]:
        """Return union of source, test, and config files."""
        return {
            **self.source_files,
            **self.test_files,
            **self.config_files,
        }

    def to_dict(self) -> dict:
        return {
            "fixture_id":             self.fixture_id,
            "repo_name":              self.repo_name,
            "family":                 self.family,
            "bug_report":             self.bug_report,
            "failing_test_command":   self.failing_test_command,
            "failing_test_output":    self.failing_test_output,
            "source_files":           self.source_files,
            "test_files":             self.test_files,
            "config_files":           self.config_files,
            "oracle_repair_procedure": self.oracle_repair_procedure,
            "expected_patch":         self.expected_patch,
            "verification_command":   self.verification_command,
            "transfer_group":         self.transfer_group,
            "difficulty":             self.difficulty,
        }
