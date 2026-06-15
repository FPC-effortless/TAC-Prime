"""
TAC-PSM-001: Benchmark Task Families

Four software-repair families with synthetic task instances:

  Family A — Import Errors          (missing import, incorrect import, renamed module)
  Family B — Dependency Conflicts   (incompatibility, conflicting reqs, dep resolution)
  Family C — Version Mismatch       (API changes, version drift, deprecation)
  Family D — Path / Module Resolution (incorrect paths, module discovery, env issues)

Each TaskInstance defines:
  - a canonical task_signature
  - a canonical_procedure (ordered steps)
  - a query_embedding (random but deterministic — seeded from signature hash)
  - evaluation logic (evaluate_procedure_on_task)

The benchmark is entirely self-contained — no real code execution required.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


EMBEDDING_DIM = 64     # small enough for fast tests; matches psm001 default


# ── Task definitions ──────────────────────────────────────────────────────────

@dataclass
class TaskFamily:
    name:     str
    label:    str                    # short label, e.g. "FamilyA"
    tasks:    List["TaskInstance"]


@dataclass
class TaskInstance:
    task_id:            str
    family:             str           # e.g. "ImportErrors"
    sub_type:           str           # e.g. "missing_import"
    description:        str
    canonical_steps:    List[str]     # the correct procedure
    distractor_steps:   List[str]     # a plausible but wrong procedure
    task_signature:     str           # canonical fingerprint
    difficulty:         float = 0.5   # [0, 1]

    def query_embedding(self, dim: int = EMBEDDING_DIM) -> np.ndarray:
        """Deterministic embedding derived from task_signature."""
        seed = int(hashlib.md5(self.task_signature.encode()).hexdigest(), 16) % (2**31)
        rng  = np.random.default_rng(seed)
        v    = rng.standard_normal(dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def family_embedding(self, dim: int = EMBEDDING_DIM) -> np.ndarray:
        """Embedding closer to family centroid than to individual task."""
        seed = int(hashlib.md5(self.family.encode()).hexdigest(), 16) % (2**31)
        rng  = np.random.default_rng(seed)
        base = rng.standard_normal(dim).astype(np.float32)
        base = base / (np.linalg.norm(base) + 1e-9)
        # Mix: 70% family centroid, 30% task noise
        task_emb = self.query_embedding(dim)
        mixed    = 0.7 * base + 0.3 * task_emb
        return mixed / (np.linalg.norm(mixed) + 1e-9)


def make_task_signature(family: str, sub_type: str, variant: int = 0) -> str:
    return f"{family}::{sub_type}::v{variant}"


# ── Family A: Import Errors ───────────────────────────────────────────────────

FAMILY_A_IMPORT_ERRORS = TaskFamily(
    name  = "Import Errors",
    label = "FamilyA",
    tasks = [
        TaskInstance(
            task_id        = "A1",
            family         = "ImportErrors",
            sub_type       = "missing_import",
            description    = "ModuleNotFoundError: No module named 'requests'",
            task_signature = make_task_signature("ImportErrors", "missing_import", 0),
            canonical_steps = [
                "Identify missing module from error traceback",
                "Check if module is listed in requirements.txt",
                "Install missing module via pip",
                "Verify installation: import module in REPL",
                "Rerun failing test to confirm fix",
            ],
            distractor_steps = [
                "Restart the Python interpreter",
                "Clear __pycache__ directory",
                "Reinstall all dependencies",
            ],
            difficulty = 0.3,
        ),
        TaskInstance(
            task_id        = "A2",
            family         = "ImportErrors",
            sub_type       = "missing_import",
            description    = "ModuleNotFoundError: No module named 'numpy'",
            task_signature = make_task_signature("ImportErrors", "missing_import", 1),
            canonical_steps = [
                "Identify missing module from error traceback",
                "Check if module is listed in requirements.txt",
                "Install missing module via pip",
                "Verify installation: import module in REPL",
                "Rerun failing test to confirm fix",
            ],
            distractor_steps = [
                "Upgrade pip to latest version",
                "Delete virtual environment and recreate",
            ],
            difficulty = 0.3,
        ),
        TaskInstance(
            task_id        = "A3",
            family         = "ImportErrors",
            sub_type       = "incorrect_import",
            description    = "ImportError: cannot import name 'DataFrame' from 'pandas'",
            task_signature = make_task_signature("ImportErrors", "incorrect_import", 0),
            canonical_steps = [
                "Identify the incorrect import name from error",
                "Look up correct attribute in module documentation",
                "Update import statement in source file",
                "Run affected tests",
                "Verify fix",
            ],
            distractor_steps = [
                "Reinstall pandas",
                "Downgrade to older version",
            ],
            difficulty = 0.5,
        ),
        TaskInstance(
            task_id        = "A4",
            family         = "ImportErrors",
            sub_type       = "renamed_module",
            description    = "ModuleNotFoundError: No module named 'sklearn' (renamed to 'scikit-learn')",
            task_signature = make_task_signature("ImportErrors", "renamed_module", 0),
            canonical_steps = [
                "Identify old module name from import statement",
                "Search for new package name in PyPI",
                "Update import statement or install renamed package",
                "Update requirements.txt",
                "Run tests",
            ],
            distractor_steps = [
                "Roll back to old Python version",
                "Use importlib.import_module workaround",
            ],
            difficulty = 0.6,
        ),
    ],
)

# ── Family B: Dependency Conflicts ────────────────────────────────────────────

FAMILY_B_DEPENDENCY_CONFLICTS = TaskFamily(
    name  = "Dependency Conflicts",
    label = "FamilyB",
    tasks = [
        TaskInstance(
            task_id        = "B1",
            family         = "DependencyConflicts",
            sub_type       = "package_incompatibility",
            description    = "ERROR: pip's dependency resolver has found incompatible versions",
            task_signature = make_task_signature("DependencyConflicts", "package_incompatibility", 0),
            canonical_steps = [
                "Run pip check to list all conflicts",
                "Identify conflicting package pair",
                "Find compatible version range using pip-tools or manual inspection",
                "Pin compatible versions in requirements.txt",
                "Reinstall dependencies in clean environment",
                "Run test suite",
            ],
            distractor_steps = [
                "Force install with --ignore-requires-python",
                "Suppress dependency warnings",
            ],
            difficulty = 0.7,
        ),
        TaskInstance(
            task_id        = "B2",
            family         = "DependencyConflicts",
            sub_type       = "conflicting_requirements",
            description    = "Package A requires lib>=2.0 but Package B requires lib<2.0",
            task_signature = make_task_signature("DependencyConflicts", "conflicting_requirements", 0),
            canonical_steps = [
                "Identify conflicting version constraints",
                "Check changelogs for breaking changes in lib 2.0",
                "Determine which package to downgrade or upgrade",
                "Create isolated virtual environment if needed",
                "Pin resolved versions",
                "Validate with full test suite",
            ],
            distractor_steps = [
                "Use --no-deps flag during install",
                "Remove one of the packages entirely",
            ],
            difficulty = 0.8,
        ),
    ],
)

# ── Family C: Version Mismatch ────────────────────────────────────────────────

FAMILY_C_VERSION_MISMATCH = TaskFamily(
    name  = "Version Mismatch",
    label = "FamilyC",
    tasks = [
        TaskInstance(
            task_id        = "C1",
            family         = "VersionMismatch",
            sub_type       = "api_change",
            description    = "AttributeError: module 'sklearn.linear_model' has no attribute 'LogisticRegression' (API changed in 1.0)",
            task_signature = make_task_signature("VersionMismatch", "api_change", 0),
            canonical_steps = [
                "Identify deprecated or renamed API from error message",
                "Check package changelog for the relevant version",
                "Find new API equivalent",
                "Update usage in source code",
                "Run tests with new API",
            ],
            distractor_steps = [
                "Downgrade package to known-good version",
                "Add try/except around deprecated call",
            ],
            difficulty = 0.6,
        ),
        TaskInstance(
            task_id        = "C2",
            family         = "VersionMismatch",
            sub_type       = "deprecation_failure",
            description    = "DeprecationWarning promoted to error: use of removed function",
            task_signature = make_task_signature("VersionMismatch", "deprecation_failure", 0),
            canonical_steps = [
                "Identify deprecated function and its replacement",
                "Update all call sites in codebase",
                "Update any documentation referencing old API",
                "Run test suite",
                "Confirm no remaining DeprecationWarnings",
            ],
            distractor_steps = [
                "Suppress all DeprecationWarnings globally",
                "Pin to older version",
            ],
            difficulty = 0.5,
        ),
    ],
)

# ── Family D: Path / Module Resolution ────────────────────────────────────────

FAMILY_D_PATH_RESOLUTION = TaskFamily(
    name  = "Path / Module Resolution",
    label = "FamilyD",
    tasks = [
        TaskInstance(
            task_id        = "D1",
            family         = "PathResolution",
            sub_type       = "incorrect_path",
            description    = "FileNotFoundError: config.yaml not found at ./config/config.yaml",
            task_signature = make_task_signature("PathResolution", "incorrect_path", 0),
            canonical_steps = [
                "Trace file path construction in source code",
                "Identify actual file location on disk",
                "Update path construction or add path resolution logic",
                "Test with both absolute and relative paths",
                "Run affected tests",
            ],
            distractor_steps = [
                "Hardcode absolute path as quick fix",
                "Create symlink at expected location",
            ],
            difficulty = 0.4,
        ),
        TaskInstance(
            task_id        = "D2",
            family         = "PathResolution",
            sub_type       = "module_discovery",
            description    = "ModuleNotFoundError: No module named 'mypackage' (sys.path issue)",
            task_signature = make_task_signature("PathResolution", "module_discovery", 0),
            canonical_steps = [
                "Print sys.path to identify search paths",
                "Locate actual package directory",
                "Add package directory to sys.path or install as editable",
                "Verify import resolves correctly",
                "Run full test suite",
            ],
            distractor_steps = [
                "Copy package files to site-packages directly",
                "Add PYTHONPATH to shell rc file only",
            ],
            difficulty = 0.5,
        ),
    ],
)

ALL_FAMILIES: List[TaskFamily] = [
    FAMILY_A_IMPORT_ERRORS,
    FAMILY_B_DEPENDENCY_CONFLICTS,
    FAMILY_C_VERSION_MISMATCH,
    FAMILY_D_PATH_RESOLUTION,
]

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_procedure_on_task(
    task:           TaskInstance,
    retrieved_steps: List[str],
    seed:           int = 42,
    noise_std:      float = 0.05,
) -> Tuple[bool, float, str]:
    """
    Simulate whether retrieved_steps solve the task.

    Returns (success, quality_score, reason).

    Logic:
      - score = Jaccard similarity between retrieved_steps and canonical_steps
      - add stochastic noise (seeded for reproducibility)
      - success iff (score + noise) > task.difficulty
    """
    rng = random.Random(seed)

    canonical_set  = set(s.lower().strip() for s in task.canonical_steps)
    retrieved_set  = set(s.lower().strip() for s in retrieved_steps)

    # Jaccard similarity
    if not canonical_set and not retrieved_set:
        jaccard = 0.0
    else:
        inter   = canonical_set & retrieved_set
        union   = canonical_set | retrieved_set
        jaccard = len(inter) / max(len(union), 1)

    # Step-overlap bonus: partial credit for similar (not identical) steps
    overlap_bonus = 0.0
    for cs in task.canonical_steps:
        for rs in retrieved_steps:
            words_c = set(cs.lower().split())
            words_r = set(rs.lower().split())
            if words_c and words_r:
                word_overlap = len(words_c & words_r) / max(len(words_c | words_r), 1)
                overlap_bonus = max(overlap_bonus, word_overlap * 0.2)

    quality  = min(1.0, jaccard + overlap_bonus)
    noise    = rng.gauss(0, noise_std)
    effective = min(1.0, max(0.0, quality + noise))

    threshold = task.difficulty
    success   = effective > threshold
    reason    = (
        f"quality={quality:.3f}  effective={effective:.3f}  "
        f"threshold={threshold:.2f}  jaccard={jaccard:.3f}"
    )
    return success, effective, reason


def oracle_steps(task: TaskInstance) -> List[str]:
    """Returns the ground-truth canonical steps — upper-bound baseline."""
    return list(task.canonical_steps)


def reset_steps() -> List[str]:
    """Returns empty procedure — no-memory baseline."""
    return []


def random_steps(all_tasks: List[TaskInstance], rng: random.Random) -> List[str]:
    """Returns steps from a randomly selected task — random retrieval baseline."""
    t = rng.choice(all_tasks)
    return list(t.canonical_steps)


def get_all_tasks() -> List[TaskInstance]:
    tasks = []
    for fam in ALL_FAMILIES:
        tasks.extend(fam.tasks)
    return tasks
