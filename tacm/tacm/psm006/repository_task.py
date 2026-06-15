"""
TAC-PSM-006: Repository-Grounded Task Definitions
===================================================

120 tasks across 6 benchmark families (20 tasks each):

  Family 1 — ImportModuleError          (missing / circular / star / relative imports)
  Family 2 — DependencyConflict         (version conflicts, transitive, yanked, platform)
  Family 3 — VersionAPIMismatch         (removed / renamed APIs, changed signatures)
  Family 4 — PathModuleResolution       (wrong path, missing sys.path, editable installs)
  Family 5 — ConfigurationFailure       (missing keys, invalid format, env vars, schema)
  Family 6 — TestAssertionRepair        (wrong expected, type mismatch, off-by-one, fixture)

Each task carries full repository context fields required by the agent and verifier.

Research level: Level 1 (simulated repository-grounded repair)
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


EMBEDDING_DIM = 64

# ── Family names ──────────────────────────────────────────────────────────────

FAMILY_IMPORT        = "ImportModuleError"
FAMILY_DEPENDENCY    = "DependencyConflict"
FAMILY_VERSION_API   = "VersionAPIMismatch"
FAMILY_PATH          = "PathModuleResolution"
FAMILY_CONFIG        = "ConfigurationFailure"
FAMILY_TEST          = "TestAssertionRepair"

ALL_FAMILY_NAMES: List[str] = [
    FAMILY_IMPORT,
    FAMILY_DEPENDENCY,
    FAMILY_VERSION_API,
    FAMILY_PATH,
    FAMILY_CONFIG,
    FAMILY_TEST,
]

# Transfer groups: repos that share the same ecosystem → cross-repo transfer test
TRANSFER_GROUP_WEB    = "web_framework"
TRANSFER_GROUP_DATA   = "data_pipeline"
TRANSFER_GROUP_CLI    = "cli_tooling"
TRANSFER_GROUP_TEST   = "test_suite"
TRANSFER_GROUP_WORKER = "async_worker"

REPO_TRANSFER_MAP: Dict[str, str] = {
    "flask-api":        TRANSFER_GROUP_WEB,
    "django-web":       TRANSFER_GROUP_WEB,
    "fastapi-service":  TRANSFER_GROUP_WEB,
    "pandas-etl":       TRANSFER_GROUP_DATA,
    "numpy-ext":        TRANSFER_GROUP_DATA,
    "scikit-pipeline":  TRANSFER_GROUP_DATA,
    "click-cli":        TRANSFER_GROUP_CLI,
    "typer-app":        TRANSFER_GROUP_CLI,
    "pytest-suite":     TRANSFER_GROUP_TEST,
    "hypothesis-tests": TRANSFER_GROUP_TEST,
    "celery-worker":    TRANSFER_GROUP_WORKER,
    "rq-jobs":          TRANSFER_GROUP_WORKER,
}


# ── Core dataclass ─────────────────────────────────────────────────────────────

@dataclass
class RepoTask:
    """
    A single repository-grounded repair task.

    Fields match the specification exactly:
      task_id, repo_name, family, bug_report, failing_test_output,
      relevant_files, expected_procedure_family, oracle_repair_steps,
      verification_rule, difficulty, transfer_group
    """
    task_id:                   str
    repo_name:                 str
    family:                    str
    bug_report:                str
    failing_test_output:       str
    relevant_files:            Dict[str, str]     # filename → content snippet
    expected_procedure_family: str
    oracle_repair_steps:       List[str]
    verification_rule:         Dict               # deterministic check spec
    difficulty:                float              # [0.0, 1.0]
    transfer_group:            str

    def task_signature(self) -> str:
        return f"{self.family}::{self.repo_name}::{self.task_id}"

    def query_embedding(self, dim: int = EMBEDDING_DIM) -> np.ndarray:
        """Deterministic embedding seeded from task signature."""
        seed = int(hashlib.md5(self.task_signature().encode()).hexdigest(), 16) % (2**31)
        rng  = np.random.default_rng(seed)
        v    = rng.standard_normal(dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def family_embedding(self, dim: int = EMBEDDING_DIM) -> np.ndarray:
        """Embedding biased toward family centroid (70% family, 30% task)."""
        fam_seed = int(hashlib.md5(self.family.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(fam_seed)
        base = rng.standard_normal(dim).astype(np.float32)
        base /= (np.linalg.norm(base) + 1e-9)
        mixed = 0.70 * base + 0.30 * self.query_embedding(dim)
        return mixed / (np.linalg.norm(mixed) + 1e-9)

    def distractor_steps(self) -> List[str]:
        """Plausible but incorrect procedure for this family."""
        return _DISTRACTOR_MAP.get(self.family, ["restart interpreter", "reinstall all deps"])


# ── Distractor library ─────────────────────────────────────────────────────────

_DISTRACTOR_MAP: Dict[str, List[str]] = {
    FAMILY_IMPORT: [
        "Restart the Python interpreter",
        "Clear __pycache__ directories",
        "Reinstall all dependencies blindly",
        "Add bare 'import *' at top of module",
    ],
    FAMILY_DEPENDENCY: [
        "Force install with --ignore-requires-python",
        "Suppress all dependency warnings",
        "Remove one of the conflicting packages entirely",
        "Downgrade Python interpreter version",
    ],
    FAMILY_VERSION_API: [
        "Downgrade package to known-good version without checking",
        "Add blanket try/except around deprecated call",
        "Monkeypatch the missing attribute at runtime",
        "Pin package to exact version from lock file without updating code",
    ],
    FAMILY_PATH: [
        "Hardcode absolute path as quick fix",
        "Create symlink at expected location without updating code",
        "Add PYTHONPATH to .bashrc only",
        "Copy package files directly to site-packages",
    ],
    FAMILY_CONFIG: [
        "Comment out the config validation entirely",
        "Set all missing keys to None",
        "Ignore environment variables, use defaults always",
        "Regenerate config from scratch without migration",
    ],
    FAMILY_TEST: [
        "Delete the failing test",
        "Add @pytest.mark.skip without investigating",
        "Replace assert with print statement",
        "Change tolerance to 1e-1 globally",
    ],
}


# ── Oracle step templates (per family and sub-type) ────────────────────────────

_ORACLE_STEPS: Dict[str, Dict[str, List[str]]] = {
    FAMILY_IMPORT: {
        "missing_import": [
            "Parse failing test output to identify missing module name",
            "Search relevant_files for import statement referencing module",
            "Check requirements.txt for the missing package entry",
            "Add missing package to requirements.txt with appropriate version pin",
            "Verify import resolves by running: python -c 'import <module>'",
            "Rerun failing test to confirm fix",
        ],
        "circular_import": [
            "Parse error traceback to identify the circular import chain",
            "Map import graph: identify which modules form the cycle",
            "Refactor by extracting shared types into a new base module",
            "Update import statements to use the new base module",
            "Run linter (isort/flake8) to verify no remaining circular refs",
            "Rerun failing test to confirm fix",
        ],
        "star_import": [
            "Locate star import statement in relevant_files",
            "Run module introspection to enumerate exported names",
            "Replace star import with explicit named imports",
            "Verify no NameError is introduced by running affected tests",
            "Rerun failing test to confirm fix",
        ],
        "relative_import": [
            "Identify relative import that fails due to package structure",
            "Verify __init__.py exists in all intermediate packages",
            "Convert relative import to absolute import using full package path",
            "Confirm package is installed as editable (pip install -e .)",
            "Rerun failing test to confirm fix",
        ],
    },
    FAMILY_DEPENDENCY: {
        "version_conflict": [
            "Run dependency audit: pip check to list all conflicts",
            "Identify the two conflicting packages and their version constraints",
            "Inspect changelogs for breaking changes across the constraint boundary",
            "Determine compatible version range satisfying both constraints",
            "Pin compatible versions in requirements.txt or pyproject.toml",
            "Reinstall in clean environment and rerun failing test",
        ],
        "transitive_conflict": [
            "Generate dependency tree: pip install pipdeptree and run pipdeptree",
            "Identify transitive dependency introducing the conflict",
            "Find a version of the top-level package that pulls in a compatible transitive dep",
            "Update top-level package pin in requirements.txt",
            "Verify with pip check that no remaining conflicts exist",
            "Rerun failing test to confirm fix",
        ],
        "yanked_package": [
            "Identify yanked package version from pip install error output",
            "Query PyPI API or pip index for available non-yanked versions",
            "Update requirements.txt to pin a non-yanked release",
            "Reinstall dependencies",
            "Rerun failing test to confirm fix",
        ],
        "platform_conflict": [
            "Identify platform-specific package causing conflict from error message",
            "Add platform marker to requirements.txt entry (e.g., sys_platform=='win32')",
            "Verify platform marker syntax with pip check",
            "Test installation on target platform",
            "Rerun failing test to confirm fix",
        ],
    },
    FAMILY_VERSION_API: {
        "removed_function": [
            "Identify removed function name and the package version that removed it",
            "Search package changelog/migration guide for the replacement function",
            "Locate all call sites of the removed function in relevant_files",
            "Replace each call site with the new function and update argument names",
            "Run static type checker (mypy) to catch any signature mismatches",
            "Rerun failing test to confirm fix",
        ],
        "changed_signature": [
            "Identify the function with changed signature from the AttributeError / TypeError",
            "Consult the package changelog for the version that changed the signature",
            "Diff old and new function signature to identify renamed or removed params",
            "Update all call sites to pass arguments under the new parameter names",
            "Rerun failing test to confirm fix",
        ],
        "renamed_class": [
            "Identify renamed class from ImportError or AttributeError in test output",
            "Find the new class name in the package's current API documentation",
            "Update import statement to use the new class name",
            "Search and replace all usages of old class name in relevant_files",
            "Rerun failing test to confirm fix",
        ],
        "deprecated_param": [
            "Identify deprecated parameter from DeprecationWarning in test output",
            "Find replacement parameter or method in package changelog",
            "Update all call sites removing deprecated parameter",
            "Add new parameter if a replacement exists",
            "Run test suite with -W error::DeprecationWarning to confirm no remaining warnings",
            "Rerun failing test to confirm fix",
        ],
    },
    FAMILY_PATH: {
        "missing_file": [
            "Identify expected file path from FileNotFoundError in test output",
            "Search relevant_files to determine actual file location in repo",
            "Trace path construction logic in source code",
            "Fix path construction: use pathlib.Path(__file__).parent for relative anchoring",
            "Test with both absolute and relative paths",
            "Rerun failing test to confirm fix",
        ],
        "wrong_cwd": [
            "Identify working directory assumption from error message",
            "Check test runner configuration for cwd setting (pytest.ini / tox.ini)",
            "Refactor path construction to use pathlib.Path(__file__).resolve() as anchor",
            "Remove hard-coded cwd assumption",
            "Rerun failing test to confirm fix",
        ],
        "sys_path_missing": [
            "Identify missing module from ModuleNotFoundError in test output",
            "Print sys.path at test invocation to identify search paths",
            "Determine correct package root directory",
            "Install package as editable: pip install -e . from repo root",
            "Verify import resolves: python -c 'import <package>'",
            "Rerun failing test to confirm fix",
        ],
        "editable_install": [
            "Verify pyproject.toml / setup.py exists at repo root",
            "Check that package is not installed or installed in wrong env",
            "Run pip install -e . to create editable install",
            "Confirm package appears in pip list as editable",
            "Rerun failing test to confirm fix",
        ],
    },
    FAMILY_CONFIG: {
        "missing_key": [
            "Parse ConfigError / KeyError from test output to identify missing key",
            "Inspect config schema file (schema.json / config.yaml.example)",
            "Add missing key with a sensible default value to config file",
            "Validate config against schema using config validation tool",
            "Rerun failing test to confirm fix",
        ],
        "invalid_format": [
            "Identify config file and format error from test output",
            "Open config file and locate the malformed section",
            "Fix formatting: YAML indentation / JSON syntax / TOML type error",
            "Validate fixed config: python -c 'import yaml; yaml.safe_load(open(\"config.yaml\"))'",
            "Rerun failing test to confirm fix",
        ],
        "env_var_missing": [
            "Identify missing environment variable from KeyError or os.environ lookup",
            "Check .env.example or documentation for required environment variables",
            "Add missing environment variable to .env file or CI/CD config",
            "Verify the variable is loaded: python -c 'import os; print(os.environ[\"VAR\"])'",
            "Rerun failing test to confirm fix",
        ],
        "schema_mismatch": [
            "Identify schema validation error from test output (field name, type, constraint)",
            "Inspect config schema definition file",
            "Locate config value that violates the schema constraint",
            "Fix config value to conform to schema (correct type / valid range / required field)",
            "Re-run schema validation to confirm no remaining errors",
            "Rerun failing test to confirm fix",
        ],
    },
    FAMILY_TEST: {
        "wrong_expected": [
            "Read failing assertion in test output to identify expected vs actual values",
            "Trace the production code path that produces the actual value",
            "Determine whether the expected value or the production code is wrong",
            "If expected is stale: update assert expected value to match correct behavior",
            "If production code is wrong: fix logic and keep expected value",
            "Rerun failing test to confirm fix",
        ],
        "type_mismatch": [
            "Read TypeError or AssertionError in test output to identify type mismatch",
            "Trace the function under test return type",
            "Identify where the type conversion or coercion is missing",
            "Add explicit type cast at the appropriate boundary",
            "Verify no other tests break with the type fix",
            "Rerun failing test to confirm fix",
        ],
        "off_by_one": [
            "Identify off-by-one error from assertion diff in test output",
            "Locate the index or range expression in production code",
            "Determine correct fence-post: open vs closed interval",
            "Fix the index/range expression",
            "Verify edge cases: empty sequence, single element, max element",
            "Rerun failing test to confirm fix",
        ],
        "missing_fixture": [
            "Identify fixture name from pytest fixture error in test output",
            "Search conftest.py files for the fixture definition",
            "If fixture is missing: add fixture to appropriate conftest.py",
            "If fixture scope is wrong: update scope parameter (function/module/session)",
            "Verify fixture is visible to the failing test module",
            "Rerun failing test to confirm fix",
        ],
    },
}


# ── Verification rule builder ──────────────────────────────────────────────────

def _make_verification_rule(
    family:       str,
    sub_type:     str,
    repo_name:    str,
    difficulty:   float,
) -> Dict:
    """
    Build a deterministic verification rule for a task.

    The verifier checks:
      1. family_match      — retrieved procedure family == expected family
      2. step_overlap      — Jaccard(applied_steps, oracle_steps) >= threshold
      3. keyword_coverage  — essential keywords appear in applied steps

    min_score = difficulty + 0.1 (harder tasks require higher quality)
    """
    keyword_map = {
        FAMILY_IMPORT:      ["import", "module", "requirements", "install", "verify"],
        FAMILY_DEPENDENCY:  ["conflict", "requirements", "version", "pip", "constraint"],
        FAMILY_VERSION_API: ["api", "changelog", "replace", "deprecated", "signature"],
        FAMILY_PATH:        ["path", "file", "sys.path", "directory", "anchor"],
        FAMILY_CONFIG:      ["config", "key", "schema", "environment", "validate"],
        FAMILY_TEST:        ["assert", "expected", "actual", "fixture", "test"],
    }
    return {
        "expected_family":    family,
        "sub_type":           sub_type,
        "keyword_match":      keyword_map.get(family, []),
        "min_step_overlap":   0.30,
        "min_score":          min(0.85, difficulty + 0.10),
        "repo_context_key":   _REPO_CONTEXT_KEY.get(family, "requirements.txt"),
    }


_REPO_CONTEXT_KEY: Dict[str, str] = {
    FAMILY_IMPORT:      "requirements.txt",
    FAMILY_DEPENDENCY:  "requirements.txt",
    FAMILY_VERSION_API: "CHANGELOG.md",
    FAMILY_PATH:        "setup.py",
    FAMILY_CONFIG:      "config.yaml",
    FAMILY_TEST:        "conftest.py",
}


# ── Task factory ───────────────────────────────────────────────────────────────

_REPOS_WEB    = ["flask-api", "django-web", "fastapi-service"]
_REPOS_DATA   = ["pandas-etl", "numpy-ext", "scikit-pipeline"]
_REPOS_CLI    = ["click-cli", "typer-app"]
_REPOS_TEST   = ["pytest-suite", "hypothesis-tests"]
_REPOS_WORKER = ["celery-worker", "rq-jobs"]
_ALL_REPOS    = _REPOS_WEB + _REPOS_DATA + _REPOS_CLI + _REPOS_TEST + _REPOS_WORKER


def _bug_report(family: str, sub_type: str, repo: str, variant: int) -> str:
    templates = {
        (FAMILY_IMPORT, "missing_import"): (
            f"[{repo}] ModuleNotFoundError when running tests (variant {variant}). "
            "The package is used in source code but absent from requirements.txt."
        ),
        (FAMILY_IMPORT, "circular_import"): (
            f"[{repo}] ImportError: cannot import name 'X' — circular import detected "
            f"between src/models.py and src/utils.py (variant {variant})."
        ),
        (FAMILY_IMPORT, "star_import"): (
            f"[{repo}] NameError after refactor: name 'Foo' is not defined. "
            f"Star import removed names from namespace (variant {variant})."
        ),
        (FAMILY_IMPORT, "relative_import"): (
            f"[{repo}] ImportError: attempted relative import beyond top-level package "
            f"(variant {variant}). Package not installed as editable."
        ),
        (FAMILY_IMPORT, "namespace_package"): (
            f"[{repo}] ModuleNotFoundError with namespace package after refactor "
            f"(variant {variant}). __init__.py missing from sub-package."
        ),
        (FAMILY_DEPENDENCY, "version_conflict"): (
            f"[{repo}] pip install fails: package A requires lib>=2.0 but package B "
            f"requires lib<2.0 (variant {variant})."
        ),
        (FAMILY_DEPENDENCY, "transitive_conflict"): (
            f"[{repo}] Transitive dependency conflict: top-level package pulls in "
            f"conflicting transitive version (variant {variant})."
        ),
        (FAMILY_DEPENDENCY, "yanked_package"): (
            f"[{repo}] WARNING: package version has been yanked from PyPI. "
            f"pip refuses to install (variant {variant})."
        ),
        (FAMILY_DEPENDENCY, "platform_conflict"): (
            f"[{repo}] Platform-specific package fails to install on CI (Linux) "
            f"but works on macOS (variant {variant})."
        ),
        (FAMILY_DEPENDENCY, "extras_conflict"): (
            f"[{repo}] Optional extras introduce conflicting transitive dependency "
            f"when installed together (variant {variant})."
        ),
        (FAMILY_VERSION_API, "removed_function"): (
            f"[{repo}] AttributeError: module has no attribute 'old_func'. "
            f"Function was removed in package v3.0 (variant {variant})."
        ),
        (FAMILY_VERSION_API, "changed_signature"): (
            f"[{repo}] TypeError: func() got unexpected keyword argument. "
            f"Parameter renamed in v2.5 (variant {variant})."
        ),
        (FAMILY_VERSION_API, "renamed_class"): (
            f"[{repo}] ImportError: cannot import name 'OldClass'. "
            f"Class was renamed in v4.0 (variant {variant})."
        ),
        (FAMILY_VERSION_API, "deprecated_param"): (
            f"[{repo}] DeprecationWarning promoted to error: deprecated parameter "
            f"passed to API call (variant {variant})."
        ),
        (FAMILY_VERSION_API, "removed_module"): (
            f"[{repo}] ModuleNotFoundError: submodule was removed in latest release "
            f"and functionality moved (variant {variant})."
        ),
        (FAMILY_PATH, "missing_file"): (
            f"[{repo}] FileNotFoundError: config/settings.yaml not found. "
            f"Path constructed relative to wrong anchor (variant {variant})."
        ),
        (FAMILY_PATH, "wrong_cwd"): (
            f"[{repo}] Tests pass locally but fail in CI: path assumes wrong "
            f"current working directory (variant {variant})."
        ),
        (FAMILY_PATH, "sys_path_missing"): (
            f"[{repo}] ModuleNotFoundError in test runner: package root not in "
            f"sys.path (variant {variant})."
        ),
        (FAMILY_PATH, "editable_install"): (
            f"[{repo}] ModuleNotFoundError: local package not importable. "
            f"Editable install missing (variant {variant})."
        ),
        (FAMILY_PATH, "data_file"): (
            f"[{repo}] FileNotFoundError: data fixture file not found at path "
            f"constructed from __file__ (variant {variant})."
        ),
        (FAMILY_CONFIG, "missing_key"): (
            f"[{repo}] KeyError: 'database.host' missing from config. "
            f"New config key added in feature branch not present in base config (variant {variant})."
        ),
        (FAMILY_CONFIG, "invalid_format"): (
            f"[{repo}] yaml.scanner.ScannerError: mapping values not allowed here. "
            f"Config file has syntax error (variant {variant})."
        ),
        (FAMILY_CONFIG, "env_var_missing"): (
            f"[{repo}] KeyError: 'SECRET_KEY' — required environment variable "
            f"not set in test environment (variant {variant})."
        ),
        (FAMILY_CONFIG, "schema_mismatch"): (
            f"[{repo}] ValidationError: config field 'timeout' expected int, "
            f"got str (variant {variant})."
        ),
        (FAMILY_CONFIG, "override_conflict"): (
            f"[{repo}] Config override mechanism causes wrong value to be used; "
            f"environment variable overrides file setting unexpectedly (variant {variant})."
        ),
        (FAMILY_TEST, "wrong_expected"): (
            f"[{repo}] AssertionError: assert 42 == 43. "
            f"Expected value is stale after business logic change (variant {variant})."
        ),
        (FAMILY_TEST, "type_mismatch"): (
            f"[{repo}] AssertionError: assert '100' == 100. "
            f"Function returns string but test expects int (variant {variant})."
        ),
        (FAMILY_TEST, "off_by_one"): (
            f"[{repo}] AssertionError: assert result == [1,2,3] but got [1,2]. "
            f"Off-by-one in slice boundary (variant {variant})."
        ),
        (FAMILY_TEST, "missing_fixture"): (
            f"[{repo}] ERRORS: fixture 'db_session' not found. "
            f"Fixture defined in wrong conftest.py scope (variant {variant})."
        ),
        (FAMILY_TEST, "async_fixture"): (
            f"[{repo}] RuntimeError: async fixture used without pytest-asyncio marker "
            f"(variant {variant})."
        ),
    }
    key = (family, sub_type)
    return templates.get(key, f"[{repo}] Repair task ({family}/{sub_type}) variant {variant}.")


def _failing_test(family: str, sub_type: str, repo: str, variant: int) -> str:
    return (
        f"FAILED tests/test_{repo.replace('-', '_')}_v{variant}.py::"
        f"test_{sub_type}_{variant} — "
        + {
            FAMILY_IMPORT:      "ImportError / ModuleNotFoundError",
            FAMILY_DEPENDENCY:  "pip install failure / ImportError",
            FAMILY_VERSION_API: "AttributeError / TypeError / DeprecationWarning",
            FAMILY_PATH:        "FileNotFoundError / ModuleNotFoundError",
            FAMILY_CONFIG:      "KeyError / ValidationError / yaml.ScannerError",
            FAMILY_TEST:        "AssertionError / pytest.FixtureError",
        }.get(family, "UnknownError")
        + f"\n  at line {20 + variant * 7}"
    )


def _relevant_files(family: str, sub_type: str, repo: str) -> Dict[str, str]:
    base = {
        "requirements.txt": f"# {repo} requirements\nrequests>=2.28\nnumpy>=1.23\n",
        f"src/{repo.replace('-', '_')}/__init__.py": "# package root\n",
        f"tests/conftest.py": "# pytest fixtures\nimport pytest\n",
    }
    extras = {
        FAMILY_IMPORT:      {f"src/{repo.replace('-','_')}/core.py": "from .models import Foo\n"},
        FAMILY_DEPENDENCY:  {"pyproject.toml": "[project]\nname = \"" + repo + "\"\n"},
        FAMILY_VERSION_API: {"CHANGELOG.md": "## v3.0\n- Removed old_func\n- Added new_func\n"},
        FAMILY_PATH:        {"setup.py": "from setuptools import setup\nsetup(name='" + repo + "')\n",
                             "config/settings.yaml": "database:\n  host: localhost\n"},
        FAMILY_CONFIG:      {"config.yaml": "database:\n  host: localhost\ntimeout: '30'\n",
                             ".env.example": "SECRET_KEY=changeme\nDATABASE_URL=sqlite:///db.sqlite3\n"},
        FAMILY_TEST:        {f"tests/test_{repo.replace('-','_')}.py":
                             "def test_sample():\n    assert compute() == 42\n"},
    }
    base.update(extras.get(family, {}))
    return base


# ── Sub-type schedules per family (4-5 sub-types × variants = 20 tasks) ───────

_SUBTYPES: Dict[str, List[str]] = {
    FAMILY_IMPORT:      ["missing_import",    "circular_import",  "star_import",
                         "relative_import",   "namespace_package"],
    FAMILY_DEPENDENCY:  ["version_conflict",  "transitive_conflict", "yanked_package",
                         "platform_conflict", "extras_conflict"],
    FAMILY_VERSION_API: ["removed_function",  "changed_signature", "renamed_class",
                         "deprecated_param",  "removed_module"],
    FAMILY_PATH:        ["missing_file",      "wrong_cwd",          "sys_path_missing",
                         "editable_install",  "data_file"],
    FAMILY_CONFIG:      ["missing_key",       "invalid_format",     "env_var_missing",
                         "schema_mismatch",   "override_conflict"],
    FAMILY_TEST:        ["wrong_expected",    "type_mismatch",       "off_by_one",
                         "missing_fixture",   "async_fixture"],
}

_DIFFICULTIES: Dict[str, List[float]] = {
    FAMILY_IMPORT:      [0.30, 0.50, 0.45, 0.55, 0.60],
    FAMILY_DEPENDENCY:  [0.60, 0.70, 0.45, 0.55, 0.65],
    FAMILY_VERSION_API: [0.55, 0.60, 0.50, 0.50, 0.65],
    FAMILY_PATH:        [0.35, 0.45, 0.50, 0.40, 0.45],
    FAMILY_CONFIG:      [0.40, 0.45, 0.35, 0.50, 0.55],
    FAMILY_TEST:        [0.40, 0.45, 0.50, 0.55, 0.60],
}


def _oracle_steps_for(family: str, sub_type: str) -> List[str]:
    """Return oracle steps for a (family, sub_type) pair with fallback."""
    fam_map = _ORACLE_STEPS.get(family, {})
    if sub_type in fam_map:
        return list(fam_map[sub_type])
    # Fallback: use first sub-type's steps, slightly modified
    fallback = list(next(iter(fam_map.values()))) if fam_map else [
        f"Identify root cause of {sub_type} in {family}",
        "Inspect relevant_files for the error source",
        "Apply targeted fix",
        "Verify fix with test runner",
        "Update documentation if needed",
    ]
    return [s.replace(list(fam_map.keys())[0] if fam_map else "x", sub_type)
            if "replace" not in s.lower() else s
            for s in fallback]


def _build_family_tasks(family: str, tasks_per_family: int = 20) -> List[RepoTask]:
    """
    Generate `tasks_per_family` RepoTask instances for a given family.

    Distribution: 5 sub-types × 4 repo variants = 20 tasks.
    Each sub-type is tested against 4 different repos from the full repo pool.
    """
    subtypes    = _SUBTYPES[family]
    difficulties = _DIFFICULTIES[family]
    tasks: List[RepoTask] = []
    repo_cycle  = _ALL_REPOS[:]

    for sub_idx, sub_type in enumerate(subtypes):
        base_diff = difficulties[sub_idx % len(difficulties)]
        # 4 variants of this sub_type against different repos
        for variant in range(4):
            repo = repo_cycle[(sub_idx * 4 + variant) % len(repo_cycle)]
            tg   = REPO_TRANSFER_MAP.get(repo, "generic")
            diff = min(0.95, base_diff + variant * 0.05)
            tid  = f"{family[:3].upper()}{sub_idx:02d}{variant:02d}"

            task = RepoTask(
                task_id                   = tid,
                repo_name                 = repo,
                family                    = family,
                bug_report                = _bug_report(family, sub_type, repo, variant),
                failing_test_output       = _failing_test(family, sub_type, repo, variant),
                relevant_files            = _relevant_files(family, sub_type, repo),
                expected_procedure_family = family,
                oracle_repair_steps       = _oracle_steps_for(family, sub_type),
                verification_rule         = _make_verification_rule(
                    family, sub_type, repo, diff
                ),
                difficulty                = diff,
                transfer_group            = tg,
            )
            tasks.append(task)
            if len(tasks) >= tasks_per_family:
                return tasks

    return tasks


# ── Public task bank ───────────────────────────────────────────────────────────

def build_task_bank(tasks_per_family: int = 20) -> Dict[str, List[RepoTask]]:
    """Build the full PSM-006 task bank: 6 families × 20 tasks = 120 total."""
    return {
        family: _build_family_tasks(family, tasks_per_family)
        for family in ALL_FAMILY_NAMES
    }


def get_all_tasks(tasks_per_family: int = 20) -> List[RepoTask]:
    bank = build_task_bank(tasks_per_family)
    out: List[RepoTask] = []
    for tasks in bank.values():
        out.extend(tasks)
    return out


def split_train_test(
    tasks: List[RepoTask],
    train_frac: float = 0.5,
    seed: int = 0,
) -> Tuple[List[RepoTask], List[RepoTask]]:
    """Split tasks into train (warm-up) and test (evaluation) sets."""
    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_frac))
    return shuffled[:n_train], shuffled[n_train:]
