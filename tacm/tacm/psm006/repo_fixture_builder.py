"""
TAC-PSM-006: Repository Fixture Builder
========================================

Constructs a simulated repository context for each RepoTask.

A RepoFixture bundles:
  - file snapshots (relevant_files)
  - parsed dependency manifest
  - bug classification metadata
  - context embedding (for retrieval)

Level 1: purely simulated — no real file I/O or package installation.
Level 2 (future): real pytest fixtures for a subset of tasks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .repository_task import RepoTask, EMBEDDING_DIM


# ── Repo fixture dataclass ────────────────────────────────────────────────────

@dataclass
class RepoFixture:
    """
    Simulated repository snapshot for a single repair task.

    Attributes
    ----------
    task_id          : links back to the originating RepoTask
    repo_name        : repository identifier
    family           : bug family
    file_snapshots   : dict of filename → content string
    dependencies     : parsed list of (package, version_spec) from requirements
    env_vars         : simulated environment variables for this repo
    context_embedding: float32 vector encoding the repair context
    bug_class        : coarse category (import / dependency / api / path / config / test)
    level            : "simulated" | "semi-real" | "real"
    """
    task_id:           str
    repo_name:         str
    family:            str
    file_snapshots:    Dict[str, str]
    dependencies:      List[Dict[str, str]]
    env_vars:          Dict[str, str]
    context_embedding: np.ndarray
    bug_class:         str
    level:             str = "simulated"

    def to_dict(self) -> dict:
        return {
            "task_id":         self.task_id,
            "repo_name":       self.repo_name,
            "family":          self.family,
            "file_snapshots":  self.file_snapshots,
            "dependencies":    self.dependencies,
            "env_vars":        self.env_vars,
            "bug_class":       self.bug_class,
            "level":           self.level,
        }


# ── Dependency manifest parser ─────────────────────────────────────────────────

def parse_requirements(content: str) -> List[Dict[str, str]]:
    """
    Parse a requirements.txt content string into (package, version_spec) dicts.
    Handles: plain names, versioned pins, comments, blank lines.
    """
    deps: List[Dict[str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for op in [">=", "<=", "==", "!=", "~=", ">"]:
            if op in line:
                parts = line.split(op, 1)
                deps.append({"package": parts[0].strip(), "op": op,
                              "version": parts[1].strip()})
                break
        else:
            deps.append({"package": line, "op": "", "version": ""})
    return deps


# ── Context embedding builder ──────────────────────────────────────────────────

def _build_context_embedding(
    task: RepoTask,
    dim:  int = EMBEDDING_DIM,
) -> np.ndarray:
    """
    Build a deterministic context embedding by mixing:
      - task query embedding (60%)
      - family embedding      (30%)
      - repo name hash        (10%)

    This simulates how a real encoder would produce a context-aware
    representation from the bug report + failing test + file contents.
    """
    task_emb   = task.query_embedding(dim)
    family_emb = task.family_embedding(dim)

    repo_seed = int(hashlib.md5(task.repo_name.encode()).hexdigest(), 16) % (2**31)
    rng       = np.random.default_rng(repo_seed)
    repo_emb  = rng.standard_normal(dim).astype(np.float32)
    repo_emb /= (np.linalg.norm(repo_emb) + 1e-9)

    mixed = 0.60 * task_emb + 0.30 * family_emb + 0.10 * repo_emb
    return mixed / (np.linalg.norm(mixed) + 1e-9)


# ── Simulated env vars ─────────────────────────────────────────────────────────

_ENV_TEMPLATES: Dict[str, Dict[str, str]] = {
    "flask-api":        {"FLASK_ENV": "testing", "SECRET_KEY": "test-secret"},
    "django-web":       {"DJANGO_SETTINGS_MODULE": "mysite.settings.test", "SECRET_KEY": "t"},
    "fastapi-service":  {"ENV": "test", "DATABASE_URL": "sqlite:///test.db"},
    "pandas-etl":       {"DATA_DIR": "/tmp/data", "LOG_LEVEL": "DEBUG"},
    "numpy-ext":        {"NUMPY_SEED": "42"},
    "scikit-pipeline":  {"SKLEARN_SEED": "0"},
    "click-cli":        {"APP_ENV": "test"},
    "typer-app":        {"APP_ENV": "test"},
    "pytest-suite":     {"PYTEST_ADDOPTS": "-v"},
    "hypothesis-tests": {"HYPOTHESIS_SEED": "0"},
    "celery-worker":    {"CELERY_ALWAYS_EAGER": "True", "BROKER_URL": "memory://"},
    "rq-jobs":          {"RQ_ASYNC": "False"},
}

_DEFAULT_ENV: Dict[str, str] = {"ENV": "test", "DEBUG": "0"}


# ── Main builder ───────────────────────────────────────────────────────────────

def build_fixture(task: RepoTask) -> RepoFixture:
    """
    Build a RepoFixture from a RepoTask (Level 1: fully simulated).

    Steps:
      1. Copy file snapshots from task.relevant_files
      2. Parse dependencies from requirements.txt snapshot
      3. Build context embedding
      4. Attach simulated env vars
    """
    snapshots = dict(task.relevant_files)

    # Enrich snapshots with synthetic bug-trigger lines
    _inject_bug_markers(snapshots, task)

    # Parse requirements
    req_content = snapshots.get("requirements.txt", "")
    deps = parse_requirements(req_content)

    # Context embedding
    ctx_emb = _build_context_embedding(task)

    # Env vars
    env = dict(_ENV_TEMPLATES.get(task.repo_name, _DEFAULT_ENV))

    return RepoFixture(
        task_id           = task.task_id,
        repo_name         = task.repo_name,
        family            = task.family,
        file_snapshots    = snapshots,
        dependencies      = deps,
        env_vars          = env,
        context_embedding = ctx_emb,
        bug_class         = _coarse_class(task.family),
        level             = "simulated",
    )


def build_fixtures(tasks: List[RepoTask]) -> Dict[str, RepoFixture]:
    """Build fixtures for a list of tasks. Returns task_id → RepoFixture."""
    return {task.task_id: build_fixture(task) for task in tasks}


# ── Bug marker injection ───────────────────────────────────────────────────────

def _inject_bug_markers(snapshots: Dict[str, str], task: RepoTask) -> None:
    """
    Insert synthetic bug-trigger lines into file snapshots.
    These make the fixture more realistic for evaluation.
    """
    from .repository_task import (
        FAMILY_IMPORT, FAMILY_DEPENDENCY, FAMILY_VERSION_API,
        FAMILY_PATH, FAMILY_CONFIG, FAMILY_TEST,
    )
    markers = {
        FAMILY_IMPORT: (
            "from nonexistent_pkg import missing_name  "
            "# BUG: missing module\n"
        ),
        FAMILY_DEPENDENCY: (
            "# BUG: conflicting requirements below\n"
            "libA>=2.0\nlibB<2.0\n"
        ),
        FAMILY_VERSION_API: (
            "result = old_api.deprecated_func(x=val)  "
            "# BUG: removed in v3.0\n"
        ),
        FAMILY_PATH: (
            "cfg = open('config/settings.yaml')  "
            "# BUG: relative path from wrong cwd\n"
        ),
        FAMILY_CONFIG: (
            "# BUG: missing required key 'database.host'\n"
            "timeout: '30'  # BUG: should be int\n"
        ),
        FAMILY_TEST: (
            "assert result == STALE_VALUE  "
            "# BUG: expected value is stale\n"
        ),
    }
    marker = markers.get(task.family, "# BUG\n")

    # Inject into the first source file that isn't a config or test file
    for fname in list(snapshots.keys()):
        if fname.endswith(".py") and "test" not in fname and "conftest" not in fname:
            snapshots[fname] = snapshots[fname] + "\n" + marker
            break


# ── Coarse classification ──────────────────────────────────────────────────────

_COARSE: Dict[str, str] = {
    "ImportModuleError":   "import",
    "DependencyConflict":  "dependency",
    "VersionAPIMismatch":  "api",
    "PathModuleResolution": "path",
    "ConfigurationFailure": "config",
    "TestAssertionRepair":  "test",
}


def _coarse_class(family: str) -> str:
    return _COARSE.get(family, "unknown")
