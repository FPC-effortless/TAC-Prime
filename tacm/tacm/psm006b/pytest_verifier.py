"""
TAC-PSM-006B: Pytest Verifier
==============================

Runs pytest against an isolated temp-directory copy of a fixture repo and
returns a real exit-code-based pass/fail result.

Execution model:
  1. Caller provides a dict of {filename: file_content} already patched.
  2. PytestVerifier writes those files to a fresh tempdir.
  3. Runs `python -m pytest <verification_command_args> --tb=short -q` as a
     subprocess with a hard timeout.
  4. Returns PytestResult with exit_code, stdout, stderr, and success flag.

Key design constraints:
  - Each run is fully isolated (separate tempdir, no shared state).
  - Subprocess inherits the current Python environment so all stdlib + pytest
    are available without extra installation.
  - Timeout defaults to 10 seconds per fixture — sufficient for stdlib-only tests.
  - Fixtures are designed to be self-contained (no pip installs at test time).

Failure classes detected:
  - verifier_instability  — exit code differs between two runs of the same fixture
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PytestResult:
    """
    Result of running pytest in an isolated temporary directory.

    Attributes
    ----------
    success       : True if pytest exited with code 0
    exit_code     : raw pytest exit code (0=pass, 1=fail, 2=error, 3=no-tests, 4=usage-err, 5=no-collect)
    stdout        : captured stdout from pytest
    stderr        : captured stderr from pytest
    timed_out     : True if the subprocess exceeded the timeout
    fixture_id    : echoed from the fixture for traceability
    variant       : e.g. "before_patch" or "after_patch"
    """
    success:    bool
    exit_code:  int
    stdout:     str
    stderr:     str
    timed_out:  bool
    fixture_id: str  = ""
    variant:    str  = ""

    def to_dict(self) -> dict:
        return {
            "success":    self.success,
            "exit_code":  self.exit_code,
            "stdout":     self.stdout[:2000],   # truncate for storage
            "stderr":     self.stderr[:1000],
            "timed_out":  self.timed_out,
            "fixture_id": self.fixture_id,
            "variant":    self.variant,
        }


class PytestVerifier:
    """
    Runs pytest in an isolated subprocess against fixture files.

    Parameters
    ----------
    timeout : seconds before killing the subprocess (default 10)
    python  : path to the Python executable (default: sys.executable)
    """

    def __init__(self, timeout: float = 10.0, python: Optional[str] = None):
        self.timeout = timeout
        self.python  = python or sys.executable

    def run(
        self,
        files:               Dict[str, str],
        verification_command: str,
        fixture_id:          str = "",
        variant:             str = "",
    ) -> PytestResult:
        """
        Write `files` to a temp dir and run `verification_command`.

        Parameters
        ----------
        files                : {filename: content} — all files to materialise
        verification_command : shell command like "pytest test_foo.py -x -q"
        fixture_id           : for traceability
        variant              : "before_patch" | "after_patch" | other label

        Returns
        -------
        PytestResult
        """
        with tempfile.TemporaryDirectory(prefix="tacpsm006b_") as tmpdir:
            tmp = Path(tmpdir)
            self._write_files(tmp, files)
            return self._run_pytest(tmp, verification_command, fixture_id, variant)

    def verify_before_and_after(
        self,
        before_files:         Dict[str, str],
        after_files:          Dict[str, str],
        verification_command: str,
        fixture_id:           str = "",
    ) -> tuple:
        """
        Run pytest on both before-patch and after-patch file sets.

        Returns
        -------
        (before_result: PytestResult, after_result: PytestResult)
        """
        before = self.run(before_files, verification_command, fixture_id, "before_patch")
        after  = self.run(after_files,  verification_command, fixture_id, "after_patch")
        return before, after

    def check_instability(
        self,
        files:               Dict[str, str],
        verification_command: str,
        fixture_id:          str = "",
        n_runs:              int = 2,
    ) -> bool:
        """
        Run pytest `n_runs` times and return True if results differ (unstable).
        """
        results = [
            self.run(files, verification_command, fixture_id, f"stability_{i}").success
            for i in range(n_runs)
        ]
        return len(set(results)) > 1

    # ── Internal helpers ────────────────────────────────────────────────────

    def _write_files(self, base: Path, files: Dict[str, str]) -> None:
        """Write all files to base directory, creating parent dirs as needed."""
        for rel_path, content in files.items():
            target = base / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def _run_pytest(
        self,
        cwd:                 Path,
        verification_command: str,
        fixture_id:          str,
        variant:             str,
    ) -> PytestResult:
        """Execute pytest subprocess and capture result."""
        # Parse the verification command into args
        args = self._build_args(verification_command)
        cmd  = [self.python, "-m", "pytest"] + args

        try:
            proc = subprocess.run(
                cmd,
                cwd      = str(cwd),
                capture_output = True,
                text     = True,
                timeout  = self.timeout,
                env      = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            return PytestResult(
                success    = proc.returncode == 0,
                exit_code  = proc.returncode,
                stdout     = proc.stdout,
                stderr     = proc.stderr,
                timed_out  = False,
                fixture_id = fixture_id,
                variant    = variant,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return PytestResult(
                success    = False,
                exit_code  = -1,
                stdout     = stdout,
                stderr     = stderr,
                timed_out  = True,
                fixture_id = fixture_id,
                variant    = variant,
            )

    @staticmethod
    def _build_args(command: str) -> List[str]:
        """
        Strip 'pytest' prefix from command string and return remaining args.
        e.g. "pytest test_foo.py -x -q" → ["test_foo.py", "-x", "-q"]
        """
        parts = command.strip().split()
        if not parts:
            return []
        # Remove leading 'pytest' or 'python -m pytest'
        if parts[0] == "pytest":
            return parts[1:]
        if parts[:3] == ["python", "-m", "pytest"]:
            return parts[3:]
        return parts
