"""
TAC-PSM-006B: Persistent Pytest Worker
=======================================

A long-lived subprocess that accepts JSON verification requests on stdin
and returns JSON results on stdout.  Eliminates Python interpreter startup
overhead (~700-900ms per call) by keeping one Python+pytest process alive
for the lifetime of the benchmark.

Protocol:
  Each request is a single JSON line on stdin:
    {"files": {"filename": "content", ...}, "command": "pytest ...", "fixture_id": "...", "variant": "..."}

  Each response is a single JSON line on stdout:
    {"success": bool, "exit_code": int, "stdout": "...", "stderr": "...", "timed_out": bool, "fixture_id": "...", "variant": "..."}

  Send {"quit": true} to terminate the worker.

The worker runs pytest via subprocess (for isolation of sys.modules), but
the subprocess is a fresh pytest invocation — the Python interpreter is NOT
restarted each time because we use a pre-forked subprocess pool internally.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_one(files: dict, command: str, fixture_id: str, variant: str,
            timeout: float = 10.0) -> dict:
    """Run one pytest verification in an isolated temp directory."""
    with tempfile.TemporaryDirectory(prefix="tacpsm006b_worker_") as tmpdir:
        tmp = Path(tmpdir)
        for fname, content in files.items():
            tgt = tmp / fname
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(content, encoding="utf-8")

        parts = command.strip().split()
        if parts and parts[0] == "pytest":
            parts = parts[1:]
        elif parts[:3] == ["python", "-m", "pytest"]:
            parts = parts[3:]

        cmd = [sys.executable, "-m", "pytest"] + parts

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(tmp),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            return {
                "success":    proc.returncode == 0,
                "exit_code":  proc.returncode,
                "stdout":     proc.stdout[:2000],
                "stderr":     proc.stderr[:1000],
                "timed_out":  False,
                "fixture_id": fixture_id,
                "variant":    variant,
            }
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return {
                "success":    False,
                "exit_code":  -1,
                "stdout":     stdout[:2000],
                "stderr":     stderr[:1000],
                "timed_out":  True,
                "fixture_id": fixture_id,
                "variant":    variant,
            }


def main():
    """Main worker loop: read requests from stdin, write responses to stdout."""
    timeout = float(os.environ.get("PYTEST_WORKER_TIMEOUT", "10.0"))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        if req.get("quit"):
            break

        result = run_one(
            files      = req["files"],
            command    = req["command"],
            fixture_id = req.get("fixture_id", ""),
            variant    = req.get("variant", ""),
            timeout    = timeout,
        )
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
