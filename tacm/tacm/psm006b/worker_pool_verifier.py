"""
TAC-PSM-006B: Worker Pool Verifier
====================================

Replaces PytestVerifier's per-call subprocess with a pool of persistent
pytest_worker.py processes.  Each worker starts once (incurring the ~900ms
Python startup cost exactly once per worker), then serves many verification
requests over its lifetime via stdin/stdout JSON protocol.

With N_WORKERS=4 and ~100ms actual test execution time, the 240-call
pre-warm completes in ~6-10 seconds instead of ~107 seconds.

Thread-safe: uses a queue to distribute work across workers.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .pytest_verifier import PytestResult


# Path to the worker script
_WORKER_SCRIPT = str(Path(__file__).parent / "pytest_worker.py")


class _Worker:
    """One persistent pytest_worker.py subprocess."""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self._lock   = threading.Lock()
        env = {**os.environ, "PYTEST_WORKER_TIMEOUT": str(timeout),
               "PYTHONDONTWRITEBYTECODE": "1"}
        self._proc = subprocess.Popen(
            [sys.executable, _WORKER_SCRIPT],
            stdin  = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text   = True,
            env    = env,
        )

    def run(self, files: Dict[str, str], command: str,
            fixture_id: str = "", variant: str = "") -> PytestResult:
        req = json.dumps({
            "files":      files,
            "command":    command,
            "fixture_id": fixture_id,
            "variant":    variant,
        }) + "\n"

        with self._lock:
            try:
                self._proc.stdin.write(req)
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("Worker process died")
                data = json.loads(line)
                return PytestResult(
                    success    = data["success"],
                    exit_code  = data["exit_code"],
                    stdout     = data.get("stdout", ""),
                    stderr     = data.get("stderr", ""),
                    timed_out  = data.get("timed_out", False),
                    fixture_id = fixture_id,
                    variant    = variant,
                )
            except Exception as exc:
                # If worker died, return a safe failure result
                return PytestResult(
                    success    = False,
                    exit_code  = -2,
                    stdout     = "",
                    stderr     = f"worker error: {exc}",
                    timed_out  = False,
                    fixture_id = fixture_id,
                    variant    = variant,
                )

    def shutdown(self):
        try:
            self._proc.stdin.write(json.dumps({"quit": True}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


class WorkerPoolVerifier:
    """
    A caching verifier backed by a pool of persistent worker processes.

    The cache (keyed by MD5 of files + command) persists for the lifetime
    of the verifier instance.  Sharing one instance across all seeds avoids
    redundant subprocess calls between seeds.

    Parameters
    ----------
    n_workers : number of persistent worker processes to spawn (default 4)
    timeout   : per-fixture pytest timeout in seconds (default 10)
    """

    def __init__(self, n_workers: int = 4, timeout: float = 10.0):
        self.timeout  = timeout
        self._cache: Dict[str, PytestResult] = {}
        self._lock    = threading.Lock()
        self.hits     = 0
        self.misses   = 0

        # Spawn workers
        self._workers: List[_Worker] = [_Worker(timeout) for _ in range(n_workers)]
        # Round-robin queue of available worker indices
        self._worker_q: queue.Queue[int] = queue.Queue()
        for i in range(n_workers):
            self._worker_q.put(i)

    def _cache_key(self, files: Dict[str, str], command: str) -> str:
        blob = command + "|" + "|".join(
            f"{k}:{v}" for k, v in sorted(files.items())
        )
        return hashlib.md5(blob.encode()).hexdigest()

    def run(
        self,
        files:                Dict[str, str],
        verification_command: str,
        fixture_id:           str = "",
        variant:              str = "",
    ) -> PytestResult:
        key = self._cache_key(files, verification_command)

        with self._lock:
            if key in self._cache:
                self.hits += 1
                cached = self._cache[key]
                return PytestResult(
                    success    = cached.success,
                    exit_code  = cached.exit_code,
                    stdout     = cached.stdout,
                    stderr     = cached.stderr,
                    timed_out  = cached.timed_out,
                    fixture_id = fixture_id,
                    variant    = variant,
                )

        # Get a free worker (blocks until one is available)
        worker_idx = self._worker_q.get()
        try:
            result = self._workers[worker_idx].run(
                files, verification_command, fixture_id, variant
            )
        finally:
            self._worker_q.put(worker_idx)

        with self._lock:
            if key not in self._cache:
                self._cache[key] = result
                self.misses += 1

        return result

    def shutdown(self):
        """Terminate all worker processes."""
        for w in self._workers:
            w.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
