"""
TAC-PSM-006B: Caching Subprocess Verifier
==========================================

Thread-safe, subprocess-correct caching wrapper around PytestVerifier.

Why not pytest.main() in forked workers?
  pytest.main() in forked child processes inherits the parent's pytest
  plugin registry, conftest hooks, and module import caches.  With
  concurrent workers this causes "import file mismatch" errors and
  corrupts oracle results (oracle < full_memory, which is impossible).
  subprocess.run() spawns a fresh Python interpreter for each call,
  completely eliminating cross-contamination.

Speed strategy:
  - Thread-safe MD5 cache deduplicates identical (files, command) calls.
  - prewarm_cache() fires all 240 deterministic states as concurrent
    subprocess calls via ThreadPoolExecutor → wall time ≈ 240/8 * 0.9s ≈ 27s.
  - Seeds 1–4 get near-instant cache hits (deterministic states are
    identical across seeds).

Cache key: MD5(sorted_file_items + command_string)
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from .pytest_verifier import PytestVerifier, PytestResult


class CachingSubprocessVerifier:
    """
    Thread-safe caching layer over PytestVerifier (subprocess-based).

    Parameters
    ----------
    timeout  : seconds before killing each subprocess (default 10)
    n_warmup_workers : threads used during map_run / prewarm (default 8)
    """

    def __init__(
        self,
        timeout:          float = 10.0,
        n_warmup_workers: int   = 8,
    ):
        self.timeout          = timeout
        self.n_warmup_workers = n_warmup_workers

        self._inner  = PytestVerifier(timeout=timeout)
        self._cache: Dict[str, PytestResult] = {}
        self._lock   = threading.Lock()

        self.hits   = 0
        self.misses = 0

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        files:               Dict[str, str],
        verification_command: str,
        fixture_id:          str = "",
        variant:             str = "",
    ) -> PytestResult:
        """
        Run pytest on `files` using `verification_command`, with caching.

        Cache key is deterministic from file contents + command, so the
        result is valid for any (seed, variant) that produces the same state.
        """
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

        # Not in cache → run subprocess
        result = self._inner.run(files, verification_command, fixture_id, variant)

        with self._lock:
            if key not in self._cache:          # double-check after acquiring
                self._cache[key] = result
                self.misses += 1
            else:
                self.hits += 1                  # lost the race, count as hit

        return result

    def map_run(self, tasks) -> list:
        """
        Batch-run (files, command, fixture_id, variant) tuples in parallel.

        Serves cache hits immediately and submits only cache misses to the
        ThreadPoolExecutor.  Results are returned in input order.
        """
        results = [None] * len(tasks)

        # Partition: hits vs misses
        miss_indices: list = []
        miss_tasks:   list = []

        for i, (files, command, fid, var) in enumerate(tasks):
            key = self._cache_key(files, command)
            with self._lock:
                if key in self._cache:
                    self.hits += 1
                    cached = self._cache[key]
                    results[i] = PytestResult(
                        success    = cached.success,
                        exit_code  = cached.exit_code,
                        stdout     = cached.stdout,
                        stderr     = cached.stderr,
                        timed_out  = cached.timed_out,
                        fixture_id = fid,
                        variant    = var,
                    )
                else:
                    miss_indices.append(i)
                    miss_tasks.append((files, command, fid, var))

        if not miss_tasks:
            return results

        # Run misses in parallel
        def _call(args):
            idx, files, command, fid, var = args
            return idx, self.run(files, command, fixture_id=fid, variant=var)

        with ThreadPoolExecutor(max_workers=self.n_warmup_workers) as ex:
            futures = {
                ex.submit(_call, (i, *t)): i
                for i, t in zip(miss_indices, miss_tasks)
            }
            for fut in as_completed(futures):
                idx, result = fut.result()
                results[idx] = result

        return results

    # ── Internal ─────────────────────────────────────────────────────────

    def shutdown(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

    @staticmethod
    def _cache_key(files: Dict[str, str], command: str) -> str:
        blob = command + "|" + "|".join(
            f"{k}:{v}" for k, v in sorted(files.items())
        )
        return hashlib.md5(blob.encode()).hexdigest()
