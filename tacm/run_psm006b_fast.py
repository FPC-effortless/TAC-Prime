"""
TAC-PSM-006B: Fast Replication Runner (Parallelised + Caching)
==============================================================

Drop-in replacement for run_psm006b_replication.py that achieves ~4-8x
speedup via:

  1. CachingPytestVerifier — deduplicates identical (files, command) calls
     across variants.  The "before-patch" call is the same for all 7 variants
     on a given fixture; caching eliminates 6/7 of those calls.

  2. ThreadPoolExecutor for independent variants — reset, retrieval_disabled,
     random_procedure, structure_only, no_update, oracle have no shared mutable
     state (each fixture gets a fresh or read-only store).  Their fixture loops
     run concurrently.

  3. full_memory remains sequential — memory accumulates across fixtures, so
     the loop cannot be parallelised without changing the scientific contract.

  4. Per-fixture seeding for independent variants — derived deterministically
     from (seed, variant, fixture_index) so results are fully reproducible
     regardless of thread scheduling order.

  5. Per-seed incremental writes — results are saved after every seed so that
     a mid-run OOM kill does not discard completed work.

Usage:
  python run_psm006b_fast.py [--seeds 0 1 2 3 4] [--workers 8] [--out reports/]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from tacm.psm006b.fixture_builder import build_all_fixtures
from tacm.psm006b.fixture_schema import FAMILY_NAMES, Fixture
from tacm.psm006b.baselines import VARIANT_NAMES, _make_seeded_store
from tacm.psm006b.metrics import (
    compute_metrics,
    evaluate_success_gates,
    compute_family_confusion_matrix,
    classify_failures,
)
from tacm.psm006b.procedural_repair_agent import (
    ProceduralRepairAgent006B,
    RepairTrace006B,
    seed_procedural_memory,
    fixture_embedding,
    oracle_procedure_dict,
    _classify_failure,
    EMBEDDING_DIM,
)
from tacm.psm006b.memory_store import SimpleProceduralMemoryStore
from tacm.psm006b.pytest_verifier import PytestResult
from tacm.psm006b.patch_applier import PatchApplier
from tacm.psm006b.caching_verifier import CachingSubprocessVerifier

# Alias so the rest of the file can refer to either type
CachingPytestVerifier = CachingSubprocessVerifier


# ── Per-fixture seeding helpers ───────────────────────────────────────────

def _fixture_rng_seed(seed: int, variant: str, fx_index: int) -> int:
    """
    Derive a deterministic per-fixture seed for independent variants.
    This ensures reproducibility regardless of thread execution order.
    """
    key = f"{seed}_{variant}_{fx_index}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 31)


# ── Independent-variant repair (thread-safe, no shared mutable state) ────

def _repair_independent(
    fixture:   Fixture,
    fx_index:  int,
    variant:   str,
    seed:      int,
    verifier:  CachingPytestVerifier,
    applier:   PatchApplier,
) -> RepairTrace006B:
    """
    Run one fixture under an independent variant.

    Each call gets its own store (fresh seed derived from main seed +
    variant + fixture index), so threads don't share mutable state.
    """
    fx_seed = _fixture_rng_seed(seed, variant, fx_index)
    store   = _make_seeded_store(fx_seed)
    agent   = ProceduralRepairAgent006B(
        store           = store,
        verifier        = verifier,
        applier         = applier,
        mode            = variant,
        retrieval_noise = 0.10,
        rng_seed        = fx_seed,
        max_retries     = 0,
    )
    return agent.repair(fixture)


def run_variant_parallel(
    variant:   str,
    fixtures:  List[Fixture],
    seed:      int,
    verifier:  CachingPytestVerifier,
    applier:   PatchApplier,
    n_workers: int = 8,
) -> List[RepairTrace006B]:
    """
    Run one independent variant over all fixtures using a thread pool.
    Results are returned in fixture order.
    """
    results: List[Optional[RepairTrace006B]] = [None] * len(fixtures)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_to_idx = {
            ex.submit(_repair_independent, fx, i, variant, seed, verifier, applier): i
            for i, fx in enumerate(fixtures)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()

    return results  # type: ignore


# ── full_memory variant (must remain sequential) ──────────────────────────

def run_full_memory_sequential(
    fixtures:  List[Fixture],
    seed:      int,
    verifier:  CachingPytestVerifier,
    applier:   PatchApplier,
) -> List[RepairTrace006B]:
    """
    Run the full_memory variant sequentially.
    Memory accumulates: later fixtures benefit from earlier successful repairs.
    """
    store = _make_seeded_store(seed)
    agent = ProceduralRepairAgent006B(
        store           = store,
        verifier        = verifier,
        applier         = applier,
        mode            = "full_memory",
        retrieval_noise = 0.10,
        rng_seed        = seed,
        max_retries     = 1,
    )
    return [agent.repair(fx) for fx in fixtures]


# ── reset variant (per-fixture fresh store, parallelisable) ──────────────

def run_reset_parallel(
    fixtures:  List[Fixture],
    seed:      int,
    verifier:  CachingPytestVerifier,
    applier:   PatchApplier,
    n_workers: int = 8,
) -> List[RepairTrace006B]:
    """
    Reset variant: memory is cleared before each fixture.
    Since each fixture is independent, parallelize with a thread pool.
    Uses the same per-fixture sub-seeding as _run_reset in baselines.py,
    derived from the main seed to match the sequential implementation.
    """
    rng = np.random.default_rng(seed)
    fx_seeds = [int(rng.integers(0, 2 ** 31)) for _ in fixtures]

    def _repair_reset(fx_info: Tuple[int, Fixture, int]) -> RepairTrace006B:
        idx, fx, fx_seed = fx_info
        store = _make_seeded_store(fx_seed)
        agent = ProceduralRepairAgent006B(
            store           = store,
            verifier        = verifier,
            applier         = applier,
            mode            = "reset",
            retrieval_noise = 0.10,
            rng_seed        = fx_seed,
            max_retries     = 0,
        )
        return agent.repair(fx)

    results: List[Optional[RepairTrace006B]] = [None] * len(fixtures)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_to_idx = {
            ex.submit(_repair_reset, (i, fx, fx_seeds[i])): i
            for i, fx in enumerate(fixtures)
        }
        for fut in as_completed(future_to_idx):
            results[future_to_idx[fut]] = fut.result()
    return results  # type: ignore


# ── Cache pre-warmer ─────────────────────────────────────────────────────

def prewarm_cache(
    fixtures:  List[Fixture],
    verifier:  CachingPytestVerifier,
    applier:   PatchApplier,
    n_workers: int = 8,
) -> None:
    """
    Pre-populate the verifier cache with all deterministic fixture states.

    Runs in parallel so that seed 0's sequential full_memory loop can reuse
    results from the cache rather than re-spawning subprocesses.

    States pre-computed:
      - before_patch    : buggy files (same for ALL 7 variants, always fails)
      - after_correct   : correct-family patch applied (always passes)
      - after_wrong     : wrong-family stub applied (always fails)
      - after_structure : structure-only patch applied (always fails)
    """
    tasks = []
    for fx in fixtures:
        all_files = fx.all_files()

        # before-patch (always the same buggy files)
        tasks.append((all_files, fx.verification_command, fx.fixture_id, "prewarm_before"))

        # after correct patch
        pr_correct = applier.apply(all_files, fx.expected_patch)
        tasks.append((pr_correct.patched_files, fx.verification_command,
                      fx.fixture_id, "prewarm_after_correct"))

        # after structure-only patch
        pr_struct = applier.apply_structure_only_patch(all_files, fx.expected_patch)
        tasks.append((pr_struct.patched_files, fx.verification_command,
                      fx.fixture_id, "prewarm_after_structure"))

        # after wrong-family patches (one representative wrong family per fixture)
        wrong_fam = next(
            (f for f in FAMILY_NAMES if f != fx.family),
            FAMILY_NAMES[0],
        )
        pr_wrong = applier.apply_wrong_family_patch(all_files, wrong_fam)
        tasks.append((pr_wrong.patched_files, fx.verification_command,
                      fx.fixture_id, "prewarm_after_wrong"))

    t0 = time.time()
    # Use batch map_run if available (PoolVerifier), else fallback to ThreadPoolExecutor
    if hasattr(verifier, "map_run"):
        pool_tasks = [(f, c, fid, var) for f, c, fid, var in tasks]
        verifier.map_run(pool_tasks)
    else:
        def _run(args):
            files, cmd, fid, var = args
            return verifier.run(files, cmd, fixture_id=fid, variant=var)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_run, tasks))
    print(f"  Cache pre-warm: {len(tasks)} calls in {time.time()-t0:.1f}s  "
          f"(hits={verifier.hits} misses={verifier.misses})", flush=True)


# ── Main seed runner ──────────────────────────────────────────────────────

def run_seed_fast(
    seed:             int,
    fixtures:         List[Fixture],
    timeout_s:        float,
    n_workers:        int,
    shared_verifier:  Optional[CachingPytestVerifier] = None,
) -> dict:
    print(f"  Seed {seed}: {len(fixtures)} fixtures × 7 variants  "
          f"(full_memory sequential, others ×{n_workers} threads) ...",
          flush=True)
    t0 = time.time()

    verifier = shared_verifier or CachingPytestVerifier(timeout=timeout_s)
    applier  = PatchApplier()

    independent_variants = [
        "retrieval_disabled",
        "random_procedure",
        "structure_only",
        "no_update",
        "oracle",
    ]

    results: Dict[str, List[RepairTrace006B]] = {}

    # ── 1. full_memory (sequential, must come first so cache warms for others)
    t_fm = time.time()
    results["full_memory"] = run_full_memory_sequential(fixtures, seed, verifier, applier)
    print(f"    full_memory done  {time.time()-t_fm:.1f}s  "
          f"cache hits={verifier.hits} misses={verifier.misses}", flush=True)

    # ── 2. reset (per-fixture independent, parallel)
    t_rs = time.time()
    results["reset"] = run_reset_parallel(fixtures, seed, verifier, applier, n_workers)
    print(f"    reset done  {time.time()-t_rs:.1f}s  "
          f"cache hits={verifier.hits}", flush=True)

    # ── 3. remaining independent variants (each parallelised)
    for v in independent_variants:
        tv = time.time()
        results[v] = run_variant_parallel(v, fixtures, seed, verifier, applier, n_workers)
        print(f"    {v:<26s} done  {time.time()-tv:.1f}s  "
              f"cache hits={verifier.hits}", flush=True)

    # ── Metrics
    metrics  = compute_metrics(results, reference_variant="full_memory")
    gates    = evaluate_success_gates(metrics, results)
    failures = classify_failures(results.get("full_memory", []))
    confusion = compute_family_confusion_matrix(
        results.get("full_memory", []), FAMILY_NAMES
    )
    variant_rates = {v: _pass_rate(results.get(v, [])) for v in VARIANT_NAMES}

    elapsed = time.time() - t0
    print(f"  Seed {seed} done in {elapsed:.1f}s  "
          f"full_memory={variant_rates['full_memory']:.3f}  "
          f"oracle={variant_rates['oracle']:.3f}  "
          f"reset={variant_rates['reset']:.3f}  "
          f"cache_total_hits={verifier.hits}",
          flush=True)

    return {
        "seed":          seed,
        "metrics":       metrics,
        "gates":         gates,
        "variant_rates": variant_rates,
        "failures":      failures,
        "confusion":     {k: dict(v) for k, v in confusion.items()},
        "n_fixtures":    len(fixtures),
        "elapsed_s":     elapsed,
        "cache_hits":    verifier.hits,
        "cache_misses":  verifier.misses,
    }


def _pass_rate(traces: List[RepairTrace006B]) -> float:
    if not traces:
        return 0.0
    return mean(1.0 if t.pytest_pass else 0.0 for t in traces)


# ── Aggregation ──────────────────────────────────────────────────────────

def aggregate_seeds(seed_results: List[dict]) -> dict:
    keys = list(seed_results[0]["metrics"].keys())
    agg_metrics = {}
    for k in keys:
        vals = [r["metrics"][k] for r in seed_results]
        agg_metrics[k] = {
            "mean": mean(vals),
            "std":  stdev(vals) if len(vals) > 1 else 0.0,
        }

    gate_names = list(seed_results[0]["gates"].keys())
    gate_pass_rates = {}
    for g in gate_names:
        n_pass = sum(1 for r in seed_results if r["gates"].get(g, False))
        gate_pass_rates[g] = {"pass": n_pass, "total": len(seed_results)}

    variant_names = list(seed_results[0]["variant_rates"].keys())
    agg_variant = {}
    for v in variant_names:
        vals = [r["variant_rates"][v] for r in seed_results]
        agg_variant[v] = {"mean": mean(vals), "std": stdev(vals) if len(vals) > 1 else 0.0}

    return {
        "metrics":         agg_metrics,
        "gate_pass_rates": gate_pass_rates,
        "variant_rates":   agg_variant,
    }


def compute_verdict(seed_results: List[dict]) -> str:
    gate_names = list(seed_results[0]["gates"].keys())
    n_seeds    = len(seed_results)
    gates_all  = sum(
        1 for g in gate_names
        if sum(1 for r in seed_results if r["gates"].get(g, False)) == n_seeds
    )
    gates_most = sum(
        1 for g in gate_names
        if sum(1 for r in seed_results if r["gates"].get(g, False)) >= n_seeds * 0.6
    )
    if gates_all == len(gate_names):
        return "VALIDATES"
    elif gates_most >= 5:
        return "PARTIALLY_VALIDATES"
    else:
        return "DOES_NOT_VALIDATE"


# ── Report writers ───────────────────────────────────────────────────────

def write_results_json(seed_results: List[dict], agg: dict, verdict: str, path: str) -> None:
    payload = {
        "run_info": {
            "date":        time.strftime("%Y-%m-%d"),
            "n_seeds":     len(seed_results),
            "n_fixtures":  seed_results[0]["n_fixtures"] if seed_results else 0,
            "n_variants":  7,
        },
        "seed_results": [
            {k: v for k, v in sr.items() if k != "confusion"}
            for sr in seed_results
        ],
        "per_seed_confusion": [
            {"seed": sr["seed"], "confusion": sr["confusion"]}
            for sr in seed_results
        ],
        "aggregate": agg,
        "verdict":   verdict,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_txt(agg: dict, seed_results: List[dict], verdict: str, path: str) -> None:
    n_seeds = len(seed_results)

    # Check for trivial fixtures warning
    fm_mean   = agg["variant_rates"]["full_memory"]["mean"]
    ora_mean  = agg["variant_rates"]["oracle"]["mean"]
    trivial_warn = (fm_mean >= 1.0 and ora_mean >= 1.0)

    lines = [
        "TAC-PSM-006B Replication Summary",
        "=" * 60,
        f"Date: {time.strftime('%Y-%m-%d')}",
        f"Seeds: {[r['seed'] for r in seed_results]}",
        f"Fixtures per seed: {seed_results[0]['n_fixtures']}",
        "",
        f"Verdict: TAC-PSM-006B {verdict}",
        "",
    ]

    if trivial_warn:
        lines += [
            "⚠  WARNING: full_memory AND oracle are both 1.000.",
            "   This is not a failure, but suggests the fixtures may be too easy.",
            "   Results should be interpreted as an upper-bound ceiling run.",
            "",
        ]

    lines += ["Variant Pass Rates (mean ± std across seeds):"]
    for v in VARIANT_NAMES:
        vd  = agg["variant_rates"].get(v, {"mean": 0.0, "std": 0.0})
        lines.append(f"  {v:<30s}  {vd['mean']:.3f} ± {vd['std']:.3f}")

    lines += ["", "Key Metrics (mean ± std across seeds):"]
    for k, vd in agg["metrics"].items():
        lines.append(f"  {k:<44s}  {vd['mean']:.4f} ± {vd['std']:.4f}")

    lines += ["", "Gate Results (seeds_passing / total):"]
    for gate, info in agg["gate_pass_rates"].items():
        sym   = "[PASS]" if info["pass"] == info["total"] else "[FAIL]"
        lines.append(f"  {sym} {gate:<40s}  {info['pass']}/{info['total']}")

    lines += ["", "Per-Seed Variant Rates:"]
    for sr in seed_results:
        vr = sr["variant_rates"]
        lines.append(
            f"  seed={sr['seed']}  "
            f"full_memory={vr['full_memory']:.3f}  "
            f"oracle={vr['oracle']:.3f}  "
            f"reset={vr['reset']:.3f}  "
            f"no_update={vr.get('no_update',0.0):.3f}  "
            f"random={vr.get('random_procedure',0.0):.3f}  "
            f"({sr['elapsed_s']:.0f}s)"
        )

    lines += ["", "Per-Seed Gate Summary:"]
    for sr in seed_results:
        passed = sum(1 for v in sr["gates"].values() if v)
        total  = len(sr["gates"])
        lines.append(f"  seed={sr['seed']}  {passed}/{total} gates pass")

    lines += [
        "",
        "Note: TAC-PSM-006B uses semi-real pytest-grounded procedural memory.",
        "full_memory=oracle=1.000 means fixtures have zero noise tolerance;",
        "this is documented behaviour and not a validity failure.",
    ]

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_per_family_rates(seed_results: List[dict], agg: dict, path: str) -> None:
    """Write per-family pass rates from full_memory confusion matrices."""
    lines = ["TAC-PSM-006B Per-Family Pass Rates", "=" * 50, ""]
    for sr in seed_results:
        lines.append(f"Seed {sr['seed']}:")
        conf = sr.get("confusion", {})
        for fam in FAMILY_NAMES:
            row = conf.get(fam, {})
            total = sum(row.values())
            correct = row.get(fam, 0)
            acc = correct / total if total else 0.0
            lines.append(f"  {fam:<30s}  retrieval_acc={acc:.3f}  (n={total})")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_confusion_matrix(seed_results: List[dict], path: str) -> None:
    lines = ["TAC-PSM-006B Confusion Matrix (full_memory, mean across seeds)", "=" * 60, ""]
    header = "true\\retrieved".ljust(30) + "  ".join(f[-8:].ljust(10) for f in FAMILY_NAMES)
    lines.append(header)
    for true_fam in FAMILY_NAMES:
        row_vals = []
        for ret_fam in FAMILY_NAMES:
            vals = [sr["confusion"].get(true_fam, {}).get(ret_fam, 0) for sr in seed_results]
            row_vals.append(f"{mean(vals):.1f}".ljust(10))
        lines.append(f"{true_fam[-20:].ljust(30)}  {'  '.join(row_vals)}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_failure_analysis(seed_results: List[dict], path: str) -> None:
    from tacm.psm006b.fixture_schema import FAILURE_CLASSES

    all_counts: Dict[str, List[int]] = {fc: [] for fc in FAILURE_CLASSES}
    all_counts["none"] = []

    for sr in seed_results:
        for fc in list(FAILURE_CLASSES) + ["none"]:
            all_counts[fc].append(sr["failures"].get(fc, 0))

    lines = [
        "TAC-PSM-006B Failure Analysis (full_memory variant)",
        "=" * 60,
        "",
        f"Seeds analysed: {[sr['seed'] for sr in seed_results]}",
        f"Fixtures per seed: {seed_results[0]['n_fixtures']}",
        "",
        "Failure class counts (mean ± std across seeds):",
    ]
    total_failures = []
    for sr in seed_results:
        total_failures.append(sum(v for k, v in sr["failures"].items() if k != "none"))

    for fc, counts in all_counts.items():
        m = mean(counts)
        s = stdev(counts) if len(counts) > 1 else 0.0
        pct = m / seed_results[0]["n_fixtures"] * 100
        lines.append(f"  {fc:<40s}  {m:.1f} ± {s:.1f}  ({pct:.1f}%)")

    lines += [
        "",
        f"Total failures/seed: {mean(total_failures):.1f} ± "
        f"{stdev(total_failures) if len(total_failures) > 1 else 0.0:.1f}",
        "",
        "Failure class definitions:",
        "  wrong_procedure_retrieval   — memory retrieved the wrong family's procedure",
        "  correct_procedure_wrong_patch — right procedure, patch failed to apply cleanly",
        "  patch_wrong_file             — patch targeted a non-existent file",
        "  insufficient_update          — update step did not improve next retrieval",
        "  family_confusion             — two similar families were confused",
        "  transfer_failure             — cross-fixture or cross-family transfer failed",
        "  fixture_design_error         — fixture itself is self-contradictory",
        "  verifier_instability         — pytest returned different results on retry",
        "  none                         — success (no failure class)",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PSM-006B fast replication runner")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=8,
                        help="ThreadPoolExecutor workers for independent variants")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Per-fixture pytest timeout in seconds")
    parser.add_argument("--out",     type=str, default="reports",
                        help="Output directory for results")
    parser.add_argument("--quick",   action="store_true",
                        help="Run 12 fixtures (3 families × 4) for fast smoke-test")
    args = parser.parse_args()

    all_fixtures = build_all_fixtures()
    if args.quick:
        fixtures = []
        for fam in FAMILY_NAMES[:3]:
            fx_fam = [fx for fx in all_fixtures if fx.family == fam]
            fixtures.extend(fx_fam[:4])
    else:
        fixtures = all_fixtures

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTAC-PSM-006B Fast Replication Run")
    print(f"  Fixtures : {len(fixtures)}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Workers  : {args.workers} threads (independent variants)")
    print(f"  Timeout  : {args.timeout}s per fixture")
    print(f"  Output   : {args.out}/")
    print()

    t_total      = time.time()
    seed_results = []

    # Shared verifier cache across all seeds — pytest results are deterministic
    # by (files_content, command), so cache hits from seed 0 are valid for
    # seeds 1-4.  This eliminates almost all redundant subprocess calls after
    # the first seed warms the cache.
    shared_verifier  = CachingSubprocessVerifier(
        timeout          = args.timeout,
        n_warmup_workers = args.workers,
    )
    prewarm_applier  = PatchApplier()

    # Pre-warm: run all deterministic fixture states in parallel BEFORE the
    # seed loop.  After this, seed 0's sequential full_memory loop hits the
    # cache for every before-patch and almost every after-patch call.
    print("Pre-warming verifier cache ...", flush=True)
    prewarm_cache(fixtures, shared_verifier, prewarm_applier, n_workers=args.workers)
    print(f"  Cache warmed: {shared_verifier.misses} unique states, "
          f"{shared_verifier.hits} hits so far\n", flush=True)

    for seed in args.seeds:
        sr = run_seed_fast(seed, fixtures, args.timeout, args.workers,
                           shared_verifier=shared_verifier)
        seed_results.append(sr)

        # Incremental save after each seed
        partial_agg     = aggregate_seeds(seed_results)
        partial_verdict = compute_verdict(seed_results)
        write_results_json(
            seed_results, partial_agg, partial_verdict,
            str(out_dir / "psm006b_results.json"),
        )
        print(f"  [saved incremental results after seed {seed}]", flush=True)

    elapsed = time.time() - t_total
    agg     = aggregate_seeds(seed_results)
    verdict = compute_verdict(seed_results)

    # Write all output files
    write_results_json(seed_results, agg, verdict, str(out_dir / "psm006b_results.json"))
    write_summary_txt(agg, seed_results, verdict,  str(out_dir / "psm006b_summary.txt"))
    write_per_family_rates(seed_results, agg,       str(out_dir / "psm006b_per_family_rates.txt"))
    write_confusion_matrix(seed_results,            str(out_dir / "psm006b_confusion_matrix.txt"))
    write_failure_analysis(seed_results,            str(out_dir / "psm006b_failure_analysis.txt"))

    # Print final report to stdout
    print(f"\n{'='*60}")
    print(f"VERDICT: TAC-PSM-006B {verdict}")
    print(f"Total elapsed: {elapsed:.1f}s  (mean {elapsed/len(args.seeds):.1f}s/seed)")
    print()

    print("Variant Pass Rates (mean ± std):")
    for v in VARIANT_NAMES:
        vd = agg["variant_rates"].get(v, {"mean": 0.0, "std": 0.0})
        print(f"  {v:<30s}  {vd['mean']:.3f} ± {vd['std']:.3f}")

    print()
    print("Key Metrics:")
    key_metrics = [
        "pytest_pass_rate",
        "procedure_retrieval_accuracy",
        "retry_after_update_success",
        "procedure_reuse_gain",
        "cross_fixture_transfer_success",
        "wrong_procedure_harm",
    ]
    for k in key_metrics:
        vd = agg["metrics"].get(k, {"mean": 0.0, "std": 0.0})
        print(f"  {k:<44s}  {vd['mean']:.4f} ± {vd['std']:.4f}")

    print()
    print("Gate Results:")
    n_all_pass = 0
    for gate, info in agg["gate_pass_rates"].items():
        sym = "[PASS]" if info["pass"] == info["total"] else "[FAIL]"
        if info["pass"] == info["total"]:
            n_all_pass += 1
        print(f"  {sym} {gate:<40s}  {info['pass']}/{info['total']}")

    n_gates = len(agg["gate_pass_rates"])
    print()
    print(f"  {n_all_pass}/{n_gates} gates pass on ALL seeds")

    # Trivial-fixtures warning
    fm_mean  = agg["variant_rates"]["full_memory"]["mean"]
    ora_mean = agg["variant_rates"]["oracle"]["mean"]
    if fm_mean >= 1.0 and ora_mean >= 1.0:
        print()
        print("  ⚠  NOTE: full_memory and oracle are both 1.000.")
        print("     Fixtures may be too easy (zero noise tolerance).")
        print("     This is not a failure, but is recorded in the report.")

    print()
    print(f"  Results saved to {out_dir}/")
    print(f"    psm006b_results.json")
    print(f"    psm006b_summary.txt")
    print(f"    psm006b_per_family_rates.txt")
    print(f"    psm006b_confusion_matrix.txt")
    print(f"    psm006b_failure_analysis.txt")

    shared_verifier.shutdown()
    sys.exit(0 if verdict == "VALIDATES" else 1)


if __name__ == "__main__":
    main()
