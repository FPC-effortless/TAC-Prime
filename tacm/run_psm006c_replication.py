"""
TAC-PSM-006C: Fast 5-Seed Replication Runner
=============================================

Runs the full PSM-006C ablation across 5 seeds and writes:

  reports/psm006c_results.json
  reports/psm006c_summary.txt
  reports/psm006c_per_family_rates.txt
  reports/psm006c_confusion_matrix.txt
  reports/psm006c_failure_analysis.txt

Usage:
  python run_psm006c_replication.py [--seeds 0 1 2 3 4] [--workers 8]
                                    [--timeout 10.0] [--out reports/]

Architecture:
  - full_memory_embedding_update   : sequential (memory + embeddings accumulate)
  - full_memory                    : sequential (memory accumulates)
  - reset                          : per-fixture fresh store, parallelisable
  - no_update / oracle             : parallelisable (no shared mutable state)
  - CachingSubprocessVerifier       : shared across variants within a seed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from tacm.psm006b.fixture_builder import build_all_fixtures
from tacm.psm006b.fixture_schema import FAMILY_NAMES, Fixture
from tacm.psm006b.patch_applier import PatchApplier
from tacm.psm006b.memory_store import SimpleProceduralMemoryStore
from tacm.psm006b.procedural_repair_agent import seed_procedural_memory
from tacm.psm006b.caching_verifier import CachingSubprocessVerifier
from tacm.psm006c.agent import ProceduralRepairAgent006C, RepairTrace006C
from tacm.psm006c.baselines import VARIANT_NAMES_006C
from tacm.psm006c.embedding_update import OnlineEmbeddingAdapter
from tacm.psm006c.metrics import (
    compute_metrics_006c,
    evaluate_success_gates_006c,
    compute_family_confusion_matrix_006c,
    classify_failures_006c,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _pass_rate(traces: List[RepairTrace006C]) -> float:
    if not traces:
        return 0.0
    return mean(1.0 if t.pytest_pass else 0.0 for t in traces)


def _make_seeded_store(seed: int) -> SimpleProceduralMemoryStore:
    s = SimpleProceduralMemoryStore()
    seed_procedural_memory(s, n_records_per_family=2, rng_seed=seed)
    return s


def _fixture_rng_seed(seed: int, variant: str, fx_index: int) -> int:
    key = f"{seed}_{variant}_{fx_index}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 31)


# ── Sequential variant runners (shared verifier) ──────────────────────────

def run_embedding_update_seq(
    fixtures: List[Fixture], seed: int, verifier, applier: PatchApplier
) -> List[RepairTrace006C]:
    store   = _make_seeded_store(seed)
    adapter = OnlineEmbeddingAdapter(lr_fail=0.10, lr_success=0.05)
    agent   = ProceduralRepairAgent006C(
        store=store, verifier=verifier, applier=applier,
        adapter=adapter, mode="full_memory_embedding_update",
        retrieval_noise=0.10, rng_seed=seed, max_retries=1,
    )
    return [agent.repair(fx) for fx in fixtures]


def run_full_memory_seq(
    fixtures: List[Fixture], seed: int, verifier, applier: PatchApplier
) -> List[RepairTrace006C]:
    store   = _make_seeded_store(seed)
    adapter = OnlineEmbeddingAdapter()
    agent   = ProceduralRepairAgent006C(
        store=store, verifier=verifier, applier=applier,
        adapter=adapter, mode="full_memory",
        retrieval_noise=0.10, rng_seed=seed, max_retries=1,
    )
    return [agent.repair(fx) for fx in fixtures]


# ── Per-fixture independent runners ──────────────────────────────────────

def _repair_independent(
    fx: Fixture, fx_idx: int, variant: str, seed: int,
    verifier, applier: PatchApplier,
) -> RepairTrace006C:
    fx_seed = _fixture_rng_seed(seed, variant, fx_idx)
    store   = _make_seeded_store(fx_seed)
    adapter = OnlineEmbeddingAdapter()
    agent   = ProceduralRepairAgent006C(
        store=store, verifier=verifier, applier=applier,
        adapter=adapter, mode=variant,
        retrieval_noise=0.10, rng_seed=fx_seed, max_retries=0,
    )
    return agent.repair(fx)


def _repair_reset(
    fx: Fixture, fx_idx: int, seed: int, verifier, applier: PatchApplier
) -> RepairTrace006C:
    rng     = np.random.default_rng(seed)
    fx_seeds = [int(rng.integers(0, 2**31)) for _ in range(fx_idx + 1)]
    fx_seed  = fx_seeds[-1]
    store   = _make_seeded_store(fx_seed)
    adapter = OnlineEmbeddingAdapter()
    agent   = ProceduralRepairAgent006C(
        store=store, verifier=verifier, applier=applier,
        adapter=adapter, mode="reset",
        retrieval_noise=0.10, rng_seed=fx_seed, max_retries=0,
    )
    return agent.repair(fx)


def run_parallel(
    variant: str, fixtures: List[Fixture], seed: int,
    verifier, applier: PatchApplier, n_workers: int = 8,
) -> List[RepairTrace006C]:
    results: List[Optional[RepairTrace006C]] = [None] * len(fixtures)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        if variant == "reset":
            futures = {
                ex.submit(_repair_reset, fx, i, seed, verifier, applier): i
                for i, fx in enumerate(fixtures)
            }
        else:
            futures = {
                ex.submit(_repair_independent, fx, i, variant, seed, verifier, applier): i
                for i, fx in enumerate(fixtures)
            }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results  # type: ignore


# ── Cache pre-warmer ──────────────────────────────────────────────────────

def prewarm_cache(
    fixtures: List[Fixture], verifier, applier: PatchApplier, n_workers: int = 8,
) -> None:
    tasks = []
    for fx in fixtures:
        all_files = fx.all_files()
        tasks.append((all_files, fx.verification_command, fx.fixture_id, "prewarm_before"))
        pr_c = applier.apply(all_files, fx.expected_patch)
        tasks.append((pr_c.patched_files, fx.verification_command, fx.fixture_id, "prewarm_correct"))
        wrong_fam = next((f for f in FAMILY_NAMES if f != fx.family), FAMILY_NAMES[0])
        pr_w = applier.apply_wrong_family_patch(all_files, wrong_fam)
        tasks.append((pr_w.patched_files, fx.verification_command, fx.fixture_id, "prewarm_wrong"))

    t0 = time.time()
    if hasattr(verifier, "map_run"):
        verifier.map_run(tasks)
    else:
        def _run(args):
            f, c, fid, v = args
            return verifier.run(f, c, fixture_id=fid, variant=v)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_run, tasks))
    print(f"  Prewarm: {len(tasks)} tasks in {time.time()-t0:.1f}s  "
          f"(hits={verifier.hits} misses={verifier.misses})", flush=True)


# ── Per-seed runner ───────────────────────────────────────────────────────

def run_seed(
    seed: int, fixtures: List[Fixture], timeout_s: float,
    n_workers: int, shared_verifier=None,
) -> dict:
    print(f"\n  Seed {seed}: {len(fixtures)} fixtures × 5 variants ...", flush=True)
    t0      = time.time()
    verifier = shared_verifier or CachingSubprocessVerifier(timeout=timeout_s)
    applier  = PatchApplier()

    results: Dict[str, List[RepairTrace006C]] = {}

    # Sequential (accumulating) variants first — they warm the cache
    tv = time.time()
    results["full_memory_embedding_update"] = run_embedding_update_seq(
        fixtures, seed, verifier, applier
    )
    print(f"    full_memory_embedding_update done  {time.time()-tv:.1f}s  "
          f"hits={verifier.hits}", flush=True)

    tv = time.time()
    results["full_memory"] = run_full_memory_seq(fixtures, seed, verifier, applier)
    print(f"    full_memory done  {time.time()-tv:.1f}s  hits={verifier.hits}", flush=True)

    # Parallel variants
    for v in ("reset", "no_update", "oracle"):
        tv = time.time()
        results[v] = run_parallel(v, fixtures, seed, verifier, applier, n_workers)
        print(f"    {v:<36s} done  {time.time()-tv:.1f}s  hits={verifier.hits}", flush=True)

    metrics  = compute_metrics_006c(results)
    gates    = evaluate_success_gates_006c(metrics, results)
    failures = classify_failures_006c(results["full_memory_embedding_update"])
    confusion = compute_family_confusion_matrix_006c(
        results["full_memory_embedding_update"], FAMILY_NAMES
    )

    variant_rates = {v: _pass_rate(results.get(v, [])) for v in VARIANT_NAMES_006C}
    elapsed       = time.time() - t0

    print(f"  Seed {seed} done {elapsed:.1f}s  "
          f"emb_update={variant_rates['full_memory_embedding_update']:.3f}  "
          f"full_memory={variant_rates['full_memory']:.3f}  "
          f"oracle={variant_rates['oracle']:.3f}  "
          f"reset={variant_rates['reset']:.3f}",
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


# ── Aggregation ───────────────────────────────────────────────────────────

def aggregate_seeds(seed_results: List[dict]) -> dict:
    metric_keys  = list(seed_results[0]["metrics"].keys())
    agg_metrics  = {}
    for k in metric_keys:
        vals = [r["metrics"][k] for r in seed_results]
        agg_metrics[k] = {"mean": mean(vals), "std": stdev(vals) if len(vals) > 1 else 0.0}

    gate_names = list(seed_results[0]["gates"].keys())
    gate_pass  = {}
    for g in gate_names:
        n_pass = sum(1 for r in seed_results if r["gates"].get(g, False))
        gate_pass[g] = {"pass": n_pass, "total": len(seed_results)}

    variant_names = list(seed_results[0]["variant_rates"].keys())
    agg_variants  = {}
    for v in variant_names:
        vals = [r["variant_rates"][v] for r in seed_results]
        agg_variants[v] = {"mean": mean(vals), "std": stdev(vals) if len(vals) > 1 else 0.0}

    return {"metrics": agg_metrics, "gate_pass_rates": gate_pass, "variant_rates": agg_variants}


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
    elif gates_most >= 4:
        return "PARTIALLY_VALIDATES"
    else:
        return "DOES_NOT_VALIDATE"


# ── Writers ───────────────────────────────────────────────────────────────

def write_json(seed_results, agg, verdict, path):
    payload = {
        "run_info": {
            "date":       time.strftime("%Y-%m-%d"),
            "n_seeds":    len(seed_results),
            "n_fixtures": seed_results[0]["n_fixtures"] if seed_results else 0,
            "n_variants": 5,
            "experiment": "TAC-PSM-006C",
        },
        "seed_results":      [{k: v for k, v in sr.items() if k != "confusion"}
                               for sr in seed_results],
        "per_seed_confusion": [{"seed": sr["seed"], "confusion": sr["confusion"]}
                                for sr in seed_results],
        "aggregate":          agg,
        "verdict":            verdict,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(agg, seed_results, verdict, path):
    lines = [
        "TAC-PSM-006C Replication Summary",
        "=" * 60,
        f"Date: {time.strftime('%Y-%m-%d')}",
        f"Seeds: {[r['seed'] for r in seed_results]}",
        f"Fixtures per seed: {seed_results[0]['n_fixtures']}",
        "",
        f"Verdict: TAC-PSM-006C {verdict}",
        "",
        "Variant Pass Rates (mean ± std across seeds):",
    ]
    for v in VARIANT_NAMES_006C:
        vd = agg["variant_rates"].get(v, {"mean": 0.0, "std": 0.0})
        marker = "  ← NEW (embedding update)" if v == "full_memory_embedding_update" else ""
        lines.append(f"  {v:<36s}  {vd['mean']:.3f} ± {vd['std']:.3f}{marker}")

    lines += ["", "Key Metrics (mean ± std):"]
    priority = [
        "pytest_pass_rate", "retry_after_update_success",
        "procedure_retrieval_accuracy", "procedure_reuse_gain",
        "embedding_update_count", "embedding_shift_norm_mean",
        "retrieval_changed_after_update", "family_changed_after_update",
        "successful_retrieval_recovery", "emb_update_vs_full_memory_gain",
        "patch_correctness",
    ]
    for k in priority:
        vd = agg["metrics"].get(k, {"mean": 0.0, "std": 0.0})
        lines.append(f"  {k:<44s}  {vd['mean']:.4f} ± {vd['std']:.4f}")

    lines += ["", "Gate Results (seeds_passing / total_seeds):"]
    for gate, info in agg["gate_pass_rates"].items():
        sym = "[PASS]" if info["pass"] == info["total"] else "[FAIL]"
        lines.append(f"  {sym} {gate:<44s}  {info['pass']}/{info['total']}")

    lines += ["", "Per-Seed Summary:"]
    for sr in seed_results:
        vr = sr["variant_rates"]
        g_pass = sum(1 for v in sr["gates"].values() if v)
        lines.append(
            f"  seed={sr['seed']}  "
            f"emb_update={vr['full_memory_embedding_update']:.3f}  "
            f"full_memory={vr['full_memory']:.3f}  "
            f"reset={vr['reset']:.3f}  "
            f"oracle={vr['oracle']:.3f}  "
            f"gates={g_pass}/7  ({sr['elapsed_s']:.0f}s)"
        )

    lines += [
        "",
        "Scientific question: Does online embedding adaptation improve",
        "procedural retrieval and repair over text-only updates?",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_per_family(seed_results, path):
    lines = ["TAC-PSM-006C Per-Family Retrieval Rates (full_memory_embedding_update)", "=" * 60, ""]
    for sr in seed_results:
        lines.append(f"Seed {sr['seed']}:")
        for fam in FAMILY_NAMES:
            row   = sr["confusion"].get(fam, {})
            total = sum(row.values())
            acc   = row.get(fam, 0) / total if total else 0.0
            lines.append(f"  {fam:<30s}  retrieval_acc={acc:.3f}  (n={total})")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_confusion(seed_results, path):
    lines = ["TAC-PSM-006C Confusion Matrix (full_memory_embedding_update, mean)", "=" * 60, ""]
    header = "true\\retrieved".ljust(30) + "  ".join(f[-8:].ljust(10) for f in FAMILY_NAMES)
    lines.append(header)
    for tf in FAMILY_NAMES:
        row_vals = []
        for rf in FAMILY_NAMES:
            vals = [sr["confusion"].get(tf, {}).get(rf, 0) for sr in seed_results]
            row_vals.append(f"{mean(vals):.1f}".ljust(10))
        lines.append(f"{tf[-20:].ljust(30)}  {'  '.join(row_vals)}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_failure_analysis(seed_results, path):
    from tacm.psm006b.fixture_schema import FAILURE_CLASSES
    all_counts = {fc: [] for fc in FAILURE_CLASSES}
    all_counts["none"] = []
    for sr in seed_results:
        for fc in list(FAILURE_CLASSES) + ["none"]:
            all_counts[fc].append(sr["failures"].get(fc, 0))
    lines = [
        "TAC-PSM-006C Failure Analysis (full_memory_embedding_update)",
        "=" * 60, "",
        f"Seeds: {[sr['seed'] for sr in seed_results]}",
        f"Fixtures per seed: {seed_results[0]['n_fixtures']}",
        "",
        "Failure class counts (mean ± std across seeds):",
    ]
    for fc, counts in all_counts.items():
        m   = mean(counts)
        s   = stdev(counts) if len(counts) > 1 else 0.0
        pct = m / seed_results[0]["n_fixtures"] * 100
        lines.append(f"  {fc:<40s}  {m:.1f} ± {s:.1f}  ({pct:.1f}%)")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PSM-006C 5-seed replication runner")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--out",     type=str, default="reports")
    parser.add_argument("--quick",   action="store_true",
                        help="12 fixtures (smoke test)")
    args = parser.parse_args()

    all_fixtures = build_all_fixtures()
    if args.quick:
        fixtures = []
        for fam in FAMILY_NAMES[:3]:
            fxs = [fx for fx in all_fixtures if fx.family == fam]
            fixtures.extend(fxs[:4])
    else:
        fixtures = all_fixtures

    print(f"\nTAC-PSM-006C Replication  ({len(fixtures)} fixtures × {len(args.seeds)} seeds)",
          flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared verifier across ALL seeds — cache persists
    shared_verifier = CachingSubprocessVerifier(timeout=args.timeout)

    print("\nPre-warming verifier cache ...", flush=True)
    applier = PatchApplier()
    prewarm_cache(fixtures, shared_verifier, applier, n_workers=args.workers)

    seed_results = []
    t_total = time.time()
    for seed in args.seeds:
        sr = run_seed(seed, fixtures, args.timeout, args.workers, shared_verifier)
        seed_results.append(sr)
        # Incremental save
        write_json(seed_results,
                   aggregate_seeds(seed_results),
                   compute_verdict(seed_results),
                   str(out_dir / "psm006c_results.json"))

    agg     = aggregate_seeds(seed_results)
    verdict = compute_verdict(seed_results)
    total_t = time.time() - t_total

    write_json(seed_results, agg, verdict, str(out_dir / "psm006c_results.json"))
    write_summary(agg, seed_results, verdict, str(out_dir / "psm006c_summary.txt"))
    write_per_family(seed_results, str(out_dir / "psm006c_per_family_rates.txt"))
    write_confusion(seed_results, str(out_dir / "psm006c_confusion_matrix.txt"))
    write_failure_analysis(seed_results, str(out_dir / "psm006c_failure_analysis.txt"))

    print(f"\n{'='*60}", flush=True)
    print(f"TAC-PSM-006C Replication Complete ({total_t:.1f}s total)", flush=True)
    print(f"Verdict: {verdict}", flush=True)
    print(f"\nVariant pass rates (mean ± std):", flush=True)
    for v in VARIANT_NAMES_006C:
        vd = agg["variant_rates"].get(v, {"mean": 0.0, "std": 0.0})
        marker = " ← NEW" if v == "full_memory_embedding_update" else ""
        print(f"  {v:<36s}  {vd['mean']:.3f} ± {vd['std']:.3f}{marker}", flush=True)
    print(f"\nGates:", flush=True)
    for gate, info in agg["gate_pass_rates"].items():
        sym = "PASS" if info["pass"] == info["total"] else "FAIL"
        print(f"  [{sym}] {gate:<44s}  {info['pass']}/{info['total']}", flush=True)
    print(f"\nResults saved to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
