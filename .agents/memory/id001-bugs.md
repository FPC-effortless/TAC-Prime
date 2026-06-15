---
name: ID001 bugs fixed
description: Two routing bugs in TAC-Prime-ID001 and one benchmark loader bug; fixes confirmed by 42/42 tests.
---

# ID001 Bug Fixes

## Bug 1: compute_route_consistency empty-dict crash
**File:** `tacm/tacm/id001/routing.py`
**Fix:** Guard against empty `route_counts` dict before computing entropy
**Root cause:** Function accessed `.values()` on empty dict, causing ZeroDivisionError

## Bug 2: compute_route_consistency uniform entropy threshold
**File:** `tacm/tacm/id001/routing.py`
**Fix:** Use `log(n_families)` not `log(n_routes)` as the uniform entropy normalizer
**Root cause:** `n_routes` was the number of distinct routes observed (could be 1), so
uniform entropy was miscalculated; `n_families` is the correct reference for maximum entropy.

## Bug 3: benchmark loader sys.modules registration
**File:** `tacm/tests/test_tacprime_id001_identity_integration.py`
**Fix:** Register module in `sys.modules` before calling `exec_module()`
**Root cause:** Module's `__init__` code referenced its own name in sys.modules during
initialization; exec_module without prior registration caused ModuleNotFoundError.

**Why important:** These bugs would cause 6 test failures (2 routing + 4 benchmark loader) and make the benchmark appear broken. After fixes: 42 passed, 16 skipped.
