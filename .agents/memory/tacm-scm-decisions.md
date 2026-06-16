---
name: TAC-SCM-REAL001 key decisions
description: Durable design decisions and quirks for the TAC-SCM-REAL001 structure-native LM implementation.
---

# TAC-SCM-REAL001 Key Decisions

## NSF and DPSL disabled by default
`enable_nsf_survival=False` and `enable_dpsl_refinement=False` in `TACSCMConfig`.
**Why:** Spec explicitly states "Explicitly Do NOT Implement Yet" for both NSF and DPSL. The modules exist and are gated by the config flags, but must default to off. Any preset that doesn't explicitly set these will leave them off.
**How to apply:** When adding new presets or changing defaults, always verify these two flags remain `False` unless a future spec explicitly enables them.

## SCMSample is a dataclass, not a dict
`make_synthetic_repair_dataset()` returns an `SCMDataset` whose `__getitem__` yields `SCMSample` instances (dataclass). Access fields via attribute: `s.input_ids`, `s.structure_id`, NOT `s["input_ids"]`.
**Why:** The dataset was designed this way; the collator handles the conversion to torch tensors internally.
**How to apply:** Any test or code iterating over the dataset must use attribute access. Use `isinstance(s, SCMSample)` guard for defensive compatibility.

## scm_diagnostics.py is torch-free
`SCMDiagnosticsTracker` and all its helpers (`DiagnosticsRow`, `_WindowStat`) have zero torch imports.
**Why:** Diagnostics must be importable and usable even without torch for config-only or dataset-only environments.
**How to apply:** Never add torch imports to `scm_diagnostics.py`. Keep it pure Python + stdlib.

## Test partitioning pattern
All TAC-SCM tests that need torch use `@needs_torch = pytest.mark.skipif(not HAS_TORCH, ...)` defined at module level. Module-level `pytest.importorskip` kills the entire module when torch is absent — never use it.
**Why:** The environment (NixOS research container) does not have torch installed; tests must collect and skip gracefully rather than failing to collect.
**How to apply:** Every new test file that touches torch must use `try/except ImportError; HAS_TORCH = True/False` + `@pytest.mark.skipif` at class or function level.

## Structure context at LM head
The spec requires "language generation must depend on structure context." This is satisfied via residual fusion inside each SCM block (step 9: `h = h + fusion_proj(cat([h, struct_summary]))`), not through explicit concatenation at the final LM head. The resulting `h` entering the LM head already carries structure information.
**Why:** Residual fusion into `h` achieves the same information flow as explicit concat at the head, with fewer parameters and cleaner gradient paths.

## Shared StructureMemory
`StructureMemory` is instantiated once in `TACSCMLanguageModel` and passed to every `IntegratedStructureLanguageBlock`. All SCM blocks read from and write to the same memory bank.
**Why:** Cross-layer structure sharing requires a single shared bank; per-layer banks would fragment the structure space.
