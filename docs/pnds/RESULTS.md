# PNDS Research Results Ledger

This ledger is append-oriented. Every non-trivial PNDS experiment, diagnostic, control, replication, or methodology correction must receive a stable experiment ID and an entry here.

## Entry template

### [PNDS-REPO-CAPABILITY-NNN] — <short title>

- **Status:** NOT_RUN
- **Protocol version:** PNDS-URP v0.1
- **Repository:** <repo>
- **Branch:** <branch>
- **Commit:** <sha>
- **Parent experiment:** <ID or N/A>
- **Date:** <YYYY-MM-DD>

**Hypothesis**

<One falsifiable claim.>

**Variables**

- Independent:
- Dependent:
- Controls:

**Configuration**

- Model:
- Dataset/task:
- Seeds:
- Optimizer:
- Learning rate:
- Steps:
- Evaluation:

**Baseline**

<Exact baseline and configuration.>

**Controls**

- Control 1:
- Control 2:
- Leakage control:
- Reset/intervention control:
- Other:

**Results**

| Metric | Baseline | Treatment | Delta | Notes |
|---|---:|---:|---:|---|
| | | | | |

**Run/artifact locations**

- <path or URL>

**Observed**

<Only directly measured facts.>

**Inferred**

<Interpretation supported by the measurements.>

**Speculative**

<Unverified explanation or future hypothesis.>

**Limitations / failures**

<What the experiment cannot establish.>

**Evidence level**

E0 / E1 / E2 / E3 / E4 / E5 / E6

**Decision**

SUPPORTED / PARTIALLY_SUPPORTED / INCONCLUSIVE / FALSIFIED / BLOCKED / NOT_RUN

**Next experiment**

<Smallest experiment that resolves the main uncertainty.>

---

## Initial repository setup

### PNDS-SETUP-001 — Universal protocol installation

- **Status:** SUPPORTED
- **Protocol version:** PNDS-URP v0.1
- **Purpose:** Establish the common research protocol and result-ledger surface.
- **Decision:** SUPPORTED
- **Next experiment:** Create the first repository-specific PNDS experiment under this protocol.
