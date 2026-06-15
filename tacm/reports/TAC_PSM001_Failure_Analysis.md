# TAC-PSM-001: Failure Analysis

Gates failed: 0 / 7

**No gate failures detected.** All success criteria were met.

If unexpected failures occur in future runs, common root causes are:

| Failure pattern | Most likely cause | Corrective experiment |
|---|---|---|
| Low retrieval accuracy | Embedding space poorly separated by family | Increase embedding dim; add family-contrastive loss |
| Low reuse gain | Retrieved procedures too generic | Increase task-signature specificity |
| No retry improvement | Fork threshold too high | Lower `fork_threshold`; improve recovery step injection |
| No transfer gain | Family embeddings too distant | Add cross-family similarity loss during store build |
| Survival instability | High decay rate | Lower `decay_rate` in training config |