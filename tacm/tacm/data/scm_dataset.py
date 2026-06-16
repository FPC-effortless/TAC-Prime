"""
TAC-SCM-REAL001: Dataset Support

Supports four data modes (mixed in any ratio):
  1. plain_text  — raw text sequences for standard LM training
  2. jsonl       — instruction/completion pairs
  3. repair      — synthetic code repair traces (from PSM fixtures)
  4. structure   — structure-labeled data (structure_id + rule + trace)

Each sample is an SCMSample dataclass.
SCMDataCollator handles tokenisation + padding for the DataLoader.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    torch   = None  # type: ignore
    Dataset = object
    _HAS_TORCH = False


# ── Sample ─────────────────────────────────────────────────────────────────────

@dataclass
class SCMSample:
    """
    One training sample.  All fields except input_ids / labels are optional
    and are used as auxiliary conditioning signals.
    """
    input_ids:    List[int]
    labels:       List[int]                        # -100 for masked positions
    task_id:      Optional[str]             = None
    structure_id: Optional[int]             = None  # ground-truth structure label
    trace:        Optional[str]             = None  # repair / reasoning trace
    rule:         Optional[str]             = None  # symbolic rule that applies
    feedback:     Optional[List[float]]     = None  # external feedback vector
    source:       str                       = "text"  # "text"|"jsonl"|"repair"|"structure"


# ── Dataset ───────────────────────────────────────────────────────────────────

class SCMDataset(Dataset):
    """
    Multi-source dataset for TAC-SCM-REAL001.

    Parameters
    ----------
    samples  : list of SCMSample (pre-loaded)
    seq_len  : maximum sequence length (samples are truncated/padded)
    pad_id   : padding token id
    """

    def __init__(
        self,
        samples: List[SCMSample],
        seq_len: int = 512,
        pad_id:  int = 0,
    ):
        self.samples = samples
        self.seq_len = seq_len
        self.pad_id  = pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SCMSample:
        s = self.samples[idx]
        # Truncate
        ids  = s.input_ids[:self.seq_len]
        labs = s.labels[:self.seq_len]
        # Pad
        pad_len = self.seq_len - len(ids)
        if pad_len > 0:
            ids  = ids  + [self.pad_id] * pad_len
            labs = labs + [-100] * pad_len
        return SCMSample(
            input_ids    = ids,
            labels       = labs,
            task_id      = s.task_id,
            structure_id = s.structure_id,
            trace        = s.trace,
            rule         = s.rule,
            feedback     = s.feedback,
            source       = s.source,
        )

    @classmethod
    def from_text_file(
        cls,
        path: str,
        tokenizer,
        seq_len: int = 512,
        stride:  int = 256,
    ) -> "SCMDataset":
        """Load plain text file; chunk into overlapping windows."""
        text = Path(path).read_text(encoding="utf-8")
        ids  = tokenizer.encode(text)
        samples = []
        for start in range(0, len(ids) - seq_len + 1, stride):
            chunk = ids[start : start + seq_len]
            samples.append(SCMSample(
                input_ids = chunk,
                labels    = chunk,
                source    = "text",
            ))
        return cls(samples, seq_len=seq_len, pad_id=tokenizer.pad_token_id or 0)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer,
        seq_len:          int  = 512,
        input_field:      str  = "input",
        output_field:     str  = "output",
        task_id_field:    Optional[str] = "task_id",
        structure_field:  Optional[str] = "structure_id",
    ) -> "SCMDataset":
        """
        Load JSONL instruction data.
        Labels mask the input part (-100) and train on the output part.
        """
        samples = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inp     = obj.get(input_field, "")
            out     = obj.get(output_field, "")
            full    = inp + out
            inp_ids = tokenizer.encode(inp, add_special_tokens=False)
            out_ids = tokenizer.encode(out, add_special_tokens=False)
            all_ids = inp_ids + out_ids
            labels  = [-100] * len(inp_ids) + out_ids

            task_id = obj.get(task_id_field) if task_id_field else None
            struct  = obj.get(structure_field) if structure_field else None
            trace   = obj.get("trace")
            rule    = obj.get("rule")

            samples.append(SCMSample(
                input_ids    = all_ids[:seq_len],
                labels       = labels[:seq_len],
                task_id      = str(task_id) if task_id is not None else None,
                structure_id = int(struct)  if struct  is not None else None,
                trace        = str(trace)   if trace   is not None else None,
                rule         = str(rule)    if rule    is not None else None,
                source       = "jsonl",
            ))
        return cls(samples, seq_len=seq_len, pad_id=tokenizer.pad_token_id or 0)

    @classmethod
    def from_repair_samples(
        cls,
        repair_data: List[Dict[str, Any]],
        tokenizer,
        seq_len: int = 512,
    ) -> "SCMDataset":
        """
        Load from a list of repair dictionaries (same format as PSM fixtures).

        Expected keys per dict:
          buggy_code  : str — broken code (input)
          fixed_code  : str — repaired code (target)
          family      : str — repair family / structure class
          trace       : str optional — repair trace
          rule        : str optional — applicable rule
        """
        samples = []
        for d in repair_data:
            buggy   = d.get("buggy_code", "")
            fixed   = d.get("fixed_code", "")
            family  = d.get("family", "unknown")
            trace   = d.get("trace")
            rule    = d.get("rule")

            inp_ids  = tokenizer.encode(buggy, add_special_tokens=False)[:seq_len // 2]
            out_ids  = tokenizer.encode(fixed, add_special_tokens=False)[:seq_len // 2]
            all_ids  = inp_ids + out_ids
            labels   = [-100] * len(inp_ids) + out_ids

            samples.append(SCMSample(
                input_ids = all_ids,
                labels    = labels,
                task_id   = family,
                trace     = trace,
                rule      = rule,
                source    = "repair",
            ))
        return cls(samples, seq_len=seq_len, pad_id=tokenizer.pad_token_id or 0)


# ── Synthetic repair dataset ───────────────────────────────────────────────────

def make_synthetic_repair_dataset(
    n_samples: int = 1000,
    n_families: int = 8,
    tokenizer = None,
    seq_len:   int = 128,
    seed:      int = 42,
) -> "SCMDataset":
    """
    Generate a synthetic repair dataset without a real tokenizer.

    Each sample is a pair of integer token sequences:
      input  = [family_id, ...random tokens...]
      labels = [family_id, ...slightly modified tokens...]

    Used for smoke tests and initial architecture validation.
    """
    rng = random.Random(seed)
    vocab = 256  # small synthetic vocab

    FAMILY_TEMPLATES = {
        i: {
            "pattern": list(range(i * 10, i * 10 + 5)),
            "fix":     list(range(i * 10 + 1, i * 10 + 6)),
        }
        for i in range(n_families)
    }

    samples = []
    for _ in range(n_samples):
        fam_id   = rng.randint(0, n_families - 1)
        tmpl     = FAMILY_TEMPLATES[fam_id]
        base_len = rng.randint(seq_len // 4, seq_len // 2)

        # Buggy: inject the pattern at a random position
        tokens = [rng.randint(10, vocab - 1) for _ in range(base_len)]
        pos    = rng.randint(0, max(0, base_len - 5))
        tokens[pos:pos + 5] = tmpl["pattern"]

        # Fixed: replace pattern with fix
        fixed  = tokens[:]
        fixed[pos:pos + 5] = tmpl["fix"]

        inp_ids = tokens[:seq_len // 2]
        out_ids = fixed[:seq_len // 2]
        all_ids = inp_ids + out_ids
        labels  = [-100] * len(inp_ids) + out_ids

        samples.append(SCMSample(
            input_ids    = all_ids,
            labels       = labels,
            task_id      = f"family_{fam_id}",
            structure_id = fam_id,
            source       = "repair",
        ))

    return SCMDataset(samples, seq_len=seq_len, pad_id=0)


# ── Collator ──────────────────────────────────────────────────────────────────

class SCMDataCollator:
    """
    DataLoader collator: stacks SCMSamples into batch tensors.

    Returns dict with:
      input_ids         : (B, T) long
      labels            : (B, T) long
      attention_mask    : (B, T) float
      structure_ids     : (B,) long or None
      has_structure_ids : bool
    """

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch: List[SCMSample]) -> Dict[str, Any]:
        max_len = max(len(s.input_ids) for s in batch)

        input_ids  = []
        labels     = []
        attn_masks = []
        struct_ids = []
        has_struct = False

        for s in batch:
            ids  = s.input_ids
            labs = s.labels
            plen = max_len - len(ids)
            mask = [1] * len(ids) + [0] * plen
            ids  = ids  + [self.pad_id] * plen
            labs = labs + [-100]       * plen

            input_ids.append(ids)
            labels.append(labs)
            attn_masks.append(mask)

            if s.structure_id is not None:
                struct_ids.append(s.structure_id)
                has_struct = True
            else:
                struct_ids.append(-1)

        if _HAS_TORCH:
            out = {
                "input_ids":      torch.tensor(input_ids,  dtype=torch.long),
                "labels":         torch.tensor(labels,     dtype=torch.long),
                "attention_mask": torch.tensor(attn_masks, dtype=torch.float),
            }
            if has_struct:
                out["structure_ids"] = torch.tensor(struct_ids, dtype=torch.long)
        else:
            out = {
                "input_ids":      input_ids,
                "labels":         labels,
                "attention_mask": attn_masks,
            }
            if has_struct:
                out["structure_ids"] = struct_ids
        return out
