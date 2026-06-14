"""
TAC-SM Repository Repair Agent Loop

Autonomous agent cycle:
  Read Repository
  → Analyze Bug Report
  → Retrieve Structures + Procedures
  → Select Family + Specialist
  → Generate Repair Plan
  → Generate Patch
  → Run Tests
  → Verify Outcome
  → Update Structure Memory
  → Store Successful Structure

Outputs:
  { patch, explanation, structure_trace }
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .memory import StructureRecord
from .procedural_memory import ProcedureRecord
from .verifier import VerifierOutput
from .survival import LifecycleState


@dataclass
class BugReport:
    repo_path:   str
    description: str
    failing_test: Optional[str]       = None
    error_output: Optional[str]       = None
    affected_files: List[str]         = field(default_factory=list)
    task_type:    str                 = "Unknown"
    family_hint:  Optional[str]       = None   # e.g. "CodeRepair"


@dataclass
class RepairPlan:
    steps:         List[str]
    target_files:  List[str]
    family_id:     int
    expert_id:     int
    retrieved_structures: List[StructureRecord] = field(default_factory=list)
    retrieved_procedures: List[ProcedureRecord] = field(default_factory=list)
    confidence:    float = 0.0


@dataclass
class Patch:
    file_path:   str
    original:    str
    patched:     str
    diff:        str
    explanation: str


@dataclass
class AgentTrace:
    bug_report:       BugReport
    repair_plan:      RepairPlan
    patches:          List[Patch]
    test_results:     Dict[str, bool]
    verifier_output:  Optional[Dict]
    structure_ids:    List[str]
    procedure_id:     Optional[str]
    success:          bool
    elapsed_sec:      float
    iteration:        int


class RepositoryRepairAgent:
    """
    Stateful agent that uses the TAC-SM model to autonomously repair repositories.

    External callables (injected at construction):
      read_file(path) → str
      write_file(path, content) → None
      run_tests(repo_path, test_file=None) → dict {test_name: bool}
      compute_diff(original, patched) → str
      encode_text(text) → torch.Tensor  (embedding_dim,)
    """

    def __init__(
        self,
        model,                        # TACSM instance
        read_file:    Callable,
        write_file:   Callable,
        run_tests:    Callable,
        compute_diff: Callable,
        encode_text:  Callable,
        max_iterations: int = 3,
        verbose: bool = True,
    ):
        self.model          = model
        self.read_file      = read_file
        self.write_file     = write_file
        self.run_tests      = run_tests
        self.compute_diff   = compute_diff
        self.encode_text    = encode_text
        self.max_iterations = max_iterations
        self.verbose        = verbose

    # ── Main Agent Loop ───────────────────────────────────────────────────────

    def repair(self, bug: BugReport) -> AgentTrace:
        """
        Run the full agent loop for a single bug report.
        Returns an AgentTrace with all intermediate artefacts.
        """
        t0        = time.time()
        patches   = []
        trace_ids = []
        proc_id   = None
        success   = False
        plan      = None

        for iteration in range(self.max_iterations):
            self._log(f"\n=== Iteration {iteration + 1} ===")

            # 1. Read repository context
            repo_context = self._read_context(bug)
            self._log(f"Read {len(bug.affected_files)} affected files")

            # 2. Build task embedding
            task_text    = self._build_task_text(bug, repo_context)
            task_emb     = self.encode_text(task_text)              # (emb_dim,)

            # 3. Retrieve structures + procedures
            family_name  = bug.family_hint or "CodeRepair"
            retrieved_s  = self.model.struct_memory.retrieve(task_emb, top_k=8)
            retrieved_p  = self.model.proc_memory.retrieve(task_emb, family=family_name, top_k=4)
            self._log(f"Retrieved {len(retrieved_s)} structures, {len(retrieved_p)} procedures")

            # 4. Build repair plan using model
            plan = self._plan_repair(bug, task_emb, retrieved_s, retrieved_p)
            self._log(f"Plan: family={plan.family_id}, expert={plan.expert_id}, confidence={plan.confidence:.2f}")
            self._log(f"Steps: {plan.steps}")

            # 5. Generate patches
            new_patches = self._generate_patches(bug, repo_context, plan)
            patches.extend(new_patches)
            self._log(f"Generated {len(new_patches)} patches")

            # 6. Apply patches
            self._apply_patches(new_patches)

            # 7. Run tests
            test_results = self.run_tests(bug.repo_path, bug.failing_test)
            passed       = sum(1 for v in test_results.values() if v)
            total        = len(test_results)
            success      = (total > 0) and all(test_results.values())
            self._log(f"Tests: {passed}/{total} passed  |  success={success}")

            # 8. Verify outcome using model verifier
            verif_out = self._verify(task_emb, success, test_results)
            verif_dict = verif_out.to_dict() if verif_out is not None else {}

            # 9. Update Structure Memory
            structure_ids = self._update_memory(
                task_emb, plan, success, retrieved_s
            )
            trace_ids.extend(structure_ids)

            # 10. Write procedure if succeeded on first try
            if success and iteration == 0 and plan.steps:
                proc_id = self.model.proc_memory.write(
                    family       = family_name,
                    task_type    = bug.task_type,
                    steps        = plan.steps,
                    success_rate = 1.0 if success else 0.0,
                )
                self._log(f"Stored procedure: {proc_id}")

            if success:
                break

            # Revert patches if failed (to try again)
            if iteration < self.max_iterations - 1:
                self._revert_patches(new_patches)

        elapsed = time.time() - t0
        self._log(f"\nAgent loop done in {elapsed:.1f}s | success={success}")

        return AgentTrace(
            bug_report       = bug,
            repair_plan      = plan or RepairPlan([], [], 0, 0),
            patches          = patches,
            test_results     = test_results if 'test_results' in dir() else {},
            verifier_output  = verif_dict if 'verif_dict' in dir() else {},
            structure_ids    = trace_ids,
            procedure_id     = proc_id,
            success          = success,
            elapsed_sec      = elapsed,
            iteration        = iteration + 1,
        )

    # ── Step Implementations ──────────────────────────────────────────────────

    def _read_context(self, bug: BugReport) -> Dict[str, str]:
        ctx = {}
        for f in bug.affected_files:
            try:
                ctx[f] = self.read_file(f)
            except Exception as e:
                ctx[f] = f"[read error: {e}]"
        return ctx

    def _build_task_text(self, bug: BugReport, ctx: Dict[str, str]) -> str:
        parts = [
            f"BUG: {bug.description}",
            f"TYPE: {bug.task_type}",
        ]
        if bug.error_output:
            parts.append(f"ERROR: {bug.error_output[:500]}")
        for fpath, content in list(ctx.items())[:3]:
            parts.append(f"FILE {fpath}:\n{content[:300]}")
        return "\n".join(parts)

    def _plan_repair(
        self,
        bug:         BugReport,
        task_emb:    torch.Tensor,
        structures:  List[StructureRecord],
        procedures:  List[ProcedureRecord],
    ) -> RepairPlan:
        """
        Build a repair plan. In a full system this calls the model with
        the context. Here we implement the planning interface — actual
        model generation is done in the training / inference pipeline.
        """
        # Determine family from bug hint or most-used retrieved structure
        family_id = 0   # Default: CodeRepair
        expert_id = 0
        confidence = 0.5

        if structures:
            from collections import Counter
            fam_votes = Counter(r.family_id for r in structures[:4])
            family_id = fam_votes.most_common(1)[0][0]
            exp_votes = Counter(r.expert_id for r in structures[:4])
            expert_id = exp_votes.most_common(1)[0][0]
            avg_score = sum(r.overall_score() for r in structures[:4]) / 4
            confidence = min(avg_score + 0.2, 1.0)

        # Use best procedure's steps if available
        steps = []
        if procedures:
            best_proc = max(procedures, key=lambda p: p.success_rate)
            steps = [s.description for s in best_proc.steps]
        else:
            steps = [
                "Analyze error message",
                "Identify affected code region",
                "Generate candidate patch",
                "Validate with tests",
            ]

        return RepairPlan(
            steps                = steps,
            target_files         = bug.affected_files,
            family_id            = family_id,
            expert_id            = expert_id,
            retrieved_structures = structures,
            retrieved_procedures = procedures,
            confidence           = confidence,
        )

    def _generate_patches(
        self,
        bug:      BugReport,
        ctx:      Dict[str, str],
        plan:     RepairPlan,
    ) -> List[Patch]:
        """
        Generates patches. In production, this calls self.model.generate_greedy()
        with the bug context encoded as tokens. Here we return a stub structure
        so the interface is complete and testable.
        """
        patches = []
        for fpath in plan.target_files[:2]:   # Limit to 2 files per iteration
            original = ctx.get(fpath, "")
            if not original:
                continue
            # Stub: in real usage, model generates the patched content
            patched = original   # <- replaced by model output in full pipeline
            diff    = self.compute_diff(original, patched)
            patches.append(Patch(
                file_path   = fpath,
                original    = original,
                patched     = patched,
                diff        = diff,
                explanation = f"Repair attempt for {bug.task_type} via family {plan.family_id}",
            ))
        return patches

    def _apply_patches(self, patches: List[Patch]):
        for p in patches:
            self.write_file(p.file_path, p.patched)

    def _revert_patches(self, patches: List[Patch]):
        for p in patches:
            self.write_file(p.file_path, p.original)

    def _verify(
        self,
        task_emb:    torch.Tensor,
        success:     bool,
        test_results: Dict[str, bool],
    ) -> Optional["VerifierOutput"]:
        """Run verifier head over task embedding."""
        try:
            self.model.eval()
            with torch.no_grad():
                # Pool task embedding into (1, 1, d_model) dummy sequence
                hidden = task_emb.unsqueeze(0).unsqueeze(0)
                if hidden.shape[-1] != self.model.cfg.transformer.d_model:
                    # Project from emb_dim to d_model
                    hidden = F.pad(
                        hidden,
                        (0, self.model.cfg.transformer.d_model - hidden.shape[-1]),
                    )
                return self.model.verifier(hidden)
        except Exception:
            return None

    def _update_memory(
        self,
        task_emb:    torch.Tensor,
        plan:        RepairPlan,
        success:     bool,
        retrieved:   List[StructureRecord],
    ) -> List[str]:
        """Write new structure + update retrieved ones."""
        written = []

        # Write new structure
        sid = self.model.struct_memory.write(
            embedding      = task_emb,
            family_id      = plan.family_id,
            expert_id      = plan.expert_id,
            task_type      = "repair",
            success_score  = 1.0 if success else 0.0,
            survival_score = 1.0,
        )
        if sid:
            self.model.lifecycle.register(sid)
            written.append(sid)

        # Update retrieved structures
        for r in retrieved[:4]:
            success_d  =  0.05 if success else -0.02
            transfer_d =  0.03 if success else 0.0
            survival_d =  0.02 if success else -0.01
            self.model.struct_memory.update(
                r.structure_id,
                success_delta  = success_d,
                transfer_delta = transfer_d,
                survival_delta = survival_d,
            )

        return written

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
