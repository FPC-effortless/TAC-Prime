"""
TAC-PSM-006B: Patch Applier
=============================

Applies a procedure-derived patch to a fixture's file set and returns the
patched file dict that PytestVerifier can materialise in a temp directory.

Patch format (expected_patch in Fixture):
  {
    "filename": {
      "old": "<exact string to replace>",
      "new": "<replacement string>"
    },
    ...
  }

Special cases:
  - "old" == "" → file is created from scratch (new file)
  - "new" == "" → file content is deleted (empty file left behind)
  - filename not in source files → creates a new file

Patch generation strategy:
  TAC selects a procedure from procedural memory.  The procedure's family
  determines which patch template to use.  If the family matches the fixture,
  the oracle patch is applied (correct repair).  If the family does not match,
  a wrong-family patch is attempted (incorrect repair → pytest still fails).

Failure classes raised here:
  - patch_wrong_file      — patch targets a file not present in all_files
  - correct_procedure_wrong_patch — oracle file targeted but replacement fails
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PatchResult:
    """
    Outcome of applying a patch to a file set.

    Attributes
    ----------
    success            : True if every patch operation applied cleanly
    patched_files      : full {filename: content} after patching
    files_modified     : list of filenames actually changed
    files_created      : list of filenames newly created
    patch_errors       : list of (filename, reason) for failed ops
    failure_class      : PSM-006B failure class if something went wrong
    """
    success:        bool
    patched_files:  Dict[str, str]
    files_modified: List[str]        = field(default_factory=list)
    files_created:  List[str]        = field(default_factory=list)
    patch_errors:   List[tuple]      = field(default_factory=list)
    failure_class:  Optional[str]    = None

    def to_dict(self) -> dict:
        return {
            "success":        self.success,
            "files_modified": self.files_modified,
            "files_created":  self.files_created,
            "patch_errors":   self.patch_errors,
            "failure_class":  self.failure_class,
        }


class PatchApplier:
    """
    Applies a patch dict to a file set, returning the modified file dict.

    Usage
    -----
    applier = PatchApplier()
    result  = applier.apply(all_files, expected_patch)
    if result.success:
        verifier.run(result.patched_files, verification_command)
    """

    def apply(
        self,
        all_files: Dict[str, str],
        patch:     Dict[str, Dict[str, str]],
    ) -> PatchResult:
        """
        Apply `patch` to `all_files`.

        Parameters
        ----------
        all_files : {filename: content} — current state of all fixture files
        patch     : {filename: {"old": str, "new": str}} — changes to apply

        Returns
        -------
        PatchResult with updated file dict
        """
        patched      = dict(all_files)   # shallow copy — we replace values
        modified     = []
        created      = []
        errors       = []
        failure_class: Optional[str] = None

        if not patch:
            # No-op patch: return unmodified files (used for families with
            # no patch needed — fixture already passes with the source files
            # provided, or the test itself is already self-consistent)
            return PatchResult(
                success       = True,
                patched_files = patched,
                files_modified = [],
                files_created  = [],
                patch_errors   = [],
            )

        for filename, op in patch.items():
            old_str = op.get("old", "")
            new_str = op.get("new", "")

            if filename not in patched:
                if old_str == "":
                    # Create new file
                    patched[filename] = new_str
                    created.append(filename)
                else:
                    errors.append((filename, "file_not_found"))
                    failure_class = "patch_wrong_file"
                continue

            current = patched[filename]

            if old_str == "":
                # Append to file (or replace entire content if new == "")
                patched[filename] = current + new_str
                modified.append(filename)
            elif old_str in current:
                patched[filename] = current.replace(old_str, new_str, 1)
                modified.append(filename)
            else:
                errors.append((filename, f"old_string_not_found: {old_str[:60]!r}"))
                failure_class = "correct_procedure_wrong_patch"

        success = len(errors) == 0
        return PatchResult(
            success       = success,
            patched_files = patched,
            files_modified = modified,
            files_created  = created,
            patch_errors   = errors,
            failure_class  = failure_class if not success else None,
        )

    def apply_wrong_family_patch(
        self,
        all_files:      Dict[str, str],
        wrong_family:   str,
    ) -> PatchResult:
        """
        Apply a stub incorrect patch for `wrong_family`.

        Used by random-procedure and wrong-procedure baselines to simulate
        what happens when the agent retrieves a procedure from the wrong family
        and applies it.  The patch does not fix the actual bug, so pytest will
        still fail.  We make a syntactically valid but semantically wrong change
        to demonstrate that wrong procedures cause active harm.

        The stub adds a clearly wrong comment to a source file (or creates a
        dummy file), leaving the actual bug unfixed.
        """
        patched = dict(all_files)
        # Add a clearly-wrong stub change: append a comment that does nothing
        stub_content = f"# TAC wrong-family patch stub: procedure from {wrong_family}\n"

        # Find the first source file to annotate, or create a stub
        target = None
        for fname in patched:
            if fname.endswith(".py") and "test_" not in fname:
                target = fname
                break

        if target:
            patched[target] = patched[target] + stub_content
            return PatchResult(
                success        = True,   # patch applied, but wrong
                patched_files  = patched,
                files_modified = [target],
            )
        else:
            # No non-test .py file: create a stub
            patched["_wrong_patch_stub.py"] = stub_content
            return PatchResult(
                success       = True,
                patched_files = patched,
                files_created = ["_wrong_patch_stub.py"],
            )

    def apply_structure_only_patch(
        self,
        all_files: Dict[str, str],
        patch:     Dict[str, Dict[str, str]],
    ) -> PatchResult:
        """
        Apply only the structural part of the patch (file path and op type),
        but replace the 'new' string with a no-op stub.

        Used by the structure-memory-only baseline, which knows the correct
        file to patch but generates wrong content.
        """
        structure_patch = {
            fname: {"old": op["old"], "new": "# structure-only stub\n"}
            for fname, op in patch.items()
        }
        return self.apply(all_files, structure_patch)
