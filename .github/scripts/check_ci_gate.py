"""Assert every job in test_on_push.yml is either merge-gated or explicitly advisory.

`ci_gate` is the sole required status check on `main`, so a job missing from its
`needs` list runs but cannot block a merge. That is a legitimate choice for flaky
platforms, but it must be deliberate: give such jobs the ADVISORY_SUFFIX name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parents[1] / "workflows" / "test_on_push.yml"
GATE = "ci_gate"
# Naming convention rather than a hand-kept list, so the job name cannot disagree
# with the gating it gets.
ADVISORY_SUFFIX = "_advisory"


def main() -> int:
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    if GATE not in jobs:
        print(f"{WORKFLOW}: no `{GATE}` job found", file=sys.stderr)
        return 1

    gated = set(jobs[GATE]["needs"])
    candidates = set(jobs) - {GATE}
    advisory = {job for job in candidates if job.endswith(ADVISORY_SUFFIX)}

    errors = []
    if unknown := gated - candidates:
        errors.append(f"{GATE}.needs references non-existent job(s): {sorted(unknown)}")
    if both := gated & advisory:
        errors.append(
            f"job(s) named *{ADVISORY_SUFFIX} but listed in {GATE}.needs: {sorted(both)}"
        )
    if ungated := candidates - gated - advisory:
        errors.append(
            f"job(s) neither in {GATE}.needs nor named *{ADVISORY_SUFFIX}: {sorted(ungated)}\n"
            f"  add them to {GATE}.needs to gate merges on them, or rename them with "
            f"the {ADVISORY_SUFFIX} suffix if they should not block a merge"
        )

    for error in errors:
        print(f"{WORKFLOW}: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
