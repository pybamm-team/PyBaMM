---
name: release
description: Cut a PyBaMM major release (vYY.MM.0) by branching off main, running update_version.py, opening a release PR, merging it, then publishing the GitHub release. Use when the user says "release", "cut a release", "make the YY.MM release", or "do the major release".
---

# Major release

Walks the user through a PyBaMM major release as defined in `.github/release_workflow.md` ("Creating a major release"). The skill is interactive — pause and ask for `y` confirmation before each command that mutates anything beyond a local file edit, and **always** before remote-visible or irreversible actions (push, PR creation, merge, GitHub release creation).

For patch releases (vYY.MM.x with x >= 1), use the `/patch-release` skill instead.

## Inputs to gather up-front

Ask the user (if not already provided):

1. **New release version** — `YY.MM.0` (year and month from today's date, third digit always `0`).
2. **Confirm the `# [Unreleased]` block in `CHANGELOG.md` is the release notes** — read it and recap to the user. If anything still needs to land before the release is cut, pause and let them merge it first.

Recap the plan (branch name, version, summary of `# [Unreleased]` content) and ask for confirmation before doing anything.

Use TaskCreate to track these steps:

1. Create release/vYY.MM.0 branch from main
2. Run update_version.py YY.MM.0 and commit
3. Push and open PR to main
4. Wait for CI, then merge
5. Create GitHub release vYY.MM.0 from main
6. Verify `pip install pybamm==YY.MM.0`

Mark each in_progress when starting and completed when done.

---

## Step 1 — Create the release branch

```bash
git checkout main
git pull --ff-only
git checkout -b release/vYY.MM.0
```

**Confirm before running.**

## Step 2 — Run update_version.py

```bash
python scripts/update_version.py YY.MM.0
```

Pitfall: the script does `import pybamm; pybamm.root_dir()`, which resolves to the **installed** package path if `pybamm` is on `sys.path` from a venv. If you see `FileNotFoundError: ... CITATION.cff` pointing somewhere outside the repo, re-run with the local source on PYTHONPATH:

```bash
PYTHONPATH=$(pwd)/src python scripts/update_version.py YY.MM.0
```

Verify the diff:
- `CITATION.cff` — `version: "YY.MM.0"`
- `CHANGELOG.md` — new `# [vYY.MM.0](...) - <today>` header inserted after `# [Unreleased]`

Commit:

```bash
git add CITATION.cff CHANGELOG.md
git commit -m "Update version to YY.MM.0 and changelog"
```

**Confirm before committing.**

## Step 3 — Push and open the release PR

This is the first remote-visible action — confirm explicitly.

```bash
git push -u origin release/vYY.MM.0
gh pr create --base main --head release/vYY.MM.0 \
  --title "Release vYY.MM.0 changelog" \
  --body "<short description: this is the release PR for vYY.MM.0; per release_workflow.md, merging this triggers the release tag/publish>"
```

## Step 4 — Wait for CI, then merge

Poll until checks complete:

```bash
gh pr checks <pr-number>
```

When CI is green, **confirm with the user** before merging. Merging is required before tagging — the GitHub release is cut from `main`, not from the release branch.

```bash
gh pr merge <pr-number> --merge   # or --squash, matching the repo's conventions; check recent merges
```

After merge, update local main:

```bash
git checkout main
git pull --ff-only
```

## Step 5 — Create the GitHub release

This is **irreversible**: it tags the repo and triggers `publish_pypi.yml` to publish to PyPI. Read out the title, target branch, and notes body to the user, then ask for explicit final confirmation.

Notes body should be the new version's changelog section copied from `CHANGELOG.md` (everything between the `# [vYY.MM.0]` header and the next `# [...]` header — but without those headers themselves).

```bash
gh release create vYY.MM.0 \
  --target main \
  --title "vYY.MM.0" \
  --notes "$(cat <<'EOF'
## Features

- ...

## Bug fixes

- ...
EOF
)"
```

Important: `--target main` is correct for major releases (unlike patch releases, which target the release branch).

After creating, monitor the publish workflow (typically a couple of minutes):

```bash
gh run list --workflow publish_pypi.yml --limit 3
gh run watch <run-id> --exit-status   # use the run ID from the list above
```

## Step 6 — Verify the install

Once the workflow completes, install the published package in a clean venv:

```bash
python3 -m venv /tmp/verify-pybamm-YY-MM-0
/tmp/verify-pybamm-YY-MM-0/bin/pip install --quiet pybamm==YY.MM.0
/tmp/verify-pybamm-YY-MM-0/bin/python -c "import pybamm; print(pybamm.__version__)"
rm -rf /tmp/verify-pybamm-YY-MM-0
```

Expect the version line to print `YY.MM.0`.

## Conda-forge

Conda-forge auto-triggers from the PyPI release. Only intervene if API/console-script/entry-point/Python-version/metadata changes need to be reflected in `meta.yaml` — see [`.github/release_workflow.md`](../../.github/release_workflow.md) for the manual feedstock update procedure.

## Final report

Summarize for the user:
- Release URL (`https://github.com/pybamm-team/PyBaMM/releases/tag/vYY.MM.0`)
- Merged PR URL
- Verified install command + observed version
