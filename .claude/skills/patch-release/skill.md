---
name: patch-release
description: Cut a PyBaMM patch release (vYY.MM.x for x >= 1) by cherry-picking bug fixes from main onto a release branch, publishing a GitHub release, verifying PyPI, then updating main via a separate PR. Use when the user says "patch release", "cut a patch", "release vYY.MM.x", or asks to backport bug fixes onto a previous minor version.
---

# Patch release

Walks the user through a PyBaMM patch release as defined in `.github/release_workflow.md` ("Creating a patch release"). The skill is interactive — pause and ask for `y` confirmation before each command that mutates anything beyond a local file edit, and **always** before remote-visible or irreversible actions (push, GitHub release creation, branch deletion).

## Inputs to gather up-front

Ask the user (if not already provided):

1. **New patch version** — `YY.MM.x` (e.g. `26.4.2`).
2. **Previous tag** — usually `vYY.MM.{x-1}`. Confirm by running `git tag --sort=-version:refname | head -10`.
3. **Bug-fix commits** to cherry-pick — typically the user pastes the changelog entries from `# [Unreleased]` on `main`. Resolve each entry to its merge-commit SHA via `git log --oneline <prev-tag>..main` so the order matches main's history.

Recap the plan to the user (branch name, prior tag, ordered list of cherry-picks, target version) and ask for confirmation before doing anything.

Use TaskCreate to track these steps:

1. Create release/vYY.MM.x branch from vYY.MM.{x-1}
2. Cherry-pick bug fixes with `-x`
3. Run update_version.py YY.MM.x and commit
4. Push release/vYY.MM.x to origin
5. Create GitHub release vYY.MM.x from release branch
6. Verify `pip install pybamm==YY.MM.x`
7. Update changelog on main via separate PR
8. Delete release branch (after tagging)

Mark each in_progress when starting and completed when done.

---

## Step 1 — Create the release branch

```bash
git checkout -b release/vYY.MM.x vYY.MM.{x-1}
```

Verify HEAD matches the tag with `git log --oneline -3`. **Confirm before running.**

## Step 2 — Cherry-pick the bug fixes

Cherry-pick **in chronological order on `main`** (oldest first) so the resulting branch's history reads naturally. For each commit:

```bash
git cherry-pick -x <sha>
```

The `-x` flag records `(cherry picked from commit <sha>)` in the message — it's mandatory.

**Pause for `y` before each cherry-pick** so the user can react to conflicts one at a time.

### Resolving CHANGELOG conflicts

Most fix commits bring a `## Bug fixes` entry in `CHANGELOG.md`. Conflicts are common because `main`'s `# [Unreleased]` section accumulates entries that the previous tag doesn't have. Resolve as follows:

- **Drop entries that don't belong to this patch** — typically `## Features` entries destined for the next minor release.
- **Keep the bug-fix entry**, ordered so newer cherry-picks land at the bottom of `## Bug fixes` (matching how the entries appear on `main`).
- **Add the PR link if missing** — some original commits land without their `([#NNNN](https://github.com/pybamm-team/PyBaMM/pull/NNNN))` link because it was added in a follow-up "fix changelog" commit that you're intentionally not cherry-picking. In that case, append the link by hand.

After fixing, `git add CHANGELOG.md` then continue without rewriting the message:

```bash
GIT_EDITOR=true git cherry-pick --continue
```

Verify the trailer with `git log -1 --format=%B | tail -1` — it should end with `(cherry picked from commit <sha>)`.

### Sanity check after all cherry-picks

Read `CHANGELOG.md` and confirm every bug-fix entry the user listed is present with the right PR link. If a commit landed cleanly but **didn't** carry a CHANGELOG entry (because its entry lives in a separate "fix changelog" commit on main), add the missing line manually now — leave it staged but uncommitted; it will fold into the version-bump commit in the next step.

## Step 3 — Run update_version.py

```bash
python scripts/update_version.py YY.MM.x
```

Pitfall: the script does `import pybamm; pybamm.root_dir()`, which resolves to the **installed** package path if `pybamm` is on `sys.path` from a venv. If you see `FileNotFoundError: ... CITATION.cff` pointing somewhere outside the repo, re-run with the local source on PYTHONPATH:

```bash
PYTHONPATH=$(pwd)/src python scripts/update_version.py YY.MM.x
```

Verify the diff:
- `CITATION.cff` — `version: "YY.MM.x"`
- `CHANGELOG.md` — new `# [vYY.MM.x](...) - <today>` header inserted after `# [Unreleased]`

Then commit (folding in any manual CHANGELOG fix-ups from Step 2):

```bash
git add CITATION.cff CHANGELOG.md
git commit -m "Update version to YY.MM.x and changelog"
```

**Confirm before committing.**

## Step 4 — Push the release branch

This is the first remote-visible action — confirm explicitly.

```bash
git push -u origin release/vYY.MM.x
```

## Step 5 — Create the GitHub release

This is **irreversible**: it tags the repo and triggers `publish_pypi.yml` to publish to PyPI. Read out the title, target branch, and notes body to the user, then ask for explicit final confirmation.

Notes body should be the new version's changelog section (just the `## Bug fixes` block — without the `# [vYY.MM.x] - <date>` header).

```bash
gh release create vYY.MM.x \
  --target release/vYY.MM.x \
  --title "vYY.MM.x" \
  --notes "$(cat <<'EOF'
## Bug fixes

- <fix 1> ([#NNNN](...))
- <fix 2> ([#NNNN](...))
...
EOF
)"
```

Important: `--target release/vYY.MM.x` is required. Releases must be cut from the release branch, not `main`, so the tag contains only the bug fixes.

After creating, monitor the publish workflow (typically a couple of minutes):

```bash
gh run list --workflow publish_pypi.yml --limit 3
gh run watch <run-id> --exit-status   # use the run ID from the list above
```

## Step 6 — Verify the install

Once the workflow completes, install the published package in a clean venv:

```bash
python3 -m venv /tmp/verify-pybamm-YY-MM-x
/tmp/verify-pybamm-YY-MM-x/bin/pip install --quiet pybamm==YY.MM.x
/tmp/verify-pybamm-YY-MM-x/bin/python -c "import pybamm; print(pybamm.__version__)"
rm -rf /tmp/verify-pybamm-YY-MM-x
```

Expect the version line to print `YY.MM.x`.

## Step 7 — Update the changelog on main via a separate PR

The release workflow doc is explicit: **do not merge `release/vYY.MM.x` back to `main`** — it would create duplicate commits with different hashes. Instead, open a small PR on `main` that mirrors the changelog change.

```bash
git checkout main
git pull --ff-only
git checkout -b update-changelog-vYY.MM.x
```

Edit `CHANGELOG.md`:
- Move the bug-fix entries currently under `# [Unreleased]` into a new `# [vYY.MM.x](...) - <release-date>` section.
- Leave any unrelated Unreleased content (e.g. `## Features` for the next minor) in place.

Edit `CITATION.cff` to bump the version to `YY.MM.x`.

Then:

```bash
git add CHANGELOG.md CITATION.cff
git commit -m "Update changelog and citation for vYY.MM.x"
git push -u origin update-changelog-vYY.MM.x
gh pr create --base main --head update-changelog-vYY.MM.x \
  --title "Release vYY.MM.x changelog" \
  --body "<short description noting the release was cut from release/vYY.MM.x and main is being updated separately per release_workflow.md>"
```

**Confirm before pushing and again before opening the PR.**

## Step 8 — Delete the release branch (optional, after tagging)

Per the workflow doc, the release branch can be deleted once the tag exists. The `vYY.MM.x` tag persists, so the release is unaffected.

```bash
git checkout main
git branch -d release/vYY.MM.x
git push origin --delete release/vYY.MM.x
```

**Confirm before deleting the remote branch.**

## Conda-forge

Conda-forge auto-triggers from the PyPI release. Only intervene if API/console-script/entry-point/Python-version/metadata changes need to be reflected in `meta.yaml` — see [`.github/release_workflow.md`](../../.github/release_workflow.md) for the manual feedstock update procedure.

## Final report

Summarize for the user:
- Release URL (`https://github.com/pybamm-team/PyBaMM/releases/tag/vYY.MM.x`)
- Verified install command + observed version
- Changelog-on-main PR URL
- Whether the release branch was deleted
