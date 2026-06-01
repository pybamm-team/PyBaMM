# PyBaMM Release Policy

This document is the source of truth for how PyBaMM is versioned, when releases happen, what counts as a breaking change, and how breaking changes and deprecations are communicated. For a quick summary aimed at users, see the [Versioning section of the README](https://github.com/pybamm-team/PyBaMM/blob/main/README.md#versioning).

## At a glance

- **Versioning:** [CalVer](https://calver.org/) in the form `YY.MM.N.P` — year, month, feature release index within that month, patch level.
- **Cadence:** We release when there's something worth releasing, with no fixed schedule. Breaking changes target the first feature release of the month (`YY.MM.0.0`).
- **Breaking changes:** Documented under `## Breaking changes` in [`CHANGELOG.md`](https://github.com/pybamm-team/PyBaMM/blob/main/CHANGELOG.md). Public APIs that are removed or renamed ship a `DeprecationWarning` for at least two prior feature releases first (see the Deprecation policy section for the documented exceptions).
- **Pre-announcement:** The `# [Unreleased]` section at the top of `CHANGELOG.md` on `main` is the canonical preview of what's about to ship.

## Version scheme

PyBaMM versions take the form `YY.MM.N.P`:

| Component | Meaning | Example |
| --- | --- | --- |
| `YY` | Two-digit year | `27` |
| `MM` | Month of release (1–12, no leading zero) | `1`, `11` |
| `N` | Feature release within that month, **0-indexed** — `0` for the first | `0`, `1` |
| `P` | Patch level — `0` for the feature release itself; `1`, `2`, … for patches off it | `0`, `1`, `2` |

**Worked examples:**

- `27.1.0.0` — first feature release in January 2027.
- `27.1.0.1` — first patch off `27.1.0.0`.
- `27.1.1.0` — second feature release in January 2027 (only when two feature releases land in the same calendar month).
- `27.1.1.1` — first patch off `27.1.1.0`.

**Cutover.** The first feature release tagged after the policy lands is the first to use `YY.MM.N.P`. Earlier tags (`v26.x.N`) remain in their original form; we do not retroactively retag.

## Release cadence

- We release when there's a meaningful body of work to ship. No fixed monthly or quarterly schedule.
- Patch releases are cut from the corresponding feature release branch as fixes warrant.
- Breaking changes target the **first feature release of the month** (`YY.MM.0.0`). Subsequent feature releases in the same month (`YY.MM.1.0`, `YY.MM.2.0`, …) should not contain breaking changes unless absolutely required. Exceptions must be justified in the PR description and in the `## Breaking changes` changelog entry, alongside the standard deprecation note.

## What counts as a breaking change

A change is **breaking** if it does any of the following:

- Removes or renames a public Python API.
- Changes the signature, defaults, or return type of a public function in a way existing user code would notice.
- Changes the **default** model, solver, or option such that an unchanged user script produces materially different numerical output.
- Drops support for a previously-supported Python version, OS, or core dependency major version.
- Changes the on-disk format of saved models or solutions without a backward-compatible reader.

A change is **not** breaking if it:

- Adds new optional kwargs to existing public functions.
- Adds new submodels, solvers, or options that opt-in via explicit configuration.
- Improves performance.
- Fixes a bug (corrects behaviour that was previously wrong).
- Changes numerical output of a *non-default* model or solver. Algorithmic improvements under the same name and same defaults are not breaking, even if outputs shift.

## "Public API" defined

**Public API = anything documented at [docs.pybamm.org](https://docs.pybamm.org).**

If a class, function, or method appears in the rendered API reference, it is in PyBaMM's public contract. Anything else — internal modules, leading-underscore names, undocumented helpers — is private and may change between any two releases without a `## Breaking changes` entry.

The rendered docs build is the auditable source of truth. A PR reviewer can answer "is this public?" by checking whether the symbol is reachable from the API reference index.

## Deprecation policy

Removing or renaming a public API must ship a `DeprecationWarning` in at least **two prior feature releases** before the removal release.

- The deprecation lands with a `## Deprecated` entry in the changelog.
- The eventual removal lands with a `## Breaking changes` entry that references the original deprecation.

**Exceptions.** The two-release floor may be skipped only when one of the following applies. Each exception must be justified in the PR description *and* in the `## Breaking changes` changelog entry:

- (a) The API is fundamentally unsafe or broken — keeping it around for two releases would cause harm.
- (b) The API was never advertised in the docs — it was reachable but not part of the public contract.
- (c) Deprecation is technically impossible — for example, removing a required positional argument, where there is no behaviour to warn from.

## Pre-announcement

The `# [Unreleased]` section at the top of `CHANGELOG.md` on `main` is the canonical pre-announcement channel. Downstream package maintainers, users, and contributors can read that section at any time to see what is about to ship.

The following are **not** required by policy but may be chosen by maintainers for exceptional breaks (e.g. removing a long-standing core class, dropping a supported Python version):

- Release candidate (RC) tags on PyPI.
- Pinned GitHub tracking issues.
- `[BREAKING]` prefix on GitHub release titles.
- Announcements on Slack or pybamm.org.

## Changelog conventions

Within each release block (and within `# [Unreleased]`), sections appear in this order, omitting any that are empty:

```
## Breaking changes
## Deprecated
## Features
## Bug fixes
```

The ordering is **scariest first**: anyone scanning the changelog sees breaks and deprecations before features.

**Entry format.** Each entry is a single bullet ending in a PR link:

```markdown
- Short imperative description of the change. ([#1234](https://github.com/pybamm-team/PyBaMM/pull/1234))
```

Use full GitHub URLs (not bare `#1234`) so the rendered markdown on docs.pybamm.org links correctly. `## Breaking changes` and `## Deprecated` entries must include a one-line migration note describing how users adapt.

**Historical entries are not rewritten.** The new section ordering and the new `## Deprecated` section apply from the next release block forward. Existing release blocks keep whatever they currently have.

## Release-manager checklist

The version string in the built distribution is supplied by `hatch-vcs` from the VCS tag (`pyproject.toml` has `dynamic = ["version"]` with `version.source = "vcs"`, writing to `src/pybamm/_version.py`). The `scripts/update_version.py` helper updates `CITATION.cff` and prepends a new dated heading to `CHANGELOG.md`; its version-string handling is format-agnostic and accepts the four-component form unchanged.

### Cutting a feature release

A feature release is `YY.MM.N.0` — the patch component is `0`. The first feature release in a given calendar month uses `N=0`; subsequent feature releases in the same month use `N=1`, `N=2`, etc.

1. Confirm `# [Unreleased]` in `CHANGELOG.md` accurately reflects what's about to ship — every breaking change, deprecation, feature, and bug fix has an entry with a PR link, and entries are grouped under the four sections (`## Breaking changes`, `## Deprecated`, `## Features`, `## Bug fixes`) in that order.
2. Create and check out a release branch from `main`: `git checkout -b release/vYY.MM.N.0`.
3. Run `uv run python scripts/update_version.py YY.MM.N.0` to update `CITATION.cff` and prepend a dated heading to `CHANGELOG.md`.
4. Push the branch and open a PR to `main`. Ensure CI passes, then merge.
5. From `main` at the merge commit, create a GitHub _release_ with the tag `vYY.MM.N.0`. Copy the relevant `CHANGELOG.md` block into the release description. This triggers `publish_pypi.yml` and creates the PyPI release automatically.
6. Verify the release installs cleanly: `pip install pybamm==YY.MM.N.0`.

### Cutting a patch release

A patch release is `YY.MM.N.P` where `P >= 1`. Patches are cut from the previous tag in the same feature line so the release contains only the bug fixes, not unrelated changes that have landed on `main` since the feature release.

1. Ensure all bug fixes are merged to `main` first via normal PRs.
2. Create a new branch from the previous tag in the same feature line: `git checkout -b release/vYY.MM.N.P vYY.MM.N.{P-1}` (e.g. `release/v27.1.0.1` from `v27.1.0.0`).
3. Cherry-pick the bug fixes onto the new branch, recording the original SHA with `-x`:
   ```bash
   git cherry-pick -x <commit-sha-from-main>
   ```
4. Run `uv run python scripts/update_version.py YY.MM.N.P` to update `CITATION.cff` and prepend a dated heading to `CHANGELOG.md`. Commit the result on the release branch.
5. Create a GitHub _release_ with the tag `vYY.MM.N.P` from the `release/vYY.MM.N.P` branch (NOT from `main`). Copy the relevant `CHANGELOG.md` block into the release description. This triggers `publish_pypi.yml`.
6. Verify the release installs cleanly: `pip install pybamm==YY.MM.N.P`.
7. Update the changelog on `main` separately. **Do not merge the release branch back to `main`** — that would duplicate commits with new hashes. Instead:
   ```bash
   git checkout main
   git checkout -b update-changelog-vYY.MM.N.P
   ```
   Edit `CHANGELOG.md` to add the new dated `vYY.MM.N.P` block (moving the entries out of `# [Unreleased]`), and update `CITATION.cff`. Open a PR to `main`.
8. Delete the release branch after tagging — it is no longer needed.

### Conda-forge

The conda-forge release flow is triggered automatically after a stable PyPI release: the conda-forge bot opens a PR against [pybamm-feedstock](https://github.com/conda-forge/pybamm-feedstock), which maintainers review and approve.

When a release touches the API, console scripts, entry points, optional dependencies, supported Python versions, or core project metadata, update `meta.yaml` in [pybamm-feedstock](https://github.com/conda-forge/pybamm-feedstock) following [the conda-forge maintainer docs](https://conda-forge.org/docs/maintainer/updating_pkgs.html#updating-the-feedstock-repository) and re-render the recipe. Push updates directly to the bot's automated PR where possible. Manual PRs must bump the `build` number in `meta.yaml` and be opened from a personal fork.
