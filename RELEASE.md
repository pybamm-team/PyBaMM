# PyBaMM Release Policy

This document is the source of truth for how PyBaMM is versioned, when releases happen, what counts as a breaking change, and how breaking changes and deprecations are communicated. For a quick summary aimed at users, see the [Versioning section of the README](https://github.com/pybamm-team/PyBaMM/blob/main/README.md#versioning).

## At a glance

- **Versioning:** [CalVer](https://calver.org/) in the form `YY.MM.N.P` — year, month, feature release index within that month, patch level.
- **Cadence:** We release when there's something worth releasing, with no fixed schedule. Breaking changes target the first feature release of the month (`YY.MM.0.0`).
- **Breaking changes:** Documented under `## Breaking changes` in [`CHANGELOG.md`](https://github.com/pybamm-team/PyBaMM/blob/main/CHANGELOG.md). Public APIs that are removed or renamed ship a `DeprecationWarning` for at least two prior feature releases first (see the Deprecation policy section for the documented exceptions).
- **Pre-announcement:** The changelog fragments in `changelog.d/` on `main` are the canonical preview of what's about to ship.

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

## Monorepo: independent package releases

PyBaMM, `pybammsolvers`, and `pybamm-model-zoo` live in one repository (a UV
workspace under `packages/`) but release **independently to PyPI**,
discriminated by tag namespace. See
`docs/superpowers/specs/2026-06-11-pybamm-monorepo-design.md` for the full design.

- Each package's publish workflow fires only for its own tag namespace (the
  pre-monorepo `github.repository` guard no longer distinguishes them).
- `pybammsolvers` releases use the `pybammsolvers-v*` namespace.
- `pybamm-model-zoo` releases use the `pybamm-model-zoo-v*` namespace.
- **Legacy solver tags** (`v0.8.0`, `v0.8.1`, `v0.8.2`) carried over from the
  standalone `pybammsolvers` repo are **not** part of PyBaMM's release history and
  must not drive PyBaMM's computed version. They are either dropped or re-tagged
  under `pybammsolvers-v*` during the migration.
- The migration was integrated with a **merge commit (not a squash)** so the full
  commit history of both packages is preserved; PyBaMM's existing commit SHAs are
  unchanged (no history rewrite).

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

The changelog fragments in `changelog.d/` on `main` are the canonical pre-announcement channel. Downstream package maintainers, users, and contributors can render them at any time, with `uvx towncrier build --draft --version Unreleased`, to see what is about to ship.

The following are **not** required by policy but may be chosen by maintainers for exceptional breaks (e.g. removing a long-standing core class, dropping a supported Python version):

- Release candidate (RC) tags on PyPI.
- Pinned GitHub tracking issues.
- `[BREAKING]` prefix on GitHub release titles.
- Announcements on Slack or pybamm.org.

## Changelog conventions

Unreleased entries are fragments in `changelog.d/`, one file per entry named `<PR number>.<type>.md` (see `changelog.d/README.md`). At release, towncrier builds them into a release block whose sections appear in this order, omitting any that are empty:

```
## Breaking changes
## Deprecated
## Features
## Bug fixes
## Optimizations
```

The ordering is **scariest first**: anyone scanning the changelog sees breaks and deprecations before features.

**Entry format.** A fragment holds the entry text only; towncrier renders it as a single bullet ending in a full PR link taken from the file name:

```markdown
- Short imperative description of the change. ([#1234](https://github.com/pybamm-team/PyBaMM/pull/1234))
```

`breaking` and `deprecated` entries must include a one-line migration note describing how users adapt.

**Historical entries are not rewritten.** The new section ordering and the new `## Deprecated` section apply from the next release block forward. Existing release blocks keep whatever they currently have.

## Release-manager checklist

The version string in the built distribution is supplied by `hatch-vcs` from the VCS tag (`pyproject.toml` has `dynamic = ["version"]` with `version.source = "vcs"`, writing to `src/pybamm/_version.py`). The `scripts/update_version.py` helper updates `CITATION.cff` and runs `towncrier build`, which collects the `changelog.d/` fragments into a new dated block of `CHANGELOG.md` and deletes them (it needs `uv` on the `PATH`); its version-string handling is format-agnostic and accepts the four-component form unchanged.

### Cutting a feature release

A feature release is `YY.MM.N.0` — the patch component is `0`. The first feature release in a given calendar month uses `N=0`; subsequent feature releases in the same month use `N=1`, `N=2`, etc.

1. Confirm the fragments in `changelog.d/` accurately reflect what's about to ship — every breaking change, deprecation, feature, and bug fix has a fragment of the right type. Preview the block with `uvx towncrier build --draft --version YY.MM.N.0`.
2. Create and check out a release branch from `main`: `git checkout -b release/vYY.MM.N.0`.
3. Run `uv run python scripts/update_version.py YY.MM.N.0` to update `CITATION.cff` and build the fragments into a dated block of `CHANGELOG.md`.
4. Push the branch and open a PR to `main`. Ensure CI passes, then merge.
5. From `main` at the merge commit, create a GitHub _release_ with the tag `pybamm-vYY.MM.N.0`. Copy the relevant `CHANGELOG.md` block into the release description. This triggers `publish_pypi.yml` and creates the PyPI release automatically. (Monorepo: PyBaMM release tags use the `pybamm-v` prefix so they are distinct from `pybammsolvers-v*`; see "Monorepo: independent package releases" above.)
6. Verify the release installs cleanly: `pip install pybamm==YY.MM.N.0`.

### Cutting a patch release

A patch release is `YY.MM.N.P` where `P >= 1`. Patches are cut from the previous tag in the same feature line so the release contains only the bug fixes, not unrelated changes that have landed on `main` since the feature release.

1. Ensure all bug fixes are merged to `main` first via normal PRs.
2. Create a new branch from the previous tag in the same feature line: `git checkout -b release/vYY.MM.N.P pybamm-vYY.MM.N.{P-1}` (e.g. `release/v27.1.0.1` from `pybamm-v27.1.0.0`).
3. Cherry-pick the bug fixes onto the new branch, recording the original SHA with `-x`:
   ```bash
   git cherry-pick -x <commit-sha-from-main>
   ```
4. Check that `changelog.d/` holds only the cherry-picked fixes' fragments: `git diff --name-only --diff-filter=A pybamm-vYY.MM.N.{P-1} HEAD -- changelog.d` lists them, and any other fragment left over from `main` at the previous tag must be `git rm`ed in a separate commit. Then run `uv run python scripts/update_version.py YY.MM.N.P` to update `CITATION.cff` and build those fragments into a dated block of `CHANGELOG.md`. Commit the result on the release branch.
5. Create a GitHub _release_ with the tag `pybamm-vYY.MM.N.P` from the `release/vYY.MM.N.P` branch (NOT from `main`). Copy the relevant `CHANGELOG.md` block into the release description. This triggers `publish_pypi.yml`.
6. Verify the release installs cleanly: `pip install pybamm==YY.MM.N.P`.
7. Update the changelog on `main` separately. **Do not merge the release branch back to `main`** — that would duplicate commits with new hashes. Instead:
   ```bash
   git checkout main
   git checkout -b update-changelog-vYY.MM.N.P
   ```
   Copy the new dated `vYY.MM.N.P` block from the release branch's `CHANGELOG.md` into `CHANGELOG.md`, `git rm` the fragments the release commit consumed (`git show --name-only --diff-filter=D --format= pybamm-vYY.MM.N.P -- changelog.d`), and update `CITATION.cff`. Open a PR to `main`.
8. Delete the release branch after tagging — it is no longer needed.

### Cutting a `pybammsolvers` release

`pybammsolvers` releases independently of PyBaMM. **`packages/pybammsolvers/src/pybammsolvers/version.py` is the single source of truth for its published version**, *not* the release tag — the `pybammsolvers-v*` tag namespace only routes the workflow. The version is a static string (dynamic VCS versioning is avoided because it would force a build during dependency resolution, which build-tool-less resolvers like Dependabot cannot do); the `sync-pybammsolvers-version` pre-commit hook mirrors `version.py` into the two static copies — `pyproject.toml` (`[project] version`, which stamps the built distribution) and `vcpkg.json` (`version-string`) — so a bump touches only one file. Keep the tag and `version.py` in lockstep, or the wrong version ships.

1. Bump `__version__` in `packages/pybammsolvers/src/pybammsolvers/version.py` (the pre-commit hook mirrors it into `pyproject.toml` and `vcpkg.json`) and record the change in a `changelog.d/` fragment. Open a PR to `main`, ensure CI passes, then merge.
2. From `main` at the merge commit, create a GitHub _release_ with the tag `pybammsolvers-vX.Y.Z`, where `X.Y.Z` **exactly matches** the new `version.py` value. This triggers `release_solvers.yml`, which builds wheels + sdist and publishes to PyPI. The `check_version` job in that workflow fails the release if the tag, `version.py`, or the `pyproject.toml`/`vcpkg.json` mirrors disagree; PyPI separately rejects a re-upload of an already-published version.
3. Verify the release installs cleanly: `pip install pybammsolvers==X.Y.Z`.

### Cutting a `pybamm-model-zoo` release

`pybamm-model-zoo` releases independently of PyBaMM and is versioned `X.Y.Z`. **`version` in `packages/pybamm-model-zoo/pyproject.toml` is the single source of truth for its published version**; the `pybamm-model-zoo-v*` tag only routes the workflow. It is a static string for the same reason as `pybammsolvers`'s. The zoo keeps its own changelog, `packages/pybamm-model-zoo/CHANGELOG.md`.

1. In `packages/pybamm-model-zoo/`, bump `version` in `pyproject.toml` and move the `## [Unreleased]` entries of `CHANGELOG.md` under a new `## [X.Y.Z] - YYYY-MM-DD` heading. The zoo's `pybamm>=` floor in the same `pyproject.toml` is every model's floor, raised by the pull request of any model that needs a newer PyBaMM. Open a PR to `main`, ensure CI passes, then merge.
2. From `main` at the merge commit, create a GitHub _release_ with the tag `pybamm-model-zoo-vX.Y.Z`, where `X.Y.Z` **exactly matches** the new `version`. This triggers `release_model_zoo.yml`, which builds the sdist and wheel, installs the wheel with its `zoo-all` extra into two fresh environments, one with the newest PyBaMM and one with the oldest its `pybamm>=` floor admits, and fails the release if the tag does not match `version`, the wheel is missing any committed file under `src/pybamm_model_zoo/`, or any model fails its model contract checks in either environment. Publishing then waits for a maintainer to approve the `pypi` environment.
3. Verify the release installs cleanly: `pip install pybamm-model-zoo==X.Y.Z`.

To rehearse a release without publishing, run `release_model_zoo.yml` from the Actions tab: a manual run builds and checks the distributions but never uploads them.

**First release only.** PyPI has no `pybamm-model-zoo` project until the first upload, so before step 2 a PyPI account that will own the project must add a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) with owner `pybamm-team`, repository `PyBaMM`, workflow `release_model_zoo.yml`, and environment `pypi`. The first publish creates the project; add the other maintainers as PyPI owners afterwards.

### Conda-forge

The conda-forge release flow is triggered automatically after a stable PyPI release: the conda-forge bot opens a PR against [pybamm-feedstock](https://github.com/conda-forge/pybamm-feedstock), which maintainers review and approve.

When a release touches the API, console scripts, entry points, optional dependencies, supported Python versions, or core project metadata, update `meta.yaml` in [pybamm-feedstock](https://github.com/conda-forge/pybamm-feedstock) following [the conda-forge maintainer docs](https://conda-forge.org/docs/maintainer/updating_pkgs/#updating-recipes) and re-render the recipe. Push updates directly to the bot's automated PR where possible. Manual PRs must bump the `build` number in `meta.yaml` and be opened from a personal fork.
