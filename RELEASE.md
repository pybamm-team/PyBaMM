# PyBaMM Release Policy

This document is the source of truth for how PyBaMM is versioned, when releases happen, what counts as a breaking change, and how breaking changes and deprecations are communicated. For a quick summary aimed at users, see the [Versioning section of the README](https://github.com/pybamm-team/PyBaMM/blob/main/README.md#versioning).

## At a glance

- **Versioning:** [CalVer](https://calver.org/) in the form `YY.MM.N.P` — year, month, feature release index within that month, patch level.
- **Cadence:** We release when there's something worth releasing. No fixed schedule.
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
- We try to avoid trickling breaking changes across many small releases — batching is preferred where convenient — but this is guidance only. The policy does not depend on hitting a calendar target.

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

When cutting a release:

1. Confirm `# [Unreleased]` accurately reflects what's about to ship — every breaking change, deprecation, feature, and bug fix has an entry with a PR link.
2. Run `uv run python scripts/update_version.py <version>` (e.g. `uv run python scripts/update_version.py 27.1.0.0`) to update `CITATION.cff` and add a new dated heading to `CHANGELOG.md`. The script's regex is format-agnostic and accepts the four-component form unchanged.
3. Verify section ordering inside the new release block is Breaking → Deprecated → Features → Bug fixes.
4. Tag and push: `git tag v<version> && git push --tags`. The version string is supplied by `hatch-vcs` from the VCS tag (`pyproject.toml` has `dynamic = ["version"]` with `version.source = "vcs"`, writing to `src/pybamm/_version.py`).
5. Create the GitHub release. Mirror the changelog block into the release notes body.
