# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context.

Fixes # (issue)

## Type of change

Add an entry under `# [Unreleased]` in [CHANGELOG.md](https://github.com/pybamm-team/PyBaMM/blob/main/CHANGELOG.md), in one of `## Breaking changes`, `## Deprecated`, `## Features`, or `## Bug fixes` (include PR number; breaking and deprecation entries need a one-line migration note). Internal-only PRs — refactor, docs, CI, tests — can skip this. See [RELEASE.md](https://github.com/pybamm-team/PyBaMM/blob/main/RELEASE.md) for the full policy.

# Important checks:

Please confirm the following before marking the PR as ready for review:
- No style issues: `nox -s pre-commit`
- All tests pass: `nox -s tests`
- The documentation builds: `nox -s doctests`
- Code is commented for hard-to-understand areas
- Tests added that prove fix is effective or that feature works
