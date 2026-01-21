# Release workflow

This file contains the workflow required to make a `PyBaMM` release on
GitHub, PyPI, and conda-forge by the maintainers.

## Creating a major release

1. Create and checkout a new release branch (e.g., `release/vYY.MM.0`) from the `main` branch. The year and month are taken from the date of the release. The final number represents the bug fix version, which is zero for a new major release.

2. Run `scripts/update_version.py` to update `CITATION.cff` and `CHANGELOG.md`, then create a PR from `release/vYY.MM.0` to `main`.

3. Ensure CI passes on the PR, then merge it.

4. Create a new GitHub _release_ with the tag `vYY.MM.0` from the `main` branch and a description copied from `CHANGELOG.md`. This will automatically trigger `publish_pypi.yml` and create a _release_ on PyPI.

5. Verify the release installs correctly: `pip install pybamm==YY.MM.0`

## Creating a patch release

If a new bugfix release is required after the release of `vYY.MM.{x-1}`:

1. Create a new branch `release/vYY.MM.x` from the `vYY.MM.{x-1}` tag.

2. Cherry-pick the bug fixes to `release/vYY.MM.x` once each fix is merged into `main`. Add CHANGELOG entries under the `vYY.MM.x` heading.

3. Run `scripts/update_version.py` to update `CITATION.cff` and `CHANGELOG.md`, then commit the changes.

4. Create a new GitHub release with the tag `vYY.MM.x` from the `release/vYY.MM.x` branch (not `main`) and a description copied from `CHANGELOG.md`. This ensures the release contains only the bugfixes, not all changes on `main`. This will automatically trigger `publish_pypi.yml` and create a _release_ on PyPI.

5. Verify the release installs correctly: `pip install pybamm==YY.MM.x`

6. Create a PR from `release/vYY.MM.x` to `main` to sync the changelog and version updates, then merge it.

## Conda-forge

- The conda-forge release workflow will automatically be triggered following a stable PyPI release.
- If changes are made to the API, console scripts, entry points, optional dependencies, supported Python versions, or core project metadata, update the `meta.yaml` file in the [pybamm-feedstock][PYBAMM_FEED] repository by following the [conda-forge documentation][FEED_GUIDE] and re-rendering the recipe.
- Updates should be carried out directly by pushing changes to the automated PR created by the conda-forge-bot. A manual PR can also be created if needed. Manual PRs **must** bump the build number in `meta.yaml` and **must** be from a personal fork of the repository.

[PYBAMM_FEED]: https://github.com/conda-forge/pybamm-feedstock
[FEED_GUIDE]: https://conda-forge.org/docs/maintainer/updating_pkgs.html#updating-the-feedstock-repository
