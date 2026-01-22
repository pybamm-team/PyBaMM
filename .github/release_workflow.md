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

1. Ensure all bug fixes are merged to `main` first via normal PRs.

2. Create a new branch `release/vYY.MM.x` from the `vYY.MM.{x-1}` tag.

3. Cherry-pick the bug fixes to `release/vYY.MM.x` using the `-x` flag to record the original commit hash:
   ```bash
   git cherry-pick -x <commit-sha-from-main>
   ```

4. Run `scripts/update_version.py` to update `CITATION.cff` and `CHANGELOG.md`, then commit the changes.

5. Create a new GitHub release with the tag `vYY.MM.x` from the `release/vYY.MM.x` branch (not `main`) and a description copied from `CHANGELOG.md`. This ensures the release contains only the bugfixes, not all changes on `main`. This will automatically trigger `publish_pypi.yml` and create a _release_ on PyPI.

6. Verify the release installs correctly: `pip install pybamm==YY.MM.x`

7. Update the changelog on `main` separately. **Do not merge the release branch back to `main`** as this creates duplicate commits with different hashes. Instead:
   ```bash
   git checkout main
   git checkout -b update-changelog-vYY.MM.x
   ```
   Edit `CHANGELOG.md` to add the `vYY.MM.x` release section (moving entries from Unreleased) and update `CITATION.cff` with the new version. Create a PR to `main` with these changes.

8. The release branch can be deleted after tagging since it is no longer needed.

## Conda-forge

- The conda-forge release workflow will automatically be triggered following a stable PyPI release.
- If changes are made to the API, console scripts, entry points, optional dependencies, supported Python versions, or core project metadata, update the `meta.yaml` file in the [pybamm-feedstock][PYBAMM_FEED] repository by following the [conda-forge documentation][FEED_GUIDE] and re-rendering the recipe.
- Updates should be carried out directly by pushing changes to the automated PR created by the conda-forge-bot. A manual PR can also be created if needed. Manual PRs **must** bump the build number in `meta.yaml` and **must** be from a personal fork of the repository.

[PYBAMM_FEED]: https://github.com/conda-forge/pybamm-feedstock
[FEED_GUIDE]: https://conda-forge.org/docs/maintainer/updating_pkgs.html#updating-the-feedstock-repository
