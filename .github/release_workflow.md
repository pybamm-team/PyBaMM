# Release workflow

This file contains the workflow required to make a `PyBaMM` release on
GitHub, PyPI, and conda-forge by the maintainers.

## Creating a release

1. Checkout the latest develop branch, then create a local tag `vYY.MM.0`
   with the new version. The year and month are taken from the date of the
   release. The final number represents the bug fix version, which is zero for
   a new release.

2. Run `scripts/update_version.py` to update the following files:

   - `CITATION.cff`
   - `CHANGELOG.md`

   These changes should be pushed to a release branch `vYY.MM.0`
   and a PR from `vYY.MM.0` to `main` needs to be created.

3. After the release PR has been merged, create a new GitHub _release_ with the
   tag `vYY.MM.0` from the `vYY.MM` branch and a description copied from
   `CHANGELOG.md`.

4. This release will automatically trigger `publish_pypi.yml` and create a
   _release_ on PyPI.

## Bug fix releases

If a new bugfix release is required after the release of `vYY.MM.{x-1}` -

1. Create a new branch for the `vYY.MM.x` release using the `vYY.MM.{x-1}` tag.

2. Cherry-pick the bug fixes to `vYY.MM.x` branch once the fix is
   merged into `develop`. The CHANGELOG entry for such fixes should go under the
   `YY.MM.x` heading in `CHANGELOG.md`

3. Run `scripts/update_version.py` manually to update:

   - `CITATION.cff`
   - `CHANGELOG.md`

   Commit the changes to your release branch.

4. Create a PR for the release and configure it to merge into the `main` branch.

5. Create a new GitHub release with the same tag (`YY.MM.x`) from the `main`
   branch and a description copied from `CHANGELOG.md`. This release will
   automatically trigger `publish_pypi.yml` and create a _release_ on PyPI.

## Other checks

Some other essential things to check throughout the release process -

- Update jax and jaxlib to the latest version in `pybamm.util` and
  `pyproject.toml`, fixing any bugs that arise.
- If changes are made to the API, console scripts, entry points, new optional
  dependencies are added, support for major Python versions is dropped or
  added, or core project information and metadata are modified at the time
  of the release, make sure to update the `meta.yaml` file in the `recipe/`
  folder of the [pybamm-feedstock][PYBAMM_FEED] repository accordingly by
  following the instructions in the [conda-forge documentation][FEED_GUIDE] and
  re-rendering the recipe.
- The conda-forge release workflow will automatically be triggered following
  a stable PyPI release, and the aforementioned updates should be carried
  out directly in the main repository by pushing changes to the automated PR
  created by the conda-forge-bot. A manual PR can also be created if the
  updates are not included in the automated PR for some reason. This manual
  PR **must** bump the build number in `meta.yaml` and **must** be from a
  personal fork of the repository.

[PYBAMM_FEED]: https://github.com/conda-forge/pybamm-feedstock
[FEED_GUIDE]: https://conda-forge.org/docs/maintainer/updating_pkgs.html#updating-the-feedstock-repository
