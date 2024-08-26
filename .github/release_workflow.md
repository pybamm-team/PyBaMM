# Release workflow

This file contains the workflow required to make a `PyBaMM` release on
GitHub, PyPI, and conda-forge by the maintainers.

## Initial release (automated)

1. The `update_version.yml` workflow will run on every 1st of January, May
   and September, updating incrementing the version to `vYY.MM.0` by running
   `scripts/update_version.py` in the following files:

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `pyproject.toml`
   - `vcpkg.json`
   - `CHANGELOG.md`

   These changes will be automatically pushed to a new branch `vYY.MM`
   and a PR from `vYY.MM` to `main` will be created.

2. Create a new GitHub _release_ with the tag `vYY.MM.0` from the `vYY.MM`
   branch and a description copied from `CHANGELOG.md`.

3. This release will automatically trigger `publish_pypi.yml` and create a
   _release_ on PyPI.

## Bug fix releases (manual)

If a new release is required after the release of `vYY.MM.{x-1}` -

1. Cherry-pick the bug fix commit to `vYY.MM` branch once the fix is merged
   into `develop`. The CHANGELOG entry for such fixes should go under the
   `YY.MM.x` heading in `CHANGELOG.md`

2. Run `update_version.yml` manually while using `append_to_tag` to specify
   the bugfix number (`.1`, `.2`, ...). This will increment the version to
   `YY.MM.x` by running `scripts/update_version.py` in the following files:

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `pyproject.toml`
   - `vcpkg.json`
   - `CHANGELOG.md`

   These changes will be automatically pushed to the existing `vYY.MM`
   branch and a PR will be created to update version strings in `main`.

3. Create a new GitHub release with the same tag (`YY.MM.x`) from the `vYY.MM`
   branch and a description copied from `CHANGELOG.md`. This release will
   automatically trigger `publish_pypi.yml` and create a _release_ on PyPI.

## Other checks

Some other essential things to check throughout the release process -

- If updating our custom vcpkg registry entries
  [pybamm-team/sundials-vcpkg-registry][SUNDIALS_VCPKG]
  or [pybamm-team/casadi-vcpkg-registry][CASADI_VCPKG] (used to build Windows
  wheels), make sure to update the baseline of the registries in
  vcpkg-configuration.json to the latest commit id.
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

[SUNDIALS_VCPKG]: (https://github.com/pybamm-team/sundials-vcpkg-registry)
[CASADI_VCPKG]: (https://github.com/pybamm-team/casadi-vcpkg-registry)
[PYBAMM_FEED]: (https://github.com/conda-forge/pybamm-feedstock)
[FEED_GUIDE]: (https://conda-forge.org/docs/maintainer/updating_pkgs.html#updating-the-feedstock-repository)
