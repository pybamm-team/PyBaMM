# Release workflow

This file contains the workflow required to make a `PyBaMM` release on GitHub, PyPI, and conda-forge by the maintainers.

## rc0 releases (automated)

1. The `update_version.yml` workflow will run on every 1st of January, May and September, updating incrementing the version to `vYY.MMrc0` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `pyproject.toml`
   - `vcpkg.json`
   - `CHANGELOG.md`

      These changes will be automatically pushed to a new branch `vYY.MM` and a PR from `vvYY.MM` to `develop` will be created (to sync the branches).

2. Create a new GitHub _pre-release_ with the tag `vYY.MMrc0` from the `vYY.MM` branch and a description copied from `CHANGELOG.md`.

3. This release will automatically trigger `publish_pypi.yml` and create a _pre-release_ on PyPI.

## rcX releases (manual)

If a new release candidate is required after the release of `rc0` -

1. Fix a bug in `vYY.MM` (no new features should be added to `vYY.MM` once `rc0` is released) and `develop` individually.

2. Run `update_version.yml` manually while using `append_to_tag` to specify the release candidate version number (`rc1`, `rc2`, ...).

3. This will increment the version to `vYY.MMrcX` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `pyproject.toml`
   - `vcpkg.json`
   - `CHANGELOG.md`

      These changes will be automatically pushed to the existing `vYY.MM` branch and a PR from `vvYY.MM` to `develop` will be created (to sync the branches).

4. Create a new GitHub _pre-release_ with the same tag (`vYY.MMrcX`) from the `vYY.MM` branch and a description copied from `CHANGELOG.md`.

5. This release will automatically trigger `publish_pypi.yml` and create a _pre-release_ on PyPI.

## Actual release (manual)

Once satisfied with the release candidates -

1. Run `update_version.yml` manually, leaving the `append_to_tag` field blank ("") for an actual release.

2. This will increment the version to `vYY.MMrcX` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `pyproject.toml`
   - `vcpkg.json`
   - `CHANGELOG.md`

      These changes will be automatically pushed to the existing `vYY.MM` branch and a PR from `vvYY.MM` to `develop` will be created (to sync the branches).

3. Next, a PR from `vYY.MM` to `main` will be generated that should be merged once all the tests pass.

4. Create a new GitHub _release_ with the same tag from the `main` branch and a description copied from `CHANGELOG.md`.

5. This release will automatically trigger `publish_pypi.yml` and create a _release_ on PyPI.

## Other checks

Some other essential things to check throughout the release process -

- If updating our custom vcpkg registory entries [pybamm-team/sundials-vcpkg-registry](https://github.com/pybamm-team/sundials-vcpkg-registry) or [pybamm-team/casadi-vcpkg-registry](https://github.com/pybamm-team/casadi-vcpkg-registry) (used to build Windows wheels), make sure to update the baseline of the registories in vcpkg-configuration.json to the latest commit id.
- Update jax and jaxlib to the latest version in `pybamm.util` and `pyproject.toml`, fixing any bugs that arise
- As the release workflow is initiated by the `release` event, it's important to note that the default `GITHUB_REF` used by `actions/checkout` during the checkout process will correspond to the tag created during the release process. Consequently, the workflows will consistently build PyBaMM based on the commit associated with this tag. Should new commits be introduced to the `vYY.MM` branch, such as those addressing build issues, it becomes necessary to manually update this tag to point to the most recent commit -
  ```
  git tag -f <tag_name> <commit_hash>
  git push -f <pybamm-team/PyBaMM_remote_name> <tag_name>  # can only be carried out by the maintainers
  ```
- If changes are made to the API, console scripts, entry points, new optional dependencies are added, support for major Python versions is dropped or added, or core project information and metadata are modified at the time of the release, make sure to update the `meta.yaml` file in the `recipe/` folder of the [conda-forge/pybamm-feedstock](https://github.com/conda-forge/pybamm-feedstock) repository accordingly by following the instructions in the [conda-forge documentation](https://conda-forge.org/docs/maintainer/updating_pkgs.html#updating-the-feedstock-repository) and re-rendering the recipe
- The conda-forge release workflow will automatically be triggered following a stable PyPI release, and the aforementioned updates should be carried out directly in the main repository by pushing changes to the automated PR created by the conda-forge-bot. A manual PR can also be created if the updates are not included in the automated PR for some reason. This manual PR **must** bump the build number in `meta.yaml` and **must** be from a personal fork of the repository.
