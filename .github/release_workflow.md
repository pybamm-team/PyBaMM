# Release workflow

This file contains the workflow required to make a `PyBaMM` release on GitHub and PyPI by the maintainers.

## rc0 releases (automated)

1. The `update_version.yml` workflow will run on every 1st of January, May and September, updating incrementing the version to `YY.MMrc0` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `vcpkg.json`
   - `docs/_static/versions.json`
   - `CHANGELOG.md`

      These changes will be automatically pushed to a new branch `YY.MM`.

2. Create a new GitHub _pre-release_ with the tag `YY.MMrc0` from the `YY.MM` branch and a description copied from `CHANGELOG.md`.

3. This release will automatically trigger `publish_pypi.yml` and create a _pre-release_ on PyPI.

## rcX releases (manual)

If a new release candidate is required after the release of `rc0` -

1. Fix a bug in `YY.MM` (no new features should be added to `YY.MM` once `rc0` is released) and `develop` individually.

2. Run `update_version.yml` manually while using `append_to_tag` to specify the release candidate version number (`rc1`, `rc2`, ...).

3. This will increment the version to `YY.MMrcX` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `vcpkg.json`
   - `docs/_static/versions.json`
   - `CHANGELOG.md`

      These changes will be automatically pushed to the existing branch `YY.MM`.

4. Create a new GitHub _pre-release_ with the same tag (`YY.MMrcX`) from the `YY.MM` branch and a description copied from `CHANGELOG.md`.

5. This release will automatically trigger `publish_pypi.yml` and create a _pre-release_ on PyPI.

## Actual release (manual)

Once satisfied with the release candidates -

1. Run `update_version.yml` manually, leaving the `append_to_tag` field blank ("") for an actual release.

2. This will increment the version to `YY.MMrcX` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `vcpkg.json`
   - `docs/_static/versions.json`
   - `CHANGELOG.md`

      These changes will be automatically pushed to the existing branch `YY.MM`.

3. Next, a PR from `YY.MM` to `main` will be generated that should be merged once all the tests pass.

4. Create a new GitHub _release_ with the same tag from the `main` branch and a description copied from `CHANGELOG.md`.

5. This release will automatically trigger `publish_pypi.yml` and create a _release_ on PyPI.

## Other checks

Some other essential things to check throughout the release process -

- If updating our custom vcpkg registory entries [pybamm-team/sundials-vcpkg-registry](https://github.com/pybamm-team/sundials-vcpkg-registry) or [pybamm-team/casadi-vcpkg-registry](https://github.com/pybamm-team/casadi-vcpkg-registry) (used to build Windows wheels), make sure to update the baseline of the registories in vcpkg-configuration.json to the latest commit id.
- Update jax and jaxlib to the latest version in `pybamm.util` and `setup.py`, fixing any bugs that arise
- Make sure the URLs in `docs/_static/versions.json` are valid
