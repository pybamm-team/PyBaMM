# Release workflow

This file contains the workflow required to make a `PyBaMM` release on GitHub and PyPI by the maintainers.

## rc0 releases (automated)

1. The `update_version.yml` workflow will run on every 1st of January, May and September, creating 2 PRs -

   1. Incrementing the version to `YY.MMrc0` by running `scripts/update_version.py` in the following files -
      - `pybamm/version.py`
      - `docs/conf.py`
      - `CITATION.cff`
      - `vcpkg.json`
      - `docs/source/_static/versions.json`

   2. A PR from `develop` to `main`

   The version PR should be merged into `develop`, and then the develop-to-main PR should be merged into `main`.

2. Once the tests pass, create a new GitHub _pre-release_ with the same tag (`YY.MMrc0`) from the `main` branch and a description copied from `CHANGELOG.md`.

3. This release will automatically trigger `publish_pypi.yml` and create a _pre-release_ on PyPI.

## rcX releases (manual)

If a new release candidate is required after the release of `rc0` -

1. Fix a bug in `main` (no new features should be added to `main` once `rc0` is released) and `develop` individually.

2. Run `update_version.yml` manually while using `append_to_tag` to specify the release candidate version number (`rc1`, `rc2`, ...).

3. This will create a PR incrementing the version to `YY.MMrcX` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `vcpkg.json`
   - `docs/source/_static/versions.json`

   The version PR should be merged into `main`, because merging it into `develop` would require merging `develop` into `main`, something we don't want (`develop` will have new features).

4. Once the tests pass, create a new GitHub _pre-release_ with the same tag from the `main` branch and a description copied from `CHANGELOG.md`.

5. This release will automatically trigger `publish_pypi.yml` and create a _pre-release_ on PyPI.

6. Manually merge `main` back to `develop` if any conflicts arise.

## Actual release (manual)

Once satisfied with the release candidates -

1. Run `update_version.yml` manually, leaving the `append_to_tag` field blank ("") for an actual release.

2. This will create a PR incrementing the version to `YY.MM` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `vcpkg.json`
   - `docs/source/_static/versions.json`

   The version PR should be merged into `main`, because merging it into `develop` would require merging `develop` into `main`, something we don't want (`develop` will have new features).

3. Once the tests pass, create a new GitHub _release_ with the same tag from the `main` branch and a description copied from `CHANGELOG.md`.

4. This release will automatically trigger `publish_pypi.yml` and create a _release_ on PyPI.

## Other checks

Some other essential things to check throughout the release process -

- Update baseline of registries in `vcpkg-configuration.json` as the latest commit id from [pybamm-team/sundials-vcpkg-registry](https://github.com/pybamm-team/sundials-vcpkg-registry)
- Update `CHANGELOG.md` with a summary of the release
- Update jax and jaxlib to the latest version in `pybamm.util` and fix any bugs that arise
- If building wheels on Windows gives a `vcpkg` related error - revert the baseline of default-registry to a stable commit in `vcpkg-configuration.json`
- Make sure the URLs in `docs/source/_static/versions.json` are valid
