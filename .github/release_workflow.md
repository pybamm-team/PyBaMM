# Release workflow

This file contains the workflow required to make a `PyBaMM` release on GitHub and PyPI by the maintainers.

## rc0 releases (automated)

1. The `update_version.yml` workflow will run on every 1st of January, May and September, updating incrementing the version to `vYY.MMrc0` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `vcpkg.json`
   - `docs/_static/versions.json`

   and the `CHANGELOG.md` file will be generated via [Towncrier](https://towncrier.readthedocs.io/en/stable/) for the updated version string. The news fragments will be deleted at the end of the process for further release candidates or for the actual release.

   If further commits and changes are required to be made to the `develop` branch, the pull request branch titled "Make release `vYY.MMrc0`" will require manual updates to the `CHANGELOG.md` file and related sorting of pull requests (i.e., not using `Towncrier`), since the `newsfragments/` directory shall remain empty after the automated release process and the `develop` branch would have updated with new commits.

   These changes will be automatically pushed to a new branch `vYY.MM` and a PR from `vvYY.MM` to `develop` will be created (to sync the branches).

2. Create a new GitHub _pre-release_ with the tag `vYY.MMrc0` from the `vYY.MM` branch and a description copied from `CHANGELOG.md`.

3. This release will automatically trigger `publish_pypi.yml` and create a _pre-release_ on PyPI.

## rcX releases (manual)

If a new release candidate is required after the release of `rc0` -

1. Fix a bug in `vYY.MM` (no new features should be added to `vYY.MM` once `rc0` is released) and `develop` individually and add a news fragment if needed.

2. Run `update_version.yml` manually while using `append_to_tag` to specify the release candidate version number (`rc1`, `rc2`, ...).

   This will increment the version to `vYY.MMrcX` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `vcpkg.json`
   - `docs/_static/versions.json`

   and the `CHANGELOG.md` file will be generated via [Towncrier](https://towncrier.readthedocs.io/en/stable/) for the updated version string. The news fragments will be deleted at the end of the process for further release candidates or for the actual release.

   These changes will be automatically pushed to the existing `vYY.MM` branch and a PR from `vvYY.MM` to `develop` will be created (to sync the branches).

3. Create a new GitHub _pre-release_ with the same tag (`vYY.MMrcX`) from the `vYY.MM` branch and a description copied from `CHANGELOG.md`.

4. This release will automatically trigger `publish_pypi.yml` and create a _pre-release_ on PyPI.

## Actual release (manual)

Once satisfied with the release candidates -

1. Run `update_version.yml` manually, leaving the `append_to_tag` field blank ("") for an actual release.

   This will increment the version to `vYY.MM` by running `scripts/update_version.py` in the following files -

   - `pybamm/version.py`
   - `docs/conf.py`
   - `CITATION.cff`
   - `vcpkg.json`
   - `docs/_static/versions.json`

   If and after the release candidates are satisfactory, the `newsfragments/` directory shall not contain any new entries and the CHANGELOG shall have been updated at the time of the release candidates already. In this case, manually edit the `CHANGELOG.md` file to remove the `rcX` suffix from the version string to be used for the actual release (i.e., `vYY.MMrcX` -> `vYY.MM`), and edit the date of the release as necessary.

   These changes will be automatically pushed to the existing `vYY.MM` branch and a PR from `vvYY.MM` to `develop` will be created (to sync the branches).

3. Next, a PR from `vYY.MM` to `main` will be generated that should be merged once all the tests pass.

4. Create a new GitHub _release_ with the same tag from the `main` branch and a description copied from `CHANGELOG.md`.

5. This release will automatically trigger `publish_pypi.yml` and create a _release_ on PyPI.

## Other checks

Some other essential things to check throughout the release process -

- If updating our custom vcpkg registory entries [pybamm-team/sundials-vcpkg-registry](https://github.com/pybamm-team/sundials-vcpkg-registry) or [pybamm-team/casadi-vcpkg-registry](https://github.com/pybamm-team/casadi-vcpkg-registry) (used to build Windows wheels), make sure to update the baseline of the registories in vcpkg-configuration.json to the latest commit id.
- Update jax and jaxlib to the latest version in `pybamm.util` and `setup.py`, fixing any bugs that arise
- Make sure the URLs in `docs/_static/versions.json` are valid
- As the release workflow is initiated by the `release` event, it's important to note that the default `GITHUB_REF` used by `actions/checkout` during the checkout process will correspond to the tag created during the release process. Consequently, the workflows will consistently build PyBaMM based on the commit associated with this tag. Should new commits be introduced to the `vYY.MM` branch, such as those addressing build issues, it becomes necessary to manually update this tag to point to the most recent commit -
  ```
  git tag -f <tag_name> <commit_hash>
  git push origin <tag_name>  # can only be carried out by the maintainers
  ```
