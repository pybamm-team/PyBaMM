- Run `scripts/update_version.py` to

  - Increment version number in
    - `pybamm/version.py`
    - `docs/conf.py`
    - `CITATION.cff`
    - `vcpkg.json`
    - `docs/source/_static/versions.json`, and check if any links fail

  - Update baseline of registries in `vcpkg-configuration.json` as the latest commit id from [pybamm-team/sundials-vcpkg-registry](https://github.com/pybamm-team/sundials-vcpkg-registry)
  - Update `CHANGELOG.md` with a summary of the release

- Update jax and jaxlib to latest version in `pybamm.util` and fix any bugs that arise
- If building wheels on Windows gives a `vcpkg` related error - revert the baseline of default-registry to a stable commit in `vcpkg-configuration.json`
