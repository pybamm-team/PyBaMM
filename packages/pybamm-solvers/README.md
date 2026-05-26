# IDAKLU solver

Standalone repository for the C/C++ solvers used in PyBaMM

## Installation

From PyPI:

```bash
pip install pybammsolvers 
```

And from conda-forge (`>=v0.7.0` only):

```bash
conda install pybammsolvers
# or mamba install pybammsolvers
```

## Solvers

The following solvers are available:

- PyBaMM's IDAKLU solver

## Development

### Local builds

For testing new solvers and unsupported architectures, local builds are possible.
The build backend is [scikit-build-core](https://scikit-build-core.readthedocs.io/);
build-time dependencies (`scikit-build-core`, `pybind11`, `cmake`, `ninja`) are
resolved automatically via PEP 517 isolation when running `pip install`.

#### Nox (Recommended)

Nox can be used to do a quick build:

```bash
pip install nox
nox
```

This will setup an environment and attempt to build the library.

#### MacOS

Mac dependencies can be installed using brew

```bash
brew install libomp
brew reinstall gcc
git submodule update --init --recursive
python install_KLU_Sundials.py
pip install .
```

#### Linux

Linux installs may vary based on the distribution, however, the basic build can
be performed with the following commands:

```bash
sudo apt-get install libopenblas-dev gcc gfortran make g++ build-essential
git submodule update --init --recursive
python install_KLU_Sundials.py
pip install .
```

#### Custom SUNDIALS / SuiteSparse paths

If SUNDIALS and SuiteSparse are installed somewhere other than `./.idaklu`,
pass paths through pip's `--config-settings`:

```bash
pip install . \
    --config-settings=cmake.define.SUNDIALS_ROOT=/path/to/sundials \
    --config-settings=cmake.define.SuiteSparse_ROOT=/path/to/suitesparse
```

#### Editable installs and rebuilding the C++ extension

`pip install -e .[dev]` installs in editable mode with auto-rebuild on import:
edits to C++ sources trigger a rebuild the next time `pybammsolvers` is
imported. To force a rebuild manually (e.g. when auto-rebuild is bypassed in
CI):

```bash
nox -s dev-rebuild
```

### Testing

The project includes comprehensive test suites:

#### Unit Tests

Test pybammsolvers functionality in isolation:

```bash
nox -s unit            # Run all unit tests
nox -s integration     # Run all integration tests
```

#### PyBaMM Integration Tests

Verify compatibility with PyBaMM:

```bash
nox -s pybamm-tests                    # Clone/update PyBaMM and run all tests
nox -s pybamm-tests -- --unit-only     # Run only unit tests
nox -s pybamm-tests -- --integration-only  # Run only integration tests
nox -s pybamm-tests -- --no-update     # Skip git pull (use current version)
nox -s pybamm-tests -- --pybamm-dir ./custom/path  # Use existing PyBaMM clone
nox -s pybamm-tests -- --branch main  # Use specific branch/tag
```

The integration tests ensure that changes to pybammsolvers don't break PyBaMM functionality.

### Benchmarks

Test for performance regressions against released PyBaMM:

```bash
nox -s benchmarks
```
