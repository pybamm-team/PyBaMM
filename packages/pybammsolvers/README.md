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

SUNDIALS and SuiteSparse are compiled automatically from the bundled git
submodules during the build, so initialise the submodules first.

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
pip install .
```

#### Linux

Linux installs may vary based on the distribution, however, the basic build can
be performed with the following commands:

```bash
sudo apt-get install libopenblas-dev gcc gfortran make g++ build-essential
git submodule update --init --recursive
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

#### Building a wheel or sdist

Build distributions with an isolated PEP 517 frontend, which installs the build
backend itself:

```bash
python -m build        # or: pipx run build
```

`uv build` does **not** work here: the workspace marks `pybammsolvers` as
`no-build-isolation-package` (so the editable auto-rebuild can find the build
tools in the project venv), and uv then does not install the build backend into
its own build environment. The release CI uses `python -m build` for this reason.

### Testing

The project includes comprehensive test suites:

#### Unit Tests

Test pybammsolvers functionality in isolation:

```bash
nox -s unit            # Run all unit tests
nox -s integration     # Run all integration tests
```

### Benchmarks

Test for performance regressions against released PyBaMM:

```bash
nox -s benchmarks
```
