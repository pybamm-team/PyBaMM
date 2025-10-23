# IDAKLU solver

Standalone repository for the C/C++ solvers used in PyBaMM

## Installation

```bash
pip install pybammsolvers 
```

## Solvers

The following solvers are available:
- PyBaMM's IDAKLU solver

## Development

### Local builds

For testing new solvers and unsupported architectures, local builds are possible.

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
git submodules update --init --recursive
pip install cmake casadi setuptools wheel "pybind11[global]"
python install_KLU_Sundials.py
pip install .
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
nox -s pybamm-tests -- --branch develop  # Use specific branch/tag
```

The integration tests ensure that changes to pybammsolvers don't break PyBaMM functionality.

### Benchmarks
Test for performance regressions against released PyBaMM:
```bash
nox -s benchmarks
```