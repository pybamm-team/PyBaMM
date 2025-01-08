# IDAKLU solver

Standalone repository for the C/C++ solvers used in PyBaMM

## Installation

```bash
pip install pybammsolvers 
```

## Solvers

The following solvers are available:
- PyBaMM's IDAKLU solver

## Local builds

For testing new solvers and unsupported architectures, local builds are possible.

### Nox

Nox can be used to do a quick build:
```bash
pip install nox
nox
```
This will setup an environment and attempt to build the library.

### MacOS

Mac dependencies can be installed using brew
```bash
brew install libomp
brew reinstall gcc
python install_KLU_Sundials.py
pip install .
```

### Linux

Linux installs may vary based on the distribution, however, the basic build can
be performed with the following commands:
```bash
sudo apt-get libopenblas-dev gcc gfortran make g++ build-essential
pip install cmake casadi setuptools wheel
python install_KLU_Sundials.py
pip install .
```
