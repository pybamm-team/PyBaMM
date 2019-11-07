# Install scikits.odes

---
**Note**

This file provides installation instructions for either Ubuntu-based distributions or Mac OS distributions. Please read carefully which lines to run in each case.

---

Users can install [scikits.odes](https://github.com/bmcage/odes) in order to use the
wrapped SUNDIALS ODE and DAE
[solvers](https://pybamm.readthedocs.io/en/latest/source/solvers/scikits_solvers.html).
The Sundials DAE solver is required to solve the DFN battery model in PyBaMM.

Before installing scikits.odes, you need to have installed:

- Python header files (`python-dev/python3-dev` on Debian/Ubuntu-based distributions, comes with python3 by default in brew)
- C compiler
- Fortran compiler (e.g. gfortran, comes with gcc in brew)
- BLAS/LAPACK install (OpenBLAS is recommended by the scikits.odes developers)
- CMake (for building Sundials)
- Sundials 4.1.0 (see instructions below)

You can install these on Ubuntu or Debian using apt-get:

```bash
sudo apt-get install python3-dev gfortran gcc cmake libopenblas-dev
```

or on a Mac OS distribution using brew:

```bash
brew install wget gcc cmake openblas
```

## Installing SUNDIALS and scikits.odes

### Option 1: install with script

We recommend that you first try to install SUNDIALS and scikits together by running the script

```bash
source scripts/install_scikits_odes.sh
```

If this works, skip to [the final section](#setting-library-path). Otherwise, try Option 2 below.

## Option 2: install manually

To install SUNDIALS 4.1.0 manually, on the command-line type:

```bash
INSTALL_DIR=`pwd`/sundials
wget https://computation.llnl.gov/projects/sundials/download/sundials-4.1.0.tar.gz
tar -xvf sundials-4.1.0.tar.gz
mkdir build-sundials-4.1.0
cd build-sundials-4.1.0/
cmake -DLAPACK_ENABLE=ON -DSUNDIALS_INDEX_TYPE=int32_t -DBUILD_ARKODE:BOOL=OFF -DEXAMPLES_ENABLE:BOOL=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ../sundials-4.1.0/
make install
rm -r ../sundials-4.1.0
```

Then install [scikits.odes](https://github.com/bmcage/odes), letting it know the sundials install location:

```bash
SUNDIALS_INST=$INSTALL_DIR pip install scikits.odes
```

## Setting library path

After this, you will need to set your `LD_LIBRARY_PATH` (for Linux) or `DYLD_LIBRARY_PATH` (for Mac) to point to the sundials
library - for Linux:

```bash
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
```

or for Mac:

```bash
export DYLD_LIBRARY_PATH=$INSTALL_DIR/lib:$DYLD_LIBRARY_PATH
```

You may wish to put one of these lines in your `.bashrc` or virtualenv `activate` script,
which will save you needing to set your `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH` every time you log in. For
example, to add this line to your virtual environment `env` you can type:

```bash
echo "export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH" >> env/bin/activate
```

for Linux or 

```bash
echo "export DYLD_LIBRARY_PATH=$INSTALL_DIR/lib:\$DYLD_LIBRARY_PATH" >> env/bin/activate
```

for Mac. 

Please see the [scikits.odes
documentation](https://scikits-odes.readthedocs.io/en/latest/installation.html) for more
detailed installation instructions.

You can also try installing the [KLU solver](INSTALL-KLU.md) if you haven't already done so, but you only need one DAE solver.