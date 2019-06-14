# PyBaMM

[![travis](https://travis-ci.org/pybamm-team/PyBaMM.svg?branch=master)](https://travis-ci.org/pybamm-team/PyBaMM)
[![appveyor](https://ci.appveyor.com/api/projects/status/xdje8jnhuj0ye1jc?svg=true)](https://ci.appveyor.com/project/martinjrobins/pybamm)
[![readthedocs](https://readthedocs.org/projects/pybamm/badge/?version=latest)](https://pybamm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pybamm-team/PyBaMM/branch/master/graph/badge.svg)](https://codecov.io/gh/pybamm-team/PyBaMM)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pybamm-team/PyBaMM/master?filepath=examples%2Fnotebooks)
[![black_code_style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Python Battery Mathematical Modelling solves continuum models for batteries, using both numerical methods and asymptotic analysis.

## How do I use PyBaMM?

PyBaMM comes with a number of [detailed examples](examples/README.md), hosted here on
github. In addition, there is a [full API documentation](http://pybamm.readthedocs.io/),
hosted on [Read The Docs](readthedocs.io).

## How can I obtain & install PyBaMM?

You'll need the following requirements:

- Python 3.5+
- Git (`git` package on Ubuntu distributions)
- Python libraries: `venv` (`python3-venv` package on Ubuntu distributions)

The first step is to get the code by cloning this repository

```bash
git clone https://github.com/pybamm-team/PyBaMM.git
cd PyBaMM
```

The safest way to install PyBaMM is to do so within a virtual environment ([introduction
to virtual environments](https://realpython.com/python-virtual-environments-a-primer/)).
To create a virtual environment `env` within your current directory type:

```bash
python3 -m venv env
```

You can then "activate" the environment using:

```bash
source env/bin/activate
```

Now all the calls to pip described below will install PyBaMM and its dependencies into
the environment `env`. When you are ready to exit the environment and go back to your
original system, just type:

```bash
deactivate
```

PyBaMM has the following python libraries as dependencies: `numpy`, `scipy`, `pandas`,
`matplotlib`. These, along with PyBaMM, can easily be installed using `pip`. First, make
sure you have activated your virtual environment as above, and that you have the latest
version of pip installed:

```bash
pip install --upgrade pip
```

Then navigate to the path where you downloaded PyBaMM to (you will already be in the
correct location if you followed the instructions above), and install both PyBaMM and
its dependencies by typing:

```bash
pip install .
```

Or, if you want to install PyBaMM as a [developer](CONTRIBUTING.md), use

```bash
pip install -e .[dev,docs]
```

To uninstall again, type

```bash
pip uninstall pybamm
```

## Optional dependencies

### [scikits.odes](https://github.com/bmcage/odes)

Users can install [scikits.odes](https://github.com/bmcage/odes) in order to use the
wrapped SUNDIALS ODE and DAE
[solvers](https://pybamm.readthedocs.io/en/latest/source/solvers/scikits_solvers.html).

Before installing odes, you need to have installed:

- Python header files (`python-dev/python3-dev` on Debian/Ubuntu-based distributions)
- C compiler
- Fortran compiler (e.g. gfortran)
- BLAS/LAPACK install (OpenBLAS is recommended by the scikits.odes developers)
- Sundials 3.1.1

To install Sundials 3.1.1, on the command-line type:

```bash
INSTALL_DIR=`pwd`/sundials
wget https://computation.llnl.gov/projects/sundials/download/sundials-3.1.1.tar.gz
tar -xvf sundials-3.1.1.tar.gz
mkdir build-sundials-3.1.1
cd build-sundials-3.1.1/
cmake -DLAPACK_ENABLE=ON -DSUNDIALS_INDEX_TYPE=int32_t -DBUILD_ARKODE:BOOL=OFF -DEXAMPLES_ENABLE:BOOL=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ../sundials-3.1.1/
make install
rm -r ../sundials-3.1.1
```

Then install [scikits.odes](https://github.com/bmcage/odes), letting it know the sundials install location:

```bash
SUNDIALS_INST=$INSTALL_DIR pip install scikits.odes
```

After this, you will need to set your `LD_LIBRARY_PATH` to point to the sundials
library:

```bash
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
```

You may wish to put this last line in your `.bashrc` or virtualenv `activate` script,
which will save you needing to set your `LD_LIBRARY_PATH` every time you log in. For
example, to add this line to your `.bashrc` you can type:

```bash
echo "export LD_LIBRARY_PATH=$INSTALL_DIR/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
```

Please see the [scikits.odes
documentation](https://scikits-odes.readthedocs.io/en/latest/installation.html) for more
detailed installation instructions.

### [scikit.fem](https://github.com/kinnala/scikit-fem)

Users can install [scikit.fem](https://github.com/kinnala/scikit-fem) in order to
generate meshes and assemble discrete operators for use in the finite element method. At present,
the finite element method is only implemented for a small number of submodels in PyBaMM
(for example, see the [current collector submodel](https://github.com/pybamm-team/PyBaMM/blob/master/pybamm/models/submodels/current_collector.py)).

Note that scikit-fem requires Python 3.6+. To install scikit-fem, on the command-line type:

```bash
pip install scikit-fem
```

Please see the [scikit.fem documentation](https://kinnala.github.io/scikit-fem-docs/learning.html) for more information.

## How can I contribute to PyBaMM?

If you'd like to help us develop PyBaMM by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](CONTRIBUTING.md) first.

## Licensing

PyBaMM is fully open source. For more information about its license, see [LICENSE](./LICENSE.txt).
