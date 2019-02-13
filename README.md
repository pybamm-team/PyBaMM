# PyBaMM

[![Build Status](https://travis-ci.org/tinosulzer/PyBaMM.svg?branch=master)](https://travis-ci.org/tinosulzer/PyBaMM)
[![Documentation Status](https://readthedocs.org/projects/pybamm/badge/?version=latest)](https://pybamm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/tinosulzer/PyBaMM/branch/master/graph/badge.svg)](https://codecov.io/gh/tinosulzer/PyBaMM)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinosulzer/PyBaMM/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Python Battery Mathematical Modelling solves continuum models for batteries, using both numerical methods and asymptotic analysis.

## How do I use PyBaMM?

PyBaMM comes with a number of [detailed examples](examples/README.md), hosted here on github. In addition, there is a [full API documentation](http://pybamm.readthedocs.io/), hosted on [Read The Docs](readthedocs.io).

## How can I install PyBaMM?

You'll need the following requirements:

- Python 2.7 or Python 3.4+
- Python libraries: `numpy` `scipy` `pandas` `matplotlib`

These can easily be installed using `pip`. To do this, first make sure you have the latest version of pip installed:

```
$ pip install --upgrade pip
```

Then navigate to the path where you downloaded PyBaMM to, and install both PyBaMM and its dependencies by typing:

```
$ pip install .
```

Or, if you want to install PyBaMM as a [developer](CONTRIBUTING.md), use

```
$ pip install -e .[dev,docs]
```

To uninstall again, type

```
$ pip uninstall pybamm
```

## Optional dependencies

### [scikits.odes](https://github.com/bmcage/odes)

Users can install [scikits.odes](https://github.com/bmcage/odes) in order to use the
wrapped SUNDIALS ODE and DAE
[solvers](https://pybamm.readthedocs.io/en/latest/source/solvers/scikits_solvers.html).

To install scikits.odes, you will need to first download and compile sundials 3.1.1:

```bash
$ INSTALL_DIR=`pwd`/sundials
$ wget https://computation.llnl.gov/projects/sundials/download/sundials-3.1.1.tar.gz
$ tar -xvf sundials-3.1.1.tar.gz
$ mkdir build-sundials-3.1.1
$ cd build-sundials-3.1.1/
$ cmake -DLAPACK_ENABLE=ON -DSUNDIALS_INDEX_TYPE=int32_t -DBUILD_ARKODE:BOOL=OFF -DEXAMPLES_ENABLE:BOOL=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ../sundials-3.1.1/
$ make install
```

Then install [scikits.odes](https://github.com/bmcage/odes), letting it know the sundials install location:

```bash
$ SUNDIALS_INST=$INSTALL_DIR pip install scikits.odes
```

After this, you will need to set your `LD_LIBRARY_PATH` to point to the sundials
library:

```bash
$ export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
```

You may wish to put this last line in your `.bashrc` or virtualenv `activate` script. 

Please see the [scikits.odes
documentation](https://scikits-odes.readthedocs.io/en/latest/installation.html) for more
detailed installation instructions. 


## How can I contribute to PyBaMM?

If you'd like to help us develop PyBaMM by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](CONTRIBUTING.md) first.

## Licensing

PyBaMM is fully open source. For more information about its license, see [LICENSE](./LICENSE.txt).
