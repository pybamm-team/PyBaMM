# PyBaMM

[![travis](https://travis-ci.org/pybamm-team/PyBaMM.svg?branch=master)](https://travis-ci.org/pybamm-team/PyBaMM)
[![Build status](https://ci.appveyor.com/api/projects/status/xdje8jnhuj0ye1jc/branch/master?svg=true)](https://ci.appveyor.com/project/martinjrobins/pybamm/branch/master)
[![readthedocs](https://readthedocs.org/projects/pybamm/badge/?version=latest)](https://pybamm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pybamm-team/PyBaMM/branch/master/graph/badge.svg)](https://codecov.io/gh/pybamm-team/PyBaMM)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pybamm-team/PyBaMM/master?filepath=examples%2Fnotebooks)
[![black_code_style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Python Battery Mathematical Modelling solves continuum models for batteries, using both numerical methods and asymptotic analysis.

## How do I use PyBaMM?

PyBaMM comes with a number of [detailed examples](examples/notebooks/README.md), hosted here on
github. In addition, there is a [full API documentation](http://pybamm.readthedocs.io/),
hosted on [Read The Docs](readthedocs.io). A set of slides giving an overview of PyBaMM
can be found
[here](https://github.com/pybamm-team/pybamm_summary_slides/blob/master/pybamm.pdf).

## How can I obtain & install PyBaMM?

### Linux

For instructions on installing PyBaMM on Debian-based distributions, please see [here](INSTALL-LINUX.md)

### Windows

We recommend using Windows Subsystem for Linux to install PyBaMM on a Windows OS, for
instructions please see [here](INSTALL-WINDOWS.md)

### Sundials with KLU sparse solver
If you wish so simulate large systems such as the 2+1D models, we recommend employing a 
sparse solver. PyBaMM currently offers a direct interface to the sparse KLU solver within Sundials. 
If you are on a linux based distribution, a bash script has been provided which should 
install everything for you correctly. Please note you will require the python header files, openblas, 
a c compiler (e.g. gcc), and cmake, all of which you should be able to install on ubuntu using
```bash
apt install python3-dev libopenblas-dev cmake gcc
```
You will likely need to prepend `sudo` to the above command.

To install KLU, from within the main PyBaMM directory type
```bash
./scripts/install_sundials_4.1.0.sh
```
Note that this script has only been tested on Ubuntu 18.04.3 LTS. If you run into issues on 
another distribution, we recommend you first have a look through `install_sundials_4.1.0.sh` 
as it may be relatively simple to modify this for your purposes. 

In principle, the install should be possible in other operating systems by following the same
process as performed in `install_sundials_4.1.0.sh`. Although there may be some issues on 
Windows in building SuiteSparse as the only build option is via a Makefile. 





## How can I contribute to PyBaMM?

If you'd like to help us develop PyBaMM by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](CONTRIBUTING.md) first.

## Licensing

PyBaMM is fully open source. For more information about its license, see [LICENSE](./LICENSE.txt).
