# PyBaMM

[![travis](https://travis-ci.org/pybamm-team/PyBaMM.svg?branch=master)](https://travis-ci.org/pybamm-team/PyBaMM)
[![Build status](https://ci.appveyor.com/api/projects/status/xdje8jnhuj0ye1jc/branch/master?svg=true)](https://ci.appveyor.com/project/martinjrobins/pybamm/branch/master)
[![readthedocs](https://readthedocs.org/projects/pybamm/badge/?version=latest)](https://pybamm.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pybamm-team/PyBaMM/branch/master/graph/badge.svg)](https://codecov.io/gh/pybamm-team/PyBaMM)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pybamm-team/PyBaMM/master?filepath=examples%2Fnotebooks)
[![black_code_style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Python Battery Mathematical Modelling solves continuum models for batteries, using both numerical methods and asymptotic analysis.

## How do I use PyBaMM?

The easiest way to use PyBaMM is to run a 1C constant-current discharge with a model of your choice with all the default settings:
```python3
import pybamm
model = pybamm.lithium_ion.DFN() # Doyle-Fuller-Newman model
sim = pybamm.Simulation(model)
sim.solve()
sim.plot()
```
However, much greater customisation is available. It is possible to change the physics, parameter values, geometry, submesh type,  number of submesh points, methods for spatial discretisation and solver for integration (see DFN [script](examples/scripts/DFN.py) or [notebook](examples/notebooks/models/dfn.ipynb)).

For new users we recommend the [Getting Started](examples/notebooks/Getting%20Started/) guides. These are intended to be very simple step-by-step guides to show the basic functionality of PyBaMM.

Further details can be found in a number of [detailed examples](examples/notebooks/README.md), hosted here on
github. In addition, there is a [full API documentation](http://pybamm.readthedocs.io/),
hosted on [Read The Docs](readthedocs.io). A set of slides giving an overview of PyBaMM
can be found
[here](https://github.com/pybamm-team/pybamm_summary_slides/blob/master/pybamm.pdf).

For further examples, see the list of repositories that use PyBaMM [here](https://github.com/pybamm-team/pybamm-example-results)

## How can I obtain & install PyBaMM?

### Linux

For instructions on installing PyBaMM on Debian-based distributions, please see [here](INSTALL-LINUX.md)

### Windows

We recommend using Windows Subsystem for Linux to install PyBaMM on a Windows OS, for
instructions please see [here](INSTALL-WINDOWS.md)

## How can I contribute to PyBaMM?

If you'd like to help us develop PyBaMM by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](CONTRIBUTING.md) first.

## Licensing

PyBaMM is fully open source. For more information about its license, see [LICENSE](./LICENSE.txt).
