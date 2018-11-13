# PyBaMM

[![Build Status](https://travis-ci.org/tinosulzer/PyBaMM.svg?branch=master)](https://travis-ci.org/tinosulzer/PyBaMM)
[![Documentation Status](https://readthedocs.org/projects/pybamm/badge/?version=latest)](https://pybamm.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

Python Battery Mathematical Modelling solves continuum models for lead-acid batteries, using both numerical methods and asymptotic analysis

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

## How can I contribute to PyBaMM?

If you'd like to help us develop PyBaMM by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](CONTRIBUTING.md) first.

## Licensing

PyBaMM is fully open source. For more information about its license, see [LICENSE](./LICENSE.txt).
