![PyBaMM_logo](https://user-images.githubusercontent.com/20817509/107091287-8ad46a80-67cf-11eb-86f5-7ebef7c72a1e.png)

#

<div align="center">

[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](http://numfocus.org)
[![Scheduled](https://github.com/pybamm-team/PyBaMM/actions/workflows/run_periodic_tests.yml/badge.svg?branch=develop)](https://github.com/pybamm-team/PyBaMM/actions/workflows/run_periodic_tests.yml)
[![readthedocs](https://readthedocs.org/projects/pybamm/badge/?version=latest)](https://docs.pybamm.org/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/pybamm-team/PyBaMM/branch/main/graph/badge.svg)](https://codecov.io/gh/pybamm-team/PyBaMM)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/main/)
[![DOI](https://zenodo.org/badge/DOI/10.5334/jors.309.svg)](https://doi.org/10.5334/jors.309)
[![release](https://img.shields.io/github/v/release/pybamm-team/PyBaMM?color=yellow)](https://github.com/pybamm-team/PyBaMM/releases)
[![code style](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/pybamm-team/PyBaMM/badge)](https://scorecard.dev/viewer/?uri=github.com/pybamm-team/PyBaMM)

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-88-orange.svg)](#-contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

</div>

# PyBaMM

PyBaMM (Python Battery Mathematical Modelling) is an open-source battery simulation package
written in Python. Our mission is to accelerate battery modelling research by
providing open-source tools for multi-institutional, interdisciplinary collaboration.
Broadly, PyBaMM consists of
(i) a framework for writing and solving systems
of differential equations,
(ii) a library of battery models and parameters, and
(iii) specialized tools for simulating battery-specific experiments and visualizing the results.
Together, these enable flexible model definitions and fast battery simulations, allowing users to
explore the effect of different battery designs and modeling assumptions under a variety of operating scenarios.

[//]: # "numfocus-fiscal-sponsor-attribution"

PyBaMM uses an [open governance model](https://pybamm.org/governance/)
and is fiscally sponsored by [NumFOCUS](https://numfocus.org/). Consider making
a [tax-deductible donation](https://numfocus.org/donate-for-pybamm) to help the project
pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
  <a href="https://numfocus.org/project/pybamm">
    <img height="60px"
         src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
         align="center">
  </a>
</div>
<br>

## 💻 Using PyBaMM

The easiest way to use PyBaMM is to run a 1C constant-current discharge with a model of your choice with all the default settings:

```python3
import pybamm

model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
sim = pybamm.Simulation(model)
sim.solve([0, 3600])  # solve for 1 hour
sim.plot()
```

or simulate an experiment such as a constant-current discharge followed by a constant-current-constant-voltage charge:

```python3
import pybamm

experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/10 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 1 A until 4.1 V",
            "Hold at 4.1 V until 50 mA",
            "Rest for 1 hour",
        )
    ]
    * 3,
)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, solver=pybamm.CasadiSolver())
sim.solve()
sim.plot()
```

However, much greater customisation is available. It is possible to change the physics, parameter values, geometry, submesh type, number of submesh points, methods for spatial discretisation and solver for integration (see DFN [script](https://github.com/pybamm-team/PyBaMM/blob/develop/examples/scripts/DFN.py) or [notebook](https://github.com/pybamm-team/PyBaMM/blob/develop/docs/source/examples/notebooks/models/DFN.ipynb)).

For new users we recommend the [Getting Started](https://github.com/pybamm-team/PyBaMM/tree/develop/docs/source/examples/notebooks/getting_started/) guides. These are intended to be very simple step-by-step guides to show the basic functionality of PyBaMM, and can either be downloaded and used locally, or used online through [Google Colab](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/main/).

Further details can be found in a number of [detailed examples](https://github.com/pybamm-team/PyBaMM/tree/develop/examples), hosted here on
github. In addition, there is a [full API documentation](https://docs.pybamm.org/en/latest/source/api/index.html),
hosted on [Read The Docs](https://readthedocs.org/).
Additional supporting material can be found
[here](https://github.com/pybamm-team/pybamm-supporting-material/).

Note that the examples on the default `develop` branch are tested on the latest `develop` commit. This may sometimes cause errors when running the examples on the pybamm pip package, which is synced to the `main` branch. You can switch to the `main` branch on github to see the version of the examples that is compatible with the latest pip release.

## Versioning

PyBaMM makes releases every four months and we use [CalVer](https://calver.org/), which means that the version number is `YY.MM`. The releases happen, approximately, at the end of January, May and September. There is no difference between releases that increment the year and releases that increment the month; in particular, releases that increment the month may introduce breaking changes. Breaking changes for each release are communicated via the [CHANGELOG](CHANGELOG.md), and come with deprecation warnings or errors that are kept for at least one year (3 releases). If you find a breaking change that is not documented, or think it should be undone, please open an issue on [GitHub](https://github.com/pybamm-team/pybamm).

## 🚀 Installing PyBaMM

PyBaMM is available on GNU/Linux, MacOS and Windows.
We strongly recommend to install PyBaMM within a python virtual environment, in order not to alter any distribution python files.
For instructions on how to create a virtual environment for PyBaMM, see [the documentation](https://docs.pybamm.org/en/latest/source/user_guide/installation/gnu-linux-mac.html#user-install).

### Using pip

[![pypi](https://img.shields.io/pypi/v/pybamm?color=blue)](https://pypi.org/project/pybamm/)
[![downloads](https://img.shields.io/pypi/dm/pybamm?color=blue)](https://pypi.org/project/pybamm/)

```bash
pip install pybamm
```

### Using conda

PyBaMM is available as a conda package through the conda-forge channel.

[![conda_forge](https://img.shields.io/conda/vn/conda-forge/pybamm?color=green)](https://anaconda.org/conda-forge/pybamm)
[![downloads](https://img.shields.io/conda/dn/conda-forge/pybamm?color=green)](https://anaconda.org/conda-forge/pybamm)

```bash
conda install -c conda-forge pybamm
```

### Optional solvers

The following solvers are optionally available:

- [jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)-based solver, see [the documentation](https://docs.pybamm.org/en/latest/source/user_guide/installation/gnu-linux-mac.html#optional-jaxsolver).

## 📖 Citing PyBaMM

If you use PyBaMM in your work, please cite our paper

> Sulzer, V., Marquis, S. G., Timms, R., Robinson, M., & Chapman, S. J. (2021). Python Battery Mathematical Modelling (PyBaMM). _Journal of Open Research Software, 9(1)_.

You can use the BibTeX

```
@article{Sulzer2021,
  title = {{Python Battery Mathematical Modelling (PyBaMM)}},
  author = {Sulzer, Valentin and Marquis, Scott G. and Timms, Robert and Robinson, Martin and Chapman, S. Jon},
  doi = {10.5334/jors.309},
  journal = {Journal of Open Research Software},
  publisher = {Software Sustainability Institute},
  volume = {9},
  number = {1},
  pages = {14},
  year = {2021}
}
```

We would be grateful if you could also cite the relevant papers. These will change depending on what models and solvers you use. To find out which papers you should cite, add the line

```python3
pybamm.print_citations()
```

to the end of your script. This will print BibTeX information to the terminal; passing a filename to `print_citations` will print the BibTeX information to the specified file instead. A list of all citations can also be found in the [citations file](https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/CITATIONS.bib). In particular, PyBaMM relies heavily on [CasADi](https://web.casadi.org/publications/).
See [CONTRIBUTING.md](https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md#citations) for information on how to add your own citations when you contribute.

## 🛠️ Contributing to PyBaMM

If you'd like to help us develop PyBaMM by adding new methods, writing documentation, or fixing embarrassing bugs, please have a look at these [guidelines](https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md) first.

## 📫 Get in touch

For any questions, comments, suggestions or bug reports, please see the
[contact page](https://www.pybamm.org/community).

## 📃 License

PyBaMM is fully open source. For more information about its license, see [LICENSE](https://github.com/pybamm-team/PyBaMM/blob/develop/LICENSE.txt).

## ✨ Contributors

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

Click here to see [a full list](https://github.com/pybamm-team/PyBaMM/blob/develop/all_contributors.md) of our contributors' profiles.
