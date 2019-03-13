# Examples

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinosulzer/PyBaMM/master)

This page contains a number of examples showing how to use PyBaMM.

Each example was created as a _Jupyter notebook_ (http://jupyter.org/).
These notebooks can be downloaded and used locally by running
```
$ jupyter notebook
```
from your local PyBaMM repository, or used online through [Binder](https://mybinder.org/v2/gh/tinosulzer/PyBaMM/master), or you can simply copy/paste the relevant code.

## Getting started

The easiest way to start with PyBaMM is by running and comparing some of the inbuilt models:
- [Run a pre-defined model](./run-model.ipynb)
- [Compare models](./compare-models.ipynb)

It is also easy to new models or change the setting that are used:
- [Add a model](./add-model.ipynb)
- [Change the settings](./change-settings.ipynb) (parameters, discretisation or solver)

For more advanced usage, new sets of parameters, discretisations and solvers can be added:
- [Add parameters](./add-parameters.ipynb)
- [Add a discretisation](./add-discretisation.ipynb)
- [Add a solver](./add-solver.ipynb)

## Expression tree structure

PyBaMM is built around an expression tree structure, similar to FEniCS/Firedrake's
[UFL](https://fenics.readthedocs.io/projects/ufl/en/latest/).
[This](expression-tree.ipynb) notebook explains how this works, from
model creation to solution.

### Models

The following models are implemented and can easily be [used](./run-model.ipynb) or [compared](./compare-models.ipynb). We always welcome [new models](./add-model.ipynb)!

#### Lithium-ion models

- [Single-Particle Model](./models/SPM.ipynb)
- [Single-Particle Model with electrolyte](./models/SPMe.ipynb)
- [Doyle-Fuller-Newman Model](./models/DFN.ipynb)

#### Lead-acid models

- [Full porous-electrode](./models/lead-acid-full.ipynb)
- [Leading-Order Quasi-Static](./models/lead-acid-LOQS.ipynb)
- [First-Order Quasi-Static](./models/lead-acid-FOQS.ipynb)
- [Composite](./models/lead-acid-composite.ipynb)

### Discretisations

The following discretisation is implemented
- [Finite Volumes](./discretisations/finite-volumes.ipynb)

See [here](./add-discretisation.ipynb) for instructions on adding new discretisations.

### Solvers

The following solver is implemented
- [Inbuilt SciPy solver](./solvers/scipy-integrate.ipynb)

See [here](./add-solver.ipynb) for instructions on adding new solvers.
