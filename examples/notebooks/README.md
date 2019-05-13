# Examples

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pybamm-team/PyBaMM/master)

This page contains a number of examples showing how to use PyBaMM.

Each example was created as a _Jupyter notebook_ (http://jupyter.org/).
These notebooks can be downloaded and used locally by running
```
$ jupyter notebook
```
from your local PyBaMM repository, or used online through [Binder](https://mybinder.org/v2/gh/pybamm-team/PyBaMM/master), or you can simply copy/paste the relevant code.

## Getting started

The easiest way to start with PyBaMM is by running and comparing some of the inbuilt models:
- [Run the Single Particle Model (SPM)](./models/SPM.ipynb)
- [Compare models](./models/lead-acid.ipynb)

It is also easy to new models or change the setting that are used:
- [Add a model](https://pybamm.readthedocs.io/en/latest/tutorials/add-model.html)
- [Change the settings](./change-settings.ipynb) (parameters, spatial method or solver)

For more advanced usage, new sets of parameters, spatial methods and solvers can be added:
- [Add parameters](https://pybamm.readthedocs.io/en/latest/tutorials/add-parameter-values.html)
- [Add a spatial method](https://pybamm.readthedocs.io/en/latest/tutorials/add-spatial-method.html)
- [Add a solver](https://pybamm.readthedocs.io/en/latest/tutorials/add-solver.html)

## Expression tree structure

PyBaMM is built around an expression tree structure.
[This](expression_tree/expression-tree.ipynb) notebook explains how this works, from
model creation to solution.

### Models

The following models are implemented and can easily be used or [compared](./models/lead-acid.ipynb). We always welcome [new models](https://pybamm.readthedocs.io/en/latest/tutorials/add-model.html)!

#### Lithium-ion models

- [Single-Particle Model](./models/SPM.ipynb)
- [Single-Particle Model with electrolyte](./models/SPMe.ipynb)
- [Doyle-Fuller-Newman Model](./models/DFN.ipynb)

#### Lead-acid models

- [Full porous-electrode](./models/lead-acid-full.ipynb)
- [Leading-Order Quasi-Static](./models/lead-acid-LOQS.ipynb)
- [First-Order Quasi-Static](./models/lead-acid-FOQS.ipynb)
- [Composite](./models/lead-acid-composite.ipynb)

### Spatial Methods

The following spatial method is implemented
- [Finite Volumes](./spatial_methods/finite-volumes.ipynb)

See [here](https://pybamm.readthedocs.io/en/latest/tutorials/add-spatial-method.html) for instructions on adding new spatial methods.

### Solvers

The following solvers are implemented
- Scipy ODE solver
- [Scikits ODE solver](./solvers/scikits-ode-solver.ipynb)
- [Scikits DAE solver](./solvers/scikits-dae-solver.ipynb)

See [here](https://pybamm.readthedocs.io/en/latest/tutorials/add-solver.html) for instructions on adding new solvers.
