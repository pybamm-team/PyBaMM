# Getting started

For new users we recommend the [Getting Started](./Getting%20Started/) notebooks. These are intended to be very simple step-by-step guides to show the basic functionality of PyBaMM. For more detailed notebooks, please see the examples listed below.

# Examples

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pybamm-team/PyBaMM/master)

This page contains a number of examples showing how to use PyBaMM.

Each example was created as a _Jupyter notebook_ (http://jupyter.org/).
These notebooks can be downloaded and used locally by running
```
$ jupyter notebook
```
from your local PyBaMM repository, or used online through [Binder](https://mybinder.org/v2/gh/pybamm-team/PyBaMM/master), or you can simply copy/paste the relevant code.

## Using PyBaMM

The easiest way to start with PyBaMM is by running and comparing some of the inbuilt models:
- [Run the Single Particle Model (SPM)](./models/SPM.ipynb)
- [Compare models](./models/lead-acid.ipynb)
- [Comparison with COMSOL](./compare-comsol-discharge-curve.ipynb)

It is also easy to add new models or change the setting that are used:
- [Add a model (documentation)](https://pybamm.readthedocs.io/en/latest/tutorials/add-model.html)
- [Add a model (example)](./create-model.ipynb)
- [Change model options](./using-model-options_thermal-example.ipynb)
- [Using submodels](./using-submodels.ipynb)
- [Change the settings](./change-settings.ipynb) (parameters, spatial method or solver)
- [Change the applied current](./change-input-current.ipynb)

For more advanced usage, new sets of parameters, spatial methods and solvers can be added:
- [Add parameters](https://pybamm.readthedocs.io/en/latest/tutorials/add-parameter-values.html)
- [Add a spatial method](https://pybamm.readthedocs.io/en/latest/tutorials/add-spatial-method.html)
- [Add a solver](https://pybamm.readthedocs.io/en/latest/tutorials/add-solver.html)



## Expression tree structure

PyBaMM is built around an expression tree structure.

- [The expression tree notebook](expression_tree/expression-tree.ipynb) explains how this works, from model creation to solution. 
- [The broadcast notebook](expression_tree/broadcasts.ipynb) explains the different types of broadcast. 

The following notebooks are specific to different stages of the PyBaMM pipeline, such as choosing a model, spatial method, or solver.

### Models

The following models are implemented and can easily be used or [compared](./models/lead-acid.ipynb). We always welcome [new models](https://pybamm.readthedocs.io/en/latest/tutorials/add-model.html)!

#### Lithium-ion models

- [Single-Particle Model](./models/SPM.ipynb)
- [Single-Particle Model with electrolyte](./models/SPMe.ipynb)
- [Doyle-Fuller-Newman Model](./models/DFN.ipynb)

#### Lead-acid models

- [Full porous-electrode](https://pybamm.readthedocs.io/en/latest/source/models/lead_acid/full.html)
- [Leading-Order Quasi-Static](https://pybamm.readthedocs.io/en/latest/source/models/lead_acid/loqs.html)
- [Composite](https://pybamm.readthedocs.io/en/latest/source/models/lead_acid/composite.html)

### Spatial Methods

The following spatial methods are implemented
- [Finite Volumes](./spatial_methods/finite-volumes.ipynb)
- Finite Elements (only for 2D current collector domains)

See [here](https://pybamm.readthedocs.io/en/latest/tutorials/add-spatial-method.html) for instructions on adding new spatial methods.

### Solvers

The following solvers are implemented
- Scipy ODE solver
- [Scikits ODE solver](./solvers/scikits-ode-solver.ipynb)
- [Scikits DAE solver](./solvers/scikits-dae-solver.ipynb)
- CasADi DAE solver

See [here](https://pybamm.readthedocs.io/en/latest/tutorials/add-solver.html) for instructions on adding new solvers.
