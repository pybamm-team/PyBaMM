# Getting started

For new users we recommend the [Getting Started](./Getting%20Started/) notebooks. These are intended to be very simple step-by-step guides to show the basic functionality of PyBaMM. For more detailed notebooks, please see the examples listed below.

# Examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/develop/)

This page contains a number of examples showing how to use PyBaMM.

Each example was created as a _Jupyter notebook_ (http://jupyter.org/).
These notebooks can be downloaded and used locally by running

```
$ jupyter notebook
```

from your local PyBaMM repository or used online through [Google Colab](https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/develop/). Alternatively, you can simply copy/paste the relevant code.

## Using PyBaMM

The easiest way to start with PyBaMM is by running and comparing some of the inbuilt models:

- [Run the Single Particle Model (SPM)](./models/SPM.ipynb)
- [Compare models](./models/lead-acid.ipynb)
- [Comparison with COMSOL](./models/compare-comsol-discharge-curve.ipynb)

It is also easy to add new models or change the setting that are used:

- [Add a model (example)](./create-model.ipynb)
- [Change model options](./models/using-model-options_thermal-example.ipynb)
- [Using submodels](./using-submodels.ipynb)
- [Change the settings](./change-settings.ipynb) (parameters, spatial method or solver)
- [Change the applied current](./parameterization/change-input-current.ipynb)

## Expression tree structure

PyBaMM is built around an expression tree structure.

- [The expression tree notebook](expression_tree/expression-tree.ipynb) explains how this works, from model creation to solution.
- [The broadcast notebook](expression_tree/broadcasts.ipynb) explains the different types of broadcast.

The following notebooks are specific to different stages of the PyBaMM pipeline, such as choosing a model, spatial method, or solver.

### Models

Several battery models are implemented and can easily be used or [compared](./models/lead-acid.ipynb). The notebooks below show the solution of each individual model.

Once you are comfortable with the expression tree structure, a good starting point to understand the models in PyBaMM is to take a look at the [basic SPM](https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/models/full_battery_models/lithium_ion/basic_spm.py) and [basic DFN](https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/models/full_battery_models/lithium_ion/basic_dfn.py), since these define the entire model (variables, equations, initial and boundary conditions, events) in a single class and so are easier to understand. However, we recommend that you subsequently use the full models as they offer much greater flexibility for coupling different physical effects and visualising a greater range of variables.

#### Lithium-ion models

- [Single-Particle Model](./models/SPM.ipynb)
- [Single-Particle Model with electrolyte](./models/SPMe.ipynb)
- [Doyle-Fuller-Newman Model](./models/DFN.ipynb)

#### Lead-acid models

- [Full porous-electrode](https://pybamm.readthedocs.io/en/latest/source/api/models/lead_acid/full.html)
- [Leading-Order Quasi-Static](https://pybamm.readthedocs.io/en/latest/source/api/models/lead_acid/loqs.html)

### Spatial Methods

The following spatial methods are implemented

- [Finite Volumes](./spatial_methods/finite-volumes.ipynb) (1D only)
- Spectral Volumes (1D only)
- Finite Elements (only for 2D current collector domains)

### Solvers

The following notebooks show examples for generic ODE and DAE solvers. Several solvers are implemented in PyBaMM and we encourage users to try different ones to find the most appropriate one for their models.

- [ODE solver](./solvers/ode-solver.ipynb)
- [DAE solver](./solvers/dae-solver.ipynb)
