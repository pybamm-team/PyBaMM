# Fundamentals

PyBaMM (Python Battery Mathematical Modelling) is an open-source battery simulation package
written in Python. Our mission is to accelerate battery modelling research by
providing open-source tools for multi-institutional, interdisciplinary collaboration.
Broadly, PyBaMM consists of

1. a framework for writing and solving systems of differential equations,
2. a library of battery models and parameters, and
3. specialized tools for simulating battery-specific experiments and visualizing the results.

Together, these enable flexible model definitions and fast battery simulations, allowing users to
explore the effect of different battery designs and modeling assumptions under a variety of operating scenarios.

> **NOTE**: This user-guide is a work-in-progress, we hope that this brief but incomplete overview will be useful to you.

## Core framework

The core of the framework is a custom computer algebra system to define mathematical equations,
and a domain specific modeling language to combine these equations into systems of differential equations
(usually partial differential equations for variables depending on space and time).
The [expression tree](https://github.com/pybamm-team/PyBaMM/blob/develop/examples/notebooks/expression_tree/expression-tree.ipynb) example gives an introduction to the computer algebra system, and the [Getting Started](https://github.com/pybamm-team/PyBaMM/tree/develop/examples/notebooks/Getting%20Started) tutorials
walk through creating models of increasing complexity.

Once a model has been defined symbolically, PyBaMM solves it using the Method of Lines. First, the equations are discretised in the spatial dimension, using the finite volume method. Then, the resulting system is solved using third-party numerical solvers. Depending on the form of the model, the system can be ordinary differential equations (ODEs) (if only `model.rhs` is defined), or algebraic equations (if only `model.algebraic` is defined), or differential-algebraic equations (DAEs) (if both `model.rhs` and `model.algebraic` are defined). Jupyter notebooks explaining the solvers can be found [here](https://github.com/pybamm-team/PyBaMM/tree/develop/examples/notebooks/solvers).

## Model and Parameter Library

PyBaMM contains an extensive library of battery models and parameters.
The bulk of the library consists of models for lithium-ion, but there are also some other chemistries (lead-acid, lithium metal).
Models are first divided broadly into common named models of varying complexity, such as the single particle model (SPM) or Doyle-Fuller-Newman model (DFN).
Most options can be applied to any model, but some are model-specific (an error will be raised if you attempt to set an option is not compatible with a model).
See [](base_battery_model) for a list of options.

The parameter library is simply a collection of python files each defining a complete set of parameters
for a particular battery chemistry, covering all major lithium-ion chemistries (NMC, LFP, NCA, ...).
External parameter sets can be linked using entry points (see [](parameter_sets)).

## Battery-specific tools

One of PyBaMM's unique features is the `Experiment` class, which allows users to define synthetic experiments using simple instructions in English

```python
pybamm.Experiment(
    [
        ("Discharge at C/10 for 10 hours or until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        "Rest for 1 hour")
    ]
    * 3,
)
```

The above instruction will conduct a standard discharge / rest / charge / rest cycle three times, with a 10 hour discharge and 1 hour rest at the end of each cycle.

The `Simulation` class handles simulating an `Experiment`, as well as calculating additional outputs such as capacity as a function of cycle number. For example, the following code will simulate the experiment above and plot the standard output variables:

```python
import pybamm
import matplotlib.pyplot as plt

# load model and parameter values
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment)
solution = sim.solve()
solution.plot()
```

Finally, PyBaMM provides cusotm visualization tools:

- [](quick_plot): for easily plotting simulation outputs in a grid, including comparing multiple simulations
- [](plot_voltage_components): for plotting the component overpotentials that make up a voltage curve

Users are not limited to these tools and can plot the output of a simulation solution by accessing the underlying numpy array for the solution variables as

```python
solution["variable name"].data
```

and using the plotting library of their choice.
