Fundamentals
============

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

.. note::
    This user-guide is a work-in-progress, we hope it is helpful in this state but it is definitely incomplete and may contain inaccuracies.

Core framework
~~~~~~~~~~~~~~
The core of the framework is a custom computer algebra system to define mathematical equations,
and a domain specific modeling language to combine these equations into systems of differential equations
(usually partial differential equations for variables depending on space and time).
The [expression tree] example gives an introduction to the computer algebra system, and the [Getting Started] tutorials
walk through creating models of increasing complexity.

Once a model has been defined symbolically, PyBaMM solves it using the Method of Lines. First, the equations are discretised in the spatial dimension (e.g. using the [finite volume] discretisation). Then, the resulting system is solved using third-party numerical solvers. Depending on the form of the model, the system can be ordinary differential equations (ODEs) (if only `model.rhs` is defined), or algebraic equations (if only `model.algebraic` is defined), or differential-algebraic equations (DAEs) (if both `model.rhs` and `model.algebraic` are defined). [Solver examples].

Model and Parameter Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~
PyBaMM contains an extensive library of battery models and parameters.
The bulk of the library consists of models for lithium-ion, but there are also some other chemistries (lead-acid, lithium metal).
Models are first divided broadly into common named models of varying complexity, such as the `single particle model` (SPM) or `Doyle-Fuller-Newman` model (DFN).
For each model, . Most options can be applied to any model, but some are model-specific (an error will be raised if you attempt to set an option is not compatible with a model).
See ... for a list of options.

The parameter library is simply a collection of python files each defining a complete set of parameters
for a particular battery chemistry, covering all major lithium-ion chemistries (NMC, LFP, NCA, ...).
External parameter sets can be linked using [entry points]().

Battery-specific tools
~~~~~~~~~~~~~~~~~~~~~~
One of PyBaMM's unique features is the `Experiment` class, which allows users to define synthetic experiments using simple English
.. code-block::
    pybamm.Experiment()


The `Simulation` class handles simulating an `Experiment`, as well as calculating additional outputs such as capacity as a function of cycle number.

Finally, PyBaMM provides cusotm visualization tools:
- The `QuickPlot` class, for easily plotting simulation outputs in a grid, including comparing multiple simulations
- The `plot_voltage_components` function, for plotting the component overpotentials that make up a voltage curve

Users are not limited to these tools and can plot the output of a simulation solution by accessing variables as `solution["variable name"].data` and using plotting libraries of their choice.