----------
Public API
----------

.. module:: pybamm
    :noindex:

PyBaMM is a Python package for mathematical modelling and simulation of battery systems. The main classes and functions that are intended to be used by the user are described in this document.
For a more detailed description of the classes and methods, see the [API documentation](api_docs).

Available PyBaMM models
-----------------------

PyBaMM includes a number of pre-implemented models, which can be used as they are or modified to suit your needs. The main models are:

- :class:`lithium_ion.SPM`: Single Particle Model
- :class:`lithium_ion.SPMe`: Single Particle Model with Electrolyte
- :class:`lithium_ion.DFN`: Doyle-Fuller-Newman

The behaviour of the models can be modified by passing in an :class:`BatteryModelOptions` object when creating the model.

Simulations
-----------

:class:`Simulation` is a class that automates the process of setting up a model and solving it, and acts as the highest-level API to PyBaMM.
Pass at least a :class:`BaseModel` object, and optionally the experiment, solver, parameter values, and geometry objects described below to the :class:`Simulation` object.
Any of these optional arguments not provided will be supplied by the defaults specified in the model.

Parameters
----------

PyBaMM models are parameterised by a set of parameters, which are stored in a :class:`ParameterValues` object. This object acts like a Python dictionary with a few extra PyBaMM specific features and methods.
Parameters in a model are represented as either :class:`Parameter` objects or :class:`FunctionParameter` objects, and the values in the :class:`ParameterValues` object are inserted into the model when it is set up for solving
to replace the :class:`Parameter` and :class:`FunctionParameter` objects. The values in the :class:`ParameterValues` object can be scalars, Python functions or expressions of type :class:`Symbol`.

Experiments
-----------

An :class:`Experiment` object represents an experimental protocol that can be used to simulate the behaviour of a battery. The particular protocol can be provided as a Python string, or as a sequences of
:class:`step.BaseStep` objects.

Solvers
-------

The two main solvers in PyBaMM are the :class:`CasadiSolver` and the :class:`IDAKLUSolver`. Both are wrappers around the Sundials suite of solvers, but the :class:`CasadiSolver` uses the CasADi library
whereas the :class:`IDAKLUSolver` is PyBaMM specific. Both solvers have many options that can be set to control the solver behaviour, see the documentation for each solver for more details.

When a model is solved, the solution is returned as a :class:`Solution` object.

Plotting
--------

A solution object can be plotted using the :meth:`Solution.plot` or :meth:`Simulation.plot` methods, which returns a :class:`QuickPlot` object.
Note that the arguments to the plotting methods of both classes are the same as :class:`QuickPlot`.

Other plotting functions are the :func:`plot_voltage_components` and :func:`plot_summary_variables` functions, which correspond to the similarly named methods of the :class:`Solution` and :class:`Simulation` classes.

Writing PyBaMM models
---------------------

Each PyBaMM model, and the custom models written by users, are written as a set of expressions that describe the model. Each of the expressions is a subclass of the :class:`Symbol` class, which represents a mathematical expression.

If you wish to create a custom model, you can use the :class:`BaseModel` class as a starting point.


Discretisation
--------------

Each PyBaMM model contains continuous operators that must be discretised before they can be solved. This is done using a :class:`Discretisation` object, which takes a :class:`Mesh` object and a dictionary of :class:`SpatialMethod` objects.

Logging
-------

PyBaMM uses the Python logging module to log messages at different levels of severity. Use the :func:`pybamm.set_logging_level` function to set the logging level for PyBaMM.
