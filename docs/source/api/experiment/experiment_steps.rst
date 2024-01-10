Experiment step functions
=========================

The following functions can be used to define steps in an experiment.

.. autofunction:: pybamm.step.string

.. autofunction:: pybamm.step.current

.. autofunction:: pybamm.step.voltage

.. autofunction:: pybamm.step.power

.. autofunction:: pybamm.step.resistance

These functions return the following step class, which is not intended to be used
directly:

.. autoclass:: pybamm.step._Step
    :members:

Step terminations
-----------------

Standard step termination events are implemented by the following classes, which are
called when the termination is specified by a specific string. These classes can be
either be called directly or via the string format specified in the class docstring

.. autoclass:: pybamm.step.CrateTermination
    :members:

.. autoclass:: pybamm.step.CurrentTermination
    :members:

.. autoclass:: pybamm.step.VoltageTermination
    :members:

The following classes can be used to define custom terminations for an experiment
step:

.. autoclass:: pybamm.step.BaseTermination
    :members:

.. autoclass:: pybamm.step.CustomTermination
    :members:
