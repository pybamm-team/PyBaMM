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