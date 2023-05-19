Experiment step functions
=========================

The following functions can be used to define steps in an experiment.

.. autofunction:: pybamm.experiment.string

.. autofunction:: pybamm.experiment.current

.. autofunction:: pybamm.experiment.voltage

.. autofunction:: pybamm.experiment.power

.. autofunction:: pybamm.experiment.resistance

These functions return the following step class, which is not intended to be used
directly:

.. autoclass:: pybamm.experiment._Step
    :members: