.. _CONTRIBUTING.md: https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md


Adding a Solver
===============

As with any contribution to PyBaMM, please follow the workflow in CONTRIBUTING.md_.
In particular, start by creating an issue to discuss what you want to do - this is a good way to avoid wasted coding hours!

The role of solvers
-------------------

All models in PyBaMM are implemented as `expression trees <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/expression-tree.ipynb>`_.
After the model has been created, parameters have been set, and the model has been discretised, the model is now a linear algebra object with the following attributes:

model.concatenated_rhs
  A :class:`pybamm.Symbol` nodes that can be evaluated at a state (``t``, ``y``) and returns the value of all the differential equations at that state, concatenated into a single vector
model.concatenated_algebraic
  A :class:`pybamm.Symbol` nodes that can be evaluated at a state (``t``, ``y``) and returns the value of all the algebraic equations at that state, concatenated into a single vector
model.concatenated_initial_conditions
  A numpy array of initial conditions for all the differential and algebraic equations, concatenated into a single vector
model.events
  A list of :class:`pybamm.Symbol` nodes representing events at which the solver should terminate. Specifically, the solver should terminate when any of the events in ``model.events`` evaluate to zero

The role of solvers is to solve a model at a given set of time points, returning a vector of times ``t`` and a matrix of states ``y``.

Base solver classes vs specific solver classes
----------------------------------------------

There is one general base solver class, :class:`pybamm.BaseSolver`, and two specialised base classes, :class:`pybamm.OdeSolver` and :class:`pybamm.DaeSolver`. The general base class simply sets up some useful solver properties such as tolerances. The specialised base classes implement a method :meth:`self.solve()` that solves a model at a given set of time points.

The ``solve`` method unpacks the model, simplifies it by removing extraneous operations, (optionally) creates or calls the mass matrix and/or jacobian, and passes the appropriate attributes to another method, called ``integrate``, which does the time-stepping. The role of specific solver classes is simply to implement this ``integrate`` method for an arbitrary set of derivative function, initial conditions etc.

The base DAE solver class also computes a consistent set of initial conditions for the algebraic equations, using ``model.concatenated_initial_conditions`` as an initial guess.

Implementing a new solver
-------------------------

To add a new solver (e.g. My Fast DAE Solver), first create a new file (``my_fast_dae_solver.py``) in ``pybamm/solvers/``,
with a single class that inherits from either :class:`pybamm.OdeSolver` or :class:`pybamm.DaeSolver`, depending on whether the new solver can solve DAE systems. For example:

.. code-block:: python

    def MyFastDaeSolver(pybamm.DaeSolver):

Also add the class to ``pybamm/__init__.py``:

.. code-block:: python

    from .solvers.my_fast_dae_solver import MyFastDaeSolver

You can then start implementing the solver by adding the ``integrate`` function to the class (the interfaces are slightly different for an ODE Solver and a DAE Solver, see :meth:`pybamm.OdeSolver.interface` vs :meth:`pybamm.DaeSolver.interface`)

For an example of an existing solver implementation, see the Scikits DAE solver
`API docs <https://pybamm.readthedocs.io/en/latest/source/solvers/scikits_solvers.html>`_
and
`notebook <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/solvers/scikits-dae-solver.ipynb>`_.

Unit tests for the new class
----------------------------

For the new solver to be added to PyBaMM, you must add unit tests to demonstrate that it behaves as expected
(see, for example, the `Scikits solver tests <https://github.com/pybamm-team/PyBaMM/blob/master/tests/unit/test_solvers/test_scikits_solvers.py>`_).
The best way to get started would be to create a file ``test_my_fast_solver.py`` in ``tests/unit/test_solvers/`` that performs at least the
following checks:

- The ``integrate`` method works on a simple ODE/DAE model with/without jacobian, mass matrix and/or events as appropriate
- The ``solve`` method works on a simple model (in theory, if the ``integrate`` method works then the ``solve`` method should always work)

If the solver is expected to converge in a certain way as the time step is changed, you could also add a convergence test in ``tests/convergence/solvers/``.

Test on the models
------------------

In theory, any existing model can now be solved using `MyFastDaeSolver` instead of their default solvers, with no extra work from here.
To test this, add something like the following test to one of the model test files
(e.g. `DFN <https://github.com/pybamm-team/PyBaMM/blob/master/tests/unit/test_models/test_lithium_ion/test_lithium_ion_dfn.py>`_):

.. code-block:: python

    def test_my_fast_solver(self):
        model = pybamm.lithium_ion.DFN()
        solver = pybamm.MyFastDaeSolver()
        modeltest = tests.StandardModelTest(model, solver=solver)
        modeltest.test_all()

This will check that the model can run with the new solver (but not that it gives a sensible answer!).

Once you have performed the above checks, you are almost ready to merge your code into the core PyBaMM - see
`CONTRIBUTING.md workflow <https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md#c-merging-your-changes-with-pybamm>`_
for how to do this.
