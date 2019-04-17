.. _CONTRIBUTING.md: https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md


Adding a Solver
===============

As with any contribution to PyBaMM, please follow the workflow in CONTRIBUTING.md_.
In particular, start by creating an issue to discuss what you want to do - this is a good way to avoid wasted coding hours!

The role of solvers
-------------------

All models in PyBaMM are implemented as `expression trees <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/expression-tree.ipynb>`_.
After the model has been created, parameters have been set, and the model has been discretised, the model is now a linear algebra object with the following attributes:

model.rhs
  A :class:`pybamm.Symbol` that can be evaluated at a state (``t``, ``y``) and returns the value of all the differential equations at that state, concatenated into a single vector
model.algebraic
  A :class:`pybamm.Symbol` that can be evaluated at a state (``t``, ``y``) and returns the value of all the algebraic equations at that state, concatenated into a single vector
model.concatenated_initial_conditions


The role



The base solver class
---------------------



Implementing a new solver
---------------------------------

To add a new solver (e.g. My Fast DAE Solver), first create a new file (``my_fast_dae_solver.py``) in ``pybamm/solvers``,
with a single class that inherits from either :class:`pybamm.OdeSolver` or :class:`pybamm.DaeSolver`, depending on whether the new solver can solve DAE systems. For example:

    def MyFastDaeSolver(pybamm.DaeSolver):

Also add the class to `pybamm/__init__.py`:

    from .solvers.my_fast_solver import MyFastSolver

You can then start implementing the solver by adding functions to the class.
In particular, any solver *must* have the following functions (from the base class :class:`pybamm.SpatialMethod`):

- :meth:`pybamm.SpatialMethod.spatial_variable`
- :meth:`pybamm.SpatialMethod.gradient`
- :meth:`pybamm.SpatialMethod.divergence`
- :meth:`pybamm.SpatialMethod.integral`
- :meth:`pybamm.SpatialMethod.indefinite integral`
- :meth:`pybamm.SpatialMethod.boundary_value`

Optionally, a new solver can also overwrite the default behaviour for the following functions:

- :meth:`pybamm.SpatialMethod.broadcast`
- :meth:`pybamm.SpatialMethod.mass_matrix`
- :meth:`pybamm.SpatialMethod.compute_diffusivity`

For an example of an existing solver implementation, see the Scikits ODEs solver
`API docs <https://pybamm.readthedocs.io/en/latest/source/solvers/scikits_solvers.html>`_.
and
`notebook <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/solvers/scikits-solvers.ipynb>`_.

Unit tests for the new class
----------------------------

For the new solver to be added to PyBaMM, you must add unit tests to demonstrate that it behaves as expected
(see, for example, the `Finite Volume unit tests <https://github.com/pybamm-team/PyBaMM/blob/master/tests/unit/test_solvers/test_finite_volume.py>`_).
The best way to get started would be to create a file `test_my_fast_method.py` in `tests/unit/test_solvers/` that performs at least the
following checks:

- Operations return objects that have the expected shape
- Standard operations behave as expected, e.g. (in 1D) grad(x^2) = 2*x, integral(sin(x), 0, pi) = 2
- (more advanced) make sure that the operations converge at the correct rate to known analytical solutions as you decrease the grid size

Test on the models
------------------

In theory, any existing model can now be solved using `MyFastDaeSolver` instead of their default solvers, with no extra work from here.
To test this, add something like the following test to one of the model test files
(e.g. `DFN <https://github.com/pybamm-team/PyBaMM/blob/master/tests/unit/test_models/test_lithium_ion/test_lithium_ion_dfn.py>`_):

    def test_my_fast_solver(self):
        model = pybamm.lithium_ion.DFN()
        solver = pybamm.MyFastDaeSolver()

        modeltest = tests.StandardModelTest(model, solver=solver)
        modeltest.test_all()

This will check that the model can run with the new solver (but not that it gives a sensible answer!).

Once you have performed the above checks, you are almost ready to merge your code into the core PyBaMM - see
`CONTRIBUTING.md workflow <https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md#c-merging-your-changes-with-pybamm>`_
for how to do this.
