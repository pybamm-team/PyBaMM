Adding a Spatial Method
=======================

As with any contribution to PyBaMM, start by creating an issue to discuss what you want to do - this saves wasted coding hours

Creating a new Spatial Method class
-----------------------------------

To add a new Spatial Method (My Fast Method), first create a new file (`my_fast_method.py`) in ``pybamm/spatial_methods``,
with a single class that inherits from :class:`pybamm.SpatialMethod`, such as::

    def MyFastMethod(pybamm.SpatialMethod):

This class must implement the following operations from :class:`pybamm.SpatialMethod`:

- :meth:`pybamm.SpatialMethod.spatial_variable()`
- :meth:`pybamm.SpatialMethod.broadcast`
- :meth:`pybamm.SpatialMethod.gradient`
- :meth:`pybamm.SpatialMethod.divergence`
- :meth:`pybamm.SpatialMethod.integral`
- :meth:`pybamm.SpatialMethod.indefinite integral`

Optionally, a new spatial method can also overwrite the default behaviour for the following operations:

- :meth:`pybamm.SpatialMethod.boundary_value`
- :meth:`pybamm.SpatialMethod.mass_matrix`
- :meth:`pybamm.SpatialMethod.compute_diffusivity`

For an example of a spatial method implementation, see
`Finite Volume <https://github.com/pybamm-team/PyBaMM/tree/master/examples/notebooks>`_.

Unit tests for the new class
----------------------------

For the new spatial method to be added to PyBaMM, you must add unit tests to demonstrate that it behaves as expected.
(see, for example, the `Finite Volume unit tests <https://github.com/pybamm-team/PyBaMM/blob/master/tests/test_spatial_methods/test_finite_volume.py>`_).
The best way to get started would be to create a file `test_my_fast_method.py` in `tests/test_spatial_methods/` that performs the
following checks:

- Operations return objects that have the expected shape
- Standard operations behave as expected, e.g. (in 1D) grad(x^2) = 2*x, integral(sin(x), 0, pi) = 2
- (more advanced) make sure that the operations converge at the correct rate to known analytical solutions as you decrease the grid size

Test on the models
------------------

In theory, all existing models can now be discretised using `MyFastMethod` instead of their default spatial methods, with no extra work from here.
To test this, add something like the following test to one of the model test files
(e.g. `DFN <https://github.com/pybamm-team/PyBaMM/blob/master/tests/test_models/test_lithium_ion/test_lithium_ion_dfn.py>`_)::

    def test_my_fast_method(self):
        model = pybamm.lithium_ion.DFN()
        var = pybamm.standard_spatial_vars
        model.default_var_pts = {
            var.x_n: 3,
            var.x_s: 3,
            var.x_p: 3,
            var.r_n: 3,
            var.r_p: 3,
        }

        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

Housekeeping
------------

Finally:

- Add your spatial method to the API docs by copying and modifying `finite_volume.rst` as appropriate in `docs/source/spatial_methods`, and adding the appropriate line to the toctree in `docs/source/spatial_methods/index.rst`.
- Check that all the tests pass
- Create a Pull Request to merge your new spatial method into the core PyBaMM code.
