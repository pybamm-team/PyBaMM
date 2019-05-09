
.. _CONTRIBUTING.md: https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md


Adding a Model
==============

As with any contribution to PyBaMM, please follow the workflow in CONTRIBUTING.md_.
In particular, start by creating an issue to discuss what you want to do - this is a good way to avoid wasted coding hours!


The role of models
------------------

One of the main motivations for PyBaMM is to allow for new models of batteries to be easily be added, solved, tested, and compared without requiring a detailed knowledge of sophisticated numerical methods. It has therefore been our focus to make the process of adding a new model as simple as possible. To achieve this, all models in PyBaMM are implemented as `expression trees <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/expression-tree.ipynb>`_, which abstract away the details of computation. 

The fundamental building blocks of a PyBaMM expression tree are Symbols. There are different types of Symbol: Variables, Parameters, Addition, Multiplication, Gradient etc which have been created so that each component of a model written out in PyBaMM mirrors exactly the written mathematics. For example, the expression:

.. math::
    \nabla \cdot \left(D(c) \nabla c \right) + a F j

is simply written as

.. code-block:: python

    div(D(c) * grad(c)) + a * F * j

within PyBaMM. A model basically consists a set of expression trees which represent the different components of the model: the governing equations, the boundary conditions, the initial conditions, and useful outputs (e.g. the terminal voltage).  


.. After it has been created and parameters have been set, the model is passed to the :class:`pybamm.Discretisation` class,
which converts it into a linear algebra form.
For example, the object:
    .. D = pybamm.Parameter("Diffusivity")
    .. c = pybamm.Variable("Concentration")


.. might get converted to a Matrix-Vector multiplication:

.. .. code-block:: python

..     Matrix(100,100) @ y[0:100]

.. (in Python 3.5+, @ means matrix multiplication, while * is elementwise product).
.. The :class:`pybamm.Discretisation` class is a wrapper that iterates through the different parts of the model, performing the trivial conversions (e.g. Addition --> Addition),
.. and calls upon spatial methods to perform the harder conversions (e.g. grad(u) --> Matrix * StateVector, SpatialVariable --> Vector, etc).

.. Hence SpatialMethod classes only need to worry about the specific conversions, and :class:`pybamm.Discretisation` deals with the rest.

.. Implementing a new spatial method
.. ---------------------------------

.. To add a new spatial method (e.g. My Fast Method), first create a new file (``my_fast_method.py``) in ``pybamm/spatial_methods/``,
.. with a single class that inherits from :class:`pybamm.SpatialMethod`, such as:

.. .. code-block:: python

..     def MyFastMethod(pybamm.SpatialMethod):

.. and add the class to ``pybamm/__init__.py``:

.. .. code-block:: python

..     from .spatial_methods.my_fast_method import MyFastMethod

.. You can then start implementing the spatial method by adding functions to the class.
.. In particular, any spatial method *must* have the following functions (from the base class :class:`pybamm.SpatialMethod`):

.. - :meth:`pybamm.SpatialMethod.gradient`
.. - :meth:`pybamm.SpatialMethod.divergence`
.. - :meth:`pybamm.SpatialMethod.integral`
.. - :meth:`pybamm.SpatialMethod.indefinite integral`

.. Optionally, a new spatial method can also overwrite the default behaviour for the following functions:

.. - :meth:`pybamm.SpatialMethod.spatial_variable`
.. - :meth:`pybamm.SpatialMethod.broadcast`
.. - :meth:`pybamm.SpatialMethod.mass_matrix`
.. - :meth:`pybamm.SpatialMethod.process_binary_operators`
.. - :meth:`pybamm.SpatialMethod.boundary_value`

.. For an example of an existing spatial method implementation, see the Finite Volume
.. `API docs <https://pybamm.readthedocs.io/en/latest/source/spatial_methods/finite_volume.html>`_
.. and
.. `notebook <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/spatial_methods/finite-volumes.ipynb>`_.

.. Unit tests for the new class
.. ----------------------------

.. For the new spatial method to be added to PyBaMM, you must add unit tests to demonstrate that it behaves as expected
.. (see, for example, the `Finite Volume unit tests <https://github.com/pybamm-team/PyBaMM/blob/master/tests/unit/test_spatial_methods/test_finite_volume.py>`_).
.. The best way to get started would be to create a file ``test_my_fast_method.py`` in ``tests/unit/test_spatial_methods/`` that performs at least the
.. following checks:

.. - Operations return objects that have the expected shape
.. - Standard operations behave as expected, e.g. (in 1D) grad(x^2) = 2*x, integral(sin(x), 0, pi) = 2
.. - (more advanced) make sure that the operations converge at the correct rate to known analytical solutions as you decrease the grid size

.. Test on the models
.. ------------------

.. In theory, any existing model can now be discretised using ``MyFastMethod`` instead of their default spatial methods, with no extra work from here.
.. To test this, add something like the following test to one of the model test files
.. (e.g. `DFN <https://github.com/pybamm-team/PyBaMM/blob/master/tests/unit/test_models/test_lithium_ion/test_lithium_ion_dfn.py>`_):

.. .. code-block:: python

..     def test_my_fast_method(self):
..         model = pybamm.lithium_ion.DFN()
..         spatial_methods = {
..             "macroscale": pybamm.MyFastMethod,
..             "negative particle": pybamm.MyFastMethod,
..             "positive particle": pybamm.MyFastMethod,
..         }
..         modeltest = tests.StandardModelTest(model, spatial_methods=spatial_methods)
..         modeltest.test_all()

.. This will check that the model can run with the new spatial method (but not that it gives a sensible answer!).

.. Once you have performed the above checks, you are almost ready to merge your code into the core PyBaMM - see
.. `CONTRIBUTING.md workflow <https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md#c-merging-your-changes-with-pybamm>`_
.. for how to do this.
