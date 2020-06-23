
.. _CONTRIBUTING.md: https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md


Adding a Model
==============

As with any contribution to PyBaMM, please follow the workflow in CONTRIBUTING.md_.
In particular, start by creating an issue to discuss what you want to do - this is a good way to avoid wasted coding hours!

We aim here to provide an overview of how a new model is entered into PyBaMM in a form which can be eventually merged into the master branch of the PyBaMM project. However, we recommend that you first read through the notebook: `create a model <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/create-model.ipynb>`_, which goes step-by-step through the procedure for creating a model. Once you understand that procedure, you can then formalise your model following the outline provided here.

The role of models
------------------

One of the main motivations for PyBaMM is to allow for new models of batteries to be easily be added, solved, tested, and compared without requiring a detailed knowledge of sophisticated numerical methods. It has therefore been our focus to make the process of adding a new model as simple as possible. To achieve this, all models in PyBaMM are implemented as `expression trees <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/expression-tree.ipynb>`_, which abstract away the details of computation.

The fundamental building blocks of a PyBaMM expression tree are :class:`pybamm.Symbol`. There are different types of :class:`pybamm.Symbol`: :class:`pybamm.Variable`, :class:`pybamm.Parameter`, :class:`pybamm.Addition`, :class:`pybamm.Multiplication`, :class:`pybamm.Gradient` etc which have been created so that each component of a model written out in PyBaMM mirrors exactly the written mathematics. For example, the expression:

.. math::
    \nabla \cdot \left(D(c) \nabla c \right) + a F j

is simply written as

.. code-block:: python

    div(D(c) * grad(c)) + a * F * j

within PyBaMM. A model in PyBaMM is essentially an organised collection of
expression trees.

Implementing a new model
------------------------

To add a new model (e.g. My New Model), first create a new file (``my_new_model.py``) in ``pybamm/models`` (or the relevant subdirectory).
In this file create a new class which inherits from :class:`pybamm.BaseModel`
(or :class:`pybamm.LithiumIonBaseModel` if you are modelling a full lithium-ion battery or :class:`pybamm.LeadAcidBaseModel` if you are modelling a full lead acid battery):

.. code-block:: python

    class MyNewModel(pybamm.BaseModel):
        def

and add the class to ``pybamm/__init__.py``:

.. code-block:: python

    from .models.my_new_model import MyNewModel

(this line will be slightly different if you created your model in a subdirectory of models). Within your new class :class:`MyNewModel`, first create an initialisation function which calls the initialisation function of the parent class

.. code-block:: python

    def __init__(self):
        super().__init__()

Within the initialisation function of :class:`MyNewModel` you must then define the following attributes:

- ``self.rhs``
- ``self.algebraic``
- ``self.boundary_conditions``
- ``self.initial_conditions``
- ``self.variables``

You may also optionally also provide:

- ``self.events``
- ``self.default_geometry``
- ``self.default_solver``
- ``self.default_spatial_methods``
- ``self.default_submesh_types``
- ``self.default_var_pts``
- ``self.default_parameter_values``

We will go through each of these attributes in turn here for completeness but refer the user to the API documentation or example notebooks (create-model.ipnb) if further details are required.

Governing equations
~~~~~~~~~~~~~~~~~~~
The governing equations which can either be parabolic or elliptic are entered into the
``self.rhs`` and ``self.algebraic`` dictionaries, respectively. We associate each governing equation with a subject variable, which is the variable that is found when
the equation is solved. We use this subject variable as the key of the dictionary. For parabolic equations, we rearrange the equation so that the time derivative of the subject variable is the only term on the left hand side of the equation. We then simply write the resulting right hand side into the ``self.rhs`` dictionary with the subject variable as the key. For elliptic equations, we rearrange so that the left hand side of the equation if zero and then write the right hand side into the ``self.algebraic`` dictionary in the same way. The resulting dictionary should look like:

.. code-block:: python

    self.rhs = {parabolic_var1: parabolic_rhs1, parabolic_var2: parabolic_rhs2, ...}
    self.algebraic = {elliptic_var1: elliptic_rhs1, elliptic_var2: elliptic_rhs2, ...}

Boundary conditions
~~~~~~~~~~~~~~~~~~~
Boundary conditions on a variable can either be Dirichlet or Neumann (support for mixed boundary conditions will be added at a later date). For a variable :math:`c` on a one dimensional domain with a Dirichlet condition of :math:`c=1` on the left boundary and
a Neumann condition of :math:`\nabla c = 2` on the right boundary, we then have:

.. code-block:: python

    self.boundary_conditions = {c: {"left": (1, "Dirichlet"), "right": (2, "Neumann")}}


Initial conditions
~~~~~~~~~~~~~~~~~~
For a variable :math:`c` that is initially at a value of :math:`c=1`, the initial condition is included written into the model as

.. code-block:: python

    self.initial_conditions = {c: 1}

Output variables
~~~~~~~~~~~~~~~~
PyBaMM allows users to create combinations of symbols to output from their model.
For example, we might wish to output the terminal voltage which is given by
:math:`V = \phi_{s,p}|_{x=1} - \phi_{s,n}|_{x=0}`. We would first define the voltage symbol :math:`V` and then include it into the output variables dictionary in the form:

.. code-block:: python

    self.variables = {"Terminal voltage [V]": V}

Note that we indicate that the quanitity is dimensional by including the dimensions, Volts in square brackets. We do this to distinguish between dimensional and dimensionless outputs which may otherwise share the same name.

Note that if your model inherits from :class:`pybamm.StandardBatteryBaseModel`, then there is a standard set of output parameters which is enforced to ensure consistency across models so that they can be easily tested and compared.


Events
~~~~~~
Events can be added to stop computation when the event occurs. For example, we may wish to terminate our computation when the terminal voltage :math:`V` reaches some minimum voltage during a discharge :math:`V_{min}`. We do this by adding the following to the events dictionary:

.. code-block:: python

    self.events["Minimum voltage cut-off"] = V - V_min

Events will stop the solver whenever they return 0.

Setting defaults
~~~~~~~~~~~~~~~~
It can be useful for testing, and quickly running a model to have a default setup. Each of the defaults listed above should adhere to the API requirements but in short, we require ``self.default_geometry`` to be a dictionary of the right format (see :meth:`pybamm.battery_geometry`), ``self.default_solver`` to be an instance of :class:`pybamm.BaseSolver`, and
``self.default_parameter_values`` to be an instance of :class:`pybamm.ParameterValues`. We also require that ``self.default_submesh_types`` is a dictionary with keys which are strings corresponding to the regions of the battery (e.g. "negative electrode") and values which are an instance of :class:`pybamm.SubMesh1D`. The ``self.default_spatial_methods`` attribute is also required to be a dictionary with keys corresponding to the regions of the battery but with values which are an instance of
:class:`pybamm.SpatialMethod`. Finally, ``self.default_var_pts`` is required to be a dictionary with keys which are an instance of :class:`pybamm.SpatialVariable` and values which are integers.


Using submodels
~~~~~~~~~~~~~~~
The inbuilt models in PyBaMM do not add all the model attributes within their own file. Instead, they make use of inbuilt submodel (a particle model, an electrolyte model, etc). There are two main reasons for this. First, the code in the submodels can then be used by multiple models cutting down on repeated code. This makes it easier to maintain the codebase because fixing an issue in a submodel fixes that issue everywhere the submodel is called (instead of having to track down the issue in every model). Secondly, it allows for the user to easily switch a submodel out for another and study the effect. For example, we may be using standard diffusion in the particles but decide that we would like to switch in particles which are phase separating. With submodels all we need to do is switch the submodel instead of re-writing the whole sections of the model. Submodel contributions are highly encouraged so where possible, try to divide your model into submodels.

In addition to calling submodels, common sets of variables and parameters found in
lithium-ion and lead acid batteries are provided in
`standard_variables.py`,
`standard_parameters_lithium_ion.py`,
`standard_parameters_lead_acid.py`,
`electrical_parameters.py`,
`geometric_parameters.py`,
and `standard_spatial_vars.py`
which we encourage use of to save redefining the same parameters and variables in
every model and submodel.


Unit tests for a MyNewModel
---------------------------
We strongly recommend testing your model to ensure that it is behaving correctly. To do
this, first create a new file ``test_my_new_model.py`` within
``tests/integration/test_models`` (or the appropriate subdirectory). Within this file,
add the following code

.. code-block:: python

    import pybamm
    import unittest

    class TestMyNewModel(unittest.TestCase):
        def my_first_test(self):
            # add test here

    if __name__ == "__main__":
        print("Add -v for more debug output")
        import sys

        if "-v" in sys.argv:
            debug = True
        unittest.main()

We can now add functions such as :meth:`my_first_test` to :class:`TestMyNewModel` which run specific tests. As a first test, we recommend you make use of :class:`tests.StandardModelTest` which runs a suite of basic tests. If your new model is a full model of a battery and therefore inherits from :class:`pybamm.StandardBatteryBaseModel` then :class:`tests.StandardBatteryTest` will also check the set of outputs are producing reasonable behaviour.

Please see the tests of the inbuilt models to get a further idea of how to test the your model.
