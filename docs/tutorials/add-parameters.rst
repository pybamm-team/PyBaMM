.. _CONTRIBUTING.md: https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md


Adding Parameter Values
=======================

As with any contribution to PyBaMM, please follow the workflow in CONTRIBUTING.md_.
In particular, start by creating an issue to discuss what you want to do - this is a good way to avoid wasted coding hours!

The role of parameters values
----------------------

All models in PyBaMM are implemented as `expression trees <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/expression-tree.ipynb>`_.
At the stage of creating a model, we use :class:`pybamm.Parameter` and :class:`pybamm.FunctionParameter` objects to represent parameters and functions respectively.

We then create a :class:`ParameterValues` class, using a specific set of parameters, to iterate through the model and replace any :class:`pybamm.Parameter` objects with a :class:`pybamm.Scalar` and any :class:`pybamm.FunctionParameter` objects with a :class:`pybamm.Function`.

For an example of how the parameter values work, see the
`parameter values notebook <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/spatial_methods/finite-volumes.ipynb>`_.

Adding a set of parameters values
---------------------------------

Parameter sets should be added as csv files in the appropriate chemistry folder in ``input/parameters/`` (add a new folder if no parameters exist for that chemistry yet).
The expected structure of the csv file is

+------------+------------+-----------+-----------+-----------+
| Name       | Value      | Units     | Reference | Notes     |
+============+============+===========+===========+===========+
| Example    | 13         | m.s-2     | bloggs2019| an example|
+------------+------------+-----------+-----------+-----------+

Empty lines, and lines starting with ``#``, will be ignored.

Adding a function
-----------------

Functions should be added as Python functions under a file with the same name in the appropriate chemistry folder in ``input/parameters/``.
These Python functions should be documented with references explaining where they were obtained.
For example, we would put the following Python function in a file ``input/parameters/lead-acid/electrolyte_diffusivity_Bloggs2019.py``

.. code-block:: python

    def electrolyte_diffusivity_Bloggs2019(c_e):
        """
        Dimensional Fickian diffusivity in the electrolyte [m2.s-1], from [1]_, as a
        function of the electrolyte concentration c_e [mol.m-3].

        References
        ----------
        .. [1] J Bloggs, AN Other. A set of parameters. A Chemistry Journal,
               123(4):567-573, 2019.

        """
        return (1.75 + 260e-6 * c_e) * 1e-9

Unit tests for the new class
----------------------------

You might want to add some unit tests to show that the parameters combine as expected
(see e.g. `lithium-ion parameter tests <https://github.com/pybamm-team/PyBaMM/blob/master/tests/unit/test_parameters/test_dimensionless_parameter_values_lithium_ion.py>`_), but this is not crucial.

Test on the models
------------------

In theory, any existing model can now be solved using the new parameters instead of their default parameters, with no extra work from here.
To test this, add something like the following test to one of the model test files
(e.g. `DFN <https://github.com/pybamm-team/PyBaMM/blob/master/tests/test_models/test_lithium_ion/test_lithium_ion_dfn.py>`_):

.. code-block:: python

    def test_my_new_parameters(self):
        model = pybamm.lithium_ion.DFN()
        input_path = os.path.join(os.getcwd(), "path", "to", "functions")
        parameter_values = pybamm.ParameterValues(
            "path/to/parameter/file.csv",
            {
                "Typical current density": 1,
                "Current function": os.path.join(
                    os.getcwd(),
                    "pybamm",
                    "parameters",
                    "standard_current_functions",
                    "constant_current.py",
                ),
                "First function": os.path.join(input_path, "first_function.py"),
                "Second function": os.path.join(input_path, "second_function.py"),
            },
        )

        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

This will check that the model can run with the new parameters (but not that it gives a sensible answer!).

Once you have performed the above checks, you are almost ready to merge your code into the core PyBaMM - see
`CONTRIBUTING.md workflow <https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md#c-merging-your-changes-with-pybamm>`_
for how to do this.
