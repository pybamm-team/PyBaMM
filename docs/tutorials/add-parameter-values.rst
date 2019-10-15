.. _CONTRIBUTING.md: https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md


Adding Parameter Values
=======================

As with any contribution to PyBaMM, please follow the workflow in CONTRIBUTING.md_.
In particular, start by creating an issue to discuss what you want to do - this is a good way to avoid wasted coding hours!

The role of parameter values
----------------------------

All models in PyBaMM are implemented as `expression trees <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/expression-tree.ipynb>`_.
At the stage of creating a model, we use :class:`pybamm.Parameter` and :class:`pybamm.FunctionParameter` objects to represent parameters and functions respectively.

We then create a :class:`ParameterValues` class, using a specific set of parameters, to iterate through the model and replace any :class:`pybamm.Parameter` objects with a :class:`pybamm.Scalar` and any :class:`pybamm.FunctionParameter` objects with a :class:`pybamm.Function`.

For an example of how the parameter values work, see the
`parameter values notebook <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/parameter-values.ipynb>`_.

Adding a set of parameters values
---------------------------------

There are two ways to add parameter sets:

1. **Complete parameter file**: Parameter sets should be added as csv files in the appropriate chemistry folder in ``input/parameters/`` (add a new folder if no parameters exist for that chemistry yet).
The expected structure of the csv file is

+-------------------------+----------------------+-----------------------+-------------+
| Name [Units]            | Value                | Reference             | Notes       |
+=========================+======================+=======================+=============+
| Example [m]             | 13                   | bloggs2019            | an example  |
+-------------------------+----------------------+-----------------------+-------------+

Empty lines, and lines starting with ``#``, will be ignored.

2. **Parameters for a single material**: It's possible to add parameters for a single material (e.g. anode) and then re-use existing parameters for the other materials. To do this, add a new subfolder with a ``README.md`` for references and csv file of parameters (e.g. ``input/parameters/lithium-ion/anodes/new_anode_chemistry_Bloggs2019/``). Then this file can be referenced using the ``chemistry`` option in ``ParameterValues``, e.g.

.. code-block:: python

    param = pybamm.ParameterValues(
        chemistry={
            "chemistry": "lithium-ion",
            "cell": "kokam_Marquis2019",
            "anode": "new_anode_chemistry_Bloggs2019",
            "separator": "separator_Marquis2019",
            "cathode": "lico2_Marquis2019",
            "electrolyte": "lipf6_Marquis2019",
            "experiment": "1C_discharge_from_full_Marquis2019",
        }
    )

or, equivalently in this case (since the only difference from the standard parameters from Marquis et al. is the set of anode parameters),

.. code-block:: python

    param = pybamm.ParameterValues(
        chemistry={
            **pybamm.parameter_sets.Marquis2019,
            "anode": "new_anode_chemistry_Bloggs2019",
        }
    )

Adding a function
-----------------

Functions should be added as Python functions under a file with the same name in the appropriate chemistry folder in ``input/parameters/``.
These Python functions should be documented with references explaining where they were obtained.
For example, we would put the following Python function in a file ``input/parameters/lead-acid/diffusivity_Bloggs2019.py``

.. code-block:: python

    def diffusivity_Bloggs2019(c_e):
        """
        Dimensional Fickian diffusivity in the electrolyte [m2.s-1], from [1]_, as a
        function of the electrolyte concentration c_e [mol.m-3].

        References
        ----------
        .. [1] J Bloggs, AN Other. A set of parameters. A Chemistry Journal,
               123(4):567-573, 2019.

        """
        return (1.75 + 260e-6 * c_e) * 1e-9

Then, these functions should be added to the parameter file from which they will be
called (must be in the same folder), with the tag ``[function]``, for example:

+---------------------+--------------------------------------+--------------+-------------+
| Name [Units]        | Value                                |  Reference   | Notes       |
+=====================+======================================+==============+=============+
| Example [m2.s-1]    | [function]diffusivity_Bloggs2019     | bloggs2019   | a function  |
+---------------------+--------------------------------------+--------------+-------------+



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
        parameter_values = pybamm.ParameterValues("path/to/parameter/file.csv")
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

This will check that the model can run with the new parameters (but not that it gives a sensible answer!).

Once you have performed the above checks, you are almost ready to merge your code into the core PyBaMM - see
`CONTRIBUTING.md workflow <https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md#c-merging-your-changes-with-pybamm>`_
for how to do this.
