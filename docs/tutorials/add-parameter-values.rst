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

Parameter sets are split by material into ``anodes``, ``separators``, ``cathodes``, ``electrolytes``, ``cells`` (for cell geometries and thermal properties) and ``experiments`` (for initial conditions and charge/discharge rates).
To add a new parameter set in one of these subcategories, first create a new folder in the appropriate chemistry folder: for example, to add a new anode chemistry for lithium-ion, add a subfolder ``input/parameters/lithium-ion/anodes/new_anode_chemistry_AuthorYear``. 
This subfolder should then contain:

- a csv file ``parameters.csv`` with all the relevant scalar parameters. The expected structure of the csv file is:

+-------------------------+----------------------+-----------------------+-------------+
| Name [Units]            | Value                | Reference             | Notes       |
+=========================+======================+=======================+=============+
| Example [m]             | 13                   | AuthorYear            | an example  |
+-------------------------+----------------------+-----------------------+-------------+

Empty lines, and lines starting with ``#``, will be ignored.

- a ``README.md`` file with information on where these parameters came from
- python files for any functions, which should be referenced from the ``parameters.csv`` file (see ``Adding a Function`` below)
- csv files for any data to be interpolated, which should be referenced from the ``parameters.csv`` file (see ``Adding data for interpolation`` below)

The easiest way to start is to copy an existing file (e.g. ````input/parameters/lithium-ion/anodes/graphite_mcmb2528_Marquis2019``) and replace all entries in all files as appropriate

Adding a function
-----------------

Functions should be added as Python functions under a file with the same name in the appropriate chemistry folder in ``input/parameters/``.
These Python functions should be documented with references explaining where they were obtained.
For example, we would put the following Python function in a file ``input/parameters/lithium_ion/anodes/new_anode_chemistry_AuthorYear/diffusivity_AuthorYear.py``

.. code-block:: python

    def diffusivity_AuthorYear(c_e):
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
| Example [m2.s-1]    | [function]diffusivity_AuthorYear     | AuthorYear   | a function  |
+---------------------+--------------------------------------+--------------+-------------+

Adding data for interpolation
-----------------------------

Data should be added as as csv file in the appropriate chemistry folder in ``input/parameters/``.
For example, we would put the following data in a file ``input/parameters/lithium_ion/anodes/new_anode_chemistry_AuthorYear/diffusivity_AuthorYear.csv``

+--------------------------+--------------------------+
| # concentration [mol/m3] | Diffusivity [m2/s]       |
+==========================+==========================+
| 0.000000000000000000e+00 | 4.714135898019971016e+00 |
| 2.040816326530612082e-02 | 4.708899441575220557e+00 |
| 4.081632653061224164e-02 | 4.702448345762175741e+00 |
| 6.122448979591836593e-02 | 4.694558534379876136e+00 |
| 8.163265306122448328e-02 | 4.684994372928071193e+00 |
| 1.020408163265306006e-01 | 4.673523893805322516e+00 |
| 1.224489795918367319e-01 | 4.659941254449398329e+00 |
| 1.428571428571428492e-01 | 4.644096031712390271e+00 |
+--------------------------+--------------------------+

Empty lines, and lines starting with ``#``, will be ignored.

Then, this data should be added to the parameter file from which it will be
called (must be in the same folder), with the tag ``[data]``, for example:

+---------------------+----------------------------------+--------------+-------------+
| Name [Units]        | Value                            |  Reference   | Notes       |
+=====================+==================================+==============+=============+
| Example [m2.s-1]    | [data]diffusivity_AuthorYear     | AuthorYear   | some data   |
+---------------------+----------------------------------+--------------+-------------+

Using new parameters
--------------------

If you have added a whole new set of parameters, then you can create a new parameter set in ``pybamm/parameters/parameter_sets.py``, by just adding a new dictionary to that file, for example

.. code-block:: python

    AuthorYear = {
        "chemistry": "lithium-ion",
        "cell": "new_cell_AuthorYear",
        "anode": "new_anode_AuthorYear",
        "separator": "new_separator_AuthorYear",
        "cathode": "new_cathode_AuthorYear",
        "electrolyte": "new_electrolyte_AuthorYear",
        "experiment": "new_experiment_AuthorYear",
    }

Then, to use these new parameters, use:

.. code-block:: python

    param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.AuthorYear)

Note that you can re-use existing parameter subsets instead of creating new ones (for example, you could just replace "experiment": "new_experiment_AuthorYear" with "experiment": "1C_discharge_from_full_Marquis2019" in the above dictionary).

It's also possible to add parameters for a single material (e.g. anode) and then re-use existing parameters for the other materials, without adding a parameter set to ``pybamm/parameters/parameter_sets.py``.

.. code-block:: python

    param = pybamm.ParameterValues(
        chemistry={
            "chemistry": "lithium-ion",
            "cell": "kokam_Marquis2019",
            "anode": "new_anode_chemistry_AuthorYear",
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
            "anode": "new_anode_chemistry_AuthorYear",
        }
    )

Unit tests for the new class
----------------------------

You might want to add some unit tests to show that the parameters combine as expected
(see e.g. `lithium-ion parameter tests <https://github.com/pybamm-team/PyBaMM/blob/master/tests/unit/test_parameters/test_dimensionless_parameter_values_lithium_ion.py>`_), but this is not crucial.

Test on the models
------------------

In theory, any existing model can now be solved using the new parameters instead of their default parameters, with no extra work from here.
To test this, add something like the following test to one of the model test files
(e.g. `DFN <https://github.com/pybamm-team/PyBaMM/blob/master/tests/integration/test_models/test_full_battery_models/test_lithium_ion/test_dfn.py>`_):

.. code-block:: python

    def test_my_new_parameters(self):
        model = pybamm.lithium_ion.DFN()
        parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.AuthorYear)
        modeltest = tests.StandardModelTest(model, parameter_values=parameter_values)
        modeltest.test_all()

This will check that the model can run with the new parameters (but not that it gives a sensible answer!).

Once you have performed the above checks, you are almost ready to merge your code into the core PyBaMM - see
`CONTRIBUTING.md workflow <https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md#c-merging-your-changes-with-pybamm>`_
for how to do this.
