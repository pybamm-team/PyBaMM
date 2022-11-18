:tocdepth: 3

===============
Parameters Sets
===============

PyBaMM provides :ref:`pre-defined parameters <bundled-parameter-sets>` for common
chemistries, as well as, a growing set of :ref:`third-party parameter sets <third-party-parameter-sets>`.

.. autoclass:: pybamm.parameters.parameter_sets.ParameterSets
    :members:

.. _adding-parameter-sets:

*********************
Adding Parameter Sets
*********************

Parameter sets can be added to PyBaMM by creating a python package, and
registering a `entry point`_ to ``pybamm_parameter_set``. At a minimum, the
package (``cell_parameters``) should consist of the following::

    cell_parameters
    ├── pyproject.toml        # and/or setup.cfg, setup.py
    └── src
        └── cell_parameters
            └── cell_alpha.py

.. _entry point: https://setuptools.pypa.io/en/latest/userguide/entry_point.html

The actual parameter set is defined within ``cell_alpha.py``, as shown below.
For an example, see the `Marquis2019`_ parameter sets.

.. _Marquis2019: https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/input/parameters/lithium_ion/Marquis2019.py

.. code-block:: python
    :linenos:

    import pybamm

    def get_parameter_values():
        """ Doc string for cell-alpha """
        return {
            "chemistry": "lithium_ion",
            "citation": "@book{van1995python, title={Python reference manual}}",
            ...
        }

Then register ``get_parameter_values`` to ``pybamm_parameter_set`` in ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points.pybamm_parameter_set]
    cell_alpha = "cell_parameters.cell_alpha:get_parameter_values"

If you are using ``setup.py`` or ``setup.cfg`` to setup your package, please
see SetupTools' documentation for registering `entry points`_.

.. _entry points: https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins

Finally install you package (``python -m pip install .``), to complete the process.
You will need to reinstall your package every time you add a new parameter set.
If you're actively editing the parameter set it may be helpful to install in
editing mode (``python -m pip install -e .``) instead.

Once successfully registered, your parameter set will appear within the contents
of ``pybamm.parameter_sets``, along with any other bundled or installed
third-party parameter sets.

.. doctest::

        >>> import pybamm
        >>> list(pybamm.parameter_sets)
        ['Ai2020', 'Chen2020', ...]

If you're willing to open-source your parameter set,
`let us know`_, and we can add an entry to
:ref:`third-party-parameter-sets`.

.. _let us know: https://github.com/pybamm-team/PyBaMM/issues/new/choose

.. _third-party-parameter-sets:

**************************
Third-Party Parameter Sets
**************************

Registered a new parameter set to ``pybamm_parameter_set``?
`Let us know`_, and we'll update our list.

.. _bundled-parameter-sets:

**********************
Bundled Parameter Sets
**********************

PyBaMM provides pre-defined parameter sets for several common chemistries,
listed below. See :ref:`adding-parameter-sets` for information on registering new
parameter sets with PyBaMM.

Lead-acid Parameter Sets
==========================

{% for k,v in parameter_sets.items() if v.chemistry == "lead_acid" %}
{{k}}
----------------------------
{{ parameter_sets.get_docstring(k) | safe }}
{% endfor %}

Lithium-ion Parameter Sets
==========================
{% for k,v in parameter_sets.items() if v.chemistry == "lithium_ion" %}
{{k}}
----------------------------
{{ parameter_sets.get_docstring(k) }}
{% endfor %}

