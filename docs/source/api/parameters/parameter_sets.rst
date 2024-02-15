:tocdepth: 3

.. _parameter_sets:

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
registering a `entry point`_ to ``pybamm_parameter_sets``. At a minimum, the
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
        """Doc string for cell-alpha"""
        return {
            "chemistry": "lithium_ion",
            "citation": "@book{van1995python, title={Python reference manual}}",
            # ...
        }

Then register ``get_parameter_values`` to ``pybamm_parameter_sets`` in ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points.pybamm_parameter_sets]
    cell_alpha = "cell_parameters.cell_alpha:get_parameter_values"

If you are using ``setup.py`` or ``setup.cfg`` to setup your package, please
see SetupTools' documentation for registering `entry points`_.

.. _entry points: https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins

If you're willing to open-source your parameter set,
`let us know`_, and we can add an entry to
:ref:`third-party-parameter-sets`.

.. _let us know: https://github.com/pybamm-team/PyBaMM/issues/new/choose

.. _third-party-parameter-sets:

**************************
Third-Party Parameter Sets
**************************

Registered a new parameter set to ``pybamm_parameter_sets``?
`Let us know`_, and we'll update our list.

.. _bundled-parameter-sets:

**********************
Bundled Parameter Sets
**********************

PyBaMM provides pre-defined parameter sets for several common chemistries,
listed below. See :ref:`adding-parameter-sets` for information on registering new
parameter sets with PyBaMM.

Lead-acid Parameter Sets
========================

{% for k,v in parameter_sets.items() if v.chemistry == "lead_acid" %}
{{k}}
----------------------------
{{ parameter_sets.get_docstring(k) }}
{% endfor %}

Lithium-ion Parameter Sets
==========================
{% for k,v in parameter_sets.items() if v.chemistry == "lithium_ion" %}
{{k}}
--------------------------------
{{ parameter_sets.get_docstring(k) }}
{% endfor %}

.. footbibliography::
