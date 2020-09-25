Parameters command line interface
=======================================

PyBaMM comes with a small command line interface that can be used to manage parameter sets.
By default, PyBaMM provides parameters in the "input" directory located in the pybamm package
directory.
If you wish to add new parameters, you can first pull a given parameter directory into the current
working directory using the command ``pybamm_edit_parameter`` for manual editing.
By default, PyBaMM first looks for parameter defined in the current working directory before
falling back the package directory if nothing is found locally.
If you wish to access a newly defined parameter set from anywhere in your system, you can use
``pybamm_add_parameter`` to copy a given parameter directory to the package directory.
To get a list of currently available parameter sets, use ``pybamm_list_parameters``.

.. autofunction:: pybamm.parameters_cli.add_parameter

.. autofunction:: pybamm.parameters_cli.remove_parameter

.. autofunction:: pybamm.parameters_cli.edit_parameter
