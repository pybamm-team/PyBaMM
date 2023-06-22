.. _examples:

########
Examples
########

A collection of Python scripts and Jupyter notebooks that demonstrate how to use PyBaMM.
For further examples, see the `PyBaMM case studies GitHub repository <https://github.com/pybamm-team/pybamm-example-results>`_.

.. toctree::
    :maxdepth: 2

    notebooks/index

.. _notebooks:

For new users we recommend the Getting Started notebooks. These are intended to be very simple step-by-step guides to show the basic functionality of PyBaMM. 
For more detailed notebooks, please see the examples listed below.

.. _google-colab:

You may run the notebooks in `Google Colab <https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/develop/>`_. 

This page contains a number of examples showing how to use PyBaMM. Each example was created as a `Jupyter notebook <https://jupyter.org/>`_.
These notebooks can be downloaded and used locally by running

.. code:: bash

    $ jupyter notebook

from your local PyBaMM repository or used online through `Google Colab <https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/develop/>`_.
Alternatively, you can simply copy/paste the relevant code.

Using PyBaMM
------------

The easiest way to start with PyBaMM is by running and comparing some of the inbuilt models:

-  `Run the Single Particle Model (SPM) <./notebooks/models/SPM.ipynb>`__
-  `Compare models <./notebooks/models/lead-acid.ipynb>`__
-  `Comparison with COMSOL <./notebooks/models/compare-comsol-discharge-curve.ipynb>`__

It is also easy to add new models or change the setting that are used:

-  `Add a model (example) <./creating_models/index.rst>`__
-  `Change model options <./models/using-model-options_thermal-example.ipynb>`__
-  `Using submodels <./using-submodels.ipynb>`__
-  `Change the settings <./change-settings.ipynb>`__ (parameters, spatial method, or solver)
-  `Change the applied current <./parameterization/change-input-current.ipynb>`__

Expression tree structure
-------------------------

PyBaMM is built around an expression tree structure.

-  `The expression tree notebook <expression_tree/expression-tree.ipynb>`__ explains how this works, from model creation to solution.
-  `The broadcast notebook <expression_tree/broadcasts.ipynb>`__ explains the different types of broadcast.

The following notebooks are specific to different stages of the PyBaMM
pipeline, such as choosing a model, spatial method, or solver.

Models
~~~~~~

Several battery models are implemented and can easily be used or `compared <./models/lead-acid.ipynb>`__. The notebooks below show the
solution of each individual model.

Once you are comfortable with the expression tree structure, a good
starting point to understand the models in PyBaMM is to take a look at

the `basic SPM <https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/models/full_battery_models/lithium_ion/basic_spm.py>`__
and `basic DFN <https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/models/full_battery_models/lithium_ion/basic_dfn.py>`__,

since these define the entire model (variables, equations, initial and
boundary conditions, events) in a single class and so are easier to
understand. However, we recommend that you subsequently use the full
models as they offer much greater flexibility for coupling different
physical effects and visualising a greater range of variables.

Lithium-ion models
^^^^^^^^^^^^^^^^^^

-  `Single-Particle Model <./models/SPM.ipynb>`__
-  `Single-Particle Model with electrolyte <./models/SPMe.ipynb>`__
-  `Doyle-Fuller-Newman Model <./models/DFN.ipynb>`__

Lead-acid models
^^^^^^^^^^^^^^^^

-  `Full porous-electrode <https://docs.pybamm.org/en/latest/source/api/models/lead_acid/full.html>`__
-  `Leading-Order Quasi-Static <https://docs.pybamm.org/en/latest/source/api/models/lead_acid/loqs.html>`__

Spatial Methods
~~~~~~~~~~~~~~~

The following spatial methods are implemented

-  `Finite Volumes <./spatial_methods/finite-volumes.ipynb>`__ (1D only)
-  Spectral Volumes (1D only)
-  Finite Elements (only for 2D current collector domains)

Solvers
~~~~~~~

The following notebooks show examples for generic ODE and DAE solvers.
Several solvers are implemented in PyBaMM and we encourage users to try
different ones to find the most appropriate one for their models.

-  `ODE solver <./solvers/ode-solver.ipynb>`__
-  `DAE solver <./solvers/dae-solver.ipynb>`__
