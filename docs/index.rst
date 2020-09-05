.. Root of all pybamm docs

.. _GitHub: https://github.com/pybamm-team/PyBaMM

Welcome to PyBaMM's documentation!
==================================

Python Battery Mathematical Modelling (**PyBAMM**) solves continuum models for
batteries, using both numerical methods and asymptotic analysis.

PyBaMM is hosted on GitHub_. This page provides the *API*, or *developer
documentation* for ``pybamm``.

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quickstart
=========================
PyBaMM is available on GNU/Linux, MacOS and Windows.

Using pip
----------

GNU/Linux and Windows
~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   pip install pybamm

macOS
~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   brew install sundials && pip install pybamm

Using conda
-------------
PyBaMM is available as a conda package through the conda-forge channel.

.. code:: bash

   conda install -c conda-forge pybamm

Optional solvers
-----------------
On GNU/Linux and MacOS, an optional `scikits.odes <https://scikits-odes.readthedocs.io/en/latest/>`_ -based solver is available, see :ref:`scikits.odes-label`.

Installation
============

.. toctree::
   :maxdepth: 1

   install/GNU-linux
   install/windows
   install/windows-wsl
   install/install-from-source

API documentation
====================

.. module:: pybamm

.. toctree::
   :maxdepth: 2

   source/expression_tree/index
   source/models/index
   source/parameters/index
   source/geometry/index
   source/meshes/index
   source/spatial_methods/index
   source/solvers/index
   source/experiments/index
   source/simulation
   source/plotting/index
   source/util
   source/citations
   source/parameters_cli

Examples
========

Detailed examples can be viewed on the
`GitHub examples page <https://github.com/pybamm-team/PyBaMM/tree/master/examples/notebooks>`_,
and run locally using ``jupyter notebook``, or online through
`Binder <https://mybinder.org/v2/gh/pybamm-team/PyBaMM/master?filepath=examples%2Fnotebooks>`_.

Contributing
============

There are many ways to contribute to PyBaMM:

.. toctree::
    :maxdepth: 1

    tutorials/add-parameter-values
    tutorials/add-model
    tutorials/add-spatial-method
    tutorials/add-solver

Before contributing, please read the `Contribution Guidelines <https://github.com/pybamm-team/PyBaMM/blob/master/CONTRIBUTING.md>`_.
