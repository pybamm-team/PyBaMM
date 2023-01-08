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
Following GNU/Linux and macOS solvers are optionally available:

*  `scikits.odes <https://scikits-odes.readthedocs.io/en/latest/>`_ -based solver, see `Optional - scikits.odes solver <https://pybamm.readthedocs.io/en/latest/install/GNU-linux.html#optional-scikits-odes-solver>`_.
*  `jax <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_ -based solver, see `Optional - JaxSolver <https://pybamm.readthedocs.io/en/latest/install/GNU-linux.html#optional-jaxsolver>`_.

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
   source/callbacks
   source/citations
   source/batch_study

Examples
========

Detailed examples can be viewed on the
`GitHub examples page <https://github.com/pybamm-team/PyBaMM/tree/develop/examples/notebooks>`_,
and run locally using ``jupyter notebook``, or online through
`Google Colab <https://colab.research.google.com/github/pybamm-team/PyBaMM/blob/develop/>`_.

Contributing
============

Contributions to PyBaMM and its development are welcome! If you have ideas for features, bug fixes, models, spatial methods, or solvers, we would love to hear from you.

Before contributing, please read the `Contribution Guidelines <https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md>`_.
