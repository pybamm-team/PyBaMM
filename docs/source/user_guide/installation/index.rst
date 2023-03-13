Installation
============

PyBaMM is available on GNU/Linux, MacOS and Windows.
It can be installed using pip or conda, or from source.

Using pip
----------

PyBaMM can be installed via pip from `PyPI <https://pypi.org/project/pybamm>`__

GNU/Linux and Windows
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   pip install pybamm

macOS
~~~~~

.. code:: bash

   brew install sundials && pip install pybamm

Using conda
-----------

PyBaMM is part of the `Anaconda <https://docs.continuum.io/anaconda/>`_ distribution and is available as a conda package through the conda-forge channel

.. code:: bash

   conda install -c conda-forge pybamm

Optional solvers
----------------

Following GNU/Linux and macOS solvers are optionally available:

*  `scikits.odes <https://scikits-odes.readthedocs.io/en/latest/>`_ -based solver, see `Optional - scikits.odes solver <https://pybamm.readthedocs.io/en/latest/source/user_guide/installation/GNU-linux.html#optional-scikits-odes-solver>`_.
*  `jax <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_ -based solver, see `Optional - JaxSolver <https://pybamm.readthedocs.io/en/latest/source/user_guide/installation/GNU-linux.html#optional-jaxsolver>`_.

Full installation guide
-----------------------

Installing a specific version? Installing from source? Check the advanced installation pages below

.. toctree::
   :maxdepth: 1

   GNU-linux
   windows
   windows-wsl
   install-from-source