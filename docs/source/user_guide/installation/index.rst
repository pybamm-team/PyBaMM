Installation
============

PyBaMM is available on GNU/Linux, MacOS and Windows.
It can be installed using `pip` or `conda`, or from source.

.. tab:: GNU/Linux and Windows

   .. tab:: pip

      PyBaMM can be installed via pip from `PyPI <https://pypi.org/project/pybamm>`__.

      .. code:: bash

         pip install pybamm

   .. tab:: conda

      PyBaMM is part of the `Anaconda <https://docs.continuum.io/anaconda/>`_ distribution and is available as a conda package through the conda-forge channel.

      .. code:: bash

         conda install -c conda-forge pybamm

.. tab:: macOS

   .. tab:: pip

      PyBaMM can be installed via pip from `PyPI <https://pypi.org/project/pybamm>`__.

      .. code:: bash

         brew install sundials && pip install pybamm


   .. tab:: conda

      PyBaMM is part of the `Anaconda <https://docs.continuum.io/anaconda/>`_ distribution and is available as a conda package through the conda-forge channel.

      .. code:: bash

         conda install -c conda-forge pybamm


Optional solvers
----------------

Following GNU/Linux and macOS solvers are optionally available:

*  `scikits.odes <https://scikits-odes.readthedocs.io/en/latest/>`_ -based solver, see `Optional - scikits.odes solver <https://pybamm.readthedocs.io/en/latest/source/user_guide/installation/GNU-linux.html#optional-scikits-odes-solver>`_.
*  `jax <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_ -based solver, see `Optional - JaxSolver <https://pybamm.readthedocs.io/en/latest/source/user_guide/installation/GNU-linux.html#optional-jaxsolver>`_.

Dependencies
------------
.. _install.required_dependencies:

Required dependencies
~~~~~~~~~~~~~~~~~~~~~

PyBaMM requires the following dependencies.

================================================================ ==========================
Package                                                          Minimum supported version
================================================================ ==========================
`NumPy <https://numpy.org>`__                                    1.16.0
`SciPy <https://docs.scipy.org/doc/scipy/>`__                    2.8.2
`pandas <https://pandas.pydata.org/docs/>`__                     0.24.0
`CasADi <https://web.casadi.org/docs/>`__                        3.6.0
`xarray <https://docs.xarray.dev/en/stable/>`__                  2023.04.0
================================================================ ==========================

.. _install.optional_dependencies:

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

PyBaMM has a number of optional dependencies for different functionalities.
If the optional dependency is not installed, PyBaMM will raise an ImportError when the method requiring that dependency is called.

If using pip, optional PyBaMM dependencies can be installed or managed in a file (e.g. requirements.txt or setup.py)
as optional extras (e.g.,``pybamm[dev,plot]``). All optional dependencies can be installed with ``pybamm[all]``,
and specific sets of dependencies are listed in the sections below.

.. _install.plot_dependencies:

Plot dependencies
^^^^^^^^^^^^^^^^^

Installable with ``pip install "pandas[plot]"``

=========================================================== ================== ================== ===================================================================================================================================================================================
Dependency                                                  Minimum Version    pip extra          Notes
=========================================================== ================== ================== ===================================================================================================================================================================================
`imageio <https://imageio.readthedocs.io/en/stable/>`__     2.9.0              plot               Handles diverse image data formats, including animated, volumetric, and scientific formats.
`Matplotlib <https://matplotlib.org/stable/>`__             2.0.0              plot               To plot various battery models, and analyzing battery performance.
=========================================================== ================== ================== ===================================================================================================================================================================================

Full installation guide
-----------------------

Installing a specific version? Installing from source? Check the advanced installation pages below

.. toctree::
   :maxdepth: 1

   GNU-linux
   windows
   windows-wsl
   install-from-source