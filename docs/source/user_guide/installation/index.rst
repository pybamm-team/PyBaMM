Installation
============

PyBaMM is available on GNU/Linux, MacOS and Windows.
It can be installed using ``pip`` or ``conda``, or from source.

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

         pip install pybamm


   .. tab:: conda

      PyBaMM is part of the `Anaconda <https://docs.continuum.io/anaconda/>`_ distribution and is available as a conda package through the conda-forge channel.

      .. code:: bash

         conda install -c conda-forge pybamm


Optional solvers
----------------

The following solvers are optionally available:

*  `jax <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_ -based solver, see `Optional - JaxSolver <gnu-linux-mac.html#optional-jaxsolver>`_.
*  `IREE <https://iree.dev/>`_ (`MLIR <https://mlir.llvm.org/>`_) support, see `Optional - IREE / MLIR Support <gnu-linux-mac.html#optional-iree-mlir-support>`_.

Dependencies
------------
.. _install.required_dependencies:

Required dependencies
~~~~~~~~~~~~~~~~~~~~~

PyBaMM requires the following dependencies.

=================================================================== ==========================
Package                                                             Minimum supported version
=================================================================== ==========================
`NumPy <https://numpy.org>`__                                       1.23.5
`SciPy <https://docs.scipy.org/doc/scipy/>`__                       1.9.3
`CasADi <https://web.casadi.org/docs/>`__                           3.6.3
`Xarray <https://docs.xarray.dev/en/stable/>`__                     2022.6.0
`Anytree <https://anytree.readthedocs.io/en/stable/>`__             2.8.0
`SymPy <https://docs.sympy.org/latest/index.html>`__                1.9.3
`typing-extensions <https://pypi.org/project/typing-extensions/>`__ 4.10.0
`pandas <https://pypi.org/project/pandas/>`__                       1.5.0
`pooch <https://www.fatiando.org/pooch/>`__                         1.8.1
=================================================================== ==========================

.. _install.optional_dependencies:

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

PyBaMM has a number of optional dependencies for different functionalities.
If the optional dependency is not installed, PyBaMM will raise an ImportError when the method requiring that dependency is called.

If you are using ``pip``, optional PyBaMM dependencies can be installed or managed in a file (e.g., setup.py, or pyproject.toml)
as optional extras (e.g.,``pybamm[dev,plot]``). All optional dependencies can be installed with ``pybamm[all]``,
and specific sets of dependencies are listed in the sections below.

.. _install.plot_dependencies:

Plot dependencies
^^^^^^^^^^^^^^^^^

Installable with ``pip install "pybamm[plot]"``

=========================================================== ================== ================== ==================================================================
Dependency                                                  Minimum Version    pip extra          Notes
=========================================================== ================== ================== ==================================================================
`imageio <https://imageio.readthedocs.io/en/stable/>`__     2.3.0              plot               For generating simulation GIFs.
`matplotlib <https://matplotlib.org/stable/>`__             3.6.0              plot               To plot various battery models, and analyzing battery performance.
=========================================================== ================== ================== ==================================================================

.. _install.docs_dependencies:

Docs dependencies
^^^^^^^^^^^^^^^^^

Installable with ``pip install "pybamm[docs]"``

================================================================================================= ================== ================== =======================================================================
Dependency                                                                                        Minimum Version    pip extra          Notes
================================================================================================= ================== ================== =======================================================================
`sphinx <https://www.sphinx-doc.org/en/master/>`__                                                \-                 docs               Sphinx makes it easy to create intelligent and beautiful documentation.
`pydata-sphinx-theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`__                   \-                 docs               A clean, Bootstrap-based Sphinx theme.
`sphinx_design <https://sphinx-design.readthedocs.io/en/latest/>`__                               \-                 docs               A sphinx extension for designing.
`sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/en/latest/>`__                       \-                 docs               To copy codeblocks.
`myst-parser <https://myst-parser.readthedocs.io/en/latest/>`__                                   \-                 docs               For technical & scientific documentation.
`sphinx-inline-tabs <https://sphinx-inline-tabs.readthedocs.io/en/latest/>`__                     \-                 docs               Add inline tabbed content to your Sphinx documentation.
`sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`__                 \-                 docs               For BibTeX citations.
`sphinx-autobuild <https://sphinx-extensions.readthedocs.io/en/latest/sphinx-autobuild.html>`__   \-                 docs               For re-building docs once triggered.
================================================================================================= ================== ================== =======================================================================

.. _install.examples_dependencies:

Examples dependencies
^^^^^^^^^^^^^^^^^^^^^

Installable with ``pip install "pybamm[examples]"``

================================================================================ ================== ================== ================================
Dependency                                                                       Minimum Version    pip extra          Notes
================================================================================ ================== ================== ================================
`jupyter <https://docs.jupyter.org/en/latest/>`__                                \-                 examples           For example notebooks rendering.
================================================================================ ================== ================== ================================

.. _install.dev_dependencies:

Dev dependencies
^^^^^^^^^^^^^^^^^

Installable with ``pip install "pybamm[dev]"``

================================================================================ ================== ================== =============================================================
Dependency                                                                       Minimum Version    pip extra          Notes
================================================================================ ================== ================== =============================================================
`pre-commit <https://pre-commit.com/index.html>`__                               \-                 dev                For managing and maintaining multi-language pre-commit hooks.
`ruff <https://beta.ruff.rs/docs/>`__                                            \-                 dev                For code formatting.
`nox <https://nox.thea.codes/en/stable/>`__                                      \-                 dev                For running testing sessions in multiple environments.
`pytest-cov <https://pytest-cov.readthedocs.io/en/stable/>`__                    \-                 dev                For calculating test coverage.
`parameterized <https://github.com/wolever/parameterized>`__                     \-                 dev                For test parameterization.
`pytest <https://docs.pytest.org/en/stable/>`__                                  6.0.0              dev                For running the test suites.
`pytest-doctestplus <https://github.com/scientific-python/pytest-doctestplus>`__ \-                 dev                For running doctests.
`pytest-xdist <https://pytest-xdist.readthedocs.io/en/latest/>`__                \-                 dev                For running tests in parallel across distributed workers.
`nbmake <https://github.com/treebeardtech/nbmake/>`__                            \-                 dev                A ``pytest`` plugin for executing Jupyter notebooks.
================================================================================ ================== ================== =============================================================

.. _install.cite_dependencies:

Cite dependencies
^^^^^^^^^^^^^^^^^

Installable with ``pip install "pybamm[cite]"``

=========================================================== ================== ================== =========================================
Dependency                                                  Minimum Version    pip extra          Notes
=========================================================== ================== ================== =========================================
`pybtex <https://docs.pybtex.org/>`__                       0.24.0             cite               BibTeX-compatible bibliography processor.
=========================================================== ================== ================== =========================================

.. _install.bpx_dependencies:

bpx dependencies
^^^^^^^^^^^^^^^^^

Installable with ``pip install "pybamm[bpx]"``

=========================================================== ================== ================== ==========================
Dependency                                                  Minimum Version    pip extra          Notes
=========================================================== ================== ================== ==========================
`bpx <https://pypi.org/project/bpx/>`__                     \-                 bpx                Battery Parameter eXchange
=========================================================== ================== ================== ==========================

.. _install.tqdm_dependencies:

tqdm dependencies
^^^^^^^^^^^^^^^^^

Installable with ``pip install "pybamm[tqdm]"``

=========================================================== ================== ================== ==================
Dependency                                                  Minimum Version    pip extra          Notes
=========================================================== ================== ================== ==================
`tqdm <https://tqdm.github.io/>`__                          \-                 tqdm               For logging loops.
=========================================================== ================== ================== ==================

.. _install.jax_dependencies:

Jax dependencies
^^^^^^^^^^^^^^^^^

Installable with ``pip install "pybamm[jax]"``, currently supported on Python 3.9-3.11.

========================================================================= ================== ================== =======================
Dependency                                                                Minimum Version    pip extra          Notes
========================================================================= ================== ================== =======================
`JAX <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`__  0.4.20             jax                For the JAX solver
`jaxlib <https://pypi.org/project/jaxlib/>`__                             0.4.20             jax                Support library for JAX
========================================================================= ================== ================== =======================

IREE dependencies
^^^^^^^^^^^^^^^^^^

Installable with ``pip install "pybamm[iree]"`` (requires ``jax`` dependencies to be installed).

========================================================================= ================== ================== =======================
Dependency                                                                Minimum Version    pip extra          Notes
========================================================================= ================== ================== =======================
`iree-compiler <https://iree.dev/>`__                                     20240507.886       iree               IREE compiler
========================================================================= ================== ================== =======================

Full installation guide
-----------------------

Installing a specific version? Installing from source? Check the advanced installation pages below

.. toctree::
   :maxdepth: 1

   gnu-linux-mac
   windows
   windows-wsl
   install-from-source
   install-from-docker
