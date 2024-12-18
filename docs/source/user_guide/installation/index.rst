Installation
============

PyBaMM is available on GNU/Linux, MacOS and Windows.
It can be installed using ``pip`` or ``conda``, or from source.

.. tab:: pip

   PyBaMM can be installed via pip from `PyPI <https://pypi.org/project/pybamm>`__.

   .. code:: bash

      pip install pybamm

.. tab:: conda

   PyBaMM is available as a ``conda`` package through the conda-forge channel.

   The ``pybamm`` package on conda-forge installs PyBaMM with all the `required and optional dependencies <index.html#dependencies>`_ available on conda-forge.

   .. note::

      At the moment, PyBaMM's conda-forge distribution does not include ``IDAKLUSolver``.

   .. code:: bash

      conda install -c conda-forge pybamm

   The ``pybamm-base`` package installs PyBaMM only with its `required dependencies <index.html#install-required-dependencies>`_.

   .. code:: bash

      conda install -c conda-forge pybamm-base


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
`posthog <https://posthog.com/>`__                                  3.6.5
`pyyaml <https://pyyaml.org/>`__
`platformdirs <https://platformdirs.readthedocs.io/en/latest/>`__
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
`sphinx_rtd_theme <https://pypi.org/project/sphinx-rtd-theme/>`__                                 \-                 docs               This Sphinx theme provides a great reader experience for documentation.
`pydata-sphinx-theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`__                   \-                 docs               A clean, Bootstrap-based Sphinx theme.
`sphinx_design <https://sphinx-design.readthedocs.io/en/latest/>`__                               \-                 docs               A sphinx extension for designing.
`sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/en/latest/>`__                       \-                 docs               To copy codeblocks.
`myst-parser <https://myst-parser.readthedocs.io/en/latest/>`__                                   \-                 docs               For technical & scientific documentation.
`sphinx-inline-tabs <https://sphinx-inline-tabs.readthedocs.io/en/latest/>`__                     \-                 docs               Add inline tabbed content to your Sphinx documentation.
`sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`__                 \-                 docs               For BibTeX citations.
`sphinx-autobuild <https://sphinx-extensions.readthedocs.io/en/latest/sphinx-autobuild.html>`__   \-                 docs               For re-building docs once triggered.
`sphinx-last-updated-by-git <https://pypi.org/project/sphinx-last-updated-by-git/>`__             \-                 docs               To get the "last updated" time for each Sphinx page from Git.
`nbsphinx <https://nbsphinx.readthedocs.io/en/0.9.5/>`__                                          \-                 docs               Sphinx extension that provides a source parser for .ipynb files
`ipykernel <https://pypi.org/project/ipykernel/>`__                                               \-                 docs               Provides the IPython kernel for Jupyter.
`ipywidgets <https://ipywidgets.readthedocs.io/en/latest/>`__                                     \-                 docs               Interactive HTML widgets for Jupyter notebooks and the IPython kernel.
`sphinx-gallery <https://pypi.org/project/sphinx-gallery/>`__                                     \-                 docs               Builds an HTML gallery of examples from any set of Python scripts.
`sphinx-hoverxref <https://sphinx-hoverxref.readthedocs.io/en/latest/index.html>`__               \-                 docs               Sphinx extension to show a floating window.
`sphinx-docsearch <https://sphinx-docsearch.readthedocs.io/>`__                                   \-                 docs               To replaces Sphinx’s built-in search with Algolia DocSearch.
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
`pytest-subtests <https://github.com/pytest-dev/pytest-subtests/>`__             \-                 dev                For subtests pytest fixture.
`pytest-cov <https://pytest-cov.readthedocs.io/en/stable/>`__                    \-                 dev                For calculating test coverage.
`pytest <https://docs.pytest.org/en/stable/>`__                                  6.0.0              dev                For running the test suites.
`pytest-doctestplus <https://github.com/scientific-python/pytest-doctestplus>`__ \-                 dev                For running doctests.
`pytest-xdist <https://pytest-xdist.readthedocs.io/en/latest/>`__                \-                 dev                For running tests in parallel across distributed workers.
`pytest-mock <https://pytest-mock.readthedocs.io/en/latest/index.html>`__        \-                 dev                Provides a mocker fixture.
`nbmake <https://github.com/treebeardtech/nbmake/>`__                            \-                 dev                A ``pytest`` plugin for executing Jupyter notebooks.
`importlib-metadata <https://pypi.org/project/importlib-metadata/>`__            \-                 dev                Used to read metadata from Python packages.
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
