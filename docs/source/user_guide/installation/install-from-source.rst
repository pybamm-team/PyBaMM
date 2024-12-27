.. _install-from-source:

Install from source (GNU Linux and macOS)
=========================================


.. contents::

This page describes the build and installation of PyBaMM from the source code, available on GitHub. Note that this is **not the recommended approach for most users** and should be reserved to people wanting to participate in the development of PyBaMM, or people who really need to use bleeding-edge feature(s) not yet available in the latest released version. If you do not fall in the two previous categories, you would be better off installing PyBaMM using pip or conda.

Lastly, familiarity with the Python ecosystem is recommended (pip, virtualenvs).
Here is a gentle introduction/refresher: `Python Virtual Environments: A Primer <https://realpython.com/python-virtual-environments-a-primer/>`_.


Prerequisites
---------------

The following instructions are valid for both GNU/Linux distributions and MacOS.
If you are running Windows, consider using the `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_.

To obtain the PyBaMM source code, clone the GitHub repository

.. code:: bash

	  git clone https://github.com/pybamm-team/PyBaMM.git

or download the source archive on the repository's homepage.

To install PyBaMM, you will need:

- Python 3 (PyBaMM supports versions 3.9, 3.10, 3.11, and 3.12)
- The Python headers file for your current Python version.
- A BLAS library (for instance `openblas <https://www.openblas.net/>`_).
- A C compiler (ex: ``gcc``).
- A Fortran compiler (ex: ``gfortran``).
- ``graphviz`` (optional), if you wish to build the documentation locally.
- ``pandoc`` (optional) to convert the example Jupyter notebooks when building the documentation.

You can install the above with

.. tab:: Ubuntu/Debian

	.. code:: bash

		sudo apt install python3.X python3.X-dev libopenblas-dev gcc gfortran graphviz cmake pandoc

	Where ``X`` is the version sub-number.

.. tab:: MacOS

	.. code:: bash

		brew install python openblas gcc gfortran graphviz libomp cmake pandoc

.. note::

    If you are using some other linux distribution you can install the equivalent packages for ``python3, cmake, gcc, gfortran, openblas, pandoc``.

    On Windows, you can install ``graphviz`` using the `Chocolatey <https://chocolatey.org/>`_ package manager, or follow the instructions on the `graphviz website <https://graphviz.org/download/>`_.

Finally, we recommend using `Nox <https://nox.thea.codes/en/stable/>`_.
You can install it to your local user account (make sure you are not within a virtual environment) with

.. code:: bash

	  python3.X -m pip install --user nox

Note that running ``nox`` will create new virtual environments for you to use, so you do not need to create one yourself.

Depending on your operating system, you may or may not have ``pip`` installed along Python.
If ``pip`` is not found, you probably want to install the ``python3-pip`` package.

Installing the build-time requirements
--------------------------------------

PyBaMM comes with a DAE solver based on the IDA solver provided by the SUNDIALS library.
To use this solver, you must make sure that you have the necessary SUNDIALS components
installed on your system.

The IDA-based solver is currently unavailable on windows.
If you are running windows, you can simply skip this section and jump to :ref:`pybamm-install`.

.. code:: bash

	  # in the PyBaMM/ directory
	  nox -s pybamm-requires

This will download, compile and install the SuiteSparse and SUNDIALS libraries.
Both libraries are installed in ``~/.local``.

For users requiring more control over the installation process, the ``pybamm-requires`` session supports additional command-line arguments:

- ``--install-dir``: Specify a custom installation directory for SUNDIALS and SuiteSparse.

  Example:

  .. code:: bash

      nox -s pybamm-requires -- --install-dir [custom_directory_path]

- ``--force``: Force the installation of SUNDIALS and SuiteSparse, even if they are already found in the specified directory.

  Example:

  .. code:: bash

      nox -s pybamm-requires -- --force

Manual install of build time requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you'd rather do things yourself,

1. Make sure you have CMake installed
2. Compile and install SuiteSparse (PyBaMM only requires the ``KLU`` component).
3. Compile and install SUNDIALS.
4. Clone the pybind11 repository in the ``PyBaMM/`` directory (make sure the directory is named ``pybind11``).


PyBaMM ships with a Python script that automates points 2. and 3. You can run it with

.. code:: bash

	  python scripts/install_KLU_Sundials.py

This script supports optional arguments for custom installations:

- ``--install-dir``: Specify a custom installation directory for SUNDIALS and SuiteSparse.
  By default, they are installed in ``~/.local``.

  Example:

  .. code:: bash

      python scripts/install_KLU_Sundials.py --install-dir [custom_directory_path]

- ``--force``: Force the installation of SUNDIALS and SuiteSparse, even if they are already found in the specified directory.

  Example:

  .. code:: bash

      python scripts/install_KLU_Sundials.py --force

.. _pybamm-install:

Installing PyBaMM
-----------------

You should now have everything ready to build and install PyBaMM successfully.

Using ``Nox`` (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

	# in the PyBaMM/ directory
	nox -s dev

.. note::
	It is recommended to use ``--verbose`` or ``-v`` to see outputs of all commands run.

This creates a virtual environment ``venv/`` inside the ``PyBaMM/`` directory.
It comes ready with PyBaMM and some useful development tools like `pre-commit <https://pre-commit.com/>`_ and `ruff <https://beta.ruff.rs/docs/>`_.

You can now activate the environment with

.. tab:: GNU/Linux and MacOS (bash)

	.. code:: bash

		source venv/bin/activate

.. tab:: Windows

	.. code:: bash

		venv\Scripts\activate.bat

and run the tests to check your installation.

Manual install
~~~~~~~~~~~~~~

From the ``PyBaMM/`` directory, you can install PyBaMM using

.. code:: bash

	  pip install .

If you intend to contribute to the development of PyBaMM, it is convenient to
install in "editable mode", along with all the optional dependencies and useful
tools for development and documentation:

.. code:: bash

	  pip install -e .[all,dev,docs]

If you are using ``zsh`` or ``tcsh``, you would need to use different pattern matching:

.. code:: bash

	  pip install -e '.[all,dev,docs]'

Before you start contributing to PyBaMM, please read the `contributing
guidelines <https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md>`__.

Running the tests
-----------------

Using Nox (recommended)
~~~~~~~~~~~~~~~~~~~~~~~

You can use ``Nox`` to run the unit tests and example notebooks in isolated virtual environments.

The default command

.. code:: bash

	nox

will run pre-commit, install ``Linux`` and ``macOS`` dependencies, and run the unit tests.
This can take several minutes.

To just run the unit tests, use

.. code:: bash

	nox -s unit

Similarly, to run the integration tests, use

.. code:: bash

	nox -s integration

Finally, to run the unit and the integration suites sequentially, use

.. code:: bash

	nox -s tests

Using pytest
~~~~~~~~~~~~~~~~~~~~~~

You can run unit tests for PyBaMM using

.. code:: bash

	  # in the PyBaMM/ directory
	  pytest -m unit


The above uses pytest (in your current
Python environment) to run the unit tests. This can take a few minutes.

You can also use pytest to run the doctests:

.. code:: bash

	  pytest --doctest-plus src

Refer to the `testing <https://docs.pybamm.org/en/stable/source/user_guide/contributing.html#testing>`_ docs to find out more ways to test PyBaMM using pytest.

How to build the PyBaMM documentation
-------------------------------------

The documentation is built using

.. code:: bash

	  nox -s docs

This will build the documentation and serve it locally (thanks to `sphinx-autobuild <https://github.com/GaretJax/sphinx-autobuild>`_) for preview.
The preview will be updated automatically following changes.

Doctests, examples, and coverage
--------------------------------

``Nox`` can also be used to run doctests, run examples, and generate a coverage report using:

- ``nox -s examples``: Run the Jupyter notebooks in ``docs/source/examples/notebooks/``.
- ``nox -s examples -- <path-to-notebook-1.ipynb> <path-to_notebook-2.ipynb>``: Run specific Jupyter notebooks.
- ``nox -s scripts``: Run the example scripts in ``examples/scripts/``.
- ``nox -s doctests``: Run doctests.
- ``nox -s coverage``: Measure current test coverage and generate a coverage report.
- ``nox -s quick``: Run integration tests, unit tests, and doctests sequentially.

Extra tips while using ``Nox``
------------------------------

Here are some additional useful commands you can run with ``Nox``:

- ``--verbose or -v``: Enables verbose mode, providing more detailed output during the execution of Nox sessions.
- ``--list or -l``: Lists all available Nox sessions and their descriptions.
- ``--stop-on-first-error``: Stops the execution of Nox sessions immediately after the first error or failure occurs.
- ``--envdir <path>``: Specifies the directory where Nox creates and manages the virtual environments used by the sessions. In this case, the directory is set to ``<path>``.
- ``--install-only``: Skips the test execution and only performs the installation step defined in the Nox sessions.
- ``--nocolor``: Disables the color output in the console during the execution of Nox sessions.
- ``--report output.json``: Generates a JSON report of the Nox session execution and saves it to the specified file, in this case, "output.json".
- ``nox -s docs --non-interactive``: Builds the documentation without serving it locally (using ``sphinx-build`` instead of ``sphinx-autobuild``).

Troubleshooting
---------------

**Problem:** I have made edits to source files in PyBaMM, but these are
not being used when I run my Python script.

**Solution:** Make sure you have installed PyBaMM using the ``-e`` flag,
i.e. ``pip install -e .``. This sets the installed location of the
source files to your current directory.

**Problem:** Errors when solving model
``ValueError: Integrator name ida does not exist``, or
``ValueError: Integrator name cvode does not exist``.

**Solution:** This could mean that you have not installed
``scikits.odes`` correctly, check the instructions given above and make
sure each command was successful.

One possibility is that you have not set your ``LD_LIBRARY_PATH`` to
point to the sundials library, type ``echo $LD_LIBRARY_PATH`` and make
sure one of the directories printed out corresponds to where the
SUNDIALS libraries are located.

Another common reason is that you forget to install a BLAS library such
as OpenBLAS before installing SUNDIALS. Check the cmake output when you
configured SUNDIALS, it might say:

::

   -- A library with BLAS API not found. Please specify library location.
   -- LAPACK requires BLAS

If this is the case, on a Debian or Ubuntu system you can install
OpenBLAS using ``sudo apt-get install libopenblas-dev`` (or
``brew install openblas`` for Mac OS) and then re-install SUNDIALS using
the instructions above.
