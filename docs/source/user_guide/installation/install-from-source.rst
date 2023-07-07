Install from source (developer install)
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

- Python 3 (PyBaMM supports versions 3.8, 3.9, 3.10, and 3.11)
- The Python headers file for your current Python version.
- A BLAS library (for instance `openblas <https://www.openblas.net/>`_).
- A C compiler (ex: ``gcc``).
- A Fortran compiler (ex: ``gfortran``).
- ``graphviz`` (optional), if you wish to build the documentation locally.

You can install the above with

.. tab:: Ubuntu

	.. code:: bash

		sudo apt install python3.X python3.X-dev libopenblas-dev gcc gfortran graphviz

	Where ``X`` is the version sub-number.

.. tab:: MacOS

	.. code:: bash

		brew install python openblas gcc gfortran graphviz libomp

.. note::

	On Windows, you can install ``graphviz`` using the `Chocolatey <https://chocolatey.org/>`_ package manager, or
	follow the instructions on the `graphviz website <https://graphviz.org/download/>`_.

Finally, we recommend using `Nox <https://nox.thea.codes/en/stable/>`_.
You can install it with

.. code:: bash

	  python3.X -m pip install --user nox

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

.. _pybamm-install:

Installing PyBaMM
-----------------

You should now have everything ready to build and install PyBaMM successfully.

Using Nox (recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

	# in the PyBaMM/ directory
	nox -s dev

.. note::
    It is recommended to use ``--verbose`` or ``-v`` to see outputs of all commands run.

This creates a virtual environment ``.nox/dev`` inside the ``PyBaMM/`` directory.
It comes ready with PyBaMM and some useful development tools like `pre-commit <https://pre-commit.com/>`_ and `black <https://black.readthedocs.io/en/stable/>`_.

You can now activate the environment with

.. tab:: GNU/Linux and MacOS

	.. code:: bash

		source .nox/dev/bin/activate

.. tab:: Windows

	.. code:: bash

	  	.nox\dev\Scripts\activate.bat

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

If you are using ``zsh``, you would need to use different pattern matching:

.. code:: bash

	  pip install -e '.[all,dev,docs]'

Running the tests
-----------------

Using Nox (recommended)
~~~~~~~~~~~~~~~~~~~~~~~

You can use Nox to run the unit tests and example notebooks in isolated virtual environments.

The default command

.. code:: bash

	nox

will run pre-commit, install ``Linux`` dependencies, and run the unit tests.
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

Using the test runner 
~~~~~~~~~~~~~~~~~~~~~~

You can run unit tests for PyBaMM using

.. code:: bash

	  # in the PyBaMM/ directory
	  python run-tests.py --unit


The above starts a sub-process using the current python interpreter (i.e. using your current
Python environment) and run the unit tests. This can take a few minutes.

You can also use the test runner to run the doctests:

.. code:: bash

	  python run-tests.py --doctest

There is more to the PyBaMM test runner. To see a list of all options, type

.. code:: bash

	  python run-tests.py --help

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

- ``nox -s examples``: Run the example scripts in ``examples/scripts``.
- ``nox -s doctests``: Run doctests.
- ``nox -s coverage``: Measure current test coverage and generate a coverage report.

Extra tips while using Nox
--------------------------
Here are some additional useful commands you can run with ``Nox``:

- ``--verbose or -v``: Enables verbose mode, providing more detailed output during the execution of Nox sessions.
- ``--list or -l``: Lists all available Nox sessions and their descriptions.
- ``--stop-on-first-error``: Stops the execution of Nox sessions immediately after the first error or failure occurs.
- ``--envdir <path>``: Specifies the directory where Nox creates and manages the virtual environments used by the sessions. In this case, the directory is set to ``<path>``.
- ``--install-only``: Skips the test execution and only performs the installation step defined in the Nox sessions.
- ``--nocolor``: Disables the color output in the console during the execution of Nox sessions.
- ``--report output.json``: Generates a JSON report of the Nox session execution and saves it to the specified file, in this case, "output.json".
