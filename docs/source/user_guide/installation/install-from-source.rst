Install from source
===================

.. contents::

This page describes the build and installation of PyBaMM from the source code, available on GitHub. Note that this is **not the recommended approach for most users** and should be reserved to people wanting to participate in the development of PyBaMM, or people who really need to use bleeding-edge feature(s) not yet available in the latest released version. If you do not fall in the two previous categories, you would be better off installing PyBaMM using pip or conda.

Lastly, familiarity with the Python ecosystem is recommended (pip, virtualenvs).
Here is a gentle introduction/refresher: `Python Virtual Environments: A Primer <https://realpython.com/python-virtual-environments-a-primer/>`_.


Prerequisites
---------------

.. tab:: Ubuntu/Debian

	To install PyBaMM, you will need:

	- Python 3 (PyBaMM supports versions 3.9, 3.10, 3.11, and 3.12)
	- The Python headers file for your current Python version.
	- A BLAS library (for instance `openblas <https://www.openblas.net/>`_).
	- A C compiler (ex: ``gcc``).
	- A Fortran compiler (ex: ``gfortran``).
	- ``graphviz`` (optional), if you wish to build the documentation locally.
	- ``pandoc`` (optional) to convert the example Jupyter notebooks when building the documentation.
	- ``texlive-latex-extra`` (optional) to convert model equations in latex.
	- ``dvipng`` (optional) to convert a DVI file to a PNG image.

	You can install the above with

	.. code:: bash

		sudo apt install python3.X python3.X-dev libopenblas-dev gcc gfortran graphviz cmake pandoc texlive-latex-extra dvipng

	Where ``X`` is the version sub-number.

	.. note::

		If you are using some other linux distribution you can install the equivalent packages for ``python3, cmake, gcc, gfortran, openblas, pandoc, texlive-latex-extra, dvipng``.

	Finally, we recommend using `Nox <https://nox.thea.codes/en/stable/>`_.
	You can install it to your local user account (make sure you are not within a virtual environment) with

	.. code:: bash

		python3.X -m pip install --user nox

	Note that running ``nox`` will create new virtual environments for you to use, so you do not need to create one yourself.

	Depending on your operating system, you may or may not have ``pip`` installed along Python.
	If ``pip`` is not found, you probably want to install the ``python3-pip`` package.

.. tab:: MacOS

	To install PyBaMM, you will need:

	- Python 3 (PyBaMM supports versions 3.9, 3.10, 3.11, and 3.12)
	- The Python headers file for your current Python version.
	- A BLAS library (for instance `openblas <https://www.openblas.net/>`_).
	- A C compiler (ex: ``gcc``).
	- A Fortran compiler (ex: ``gfortran``).
	- ``graphviz`` (optional), if you wish to build the documentation locally.
	- ``pandoc`` (optional) to convert the example Jupyter notebooks when building the documentation.
	- ``texlive-latex-extra`` (optional) to convert model equations in latex.
	- ``dvipng`` (optional) to convert a DVI file to a PNG image.

	You can install the above with

	.. code:: bash

		brew install python openblas gcc gfortran graphviz libomp cmake pandoc

	Finally, we recommend using `Nox <https://nox.thea.codes/en/stable/>`_.
	You can install it to your local user account (make sure you are not within a virtual environment) with

	.. code:: bash

		python3.X -m pip install --user nox

	Note that running ``nox`` will create new virtual environments for you to use, so you do not need to create one yourself.

	Depending on your operating system, you may or may not have ``pip`` installed along Python.
	If ``pip`` is not found, you probably want to install the ``python3-pip`` package.

.. tab:: Windows

	To use PyBaMM, you must have Python 3.9, 3.10, 3.11, or 3.12 installed.

	To install Python 3.X, download the installation files from `Python’s
	website <https://www.python.org/downloads/windows/>`_. Make sure to
	tick the box on ``Add Python 3.X to PATH``. For more detailed
	instructions please see the `official Python on Windows
	guide <https://docs.python.org/3.9/using/windows.html>`__.

	(Optional) If you wish to build the documentation locally, you can install ``graphviz`` using the `Chocolatey <https://chocolatey.org/>`_ package manager,

	.. code:: bash

		  choco install graphviz

	or follow the instructions on the `graphviz website <https://graphviz.org/download/>`_.

	Finally, we recommend using `Nox <https://nox.thea.codes/en/stable/>`_.
	You can install it to your local user account (make sure you are not within a virtual environment) with

	.. code:: bash

		python -m pip install --user nox

	Note that running ``nox`` will create new virtual environments for you to use, so you do not need to create one yourself.

	After installing, you must add the following location to your ``Path`` environment variable to run ``nox`` in a terminal, like Command Prompt.

	.. code::

		C:\Users\<USERNAME>\AppData\Roaming\Python\Python3<X>\Scripts

	Make sure to replace ``<USERNAME>`` with your user name and ``X`` with your Python subversion.

.. _install-build-time:

Installing the build-time requirements
--------------------------------------

PyBaMM comes with a DAE solver based on the IDA solver provided by the SUNDIALS library. To use this solver, you must ensure you have the necessary SUNDIALS components installed on your system.
To install SUNDIALS, you will need to install the following components:

.. tab:: GNU/Linux and MacOS

	.. code:: bash

		# in the project root directory
		nox -s pybamm-requires

	This will download, compile and install the SuiteSparse and SUNDIALS libraries.
	Both libraries are installed in ``PyBaMM/sundials_KLU_libs``.

	For users requiring more control over the installation process, the ``pybamm-requires`` session supports additional command-line arguments:

	- ``--install-dir``: Specify a custom installation directory for SUNDIALS and SuiteSparse.

	Example:

	.. code:: bash

		nox -s pybamm-requires -- --install-dir [custom_directory_path]

	After running this command, you need to export the environment variable ``INSTALL_DIR`` with the custom installation directory to link the libraries with the solver:

	.. code:: bash

		export INSTALL_DIR=[custom_directory_path]

	- ``--force``: Force the installation of SUNDIALS and SuiteSparse, even if they are already found in the specified directory.

	Example:

	.. code:: bash

		nox -s pybamm-requires -- --force

.. tab:: Windows

	VCPKG

	- VCPKG automatically installs the required libraries for you during the build process. To install VCPKG, follow the instructions in `Microsoft's official documentation <https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-cmd>`_.
	- Make sure to add the location to ``Path`` and a new environment variable ``VCPKG_ROOT_DIR`` with the location to VCPKG.

	C++ Compiler

	- PyBaMM uses a recent version of Microsoft Visual C++ (MSVC), which you can get using `Build Tools for Visual Studio Code 2022 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022/>`_.
	- Note that you won't need Visual Studio 2022 entirely; just ``Desktop development with C++`` will suffice.

	CMake

	- ``CMake`` is required to install the SUNDIALS and other libraries for the ``IDAKLU`` solver.
	- To install it, follow the link to the `official CMake downloads page <https://cmake.org/download/>`_.
	- Download an installer based on your system's architecture, i.e. ``x32/x64``, and check ``Add CMake to the PATH environment variable`` during installation.

Manual install of build time requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: GNU/Linux and MacOS

	If you'd rather do things yourself,

	1. Make sure you have CMake installed
	2. Compile and install SuiteSparse (PyBaMM only requires the ``KLU`` component).
	3. Compile and install SUNDIALS.


	PyBaMM ships with a Python script that automates points 2. and 3. You can run it with

	.. code:: bash

		python scripts/install_KLU_Sundials.py

	This script supports optional arguments for custom installations:

	- ``--install-dir``: Specify a custom installation directory for SUNDIALS and SuiteSparse.

	By default, they are installed in ``PyBaMM/sundials_KLU_libs``.

	Example:

	.. code:: bash

		python scripts/install_KLU_Sundials.py --install-dir [custom_directory_path]

	After running this command, you need to export the environment variable ``INSTALL_DIR`` with the custom installation directory to link the libraries with the solver:

	.. code:: bash

		export INSTALL_DIR=[custom_directory_path]

	- ``--force``: Force the installation of SUNDIALS and SuiteSparse, even if they are already found in the specified directory.

	Example:

	.. code:: bash

		python scripts/install_KLU_Sundials.py --force

.. tab:: Windows

	There isn't a method to manually build SUNDIALS on Windows. However, if you've followed the instructions provided in the previous section (:ref:`install-build-time`), you should have successfully set up SUNDIALS on your Windows system.

	With SUNDIALS already set up, you can proceed directly to the next section.

.. _pybamm-install:

Installing PyBaMM
-----------------

.. tab:: GNU/Linux and MacOS

	To obtain the PyBaMM source code, clone the GitHub repository

	.. code:: bash

		git clone https://github.com/pybamm-team/PyBaMM.git

	or download the source archive on the repository's homepage.

	You should now have everything ready to build and install PyBaMM successfully.

.. tab:: Windows

	Open a Command Prompt and navigate to the folder where you want to install PyBaMM,

	1. Obtain the PyBaMM source code by cloning the GitHub repository or downloading the source archive on the repository's homepage.

	   .. code:: bash

			git clone https://github.com/pybamm-team/PyBaMM.git

	2. PyBaMM requires setting a few environment variables to install the IDAKLU solver. To put them automatically, run the following ``.bat`` script using the following command from the project root directory.

	   .. code::

			.\Scripts\windows_setup.bat

	The script sets the following environment variables with the following defaults.

	.. code-block:: bash

		PYBAMM_USE_VCPKG: ON
		VCPKG_DEFAULT_TRIPLET: x64-windows-static-md
		VCPKG_FEATURE_FLAGS: manifests,registries

	.. note::

		Ensure you set the ``VCPKG_ROOT_DIR`` environment variable to the location where VCPKG is installed.

	You should now have everything ready to build and install PyBaMM successfully.

Using ``Nox`` (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyBaMM needs a few build-time dependencies when installing PyBaMM in "editable mode" without build-isolation. `TOML Kit <https://tomlkit.readthedocs.io/en/latest/>`_, a ``toml`` table parser, automates this process for you.

To install ``tomlkit`` to your local user account (ensure you are not within a virtual environment), use the following command:

.. code:: bash

	  python -m pip install --user tomlkit

To install PyBaMM, execute the following command

.. code:: bash

	# in the project root directory
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

	.. code::

		venv\Scripts\activate.bat

and run the tests to check your installation.

Manual install
~~~~~~~~~~~~~~

We recommend installing PyBaMM within a virtual environment to avoid altering any distribution of Python files.

.. tab:: GNU/Linux and MacOS

	To create a virtual environment ``env`` within your current directory type:

	.. code:: bash

		virtualenv env

	You can then “activate” the environment using:

	.. code:: bash

		source env/bin/activate

	Now all the calls to pip described below will install PyBaMM and its
	dependencies into the environment ``env``. When you are ready to exit
	the environment and go back to your original system, just type:

	.. code:: bash

		deactivate

	From the ``PyBaMM/`` directory inside the virtual environment, you can install PyBaMM using

	.. code:: bash

		pip install .

	If you intend to contribute to the development of PyBaMM, it is convenient to
	install in "editable mode", along with all the optional dependencies and useful
	tools for development and documentation.

	You can install PyBaMM in an "editable mode" using the following command:

	.. code:: bash

		pip install -e .[all,dev,docs]

	You can also install PyBaMM with "partial rebuilds" enabled. But, due to the ``--no-build-isolation`` flag, you first need to install the build-time dependencies inside the virtual environment:

	.. code:: bash

		pip install scikit-build-core pybind11 casadi cmake

	You can now install PyBaMM in "editable mode" with "partial rebuilds" for development using the following command:

	.. code:: bash

		pip install --no-build-isolation --config-settings=editable.rebuild=true -e .[all,dev,docs]

	If you are using ``zsh`` or ``tcsh``, you would need to use different pattern matching:

	.. code:: bash

		pip install --no-build-isolation --config-settings=editable.rebuild=true -e '.[all,dev,docs]'


.. tab:: Windows

	You can install ``virtualenv`` by executing the following command:

	.. code:: bash

		python -m pip install virtualenv

	Create a virtual environment ``venv`` within the PyBaMM root directory:

	.. code:: bash

		python -m virtualenv venv

	You can then “activate” the environment using:

	.. code:: text

		venv\Scripts\activate.bat

	Now, all the calls to pip described below will install PyBaMM and its
	dependencies into the environment ``venv``. When you are ready to exit
	the environment and go back to your original system, just type:

	.. code:: bash

		deactivate

	From the ``PyBaMM/`` directory inside the virtual environment, you can install PyBaMM using

	.. code:: bash

		pip install .

	If you intend to contribute to the development of PyBaMM, it is convenient to
	install in "editable mode", along with all the optional dependencies and useful
	tools for development and documentation.

	You can install PyBaMM in an "editable mode" using the following command:

	.. code:: bash

		pip install -e .[all,dev,docs]

	You can also install PyBaMM with "partial rebuilds" enabled. But, due to the ``--no-build-isolation`` flag, you first need to install the build-time dependencies inside the virtual environment:

	.. code:: bash

		pip install scikit-build-core pybind11

	You can now install PyBaMM in "editable mode" with "partial rebuilds" for development using the following command:

	.. code:: bash

		pip install --no-build-isolation --config-settings=editable.rebuild=true -e .[all,dev,docs]

.. note::

	The "partial rebuilds" feature is still experimental and may break. To learn more, check out `scikit-build-core's official documentation <https://scikit-build-core.readthedocs.io/en/stable/configuration.html#editable-installs/>`_.

Before you start contributing to PyBaMM, please read the `contributing
guidelines <https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md>`__.

Running the tests
-----------------

.. tab:: GNU/Linux

	Make sure to install ``texlive-latex-extra`` to pass all tests. Otherwise, you can safely ignore the failed tests needing it.

.. tab:: Windows and MacOS

	Make sure to install ``graphviz`` using the `Chocolatey <https://chocolatey.org/>`_ package manager (Windows) or using ``brew`` (MacOS) to pass all the tests. Otherwise, you can safely ignore the failed tests needing ``graphviz``.

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

Using ``pytest``
~~~~~~~~~~~~~~~~

You can run unit tests for PyBaMM inside the virtual environment using

.. code:: bash

	  pytest -m unit

You can run integration tests using

.. code:: bash

	  pytest -m integration

To run both unit and integration tests, use the following command:

.. code:: bash

	  pytest -m tests

You can also use ``pytest`` to test example notebooks.

.. code:: bash

	  pytest --nbmake docs/source/examples/

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

.. tab:: GNU/Linux and MacOS

	**Problem:** I ran a ``nox``/python build command and encountered ``Could NOT find SUNDIALS (missing: SUNDIALS_INCLUDE_DIR SUNDIALS_LIBRARIES)`` error.

	**Solution:** This error occurs when the build system, ``scikit-build-core``, can not find the SUNDIALS libraries to build the ``IDAKLU`` solver.

	1. Run the following command to ensure SUNDIALS libraries are installed:

	   .. code:: bash

			nox -s pybamm-requires -- --force

	2. If you are using a custom directory for SUNDIALS, set the ``INSTALL_DIR`` environment variable to specify the path:

	   .. code:: bash

			export INSTALL_DIR=[custom_directory_path]

	**Problem:** When installing SUNDIALS, I encountered ``CMake Error: The source "../CMakeLists.txt" does not match the source "../CMakeLists.txt" used to generate cache`` error.

	**Solution:** This error occurs when there is a delay between installing and downloading SUNDIALS libraries.

	1. Remove the following directories from the PyBaMM directory if they exist:

	   a. ``download_KLU_Sundials``
	   b. ``sundials_KLU_libs``
	   c. Any custom directory you have set for installation

	2. Re-run the command to install SUNDIALS.
	3. If you are using a custom directory, make sure to set the ``INSTALL_DIR`` environment variable:

	   .. code:: bash

			export INSTALL_DIR=[custom_directory_path]

	**Problem:** I have made edits to source files in PyBaMM, but these are
	not being used when I run my Python script.

	**Solution:** Make sure you have installed PyBaMM using the ``-e`` flag, like so:

	.. code:: bash

		pip install -e .

	If you want to install with "partial rebuilds" enabled, use this command:

	.. code:: bash

		pip install --no-build-isolation --config-settings=editable.rebuild=true -e.

	Make sure you have the build-time dependencies installed beforehand.

	These commands set the installed location of the
	source files to your current directory.

.. tab:: Windows

	**Problem:** I ran a ``nox``/python build command and encountered ``Configuring incomplete, errors occurred!`` error.

	**Solution:** This can occur when the environment variables are improperly set in the terminal.

	1. Make sure you've set environment variables before running any ``nox``/python build command.
	2. Try running the build command again in the same terminal.

	**Problem:** I have made edits to source files in PyBaMM, but these are
	not being used when I run my Python script.

	**Solution:** Make sure you have installed PyBaMM using the ``-e`` flag, like so:

	.. code:: bash

		pip install -e .

	If you want to install with "partial rebuilds" enabled, use this command:

	.. code:: bash

		pip install --no-build-isolation --config-settings=editable.rebuild=true -e.

	Make sure you have the build-time dependencies installed beforehand.

	These commands set the installed location of the
	source files to your current directory.
