Install from source (developer install)
=========================================

.. contents::

Prerequisites
---------------

The following instructions are valid for both GNU/Linux distributions and MacOS.
If you are running Windows, consider using the `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_.

To obtain the PyBaMM source code, clone the GitHub repository

.. code:: bash

	  git clone https://github.com/pybamm-team/PyBaMM.git

or download the source archive on the repository's homepage.

To install PyBaMM, you will need:

- Python 3.6 and/or 3.7
- The python headers
- A BLAS library (for instance `openblas <https://www.openblas.net/>`_)
- A C compiler (ex: :code:`gcc`)
- A Fortran compiler (ex: :code:`gfortran`)

On Ubuntu, you can install the above with

.. code:: bash

	  sudo apt install python3 python3-dev python3.7 python3.7-dev libopenblas-dev gcc gfortran

On MacOS,

.. code:: bash

	  brew install python openblas gcc gfortran

Finally, the following assumes that you have Tox installed:

.. code:: bash

	  python3.7 -m pip install --upgrade pip tox

Depending on your operating system, you may or may not have :code:`pip` installed along python.
If :code:`pip` is not found, you probably want to install the :code:`python3-pip` package.

Installation
-------------
To install PyBaMM

.. code:: bash

	  cd PyBaMM/
	  tox -e sundials
	  tox -e dev

The first command will install the Sundials library in :code:`~/.local`.
The second step creates a virtual environment, ready with PyBaMM and the useful tools `flake8 <https://flake8.pycqa.org/en/latest/>`_ and `black <https://black.readthedocs.io/en/stable/>`_.

You can now activate the environment with

.. code:: bash

	  source .tox/dev/bin/activate

Running the tests
--------------------
You can use Tox to run the unit tests and example notebooks in isolated virtual environments.

The default command

.. code:: bash

	  tox

will run the unit tests, doctests and check for style in both python3.6 and python3.7, assuming you have both versions installed.
If you want to run the tests for a specific version, say 3.6, run instead

.. code:: bash

	  tox -e py36

The documentation is built using

.. code:: bash

	  tox -e docs

This will build the documentation and serve it on the localhost (thanks to `sphinx-autobuild <https://github.com/GaretJax/sphinx-autobuild>`_) for preview.
The preview will be updated automatically following changes.

In addition, the following tox commands are available:

- :code:`tox -e examples`: Run the example scripts in :code:`examples/scripts`
- :code:`tox -e flake8`: Check for PEP8 compliance
- :code:`tox -e doctests`: Run doctests
- :code:`tox -e coverage`: Measure current test coverage




