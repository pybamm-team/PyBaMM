===================
 GNU-Linux & MacOS
===================

.. contents::

Prerequisites
=============

To use and/or contribute to PyBaMM, you must have Python 3.8, 3.9, 3.10, or 3.11 installed.

.. tab:: Debian-based distributions (Debian, Ubuntu, Linux Mint)

   To install Python 3 on Debian-based distributions (Debian, Ubuntu, Linux Mint), open a terminal and run

   .. code:: bash

      sudo apt update
      sudo apt install python3

.. tab:: Fedora/CentOS

   On Fedora or CentOS, you can use DNF or Yum. For example

   .. code:: bash

      sudo dnf install python3

.. tab:: macOS

   On macOS, you can use the ``homebrew`` package manager. First, `install
   brew <https://docs.python-guide.org/starting/install3/osx/>`__:

   .. code:: bash

      ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   then follow instructions in the link on adding ``brew`` to path, and run

   .. code:: bash

      brew install python3

Install PyBaMM
==============

.. _user-install-label:

User install
------------

We recommend to install PyBaMM within a virtual environment, in order
not to alter any distribution Python files.
First, make sure you are using Python 3.8, 3.9, 3.10, or 3.11.
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

PyBaMM can be installed via pip. On macOS, it is necessary to install the `SUNDIALS <https://computing.llnl.gov/projects/sundials/>`__
library beforehand.

.. tab:: GNU/Linux and Windows

   In a terminal, run the following command:

   .. code:: bash

      pip install pybamm

.. tab:: macOS

   In a terminal, run the following commands:

   .. code:: bash

      brew install sundials
      pip install pybamm

PyBaMM’s dependencies (such as ``numpy``, ``scipy``, etc) will be
installed automatically when you install PyBaMM using ``pip``.

For an introduction to virtual environments, see
(https://realpython.com/python-virtual-environments-a-primer/).

.. _scikits.odes-label:

Optional - scikits.odes solver
------------------------------

Users can install `scikits.odes <https://github.com/bmcage/odes>`__ in
order to use the wrapped SUNDIALS ODE and DAE
`solvers <https://pybamm.readthedocs.io/en/latest/source/api/solvers/scikits_solvers.html>`__.
Currently, only GNU/Linux and macOS are supported.

.. tab:: GNU/Linux

   In a terminal, run the following commands:

   .. code:: bash

	   apt install libopenblas-dev
	   pybamm_install_odes

   The ``pybamm_install_odes`` command is installed with PyBaMM. It automatically downloads and installs the SUNDIALS library on your
   system (under ``~/.local``), before installing ``scikits.odes`` (by running ``pip install scikits.odes``).

.. tab:: macOS

   In a terminal, run the following command:

   .. code:: bash

	  pip install scikits.odes

   Assuming that SUNDIALS was installed as described :ref:`above<user-install-label>`.

Optional - JaxSolver
--------------------

Users can install ``jax`` and ``jaxlib`` to use the Jax solver.
Currently, only GNU/Linux and macOS are supported.

GNU/Linux and macOS
~~~~~~~~~~~~~~~~~~~

.. code:: bash

	  pybamm_install_jax

The ``pybamm_install_jax`` command is installed with PyBaMM. It automatically downloads and installs jax and jaxlib on your system.

Developer install
-----------------

If you wish to contribute to PyBaMM, you should get the latest version
from the GitHub repository. To do so, you must have ``Git`` and ``graphviz``
installed. For instance, run

   .. tab:: Debian-based distributions (Debian, Ubuntu, Linux Mint)

      In a terminal, run the following command:

      .. code:: bash

         sudo apt install git graphviz

   .. tab:: macOS

      In a terminal, run the following command:

      .. code:: bash

         brew install git graphviz

To install PyBaMM, the first step is to get the code by cloning this
repository

.. code:: bash

   git clone https://github.com/pybamm-team/PyBaMM.git
   cd PyBaMM

Then, to install PyBaMM as a `developer <https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md>`__, type

.. code:: bash

   pip install -e .[dev,docs]

or on ``zsh`` shells, type

.. code:: bash
   
   pip install -e .'[dev,docs]'

To check whether PyBaMM has installed properly, you can run the tests:

.. code:: bash

   python3 run-tests.py --unit

Before you start contributing to PyBaMM, please read the `contributing
guidelines <https://github.com/pybamm-team/PyBaMM/blob/develop/CONTRIBUTING.md>`__.

Uninstall PyBaMM
================

PyBaMM can be uninstalled by running

.. code:: bash

   pip uninstall pybamm

in your virtual environment.

Troubleshooting
===============

**Problem:** I’ve made edits to source files in PyBaMM, but these are
not being used when I run my Python script.

**Solution:** Make sure you have installed PyBaMM using the ``-e`` flag,
i.e. ``pip install -e .``. This sets the installed location of the
source files to your current directory.

**Problem:** Errors when solving model
``ValueError: Integrator name ida does not exsist``, or
``ValueError: Integrator name cvode does not exsist``.

**Solution:** This could mean that you have not installed
``scikits.odes`` correctly, check the instructions given above and make
sure each command was successful.

One possibility is that you have not set your ``LD_LIBRARY_PATH`` to
point to the sundials library, type ``echo $LD_LIBRARY_PATH`` and make
sure one of the directories printed out corresponds to where the
sundials libraries are located.

Another common reason is that you forget to install a BLAS library such
as OpenBLAS before installing sundials. Check the cmake output when you
configured Sundials, it might say:

::

   -- A library with BLAS API not found. Please specify library location.
   -- LAPACK requires BLAS

If this is the case, on a Debian or Ubuntu system you can install
OpenBLAS using ``sudo apt-get install libopenblas-dev`` (or
``brew install openblas`` for Mac OS) and then re-install sundials using
the instructions above.
