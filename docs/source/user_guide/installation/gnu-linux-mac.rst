GNU/Linux & macOS
=================

.. contents::

Prerequisites
-------------

To use PyBaMM, you must have Python 3.8, 3.9, 3.10, 3.11, or 3.12 installed.

.. tab:: Debian-based distributions (Debian, Ubuntu)

   To install Python 3 on Debian-based distributions (Debian, Ubuntu), open a terminal and run

   .. code:: bash

      sudo apt-get update
      sudo apt-get install python3

.. tab:: macOS

   On macOS, you can use the ``homebrew`` package manager. First, `install
   brew <https://docs.python-guide.org/starting/install3/osx/>`__:

   .. code:: bash

      ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   then follow instructions in the link on adding ``brew`` to path, and run

   .. code:: bash

      brew install python3


Install PyBaMM
--------------

.. _user-install-label:

User install
~~~~~~~~~~~~

We recommend to install PyBaMM within a virtual environment, in order
not to alter any distribution Python files.
First, make sure you are using Python 3.8, 3.9, 3.10, 3.11, or 3.12.
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

.. tab:: GNU/Linux

   In a terminal, run the following command:

   .. code:: bash

      pip install pybamm

.. tab:: macOS

   In a terminal, run the following command:

   .. code:: bash

      pip install pybamm

PyBaMM’s required dependencies (such as ``numpy``, ``casadi``, etc) will be
installed automatically when you install PyBaMM using ``pip``.

For an introduction to virtual environments, see
(https://realpython.com/python-virtual-environments-a-primer/).

.. _scikits.odes-label:

Optional - scikits.odes solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can install `scikits.odes <https://github.com/bmcage/odes>`__ to utilize its interfaced SUNDIALS ODE and DAE `solvers <https://docs.pybamm.org/en/latest/source/api/solvers/scikits_solvers.html>`__ wrapped in PyBaMM.

.. note::

   Currently, only GNU/Linux and macOS are supported.

.. note::

   The ``scikits.odes`` solver is not supported on Python 3.12 yet. Please refer to https://github.com/bmcage/odes/issues/162.
   There is support for Python 3.8, 3.9, 3.10, and 3.11.

.. tab:: Debian/Ubuntu

   In a terminal, run the following commands:

   .. code:: bash

      sudo apt-get install libopenblas-dev cmake
      pybamm_install_odes

   This will compile and install SUNDIALS for the system (under ``~/.local``), before installing ``scikits.odes``. (Alternatively, one can install SUNDIALS without this script and run ``pip install pybamm[odes]`` to install ``pybamm`` with ``scikits.odes``.)

.. tab:: macOS

   In a terminal, run the following command:

   .. code:: bash

      brew install openblas gcc gfortran cmake
      pybamm_install_odes

The ``pybamm_install_odes`` command, installed with PyBaMM, automatically downloads and installs the SUNDIALS library on your
system (under ``~/.local``), before installing `scikits.odes <https://scikits-odes.readthedocs.io/en/stable/installation.html>`__ . (Alternatively, one can install SUNDIALS without this script and run ``pip install pybamm[odes]`` to install ``pybamm`` with `scikits.odes <https://scikits-odes.readthedocs.io/en/stable/installation.html>`__)

To avoid installation failures when using ``pip install pybamm[odes]``, make sure to set the ``SUNDIALS_INST`` environment variable. If you have installed SUNDIALS using Homebrew, set the variable to the appropriate location. For example:

.. code:: bash

   export SUNDIALS_INST=$(brew --prefix sundials)

Ensure that the path matches the installation location on your system. You can verify the installation location by running:

.. code:: bash

   brew info sundials

Look for the installation path, and use that path to set the ``SUNDIALS_INST`` variable.

Note: The location where Homebrew installs SUNDIALS might vary based on the system architecture (ARM or Intel). Adjust the path in the ``export SUNDIALS_INST`` command accordingly.

To avoid manual setup of path the ``pybamm_install_odes`` is recommended for a smoother installation process, as it takes care of automatically downloading and installing the SUNDIALS library on your system.

Optional - JaxSolver
~~~~~~~~~~~~~~~~~~~~

Users can install ``jax`` and ``jaxlib`` to use the Jax solver.

.. note::

   The Jax solver is only supported for Python versions 3.9 through 3.12.

.. code:: bash

	  pip install "pybamm[jax]"

The ``pip install "pybamm[jax]"`` command automatically downloads and installs ``pybamm`` and the compatible versions of ``jax`` and ``jaxlib`` on your system. (``pybamm_install_jax`` is deprecated.)

Uninstall PyBaMM
----------------

PyBaMM can be uninstalled by running

.. code:: bash

   pip uninstall pybamm

in your virtual environment.
