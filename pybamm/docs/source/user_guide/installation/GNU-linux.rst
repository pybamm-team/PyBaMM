GNU-Linux & MacOS
=================

.. contents::

Prerequisites
-------------

To use PyBaMM, you must have Python 3.8, 3.9, 3.10, or 3.11 installed.

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
--------------

.. _user-install-label:

User install
~~~~~~~~~~~~

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

.. tab:: GNU/Linux

   In a terminal, run the following command:

   .. code:: bash

      pip install pybamm

.. tab:: macOS

   In a terminal, run the following commands:

   .. code:: bash

      brew install sundials
      pip install pybamm

PyBaMM’s required dependencies (such as ``numpy``, ``casadi``, etc) will be
installed automatically when you install PyBaMM using ``pip``.

For an introduction to virtual environments, see
(https://realpython.com/python-virtual-environments-a-primer/).

.. _scikits.odes-label:

Optional - scikits.odes solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can install `scikits.odes <https://github.com/bmcage/odes>`__ in
order to use the wrapped SUNDIALS ODE and DAE
`solvers <https://docs.pybamm.org/en/latest/source/api/solvers/scikits_solvers.html>`__.
Currently, only GNU/Linux and macOS are supported.

.. tab:: GNU/Linux

   In a terminal, run the following commands:

   .. code:: bash

	   apt install libopenblas-dev
	   pybamm_install_odes

   The ``pybamm_install_odes`` command is installed with PyBaMM. It automatically downloads and installs the SUNDIALS library on your
   system (under ``~/.local``), before installing ``scikits.odes``. (Alternatively, one can install SUNDIALS without this script and run ``pip install pybamm[odes]`` to install ``pybamm`` with ``scikits.odes``.)

.. tab:: macOS

   In a terminal, run the following command:

   .. code:: bash

	  pip install scikits.odes

   Assuming that SUNDIALS was installed as described :ref:`above<user-install-label>`.

Optional - JaxSolver
~~~~~~~~~~~~~~~~~~~~

Users can install ``jax`` and ``jaxlib`` to use the Jax solver.

.. note::

   The Jax solver is not supported on Python 3.8. It is supported on Python 3.9, 3.10, and 3.11.

.. code:: bash

	  pip install "pybamm[jax]"

The ``pip install "pybamm[jax]"`` command automatically downloads and installs ``pybamm`` and the compatible versions of ``jax`` and ``jaxlib`` on your system. (``pybamm_install_jax`` is deprecated.)

Uninstall PyBaMM
----------------

PyBaMM can be uninstalled by running

.. code:: bash

   pip uninstall pybamm

in your virtual environment.
