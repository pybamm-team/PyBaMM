GNU/Linux & macOS
=================

.. contents::

Prerequisites
-------------

To use PyBaMM, you must have Python 3.9, 3.10, 3.11, or 3.12 installed.

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

      brew install python


Install PyBaMM
--------------

.. _user-install-label:

User install
~~~~~~~~~~~~

We recommend to install PyBaMM within a virtual environment, in order
not to alter any distribution Python files.
First, make sure you are using Python 3.9, 3.10, 3.11, or 3.12.
To create a virtual environment ``env`` within your current directory type:

.. code:: bash

   virtualenv env

or use any of your preferred environment management tool. You can then “activate”
the environment using:

.. code:: bash

   source env/bin/activate

Now all the calls to pip described below will install PyBaMM and its
dependencies into the environment ``env``. When you are ready to exit
the environment and go back to your original system, just type:

.. code:: bash

   deactivate

PyBaMM can be installed via ``pip`` or ``conda``. On macOS, it is necessary
to install the `SUNDIALS <https://computing.llnl.gov/projects/sundials/>`__
library beforehand.

.. tab:: pip

   .. code:: bash

      pip install pybamm

.. tab:: conda

   .. code:: bash

      conda install -c conda-forge pybamm-base

PyBaMM’s `required dependencies <index.html#install-required-dependencies>`_
(such as ``numpy``, ``casadi``, etc) will be installed automatically when you
install ``pybamm`` using ``pip`` or ``pybamm-base`` using ``conda``.

For an introduction to virtual environments, see
(https://realpython.com/python-virtual-environments-a-primer/).


Optional - JaxSolver
~~~~~~~~~~~~~~~~~~~~

Users can install ``jax`` and ``jaxlib`` to use the Jax solver.

.. code:: bash

	  pip install "pybamm[jax]"

The ``pip install "pybamm[jax]"`` command automatically downloads and installs ``pybamm`` and the compatible versions of ``jax`` and ``jaxlib`` on your system.

PyBaMM's full `conda-forge distribution <index.html#installation>`_ (``pybamm``) includes ``jax`` and ``jaxlib`` by default.

.. _optional-iree-mlir-support:

Optional - IREE / MLIR support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can install ``iree`` (for MLIR just-in-time compilation) to use for main expression evaluation in the IDAKLU solver. Requires ``jax``.
At the moment, PyBaMM's conda-forge distribution does not include the ``IDAKLUSolver``.

.. code:: bash

   pip install "pybamm[iree,jax]"

The ``pip install "pybamm[iree,jax]"`` command automatically downloads and installs ``pybamm`` and the compatible versions of ``jax`` and ``iree`` onto your system.

Uninstall PyBaMM
----------------

PyBaMM can be uninstalled by running

.. code:: bash

   pip uninstall pybamm

in your virtual environment.
