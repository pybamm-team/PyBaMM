GNU/Linux & macOS
=================

.. contents::

Prerequisites
-------------

To use PyBaMM, you must have Python 3.10, 3.11, 3.12, 3.13 installed.

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
First, make sure you are using Python 3.10, 3.11, 3.12, 3.13.
To create a virtual environment ``env`` within your current directory type:

.. code:: bash

   virtualenv env

or use any of your preferred environment management tools. You can then “activate”
the environment using:

.. code:: bash

   source env/bin/activate

Now all the calls to pip described below will install PyBaMM and its
dependencies into the environment ``env``. When you are ready to exit
the environment and go back to your original system, just type:

.. code:: bash

   deactivate

PyBaMM can be installed via ``pip`` or ``conda``.
library beforehand.

.. tab:: pip

   .. code:: bash

      pip install pybamm

.. tab:: conda

   .. code:: bash

      conda install -c conda-forge pybamm-base

PyBaMM’s :ref:`required dependencies <install-required-dependencies>`

(such as ``numpy``, ``casadi``, etc) will be installed automatically when you
install ``pybamm`` using ``pip`` or ``pybamm-base`` using ``conda``.

For an introduction to virtual environments, see
(https://realpython.com/python-virtual-environments-a-primer/).

.. _optional-jaxsolver:

Optional - JaxSolver
~~~~~~~~~~~~~~~~~~~~

Users can install ``jax`` and ``jaxlib`` to use the Jax solver.

.. code:: bash

	  pip install "pybamm[jax]"

The ``pip install "pybamm[jax]"`` command automatically downloads and installs ``pybamm`` and the compatible versions of ``jax`` and ``jaxlib`` on your system.

PyBaMM's full `conda-forge distribution <index.rst#installation>`_ (``pybamm``) includes ``jax`` and ``jaxlib`` by default.

Uninstall PyBaMM
----------------

PyBaMM can be uninstalled by running

.. code:: bash

   pip uninstall pybamm

in your virtual environment.
