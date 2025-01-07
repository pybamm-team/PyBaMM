Windows
=======

.. contents::

Prerequisites
-------------

To use PyBaMM, you must have Python 3.9, 3.10, 3.11, or 3.12 installed.

To install Python 3 download the installation files from `Python’s
website <https://www.python.org/downloads/windows/>`__. Make sure to
tick the box on ``Add Python 3.X to PATH``. For more detailed
instructions please see the `official Python on Windows
guide <https://docs.python.org/3.9/using/windows.html>`__.

Install PyBaMM
--------------

User install
~~~~~~~~~~~~

Launch the Command Prompt and go to the directory where you want to
install PyBaMM. You can find a reminder of how to navigate the terminal
`here <http://www.cs.columbia.edu/~sedwards/classes/2015/1102-fall/Command%20Prompt%20Cheatsheet.pdf>`__.

We recommend to install PyBaMM within a virtual environment, in order
not to alter any distribution python files.

To install ``virtualenv``, type:

.. code:: bash

   python -m pip install virtualenv

To create a virtual environment ``env`` within your current directory
type:

.. code:: bash

   python -m virtualenv env

or use any of your preferred environment management tools. You can then “activate”
the environment using:

.. code::

   env\Scripts\activate.bat

Now all the calls to pip described below will install PyBaMM and its
dependencies into the environment ``env``. When you are ready to exit
the environment and go back to your original system, just type:

.. code:: bash

   deactivate

PyBaMM can be installed via pip:

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

Installation using WSL
----------------------

If you want to install the optional PyBaMM solvers, you have to use the
Windows Subsystem for Linux (WSL). You can find the installation
instructions on the `Installation <windows-wsl.rst>`_ page.
