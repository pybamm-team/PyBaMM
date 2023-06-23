Windows
==========

.. contents::

Prerequisites
-------------

To use and/or contribute to PyBaMM, you must have Python 3.8, 3.9, 3.10, or 3.11 installed.

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

To install virtualenv type:

.. code:: bash

   python -m pip install virtualenv

To create a virtual environment ``env`` within your current directory
type:

.. code:: bash

   python -m virtualenv env

You can then “activate” the environment using:

.. code:: cmd

   env\Scripts\activate.bat

Now all the calls to pip described below will install PyBaMM and its
dependencies into the environment ``env``. When you are ready to exit
the environment and go back to your original system, just type:

.. code:: bash

   deactivate

PyBaMM can be installed via pip:

.. code:: bash

   pip install pybamm

PyBaMM’s dependencies (such as ``numpy``, ``scipy``, etc) will be
installed automatically when you install PyBaMM using ``pip``.

For an introduction to virtual environments, see
(https://realpython.com/python-virtual-environments-a-primer/).

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
instructions `here <INSTALL-WINDOWS-WSL.md>`__.
