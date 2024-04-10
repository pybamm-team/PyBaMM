Install from source (Windows Subsystem for Linux)
=================================================

To make it easier to install PyBaMM, we recommend using the Windows Subsystem for Linux (WSL) along with Visual Studio Code. This guide will walk you through the process.

Install WSL
-----------

Install Ubuntu 22.04 or 20.04 LTS as a distribution for WSL following `Microsoft's guide to install WSL <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__. For a seamless development environment, refer to `this guide <https://docs.microsoft.com/en-us/windows/wsl/setup/environment>`__.

Install PyBaMM
--------------

Get PyBaMM's Source Code
~~~~~~~~~~~~~~~~~~~~~~~~

1. Open a terminal in your Ubuntu distribution by selecting "Ubuntu" from the Start menu. You'll get a bash prompt in your home directory.

2. Install Git by typing the following command:

.. code:: bash

     sudo apt install git-core

3. Clone the PyBaMM repository:

.. code:: bash

     git clone https://github.com/pybamm-team/PyBaMM.git

4. Enter the PyBaMM Directory by running:

.. code:: bash

     cd PyBaMM

5. Follow the Installation Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the `installation instructions for PyBaMM on Linux <install-from-source.html>`__.

Using Visual Studio Code with the WSL
---------------------------------------

To use Visual Studio Code with the Windows Subsystem for Linux (WSL), follow these steps:

1. Open Visual Studio Code.
2. Install the "Remote - WSL" extension if not already installed.
3. Open the PyBaMM directory in Visual Studio Code.
4. In the bottom pane, select the "+" sign and choose "New WSL Window."
5. This opens a WSL terminal in the PyBaMM directory within the WSL.

Now you can develop and edit PyBaMM code using Visual Studio Code while utilizing the WSL environment.
