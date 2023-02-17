Windows Subsystem for Linux (WSL)
======================================

We recommend the use of Windows Subsystem for Linux (WSL) to install
PyBaMM, see the instructions below to get PyBaMM working using Windows,
WSL and VSCode.

.. contents::

Install WSL
-----------

Follow the instructions from Microsoft
`here <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`__.
When given the option, choose the Ubuntu 18.04 LTS distribution to
install. Best practices for setting up a WSL development environment can be found
`here <https://docs.microsoft.com/en-us/windows/wsl/setup/environment>`__.

Install PyBaMM
--------------

Open a terminal window in your installed Ubuntu distribution by
selecting “Ubuntu” from the start menu. This should give you a bash
prompt in your home directory.

To download the PyBaMM source code, you first need to install git, which
you can do by typing

.. code:: bash

   sudo apt install git-core

For easier integration with WSL, we recommend that you install PyBaMM in
your *Windows* Documents folder, for example by first navigating to

.. code:: bash

   $ cd /mnt/c/Users/USER_NAME/Documents

where USER_NAME is your username. Exact path to Windows documents may
vary. Now use git to clone the PyBaMM repository:

.. code:: bash

   git clone https://github.com/pybamm-team/PyBaMM.git

This will create a new directly called ``PyBaMM``, you can move to this
directory in bash using the ``cd`` command:

.. code:: bash

   cd PyBaMM

If you are unfamiliar with the linux command line, you might find it
useful to work through this
`tutorial <https://tutorials.ubuntu.com/tutorial/command-line-for-beginners>`__
provided by Ubuntu.

Now head over and follow the installation instructions for PyBaMM for
linux `here <GNU-linux.html>`__.

Use Visual Studio Code to run PyBaMM
------------------------------------

You will probably want to use a native Windows IDE such as Visual Studio
Code or the full Microsoft Visual Studio IDE. Both of these packages can
connect to WSL so that you can write Python code in a native windows
environment, while at the same time using WSL to run the code using your
installed Ubuntu distribution. The following instructions assume that
you are using Visual Studio Code.

First, setup VSCode to run within the ``PyBaMM`` directory that you
created above, using the instructions provided
`here <https://code.visualstudio.com/docs/remote/wsl>`__.

Once you have opened the ``PyBaMM`` folder in vscode, use the
``Extensions`` panel to install the ``Python`` extension from Microsoft.
Note that extensions are either installed on the Windows (Local) or on
in WSL (WSL:Ubuntu), so even if you have used VSCode previously with the
Python extension, you probably haven’t installed it in WSL. Make sure to
reload after installing the Python extension so that it is available.

If you have installed PyBaMM into the virtual environment ``env`` as in
the PyBaMM linux install guide, then VSCode should automatically start
using this environment and you should see something similar to “Python
3.6.8 64-bit (‘env’: venv)” in the bottom bar.

To test that vscode can run a PyBaMM script, navigate to the
``examples/scripts`` folder and right click on the ``create-model.py``
script. Select “Run current file in Python Interactive Window”. This
should run the script, which sets up and solves a model of SEI thickness
using PyBaMM. You should see a plot of SEI thickness versus time pop up
in the interactive window.

The Python Interactive Window in VSCode can be used to view plots, but
is restricted in functionality and cannot, for example, launch separate
windows to show plot. To setup an xserver on windows and use this to
launch windows for plotting, follow these instructions:

1. Install VcXsrv from
   `here <https://sourceforge.net/projects/vcxsrv/>`__.
2. Set the display port in the WSL command-line:
   ``echo "export DISPLAY=localhost:0.0" >>  ~/.bashrc``
3. Install python3-tk in WSL: ``sudo apt-get install python3-tk``
4. Set the matplotlib backend to TKAgg in WSL:
   ``echo "backend : TKAgg" >>  ~/.config/matplotlib/matplotlibrc``
5. Before running the code, just launch XLaunch (with the default
   settings) from within Windows. Then the code works as usual.
