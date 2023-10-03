Install from source (Docker)
============================

.. contents::

This page describes the build and installation of PyBaMM using a Dockerfile, available on GitHub. Note that this is **not the recommended approach for most users** and should be reserved to people wanting to participate in the development of PyBaMM, or people who really need to use bleeding-edge feature(s) not yet available in the latest released version. If you do not fall in the two previous categories, you would be better off installing PyBaMM using ``pip`` or ``conda``.

Prerequisites
-------------

Before you begin, make sure you have Docker installed on your system. You can download and install Docker from the official `Docker website <https://www.docker.com/get-started/>`_.
Ensure Docker installation by running:

.. code:: bash

	  docker --version

Pulling the Docker image
------------------------

Use the following command to pull the PyBaMM Docker image from Docker Hub:

.. tab:: No optional solver

      .. code:: bash

            docker pull pybamm/pybamm:latest

.. tab:: Scikits.odes solver

      .. code:: bash

            docker pull pybamm/pybamm:odes

.. tab:: JAX solver

      .. code:: bash

            docker pull pybamm/pybamm:jax

.. tab:: IDAKLU solver

      .. code:: bash

            docker pull pybamm/pybamm:idaklu

.. tab:: All solvers

      .. code:: bash

            docker pull pybamm/pybamm:all

Running the Docker container
----------------------------

Once you have pulled the Docker image, you can run a Docker container with the PyBaMM environment:

1. In your terminal, use the following command to start a Docker container from the pulled image:

.. tab:: Basic

      .. code:: bash

            docker run -it pybamm/pybamm:latest

.. tab:: ODES Solver

      .. code:: bash

            docker run -it pybamm/pybamm:odes

.. tab:: JAX Solver

      .. code:: bash

            docker run -it pybamm/pybamm:jax

.. tab:: IDAKLU Solver

      .. code:: bash

            docker run -it pybamm/pybamm:idaklu

.. tab:: All Solver

      .. code:: bash

            docker run -it pybamm/pybamm:all

2. You will now be inside the Docker container's shell. You can use PyBaMM and its dependencies as if you were in a virtual environment.

3. You can execute PyBaMM-related commands, run tests develop & contribute from the container.

Exiting the Docker container
----------------------------

To exit the Docker container's shell, you can simply type:

.. code-block:: bash

      exit

This will return you to your host machine's terminal.

Building Docker image locally from source
-----------------------------------------

If you want to build the PyBaMM Docker image locally from the PyBaMM source code, follow these steps:

1. Clone the PyBaMM GitHub repository to your local machine if you haven't already:

.. code-block:: bash

      git clone https://github.com/pybamm-team/PyBaMM.git

2. Change into the PyBaMM directory:

.. code-block:: bash

      cd PyBaMM

3. Build the Docker image using the following command:

.. code-block:: bash

      docker build -t pybamm -f scripts/Dockerfile .

4. Once the image is built, you can run a Docker container using:

.. code-block:: bash

      docker run -it pybamm

5. Activate PyBaMM development environment inside docker container using:

.. code-block:: bash

      conda activate pybamm

Building Docker images with optional arguments
----------------------------------------------

When building the PyBaMM Docker images locally, you have the option to include specific solvers by using optional arguments. These solvers include:

- ``IDAKLU``: For IDA solver provided by the SUNDIALS plus KLU.
- ``ODES``: For scikits.odes solver for ODE & DAE problems.
- ``JAX``: For Jax solver.
- ``ALL``: For all the above solvers.

To build the Docker images with optional arguments, you can follow these steps for each solver:

.. tab:: Scikits.odes solver

      .. code-block:: bash

            docker build -t pybamm:odes -f scripts/Dockerfile --build-arg ODES=true .

.. tab:: JAX solver

      .. code-block:: bash

            docker build -t pybamm:jax -f scripts/Dockerfile --build-arg JAX=true .

.. tab:: IDAKLU solver

      .. code-block:: bash

            docker build -t pybamm:idaklu -f scripts/Dockerfile --build-arg IDAKLU=true .

.. tab:: All solvers

      .. code-block:: bash

            docker build -t pybamm:all -f scripts/Dockerfile --build-arg ALL=true .

After building the Docker images with the desired solvers, use the ``docker run`` command followed by the desired image name. For example, to run a container from the image built with all optional solvers:

.. code-block:: bash

      docker run -it pybamm:all

Activate PyBaMM development environment inside docker container using:

.. code-block:: bash

      conda activate pybamm

If you want to exit the Docker container's shell, you can simply type:

.. code-block:: bash

      exit


Using Git inside a running Docker container
-------------------------------------------

.. note::
      You might require re-configuring git while running the docker container for the first time.
      You can run ``git config --list`` to ensure if you have desired git configuration already.

1. Setting up git configuration

.. code-block:: bash

      git config --global user.name "Your Name"

      git config --global user.email your@mail.com

2. Setting a git remote

.. code-block:: bash

      git remote set-url origin <fork_url>

      git remote add upstream https://github.com/pybamm-team/PyBaMM

      git fetch --all

Using Visual Studio Code inside a running Docker container
----------------------------------------------------------

You can easily use Visual Studio Code inside a running Docker container by attaching it directly. This provides a seamless development environment within the container. Here's how:

1. Install the "Docker" extension from Microsoft in your local Visual Studio Code if it's not already installed.
2. Pull and run the Docker image containing PyBaMM development environment.
3. In your local Visual Studio Code, open the "Docker" extension by clicking on the Docker icon in the sidebar.
4. Under the "Containers" section, you'll see a list of running containers. Right-click the running PyBaMM container.
5. Select "Attach Visual Studio Code" from the context menu.
6. Visual Studio Code will now connect to the container, and a new VS Code window will open up, running inside the container. You can now edit, debug, and work on your code using VS Code as if you were working directly on your local machine.
