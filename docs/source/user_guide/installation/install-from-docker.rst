Install using Docker (developer install)
=========================================

.. contents::

This page describes the build and installation of PyBaMM from the source code, available on GitHub. Note that this is **not the recommended approach for most users** and should be reserved to people wanting to participate in the development of PyBaMM, or people who really need to use bleeding-edge feature(s) not yet available in the latest released version. If you do not fall in the two previous categories, you would be better off installing PyBaMM using pip or conda.

Prerequisites
-------------
Before you begin, make sure you have Docker installed on your system. You can download and install Docker from the official `Docker website <https://www.docker.com/get-started/>`_.
Ensure Docker installation by running :

.. code:: bash

	  docker --version

Pulling the Docker Image
------------------------
Use the following command to pull the PyBaMM Docker image from Docker Hub:

.. code:: bash

      docker pull pybamm/pybamm:latest

Running the Docker Container
----------------------------

Once you have pulled the Docker image, you can run a Docker container with the PyBaMM environment:

1. In your terminal, use the following command to start a Docker container from the pulled image:

.. code-block:: bash

      docker run -it pybamm/pybamm:latest

2. You will now be inside the Docker container's shell. You can use PyBaMM and its dependencies as if you were in a virtual environment.

3. You can execute PyBaMM-related commands, run tests develop & contribute from the container.

Exiting the Docker Container
---------------------------

To exit the Docker container's shell, you can simply type:

.. code-block:: bash

      exit

This will return you to your host machine's terminal.

Building Docker Image Locally from Source
------------------------------------------

If you want to build the PyBaMM Docker image locally from the PyBaMM source code, follow these steps:

1. Clone the PyBaMM GitHub repository to your local machine if you haven't already:

.. code-block:: bash

      git clone https://github.com/pybamm-team/PyBaMM.git

2. Change into the PyBaMM directory:

.. code-block:: bash

      cd PyBaMM

3. Build the Docker image using the following command:

.. code-block:: bash

      docker build -t pybamm .

4. Once the image is built, you can run a Docker container using:

.. code-block:: bash

      docker run -it pybamm

Building Docker Images with Optional Args
-----------------------------------------

When building the PyBaMM Docker images locally, you have the option to include specific solvers by using optional arguments. These solvers include:

- IDAKLU: For IDA solver provided by the SUNDIALS plus KLU.
- ODES: For scikits.odes solver for ODE & DAE problems.
- JAX: For Jax solver.

To build the Docker images with optional arguments, you can follow these steps for each solver:

Build Docker Image with IDAKLU Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Follow the same steps as above to clone the PyBaMM repository and navigate to the source code directory.

3. Build the Docker image for IDAKLU using the following command:

.. code-block:: bash

      docker build -t pybamm:idaklu --build-arg IDAKLU=true .

Build Docker Image with ODES Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Follow the same steps as above to clone the PyBaMM repository and navigate to the source code directory.

2. Build the Docker image for ODES using the following command:

.. code-block:: bash

      docker build -t pybamm:odes --build-arg ODES=true .

Build Docker Image with JAX Solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Follow the same steps as above to clone the PyBaMM repository and navigate to the source code directory.

2. Build the Docker image for JAX using the following command:

.. code-block:: bash

      docker build -t pybamm:jax --build-arg JAX=true .


After building the Docker images with the desired solvers, use the ``docker run`` command followed by the desired image name. For example, to run a container from the image built with IDAKLU solver:

.. code-block:: bash

      docker run -it pybamm:idaklu
