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
