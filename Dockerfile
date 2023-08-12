FROM python:3.9-slim

# Set the working directory
WORKDIR /

# Install the necessary dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libopenblas-dev gcc gfortran graphviz git make g++ build-essential cmake

ENV CMAKE_C_COMPILER=/usr/bin/gcc
ENV CMAKE_CXX_COMPILER=/usr/bin/g++
ENV CMAKE_MAKE_PROGRAM=/usr/bin/make
ENV SUNDIALS_INST=root/.local
ENV LD_LIBRARY_PATH=root/.local/lib:

# Copy project files into the container
RUN git clone https://github.com/pybamm-team/PyBaMM.git

WORKDIR /PyBaMM/

# Install PyBaMM
RUN python -m pip install --upgrade pip setuptools wheel nox wget
RUN python scripts/install_KLU_Sundials.py 
RUN git clone https://github.com/pybind/pybind11.git pybind11/
RUN python -m pip install -e ".[all]"

CMD ["/bin/bash"]
