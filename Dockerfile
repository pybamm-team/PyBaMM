FROM ubuntu:22.04

# Set the working directory
WORKDIR /

# Install the necessary dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libopenblas-dev gcc gfortran graphviz git make g++ build-essential python3.9 python3-pip

ENV CMAKE_C_COMPILER=/usr/bin/gcc
ENV CMAKE_CXX_COMPILER=/usr/bin/g++
ENV CMAKE_MAKE_PROGRAM=/usr/bin/make
ENV SUNDIALS_INST=root/.local
ENV LD_LIBRARY_PATH=root/.local/lib:

RUN python3.9 -m pip install wget

# Copy project files into the container
RUN git clone https://github.com/pybamm-team/PyBaMM.git

WORKDIR /PyBaMM/

# Install virtualenv
RUN python3.9 -m pip install virtualenv

# Create and activate virtual environment
RUN virtualenv -p python3.9 venv
RUN /bin/bash -c "source venv/bin/activate"

# Install PyBaMM
RUN python3.9 -m pip install --upgrade pip setuptools wheel nox
RUN python3.9 scripts/install_KLU_Sundials.py
RUN git clone https://github.com/pybind/pybind11.git pybind11/
RUN python3.9 -m pip install -e ".[all]"

CMD ["/bin/bash"]
