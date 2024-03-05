# FROM continuumio/miniconda3:latest
FROM quay.io/pypa/manylinux2014_x86_64
WORKDIR /

# Install the necessary dependencies
RUN yum update && yum upgrade
RUN yum -y install openblas-devel gcc gcc-gfortran graphviz git cmake pandoc texlive dvipng python-devel
# RUN rm -rf /var/lib/apt/lists/*
RUN yum -y groupinstall "Development Tools"
RUN useradd -m -s /bin/bash pybamm
USER pybamm

WORKDIR /home/pybamm/

# Clone project files from Git repository
RUN git clone https://github.com/pybamm-team/PyBaMM.git

WORKDIR /home/pybamm/PyBaMM
USER root
RUN mkdir /home/pybamm/.local

ENV CMAKE_C_COMPILER=/usr/bin/gcc
ENV CMAKE_CXX_COMPILER=/usr/bin/g++
ENV CMAKE_MAKE_PROGRAM=/usr/bin/make
ENV SUNDIALS_INST=/home/pybamm/.local
ENV LD_LIBRARY_PATH=/home/pybamm/.local/lib

#RUN conda create -n pybamm python=3.11
#RUN conda init --all
#SHELL ["conda", "run", "-n", "pybamm", "/bin/bash", "-c"]
#RUN conda install -y pip
# Migrating to venv and create a virtual environment for python 3.11

ENV VIRTUAL_ENV=/home/pybamm/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN usermod -aG wheel pybamm 

RUN pip install --upgrade setuptools wheel wget
RUN pip install cmake
RUN source /home/pybamm/venv/bin/activate
RUN python scripts/install_KLU_Sundials.py --install-dir /home/pybamm/.local
RUN python scripts/install_KLU_Sundials.py 
RUN mv /home/pybamm/.local/lib64 /home/pybamm/.local/lib
RUN rm -rf pybind11 
RUN git clone https://github.com/pybind/pybind11.git
RUN pip install -e ".[all,dev,docs,jax,odes]" # ODES is not installing, installation path /root/.local in centos 

#USER pybamm

ENTRYPOINT ["/bin/bash"]

