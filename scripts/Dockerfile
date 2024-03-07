FROM quay.io/pypa/manylinux2014_x86_64
WORKDIR /

# Install the necessary dependencies
RUN yum update && yum upgrade
RUN yum -y install openblas-devel gcc gcc-gfortran graphviz git cmake pandoc texlive-* wget dvipng python-devel
RUN yum -y groupinstall "Development Tools"

RUN useradd -m -s /bin/bash pybamm
USER pybamm

WORKDIR /home/pybamm/

# Clone project files from Git repository
RUN git clone https://github.com/pybamm-team/PyBaMM.git

WORKDIR /home/pybamm/PyBaMM
USER root

# .local not present by default
RUN mkdir /home/pybamm/.local

ENV CMAKE_C_COMPILER=/usr/bin/gcc
ENV CMAKE_CXX_COMPILER=/usr/bin/g++
ENV CMAKE_MAKE_PROGRAM=/usr/bin/make
ENV SUNDIALS_INST=/home/pybamm/.local
ENV LD_LIBRARY_PATH=/home/pybamm/.local/lib

# Migrating to venv and create a virtual environment for python 3.11
ENV VIRTUAL_ENV=/home/pybamm/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade setuptools wheel wget
RUN pip install cmake
RUN source /home/pybamm/venv/bin/activate

RUN python scripts/install_KLU_Sundials.py --install-dir /home/pybamm/.local && \
python scripts/install_KLU_Sundials.py && \
mv /home/pybamm/.local/lib64 /home/pybamm/.local/lib && \
rm -rf pybind11 && \
git clone https://github.com/pybind/pybind11.git && \
pip install -e ".[all,dev,docs,jax,odes]";

USER pybamm
RUN source /home/pybamm/venv/bin/activate

ENTRYPOINT ["/bin/bash"]
