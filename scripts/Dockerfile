FROM continuumio/miniconda3:latest

WORKDIR /

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libopenblas-dev gcc gfortran graphviz git make g++ build-essential cmake pandoc texlive-latex-extra dvipng sudo
RUN rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash pybamm && echo "pybamm:pybamm" | chpasswd && adduser pybamm sudo
RUN usermod -ou 1000 -g 0 pybamm
USER pybamm

WORKDIR /home/pybamm/

RUN git clone https://github.com/pybamm-team/PyBaMM.git

WORKDIR /home/pybamm/PyBaMM

ENV CMAKE_C_COMPILER=/usr/bin/gcc
ENV CMAKE_CXX_COMPILER=/usr/bin/g++
ENV CMAKE_MAKE_PROGRAM=/usr/bin/make
ENV LD_LIBRARY_PATH=/home/pybamm/.local/lib

RUN conda create -n pybamm python=3.11
RUN conda init --all
SHELL ["conda", "run", "-n", "pybamm", "/bin/bash", "-c"]
RUN conda install -y pip

RUN pip install --upgrade --user pip setuptools wheel wget
RUN pip install cmake

RUN python scripts/install_KLU_Sundials.py && \
    rm -rf pybind11 && \
    git clone https://github.com/pybind/pybind11.git && \
    pip install --user -e ".[all,dev,docs,jax]";

ENTRYPOINT ["/bin/bash"]
