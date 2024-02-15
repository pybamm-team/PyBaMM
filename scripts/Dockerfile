FROM continuumio/miniconda3:latest

WORKDIR /

# Install the necessary dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libopenblas-dev gcc gfortran graphviz git make g++ build-essential cmake pandoc texlive-latex-extra dvipng
RUN rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash pybamm
USER pybamm

WORKDIR /home/pybamm/

# Clone project files from Git repository
RUN git clone https://github.com/pybamm-team/PyBaMM.git

WORKDIR /home/pybamm/PyBaMM

ENV CMAKE_C_COMPILER=/usr/bin/gcc
ENV CMAKE_CXX_COMPILER=/usr/bin/g++
ENV CMAKE_MAKE_PROGRAM=/usr/bin/make
ENV SUNDIALS_INST=/home/pybamm/.local
ENV LD_LIBRARY_PATH=/home/pybamm/.local/lib

RUN conda create -n pybamm python=3.11
RUN conda init --all
SHELL ["conda", "run", "-n", "pybamm", "/bin/bash", "-c"]
RUN conda install -y pip

ARG IDAKLU
ARG ODES
ARG JAX
ARG ALL

RUN pip install --upgrade --user pip setuptools wheel wget
RUN pip install cmake

RUN if [ "$IDAKLU" = "true" ]; then \
    python scripts/install_KLU_Sundials.py && \
    rm -rf pybind11 && \
    git clone https://github.com/pybind/pybind11.git && \
    pip install --user -e ".[all,dev,docs]"; \
    fi

RUN if [ "$ODES" = "true" ]; then \
    python scripts/install_KLU_Sundials.py && \
    pip install --user -e ".[all,dev,docs,odes]"; \
    fi

RUN if [ "$JAX" = "true" ]; then \
    pip install --user -e ".[all,dev,docs,jax]"; \
    fi

RUN if [ "$ALL" = "true" ]; then \
    python scripts/install_KLU_Sundials.py && \
    rm -rf pybind11 && \
    git clone https://github.com/pybind/pybind11.git && \
    pip install --user -e ".[all,dev,docs,jax,odes]"; \
    fi

RUN if [ -z "$IDAKLU" ] \
    && [ -z "$ODES" ] \
    && [ -z "$JAX" ] \
    && [ -z "$ALL" ]; then \
    pip install --user -e ".[all,dev,docs]"; \
    fi

ENTRYPOINT ["/bin/bash"]
