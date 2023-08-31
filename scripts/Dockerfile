FROM continuumio/miniconda3:latest

WORKDIR /

# Install the necessary dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libopenblas-dev gcc gfortran graphviz git make g++ build-essential cmake
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
ENV LD_LIBRARY_PATH=/home/pybamm/.local/lib:

ARG IDAKLU
ARG ODES
ARG JAX
ARG ALL

RUN conda create -n pybamm python=3.9
RUN conda init --all
SHELL ["conda", "run", "-n", "pybamm", "/bin/bash", "-c"]
RUN conda install -y pip

RUN if [ "$IDAKLU" = "true" ]; then \
    pip install --upgrade --user pip setuptools wheel wget && \
    pip install cmake==3.22 && \
    python scripts/install_KLU_Sundials.py && \
    git clone https://github.com/pybind/pybind11.git && \
    pip install --user -e ".[all,dev]"; \
    fi

RUN if [ "$ODES" = "true" ]; then \
    pip install cmake==3.22 && \
    pip install --upgrade --user pip wget && \
    python scripts/install_KLU_Sundials.py && \
    pip install --user -e ".[all,odes,dev]"; \
    fi

RUN if [ "$JAX" = "true" ]; then \
    pip install --user -e ".[jax,all,dev]";\
    fi

RUN if [ "$ALL" = "true" ]; then \
    pip install cmake==3.22 && \
    pip install --upgrade --user pip setuptools wheel wget && \
    python scripts/install_KLU_Sundials.py && \
    git clone https://github.com/pybind/pybind11.git && \
    pip install --user -e ".[all,dev,jax,odes]"; \
    fi

RUN pip install --user -e ".[all,dev]"

ENTRYPOINT ["/bin/bash"]
