FROM python:3.12-slim-bullseye
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

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

# Create a virtual environment
ENV VIRTUAL_ENV=/home/pybamm/venv
RUN uv venv $VIRTUAL_ENV
RUN #!/bin/bash && source /home/pybamm/venv/bin/activate;
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install --upgrade setuptools wheel wget cmake

RUN python scripts/install_KLU_Sundials.py && \
rm -rf pybind11 && \
git clone https://github.com/pybind/pybind11.git && \
uv pip install -e ".[all,dev,docs,jax]";

ENTRYPOINT ["/bin/bash"]
