FROM python:3.12-slim-bullseye
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y libopenblas-dev liblapack-dev gcc gfortran graphviz git make g++ build-essential cmake pandoc texlive-latex-extra dvipng sudo python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash pybamm && echo "pybamm:pybamm" | chpasswd && adduser pybamm sudo
RUN usermod -ou 1000 -g 0 pybamm
USER pybamm

WORKDIR /home/pybamm/

# TODO: Change to upstream repository
RUN git clone --branch migrate-test2 https://github.com/cringeyburger/PyBaMM.git

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

RUN uv pip install --upgrade cmake casadi scikit-build-core pybind11

RUN python scripts/install_KLU_Sundials.py
RUN mkdir -p sundials_KLU_libs/lib64/
ENV LD_LIBRARY_PATH="/home/pybamm/PyBaMM/sundials_KLU_libs/lib64/:$LD_LIBRARY_PATH"
ENV PATH="/home/pybamm/PyBaMM/sundials_KLU_libs/lib64/:$PATH"

ENV BUILD_IDAKLU=ON
RUN uv pip install -e ".[all,dev,docs,jax]" --verbose
RUN echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
RUN python -c "import pybamm; print(pybamm.IDAKLUSolver())"

RUN
ENTRYPOINT ["/bin/bash"]
