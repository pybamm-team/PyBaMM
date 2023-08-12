FROM python:3.10-slim

# Set the working directory
WORKDIR /

# Install the necessary dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libopenblas-dev gcc gfortran graphviz git make g++ build-essential python3.11-dev python3-pip

ENV CMAKE_C_COMPILER=/usr/bin/gcc
ENV CMAKE_CXX_COMPILER=/usr/bin/g++
ENV CMAKE_MAKE_PROGRAM=/usr/bin/make

# Copy project files into the container
RUN git clone https://github.com/pybamm-team/PyBaMM.git

WORKDIR /PyBaMM/

# RUN python3.11 -m pip install virtualenv
RUN apt-get install -y python3.11-venv
RUN python3.11 -m venv venv
RUN /bin/bash -c "source venv/bin/activate"

# Install PyBaMM
RUN python3.11 -m pip install --upgrade pip setuptools wheel nox
# RUN pip install -e ".[all]"

ARG ODES
ARG JAX
ARG IDAKLU


# RUN if [ "$ODES" = "true" ]; then \
#     apt-get install -y cmake && \
#     pip install wget \
#     pybamm_install_odes; \
#     fi

# RUN if [ "$JAX" = "true" ]; then \
#     pip install -e ".[jax,all]";\
#     fi

RUN pip install wget cmake 
RUN python scripts/install_KLU_Sundials.py 
RUN git clone https://github.com/pybind/pybind11.git pybind11/
RUN python3.11 -m pip install -e ".[all]"

CMD ["/bin/bash"]
