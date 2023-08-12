FROM continuumio/miniconda3:latest

WORKDIR /

# Install the necessary dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y libopenblas-dev gcc gfortran graphviz git make g++ build-essential cmake
RUN rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash myuser
USER myuser

WORKDIR /home/myuser/

# Clone project files from Git repository
RUN git clone https://github.com/pybamm-team/PyBaMM.git

WORKDIR /home/myuser/PyBaMM

ENV CMAKE_C_COMPILER=/usr/bin/gcc
ENV CMAKE_CXX_COMPILER=/usr/bin/g++
ENV CMAKE_MAKE_PROGRAM=/usr/bin/make
ENV SUNDIALS_INST=/home/myuser/.local
ENV LD_LIBRARY_PATH=/home/myuser/.local/lib:

RUN conda create -n py39 python=3.9

SHELL ["conda", "run", "-n", "py39", "/bin/bash", "-c"]
RUN conda install -y pip
RUN pip install --upgrade --user pip setuptools wheel nox wget
RUN pip install cmake==3.22
RUN python scripts/install_KLU_Sundials.py
RUN git clone https://github.com/pybind/pybind11.git
RUN pip install --user -e ".[all]"
RUN conda init --all

ENTRYPOINT ["/bin/bash", "-c", "source activate py39 && /bin/bash"]
