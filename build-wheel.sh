#!/bin/bash

# This is script is to run inside a manylinux
# docker image, i.e.
# docker run -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 /io/build-wheel.sh

set -e -x
# Compile wheels
cd /io

yum -y install openblas-devel

# Update Cmake
/opt/python/cp37-cp37m/bin/pip install cmake
ln -s /opt/python/cp37-cp37m/bin/cmake /usr/bin/cmake

# The wget python module is required to download
# SuiteSparse and Sundials.
# https://pypi.org/project/wget/
/opt/python/cp37-cp37m/bin/pip install wget

# Clone the pybind11 git repo next to the setup.py
# Required to build the idaklu extension module.
if [ ! -d "pybind11" ]
then
  git clone https://github.com/pybind/pybind11.git
fi


# Download and build SuiteSparse/Sundials
# in KLU_module_deps/
/opt/python/cp37-cp37m/bin/python scripts/setup_KLU_module_build.py

SUITESPARSE_DIR=KLU_module_deps/SuiteSparse-5.6.0
SUNDIALS_DIR=KLU_module_deps/sundials5

# Build wheels!
for PYBIN in /opt/python/cp3[67]-cp3[67]m/bin; do
    "${PYBIN}/python" setup.py bdist_wheel\
		      --suitesparse-root=${SUITESPARSE_DIR}\
		      --sundials-root=${SUNDIALS_DIR}
done

# And repair them
for whl in dist/*.whl; do
    auditwheel repair $whl -w /io/wheelhouse/
done

echo "** --- All good ! --- **"
ls /io/wheelhouse
