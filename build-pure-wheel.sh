#!/bin/bash

# This is script is to run inside a manylinux
# docker image, i.e.
# docker run -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 /io/build-wheel.sh
#
# Builds a pure python PyBaMM wheel, that is without the idaklu module extension.
# In the following, clining the pybind11 is omitted, resulting in the extension module
# compilation to be skipped.
# This pure python wheel is mostly intended to Windows support.

set -e -x
cd /io

# Build wheel!
# Using python 3.7 but the resulting wheel is not
# python version dependent
for PYBIN in /opt/python/cp37-cp37m/bin; do
    "${PYBIN}/python" setup.py bdist_wheel -d /io/wheelhouse
done

echo "** --- All good ! --- **"
ls /io/wheelhouse
