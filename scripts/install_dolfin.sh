#!/bin/bash

pip install --prefix=$VIRTUAL_ENV fenics-ffc --upgrade

FENICS_VERSION=$(python3 -c"import ffc; print(ffc.__version__)")
git clone --branch=$FENICS_VERSION https://bitbucket.org/fenics-project/dolfin
mkdir dolfin/build && cd dolfin/build && cmake -DCMAKE_INSTALL_PREFIX=$VIRTUAL_ENV .. && make install && cd ../..
cd dolfin/python && pip install --prefix=$VIRTUAL_ENV . && cd ../..
rm -rf dolfin
