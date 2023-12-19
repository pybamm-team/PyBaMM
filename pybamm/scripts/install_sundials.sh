#!/bin/bash

# This script installs both SuiteSparse
# (https://people.engr.tamu.edu/davis/suitesparse.html) and SUNDIALS
# (https://computing.llnl.gov/projects/sundials) from source. For each
# two library:
# - Archive downloaded and source code extracted in current working
#   directory.
# - Library is built and installed.
#
# Usage: ./install_sundials.sh suitesparse_version sundials_version

function prepend_python_bin_dir_to_PATH {
    python_bin_dir_cmd="print(os.path.split(sys.executable)[0])"
    python_bin_dir=$(python -c "import sys, os;$python_bin_dir_cmd")
    export PATH=$python_bin_dir:$PATH
}

function download {
    ROOT_ADDR=$1
    FILENAME=$2

    wget -q $ROOT_ADDR/$FILENAME
}

function extract {
    tar -xf $1
}

SUITESPARSE_VERSION=$1
SUNDIALS_VERSION=$2

SUITESPARSE_ROOT_ADDR=https://github.com/DrTimothyAldenDavis/SuiteSparse/archive
SUNDIALS_ROOT_ADDR=https://github.com/LLNL/sundials/releases/download/v$SUNDIALS_VERSION

SUITESPARSE_ARCHIVE_NAME=v$SUITESPARSE_VERSION.tar.gz
SUNDIALS_ARCHIVE_NAME=sundials-$SUNDIALS_VERSION.tar.gz

yum -y update
yum -y install wget
download $SUITESPARSE_ROOT_ADDR $SUITESPARSE_ARCHIVE_NAME
download $SUNDIALS_ROOT_ADDR $SUNDIALS_ARCHIVE_NAME
extract $SUITESPARSE_ARCHIVE_NAME
extract $SUNDIALS_ARCHIVE_NAME

### Compile and install SUITESPARSE ###
# SuiteSparse is required to compile SUNDIALS's
# KLU solver.

SUITESPARSE_DIR=SuiteSparse-$SUITESPARSE_VERSION
for dir in SuiteSparse_config AMD COLAMD BTF KLU
do
    make -C $SUITESPARSE_DIR/$dir library
    make -C $SUITESPARSE_DIR/$dir install INSTALL=/usr
done

### Compile and install SUNDIALS ###

# Building cmake requires cmake >= 3.5
python -m pip install cmake
prepend_python_bin_dir_to_PATH

# Building SUNDIALS requires a BLAS library
yum -y install openblas-devel

mkdir -p build_sundials
cd build_sundials
KLU_INCLUDE_DIR=/usr/local/include
KLU_LIBRARY_DIR=/usr/local/lib
SUNDIALS_DIR=sundials-$SUNDIALS_VERSION
cmake -DENABLE_LAPACK=ON\
      -DSUNDIALS_INDEX_SIZE=32\
      -DEXAMPLES_ENABLE:BOOL=OFF\
      -DENABLE_KLU=ON\
      -DENABLE_OPENMP=ON\
      -DKLU_INCLUDE_DIR=$KLU_INCLUDE_DIR\
      -DKLU_LIBRARY_DIR=$KLU_LIBRARY_DIR\
      -DCMAKE_INSTALL_PREFIX=/usr\
      ../$SUNDIALS_DIR
make install
