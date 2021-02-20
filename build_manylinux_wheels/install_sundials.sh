#!/bin/bash

yum -y update
yum -y install wget openblas-devel

python_bin_dir_cmd="print(os.path.split(sys.executable)[0])"
python_bin_dir=$(python -c "import sys, os;$python_bin_dir_cmd")
export PATH=$python_bin_dir:$PATH
python -m pip install cmake

echo CMAKE_VERSION: $(cmake --version)

mkdir /deps
wget -q https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v5.7.2.tar.gz .
wget -q https://computing.llnl.gov/projects/sundials/download/sundials-5.3.0.tar.gz .
tar -xf v5.7.2.tar.gz --directory /deps
tar -xf sundials-5.3.0.tar.gz --directory /deps

SUITESPARSE_DIR=/deps/SuiteSparse-5.7.2
SUNDIALS_DIR=/deps/sundials-5.3.0

for dir in SuiteSparse_config AMD COLAMD BTF KLU
do
    cd $SUITESPARSE_DIR/$dir;
    make library
    make install INSTALL=/usr
    cd ../
done

KLU_INCLUDE_DIR=/usr/include
KLU_LIBRARY_DIR=/usr/lib
mkdir -p /deps/build_sundials
cd /deps/build_sundials
cmake -DLAPACK_ENABLE=ON\
      -DSUNDIALS_INDEX_SIZE=32\
      -DEXAMPLES_ENABLE:BOOL=OFF\
      -DKLU_ENABLE=ON\
      -DKLU_INCLUDE_DIR=$KLU_INCLUDE_DIR\
      -DKLU_LIBRARY_DIR=$KLU_LIBRARY_DIR\
      -DCMAKE_INSTALL_PREFIX=/usr\
      $SUNDIALS_DIR
make install

