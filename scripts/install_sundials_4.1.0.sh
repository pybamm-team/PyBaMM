#!/bin/bash
CURRENT_DIR=`pwd`

# install sundials-4.1.0
SUNDIALS_URL=https://computing.llnl.gov/projects/sundials/download/sundials-4.1.0.tar.gz
SUNDIALS_NAME=sundials-4.1.0.tar.gz

mkdir $TMP_DIR
INSTALL_DIR=$CURRENT_DIR/sundials4

wget $SUNDIALS_URL -O $SUNDIALS_NAME
tar -xvf $SUNDIALS_NAME

# replace the sundials cmake file by a modified version that finds the KLU libraries and headers
cd sundials-4.1.0
cp $CURRENT_DIR/scripts/replace-cmake/CMakeLists.txt .

mkdir build-sundials-4.1.0
cd build-sundials-4.1.0/

cmake -DBLAS_ENABLE=ON\
      -DLAPACK_ENABLE=ON\
      -DSUNDIALS_INDEX_SIZE=32\
      -DBUILD_ARKODE=OFF\
      -DBUILD_CVODE=OFF\
      -DBUILD_CVODES=OFF\
      -DBUILD_IDAS=OFF\
      -DBUILD_KINSOL=OFF\
      -DEXAMPLES_ENABLE:BOOL=OFF\
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ../sundials-4.1.0/\
      -DKLU_ENABLE=ON\
      ../sundials-4.1.0


make clean
make install
cd $CURRENT_DIR
rm -rf build-sundials-4.1.0
rm -rf sundials-4.1.0
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
export SUNDIALS_INST=$INSTALL_DIR

# get pybind11
cd $CURRENT_DIR
mkdir -p third-party
cd third-party
rm -rf pybind11 # just remove it if it is already there
git clone https://github.com/pybind/pybind11.git

cd $CURRENT_DIR
pip install pybind11 # also do a pip install for good measure

PY_VERSION=PY_VER=$(python --version 2>&1 | awk '{print $2}')
cmake -DPYBIND11_PYTHON_VERSION=${PY_VERSION} .
make clean
make

# remove cmakefiles etc just to clean things up 
rm -rf CMakeFiles
rm CMakeCache.txt
rm cmake_install.cmake
