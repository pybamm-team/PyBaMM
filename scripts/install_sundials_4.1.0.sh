#!/bin/bash
CURRENT_DIR=`pwd`

# build SparseSuite to use KLU sparse linear solver
SUITESPARSE_URL=http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-5.4.0.tar.gz
SUITESPARSE_NAME=SuiteSparse-5.4.0.tar.gz
wget $SUITESPARSE_URL -O $SUITESPARSE_NAME
tar -xvf $SUITESPARSE_NAME
SUITESPARSE_DIR=$CURRENT_DIR/SuiteSparse
cd $SUITESPARSE_DIR
make
cd $CURRENT_DIR
rm $SUITESPARSE_NAME

# sparse suite library paths
KLU_LIB=$SUITESPARSE_DIR/KLU/Lib/libklu.a
AMD_LIB=$SUITESPARSE_DIR/AMD/Lib/libamd.a
COLAMD_LIB=$SUITESPARSE_DIR/COLAMD/Lib/libcolamd.a
BTF_LIB=$SUITESPARSE_DIR/BTF/Lib/libbtf.a
SUITESPARSE_CONFIG_LIB=$SUITESPARSE_DIR/SuiteSparse_config/libsuitesparseconfig.a

# sparse suite header directories
KLU_INCLUDE=$SUITESPARSE_DIR/KLU/Include
AMD_INCLUDE=$SUITESPARSE_DIR/AMD/Include
COLAMD_INCLUDE=$SUITESPARSE_DIR/COLAMD/Include
BTF_INCLUDE=$SUITESPARSE_DIR/BTF/Include
SUITESPARSE_CONFIG_INCLUDE=$SUITESPARSE_DIR/SuiteSparse_config

# install sundials-4.1.0
SUNDIALS_URL=https://computing.llnl.gov/projects/sundials/download/sundials-4.1.0.tar.gz
SUNDIALS_NAME=sundials-4.1.0.tar.gz

TMP_DIR=$CURRENT_DIR/tmp
mkdir $TMP_DIR
INSTALL_DIR=$CURRENT_DIR/sundials4

cd $TMP_DIR
wget $SUNDIALS_URL -O $SUNDIALS_NAME
tar -xvf $SUNDIALS_NAME

# replace the sundials cmake file by a modified version that finds the KLU libraries and headers
cd sundials-4.1.0
cp $CURRENT_DIR/scripts/replace-cmake/CMakeLists.txt .

cd $TMP_DIR
mkdir build-sundials-4.1.0
cd build-sundials-4.1.0/

# need to turn on blas, LAPACK, KLU libraries etc
# only build it ida solver (not idas, or cvode etc)
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
      -DKLU_INCLUDE_DIR=$KLU_INCLUDE\
      -DAMD_INCLUDE_DIR=$AMD_INCLUDE\
      -DCOLAMD_INCLUDE_DIR=$COLAMD_INCLUDE\
      -DBTF_INCLUDE_DIR=$BTF_INCLUDE\
      -DSUITESPARSECONFIG_INCLUDE_DIR=$SUITESPARSE_CONFIG_INCLUDE\
      -DKLU_LIBRARY=$KLU_LIB\
      -DAMD_LIBRARY=$AMD_LIB\
      -DCOLAMD_LIBRARY=$COLAMD_LIB\
      -DBTF_LIBRARY=$BTF_LIB\
      -DSUITESPARSECONFIG_LIBRARY=$SUITESPARSE_CONFIG_LIB\
      ../sundials-4.1.0

make install
cd $CURRENT_DIR
rm -rf $TMP_DIR
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
export SUNDIALS_INST=$INSTALL_DIR
export SUITESPARSE=$SUITESPARSE_DIR

# get pybind11
cd $CURRENT_DIR
mkdir -p third-party
cd third-party
# if already cloned then pull otherwise clone pybind11
if cd pybind11; then git pull; else git clone https://github.com/pybind/pybind11.git; fi

cd $CURRENT_DIR
pip install pybind11 # also do a pip install for good measure
cmake .
make

# remove cmakefiles etc just to clean things up 
rm -rf CMakeFiles
rm CMakeCache.txt
rm cmake_install.cmake
