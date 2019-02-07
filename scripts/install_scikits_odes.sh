#!/bin/bash

SUNDIALS_URL=https://computation.llnl.gov/projects/sundials/download/sundials-3.1.1.tar.gz
CURRENT_DIR=`pwd`
TMP_DIR=$CURRENT_DIR/tmp
mkdir $TMP_DIR
INSTALL_DIR=$CURRENT_DIR/sundials

cd $TMP_DIR
wget $SUNDIALS_URL
tar -xvf sundials-3.1.1.tar.gz
mkdir build-sundials-3.1.1
cd build-sundials-3.1.1/
cmake -DLAPACK_ENABLE=ON -DSUNDIALS_INDEX_TYPE=int32_t -DBUILD_ARKODE:BOOL=OFF -DEXAMPLES_ENABLE:BOOL=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ../sundials-3.1.1/
make install
cd $CURRENT_DIR
rm -rf $TMP_DIR
export LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH
export SUNDIALS_INST=$INSTALL_DIR
pip install scikits.odes

