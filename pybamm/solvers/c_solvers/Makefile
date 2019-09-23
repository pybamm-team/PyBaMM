SHELL = sh

# need to find a way to make this user specific..

# sundials path
SUNDIALS_DIR = /home/scott/Projects/PyBaMM/sundials4
SUNDIALS_INCLUDE = -I${SUNDIALS_DIR}/include
SUNDIALS_LIB = ${SUNDIALS_DIR}/lib

# add all the suitesparse includes and libraries
SUITESPARSE_DIR = /home/scott/Projects/PyBaMM/SuiteSparse

KLU_INCLUDE = -I${SUITESPARSE_DIR}/KLU/Include
AMD_INCLUDE = -I${SUITESPARSE_DIR}/AMD/Include
COLAMD_INCLUDE = -I${SUITESPARSE_DIR}/COLAMD/Include
BTF_INCLUDE = -I${SUITESPARSE_DIR}/BTF/Include
SUITESPARSE_CONFIG_INCLUDE = -I${SUITESPARSE_DIR}/SuiteSparse_config

SUITESPARSE_INCLUDES = ${KLU_INCLUDE} ${AMD_INCLUDE} ${COLAMD_INCLUDE} ${BTF_INCLUDE} ${SUITESPARSE_CONFIG_INCLUDE}

KLU_LIB = ${SUITESPARSE_DIR}/KLU/Lib/libklu.a
AMD_LIB = ${SUITESPARSE_DIR}/AMD/Lib/libamd.a
COLAMD_LIB = ${SUITESPARSE_DIR}/COLAMD/Lib/libcolamd.a
BTF_LIB = ${SUITESPARSE_DIR}/BTF/Lib/libbtf.a
SUITESPARSE_CONFIG_LIB = ${SUITESPARSE_DIR}/SuiteSparse_config/libsuitesparseconfig.a

SUITESPARSE_LIBS = ${KLU_LIB} ${AMD_LIB} ${COLAMD_LIB} ${BTF_LIB} ${SUITESPARSE_CONFIG_LIB}

all: klu

klu:
	g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ${SUNDIALS_INCLUDE} ${SUITESPARSE_INCLUDES} -o klu`python3-config --extension-suffix` klu.cpp -L${SUNDIALS_LIB} -lsundials_sunmatrixsparse -lsundials_ida -lsundials_nvecserial -lm /usr/lib/x86_64-linux-gnu/librt.so -Wl,-rpath,${SUNDIALS_LIB} -lsundials_sunlinsolklu ${SUITESPARSE_LIBS} 

clean:
	rm *.so
