cmake_minimum_required(VERSION 3.10)
project(klu)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(third-party/pybind11)
pybind11_add_module(klu pybamm/solvers/c_solvers/klu.cpp)

# Sundials
set(SUNDIALS_INCLUDE "sundials4/include")
TARGET_INCLUDE_DIRECTORIES(klu PRIVATE ${SUNDIALS_INCLUDE})

find_library(SUNMATSPARSE sundials_sunmatrixsparse PATHS "sundials4/lib" NO_DEFAULT_PATH)
find_library(IDA sundials_ida PATHS "sundials4/lib" NO_DEFAULT_PATH)
find_library(NVECTOR sundials_nvecserial PATHS "sundials4/lib" NO_DEFAULT_PATH)
find_library(SUNKLU sundials_sunlinsolklu PATHS "sundials4/lib" NO_DEFAULT_PATH)
TARGET_LINK_LIBRARIES(klu PRIVATE ${SUNMATSPARSE} ${IDA} ${NVECTOR} ${SUNKLU})

# System libraries
find_library(RT rt)
TARGET_LINK_LIBRARIES(klu PRIVATE ${RT})

## KLU ------
## Add the includes
set(KLU_INCLUDE_DIR "SuiteSparse/KLU/Include")
set(AMD_INCLUDE_DIR "SuiteSparse/AMD/Include")
set(COLAMD_INCLUDE_DIR "SuiteSparse/COLAMD/Include")
set(BTF_INCLUDE_DIR "SuiteSparse/BTF/Include")
set(SUITESPARSECONFIG_INCLUDE_DIR "SuiteSparse/SuiteSparse_config")
TARGET_INCLUDE_DIRECTORIES(klu PRIVATE 
    ${AMD_INCLUDE_DIR} 
    ${COLAMD_INCLUDE_DIR} 
    ${BTF_INCLUDE_DIR} 
    ${SUITESPARSECONFIG_INCLUDE_DIR} 
    ${KLU_INCLUDE_DIR}
    )

## Link Libraries
find_library(KLU klu PATHS "SuiteSparse/KLU/Lib" NO_DEFAULT_PATH)
find_library(AMD amd PATHS "SuiteSparse/AMD/Lib" NO_DEFAULT_PATH)
find_library(COLAMD colamd PATHS "SuiteSparse/COLAMD/Lib" NO_DEFAULT_PATH)
find_library(BTF btf PATHS "SuiteSparse/BTF/Lib" NO_DEFAULT_PATH)
find_library(SUITESPARSE_CONFIG suitesparseconfig PATHS "SuiteSparse/SuiteSparse_config" NO_DEFAULT_PATH)
TARGET_LINK_LIBRARIES(klu PRIVATE 
    ${KLU} 
    ${AMD} 
    ${COLAMD} 
    ${BTF} 
    ${SUITESPARSE_CONFIG}
    )

# ---------------

