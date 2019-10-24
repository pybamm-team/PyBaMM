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

# link suitesparse
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})
find_package(SuiteSparse)
include_directories(${SuiteSparse_INCLUDE_DIRS})
target_link_libraries(klu PRIVATE ${SuiteSparse_LIBRARIES})
