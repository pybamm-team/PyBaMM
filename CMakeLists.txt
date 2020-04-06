cmake_minimum_required(VERSION 3.10)
project(idaklu)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT PYBIND11_DIR)
  set(PYBIND11_DIR pybind11)
endif()

add_subdirectory(${PYBIND11_DIR})
pybind11_add_module(idaklu pybamm/solvers/c_solvers/idaklu.cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})
# Sundials
find_package(SUNDIALS)
target_include_directories(idaklu PRIVATE ${SUNDIALS_INCLUDE_DIR})
target_link_libraries(idaklu PRIVATE ${SUNDIALS_LIBRARIES})

# link suitesparse
find_package(SuiteSparse OPTIONAL_COMPONENTS KLU AMD COLAMD BTF)
include_directories(${SuiteSparse_INCLUDE_DIRS})
target_link_libraries(idaklu PRIVATE ${SuiteSparse_LIBRARIES})
