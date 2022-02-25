cmake_minimum_required(VERSION 3.10)

if(DEFINED ENV{VCPKG_ROOT_DIR} AND NOT DEFINED VCPKG_ROOT_DIR)
  set(VCPKG_ROOT_DIR "$ENV{VCPKG_ROOT_DIR}"
    CACHE STRING "Vcpkg root directory")
endif()

if(DEFINED VCPKG_ROOT_DIR)
  set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT_DIR}/scripts/buildsystems/vcpkg.cmake
    CACHE STRING "Vcpkg toolchain file")
endif()

if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
  set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}"
    CACHE STRING "Vcpkg target triplet")
endif()

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
# if using vcpkg, use config mode to
# find suitesparse. Otherwise, use FindSuiteSparse module
if(DEFINED VCPKG_ROOT_DIR)
  find_package(SuiteSparse CONFIG REQUIRED)
else()
  find_package(SuiteSparse REQUIRED)
endif()
include_directories(${SuiteSparse_INCLUDE_DIRS})
target_link_libraries(idaklu PRIVATE ${SuiteSparse_LIBRARIES})
