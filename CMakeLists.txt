cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0074 NEW)
set(CMAKE_VERBOSE_MAKEFILE ON)

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

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    # MSVC does not support variable length arrays (vla)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=vla")
endif()

# casadi seems to compile without the newer versions of std::string
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

if(NOT PYBIND11_DIR)
  set(PYBIND11_DIR pybind11)
endif()

add_subdirectory(${PYBIND11_DIR})
pybind11_add_module(idaklu
  pybamm/solvers/c_solvers/idaklu/casadi_functions.cpp
  pybamm/solvers/c_solvers/idaklu/casadi_functions.hpp
  pybamm/solvers/c_solvers/idaklu/casadi_solver.cpp
  pybamm/solvers/c_solvers/idaklu/casadi_solver.hpp
  pybamm/solvers/c_solvers/idaklu/CasadiSolver.cpp
  pybamm/solvers/c_solvers/idaklu/CasadiSolver.hpp
  pybamm/solvers/c_solvers/idaklu/CasadiSolverOpenMP.cpp
  pybamm/solvers/c_solvers/idaklu/CasadiSolverOpenMP.hpp
  pybamm/solvers/c_solvers/idaklu/CasadiSolverOpenMP_solvers.cpp
  pybamm/solvers/c_solvers/idaklu/CasadiSolverOpenMP_solvers.hpp
  pybamm/solvers/c_solvers/idaklu/casadi_sundials_functions.cpp
  pybamm/solvers/c_solvers/idaklu/casadi_sundials_functions.hpp
  pybamm/solvers/c_solvers/idaklu/common.hpp
  pybamm/solvers/c_solvers/idaklu/python.hpp
  pybamm/solvers/c_solvers/idaklu/python.cpp
  pybamm/solvers/c_solvers/idaklu/solution.cpp
  pybamm/solvers/c_solvers/idaklu/solution.hpp
  pybamm/solvers/c_solvers/idaklu/options.hpp
  pybamm/solvers/c_solvers/idaklu/options.cpp
  pybamm/solvers/c_solvers/idaklu.cpp
)

if (NOT DEFINED USE_PYTHON_CASADI)
  set(USE_PYTHON_CASADI TRUE)
endif()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
    "import casadi as _; print(_.__path__[0])"
    OUTPUT_VARIABLE CASADI_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if (CASADI_DIR)
  file(TO_CMAKE_PATH ${CASADI_DIR} CASADI_DIR)
  message("Found python casadi path: ${CASADI_DIR}")
endif()

if(${USE_PYTHON_CASADI})
  message("Trying to link against python casadi package")
  find_package(casadi CONFIG PATHS ${CASADI_DIR} REQUIRED)
else()
  message("Trying to link against any casadi package apart from the python one")
  set(CMAKE_IGNORE_PATH "${CASADI_DIR}/cmake")
  find_package(casadi CONFIG REQUIRED)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})
# Sundials
find_package(SUNDIALS REQUIRED)
message("SUNDIALS found in ${SUNDIALS_INCLUDE_DIR}: ${SUNDIALS_LIBRARIES}")
target_include_directories(idaklu PRIVATE ${SUNDIALS_INCLUDE_DIR})
target_link_libraries(idaklu PRIVATE ${SUNDIALS_LIBRARIES} casadi)

# link suitesparse
# if using vcpkg, use config mode to
# find suitesparse. Otherwise, use FindSuiteSparse module
if(DEFINED VCPKG_ROOT_DIR)
  find_package(SuiteSparse CONFIG REQUIRED)
else()
  find_package(SuiteSparse REQUIRED)
  message("SuiteSparse found in ${SuiteSparse_INCLUDE_DIRS}: ${SuiteSparse_LIBRARIES}")
endif()
include_directories(${SuiteSparse_INCLUDE_DIRS})
target_link_libraries(idaklu PRIVATE ${SuiteSparse_LIBRARIES})
