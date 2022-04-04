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

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# casadi seems to compile without the newer versions of std::string
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

if(NOT PYBIND11_DIR)
  set(PYBIND11_DIR pybind11)
endif()

add_subdirectory(${PYBIND11_DIR})
pybind11_add_module(idaklu 
  pybamm/solvers/c_solvers/idaklu_python.cpp
  pybamm/solvers/c_solvers/idaklu_python.hpp
  pybamm/solvers/c_solvers/idaklu.cpp
  pybamm/solvers/c_solvers/idaklu.hpp
  pybamm/solvers/c_solvers/idaklu_casadi.cpp
  pybamm/solvers/c_solvers/idaklu_casadi.hpp
  pybamm/solvers/c_solvers/solution.cpp
  pybamm/solvers/c_solvers/solution.hpp
)

if(NOT CASADI_DIR)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c 
    "import casadi as _; print(_.__path__[0])"
    OUTPUT_VARIABLE CASADI_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_package(casadi CONFIG PATHS ${CASADI_DIR} REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})
# Sundials
find_package(SUNDIALS)
target_include_directories(idaklu PRIVATE ${SUNDIALS_INCLUDE_DIR})
target_link_libraries(idaklu PRIVATE ${SUNDIALS_LIBRARIES} casadi)

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
