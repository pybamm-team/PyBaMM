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

set(CUSTOM_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sundials_KLU_libs)

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

# Add pybind11
if(NOT PYBIND11_DIR)
  set(PYBIND11_DIR pybind11)
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if(DEFINED ENV{CIBUILDWHEEL})
  set(BUILD_IDAKLU ON)
elseif(DEFINED ENV{BUILD_IDAKLU})
  set(BUILD_IDAKLU "$ENV{BUILD_IDAKLU}")
else()
  set(BUILD_IDAKLU OFF)
endif()

message(STATUS "BUILD_IDAKLU is set to: ${BUILD_IDAKLU}")

if(BUILD_IDAKLU)
  message(STATUS "Building with IDAKLU support")
  # Check Casadi build flag
  if(NOT DEFINED PYBAMM_IDAKLU_EXPR_CASADI)
    set(PYBAMM_IDAKLU_EXPR_CASADI ON)
  endif()
  message("PYBAMM_IDAKLU_EXPR_CASADI: ${PYBAMM_IDAKLU_EXPR_CASADI}")

  # Casadi PyBaMM source files
  set(IDAKLU_EXPR_CASADI_SOURCE_FILES "")
  if(${PYBAMM_IDAKLU_EXPR_CASADI} STREQUAL "ON" )
    add_compile_definitions(CASADI_ENABLE)
    set(IDAKLU_EXPR_CASADI_SOURCE_FILES
      pybamm/solvers/c_solvers/idaklu/Expressions/Casadi/CasadiFunctions.cpp
      pybamm/solvers/c_solvers/idaklu/Expressions/Casadi/CasadiFunctions.hpp
    )
  endif()

  # Check IREE build flag
  if(NOT DEFINED PYBAMM_IDAKLU_EXPR_IREE)
    set(PYBAMM_IDAKLU_EXPR_IREE OFF)
  endif()
  message("PYBAMM_IDAKLU_EXPR_IREE: ${PYBAMM_IDAKLU_EXPR_IREE}")

  # IREE (MLIR expression evaluation) PyBaMM source files
  set(IDAKLU_EXPR_IREE_SOURCE_FILES "")
  if(${PYBAMM_IDAKLU_EXPR_IREE} STREQUAL "ON" )
    add_compile_definitions(IREE_ENABLE)
    # Source file list
    set(IDAKLU_EXPR_IREE_SOURCE_FILES
      pybamm/solvers/c_solvers/idaklu/Expressions/IREE/iree_jit.cpp
      pybamm/solvers/c_solvers/idaklu/Expressions/IREE/iree_jit.hpp
      pybamm/solvers/c_solvers/idaklu/Expressions/IREE/IREEFunctions.cpp
      pybamm/solvers/c_solvers/idaklu/Expressions/IREE/IREEFunctions.hpp
      pybamm/solvers/c_solvers/idaklu/Expressions/IREE/ModuleParser.cpp
      pybamm/solvers/c_solvers/idaklu/Expressions/IREE/ModuleParser.hpp
    )
  endif()

  # The complete (all dependencies) sources list should be mirrored in setup.py
  pybind11_add_module(idaklu
    # pybind11 interface
    pybamm/solvers/c_solvers/idaklu.cpp
    # IDAKLU solver (SUNDIALS)
    pybamm/solvers/c_solvers/idaklu/idaklu_solver.hpp
    pybamm/solvers/c_solvers/idaklu/IDAKLUSolver.cpp
    pybamm/solvers/c_solvers/idaklu/IDAKLUSolver.hpp
    pybamm/solvers/c_solvers/idaklu/IDAKLUSolverOpenMP.inl
    pybamm/solvers/c_solvers/idaklu/IDAKLUSolverOpenMP.hpp
    pybamm/solvers/c_solvers/idaklu/IDAKLUSolverOpenMP_solvers.cpp
    pybamm/solvers/c_solvers/idaklu/IDAKLUSolverOpenMP_solvers.hpp
    pybamm/solvers/c_solvers/idaklu/sundials_functions.inl
    pybamm/solvers/c_solvers/idaklu/sundials_functions.hpp
    pybamm/solvers/c_solvers/idaklu/IdakluJax.cpp
    pybamm/solvers/c_solvers/idaklu/IdakluJax.hpp
    pybamm/solvers/c_solvers/idaklu/common.hpp
    pybamm/solvers/c_solvers/idaklu/python.hpp
    pybamm/solvers/c_solvers/idaklu/python.cpp
    pybamm/solvers/c_solvers/idaklu/Solution.cpp
    pybamm/solvers/c_solvers/idaklu/Solution.hpp
    pybamm/solvers/c_solvers/idaklu/Options.hpp
    pybamm/solvers/c_solvers/idaklu/Options.cpp
    # IDAKLU expressions / function evaluation [abstract]
    pybamm/solvers/c_solvers/idaklu/Expressions/Expressions.hpp
    pybamm/solvers/c_solvers/idaklu/Expressions/Base/Expression.hpp
    pybamm/solvers/c_solvers/idaklu/Expressions/Base/ExpressionSet.hpp
    pybamm/solvers/c_solvers/idaklu/Expressions/Base/ExpressionTypes.hpp
    # IDAKLU expressions - concrete implementations
    ${IDAKLU_EXPR_CASADI_SOURCE_FILES}
    ${IDAKLU_EXPR_IREE_SOURCE_FILES})

  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(USE_PYTHON_CASADI FALSE)
  else()
    set(USE_PYTHON_CASADI TRUE)
  endif()

  # Use importlib to find the casadi path without importing it. This is useful
  # to find the path for the build-time dependency, not the run-time dependency.
  execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c
      "import importlib.util; print(next(iter(importlib.util.find_spec('casadi').submodule_search_locations)))"
      OUTPUT_VARIABLE CASADI_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (CASADI_DIR)
    file(TO_CMAKE_PATH ${CASADI_DIR} CASADI_DIR)
    message("Found Python casadi path: ${CASADI_DIR}")
  endif()

  if(${USE_PYTHON_CASADI})
    message("Trying to link against Python casadi package")
    find_package(casadi CONFIG PATHS ${CASADI_DIR} REQUIRED NO_DEFAULT_PATH)
  else()
    message("Trying to link against any casadi package apart from the Python one")
    set(CMAKE_IGNORE_PATH "${CASADI_DIR}/cmake")
    find_package(casadi CONFIG REQUIRED)
  endif()

  set_target_properties(
    idaklu PROPERTIES
    INSTALL_RPATH "${CASADI_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})

  # Sundials
  set(SUNDIALS_ROOT ${CUSTOM_INSTALL_DIR})
  message("SUNDIALS_ROOT: ${SUNDIALS_ROOT}")
  find_package(SUNDIALS REQUIRED)
  message("SUNDIALS found in ${SUNDIALS_INCLUDE_DIR}: ${SUNDIALS_LIBRARIES}")
  target_include_directories(idaklu PRIVATE ${SUNDIALS_INCLUDE_DIR})
  target_link_libraries(idaklu PRIVATE ${SUNDIALS_LIBRARIES} casadi)

  # link suitesparse
  # if using vcpkg, use config mode to
  # find suitesparse. Otherwise, use FindSuiteSparse module
  set(SUITESPARSE_ROOT ${CUSTOM_INSTALL_DIR})
  message("SUITESPARSE_ROOT: ${SUITESPARSE_ROOT}")
  if(DEFINED VCPKG_ROOT_DIR)
    find_package(SUITESPARSE CONFIG REQUIRED)
  else()
    find_package(SUITESPARSE REQUIRED)
    message("SuiteSparse found in ${SuiteSparse_INCLUDE_DIRS}: ${SuiteSparse_LIBRARIES}")
  endif()
  include_directories(${SUITESPARSE_INCLUDE_DIRS})
  target_link_libraries(idaklu PRIVATE ${SUITESPARSE_LIBRARIES})

  # IREE (MLIR compiler and runtime library) build settings
  if(${PYBAMM_IDAKLU_EXPR_IREE} STREQUAL "ON" )
    set(IREE_BUILD_COMPILER ON)
    set(IREE_BUILD_TESTS OFF)
    set(IREE_BUILD_SAMPLES OFF)
    add_subdirectory(iree EXCLUDE_FROM_ALL)
    set(IREE_COMPILER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler")
    target_include_directories(idaklu SYSTEM PRIVATE "${IREE_COMPILER_ROOT}/bindings/c/iree/compiler")
    target_compile_options(idaklu PRIVATE ${IREE_DEFAULT_COPTS})
    target_link_libraries(idaklu PRIVATE iree_compiler_bindings_c_loader)
    target_link_libraries(idaklu PRIVATE iree_runtime_runtime)
  endif()

  install(TARGETS idaklu DESTINATION pybamm/solvers)
else()
  message(STATUS "Building without IDAKLU support")
endif()
