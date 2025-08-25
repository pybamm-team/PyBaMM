cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0074 NEW)
set(CMAKE_VERBOSE_MAKEFILE ON)

if (DEFINED Python_EXECUTABLE)
    set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
endif ()

if (DEFINED ENV{VCPKG_ROOT_DIR} AND NOT DEFINED VCPKG_ROOT_DIR)
    set(VCPKG_ROOT_DIR "$ENV{VCPKG_ROOT_DIR}"
            CACHE STRING "Vcpkg root directory")
endif ()

if (DEFINED VCPKG_ROOT_DIR)
    set(CMAKE_TOOLCHAIN_FILE ${VCPKG_ROOT_DIR}/scripts/buildsystems/vcpkg.cmake
            CACHE STRING "Vcpkg toolchain file")
endif ()

if (DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}"
            CACHE STRING "Vcpkg target triplet")
endif ()

project(idaklu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (NOT MSVC)
    # MSVC does not support variable length arrays (vla)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=vla")
endif ()

# casadi seems to compile without the newer versions of std::string
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

find_package(pybind11 CONFIG REQUIRED)

# Check Casadi build flag
if (NOT DEFINED PYBAMM_IDAKLU_EXPR_CASADI)
    set(PYBAMM_IDAKLU_EXPR_CASADI ON)
endif ()
message("PYBAMM_IDAKLU_EXPR_CASADI: ${PYBAMM_IDAKLU_EXPR_CASADI}")

# Casadi PyBaMM source files
set(IDAKLU_EXPR_CASADI_SOURCE_FILES "")
if (${PYBAMM_IDAKLU_EXPR_CASADI} STREQUAL "ON")
    add_compile_definitions(CASADI_ENABLE)
    set(IDAKLU_EXPR_CASADI_SOURCE_FILES
            src/pybammsolvers/idaklu_source/Expressions/Casadi/CasadiFunctions.cpp
            src/pybammsolvers/idaklu_source/Expressions/Casadi/CasadiFunctions.hpp
    )
endif ()

# The complete (all dependencies) sources list should be mirrored in setup.py
pybind11_add_module(idaklu
        # pybind11 interface
        src/pybammsolvers/idaklu.cpp
        # IDAKLU solver (SUNDIALS)
        src/pybammsolvers/idaklu_source/idaklu_solver.hpp
        src/pybammsolvers/idaklu_source/IDAKLUSolver.cpp
        src/pybammsolvers/idaklu_source/IDAKLUSolver.hpp
        src/pybammsolvers/idaklu_source/IDAKLUSolverGroup.cpp
        src/pybammsolvers/idaklu_source/IDAKLUSolverGroup.hpp
        src/pybammsolvers/idaklu_source/IDAKLUSolverOpenMP.inl
        src/pybammsolvers/idaklu_source/IDAKLUSolverOpenMP.hpp
        src/pybammsolvers/idaklu_source/IDAKLUSolverOpenMP_solvers.cpp
        src/pybammsolvers/idaklu_source/IDAKLUSolverOpenMP_solvers.hpp
        src/pybammsolvers/idaklu_source/sundials_functions.inl
        src/pybammsolvers/idaklu_source/sundials_functions.hpp
        src/pybammsolvers/idaklu_source/IdakluJax.cpp
        src/pybammsolvers/idaklu_source/IdakluJax.hpp
        src/pybammsolvers/idaklu_source/common.hpp
        src/pybammsolvers/idaklu_source/common.cpp
        src/pybammsolvers/idaklu_source/Solution.cpp
        src/pybammsolvers/idaklu_source/Solution.hpp
        src/pybammsolvers/idaklu_source/SolutionData.cpp
        src/pybammsolvers/idaklu_source/SolutionData.hpp
        src/pybammsolvers/idaklu_source/observe.cpp
        src/pybammsolvers/idaklu_source/observe.hpp
        src/pybammsolvers/idaklu_source/Options.hpp
        src/pybammsolvers/idaklu_source/Options.cpp
        # IDAKLU expressions / function evaluation [abstract]
        src/pybammsolvers/idaklu_source/Expressions/Expressions.hpp
        src/pybammsolvers/idaklu_source/Expressions/Base/Expression.hpp
        src/pybammsolvers/idaklu_source/Expressions/Base/ExpressionSet.hpp
        src/pybammsolvers/idaklu_source/Expressions/Base/ExpressionTypes.hpp
        # IDAKLU expressions - concrete implementations
        ${IDAKLU_EXPR_CASADI_SOURCE_FILES}
)

if (NOT DEFINED USE_PYTHON_CASADI)
    set(USE_PYTHON_CASADI TRUE)
endif ()


execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c
        "import os; import sysconfig; print(os.path.join(sysconfig.get_path('purelib'), 'casadi', 'cmake'))"
        OUTPUT_VARIABLE CASADI_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if (CASADI_DIR)
    file(TO_CMAKE_PATH ${CASADI_DIR} CASADI_DIR)
    message("Found Python casadi path: ${CASADI_DIR}")
else ()
    message(FATAL_ERROR "Did not find casadi path}")
endif ()

if (${USE_PYTHON_CASADI})
    message("Trying to link against Python casadi package in ${CASADI_DIR}")
    find_package(casadi CONFIG PATHS ${CASADI_DIR} REQUIRED NO_DEFAULT_PATH)
else ()
    message("Trying to link against any casadi package apart from the Python one")
    set(CMAKE_IGNORE_PATH "${CASADI_DIR}/cmake")
    find_package(casadi CONFIG REQUIRED)
endif ()

set_target_properties(
        idaklu PROPERTIES
        INSTALL_RPATH "${CASADI_DIR}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
)

# openmp
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    execute_process(
            COMMAND "brew" "--prefix"
            OUTPUT_VARIABLE HOMEBREW_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (OpenMP_ROOT)
        set(OpenMP_ROOT "${OpenMP_ROOT}:${HOMEBREW_PREFIX}/opt/libomp")
    else ()
        set(OpenMP_ROOT "${HOMEBREW_PREFIX}/opt/libomp")
    endif ()
endif ()
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(idaklu PRIVATE OpenMP::OpenMP_CXX)
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})
# Sundials
find_package(SUNDIALS REQUIRED)
message("SUNDIALS found in ${SUNDIALS_INCLUDE_DIR}: ${SUNDIALS_LIBRARIES}")
target_include_directories(idaklu PRIVATE ${SUNDIALS_INCLUDE_DIR})
target_link_libraries(idaklu PRIVATE ${SUNDIALS_LIBRARIES} casadi)

# link suitesparse
# if using vcpkg, use config mode to
# find suitesparse. Otherwise, use FindSuiteSparse module
if (DEFINED VCPKG_ROOT_DIR)
    find_package(SuiteSparse CONFIG REQUIRED)
else ()
    find_package(SuiteSparse REQUIRED)
    message("SuiteSparse found in ${SuiteSparse_INCLUDE_DIRS}: ${SuiteSparse_LIBRARIES}")
endif ()
include_directories(${SuiteSparse_INCLUDE_DIRS})
target_link_libraries(idaklu PRIVATE ${SuiteSparse_LIBRARIES})
