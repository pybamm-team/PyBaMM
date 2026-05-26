cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0074 NEW)
set(CMAKE_VERBOSE_MAKEFILE ON)

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

# Require vcpkg on Windows unless building via conda (which sets CONDA_BUILD=1)
if(WIN32 AND NOT DEFINED VCPKG_ROOT_DIR AND NOT DEFINED ENV{VCPKG_ROOT_DIR} AND NOT DEFINED ENV{CONDA_BUILD})
    message(FATAL_ERROR "Windows builds require VCPKG_ROOT_DIR (CMake variable or environment variable) to be set.")
endif()

project(idaklu)

# Default to .idaklu (install_KLU_Sundials.py output).
if(NOT WIN32 AND NOT DEFINED SuiteSparse_ROOT)
    set(SuiteSparse_ROOT "${CMAKE_SOURCE_DIR}/.idaklu")
endif()
if(NOT WIN32 AND NOT DEFINED SUNDIALS_ROOT)
    set(SUNDIALS_ROOT "${CMAKE_SOURCE_DIR}/.idaklu")
endif()

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

# Casadi PyBaMM source files
set(IDAKLU_EXPR_CASADI_SOURCE_FILES
        src/pybammsolvers/idaklu_source/Expressions/Casadi/CasadiFunctions.cpp
        src/pybammsolvers/idaklu_source/Expressions/Casadi/CasadiFunctions.hpp
)

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
        src/pybammsolvers/idaklu_source/SolutionData.hpp
        src/pybammsolvers/idaklu_source/observe.cpp
        src/pybammsolvers/idaklu_source/observe.hpp
        src/pybammsolvers/idaklu_source/Options.hpp
        src/pybammsolvers/idaklu_source/Options.cpp
        src/pybammsolvers/idaklu_source/StandaloneNewtonSolver.hpp
        src/pybammsolvers/idaklu_source/StandaloneNewtonSolver.cpp
        # IDAKLU expressions / function evaluation [abstract]
        src/pybammsolvers/idaklu_source/Expressions/Expressions.hpp
        src/pybammsolvers/idaklu_source/Expressions/Base/Expression.hpp
        src/pybammsolvers/idaklu_source/Expressions/Base/ExpressionSet.hpp
        src/pybammsolvers/idaklu_source/Expressions/Base/ExpressionTypes.hpp
        # IDAKLU expressions - concrete implementations
        ${IDAKLU_EXPR_CASADI_SOURCE_FILES}
)

if (NOT DEFINED USE_PYTHON_CASADI)
    # Windows uses vcpkg for CasADi (no pip wheel exists); Linux/macOS use pip.
    if (WIN32)
        set(USE_PYTHON_CASADI FALSE)
    else()
        set(USE_PYTHON_CASADI TRUE)
    endif()
endif ()

if (${USE_PYTHON_CASADI})
    execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c
            "import os; import sysconfig; print(os.path.join(sysconfig.get_path('purelib'), 'casadi', 'cmake'))"
            OUTPUT_VARIABLE CASADI_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (CASADI_DIR)
        file(TO_CMAKE_PATH ${CASADI_DIR} CASADI_DIR)
        message("Found Python casadi path: ${CASADI_DIR}")
    else ()
        message(FATAL_ERROR "Did not find casadi path")
    endif ()

    message("Trying to link against Python casadi package in ${CASADI_DIR}")
    if (EXISTS "${CASADI_DIR}/casadiConfig.cmake" OR EXISTS "${CASADI_DIR}/casadi-config.cmake")
        find_package(casadi CONFIG PATHS ${CASADI_DIR} NO_DEFAULT_PATH)
    else ()
        message(WARNING "CasADi CMake config not found in ${CASADI_DIR}. Proceeding without find_package; using include and library paths discovered from the Python package.")
    endif ()

    execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c
            "import casadi; from pathlib import Path; print(Path(casadi.__file__).parent / 'include')"
            OUTPUT_VARIABLE CASADI_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (CASADI_INCLUDE_DIR)
        file(TO_CMAKE_PATH ${CASADI_INCLUDE_DIR} CASADI_INCLUDE_DIR)
        message("Found Python CasADi include directory: ${CASADI_INCLUDE_DIR}")
        target_include_directories(idaklu PRIVATE ${CASADI_INCLUDE_DIR})
    else ()
        message(FATAL_ERROR "Could not find CasADi include directory")
    endif ()

    execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c
            "import casadi; from pathlib import Path; import glob; lib_dir = Path(casadi.__file__).parent; lib_files = list(lib_dir.glob('*casadi*')); print(str(lib_dir) if lib_files else '')"
            OUTPUT_VARIABLE CASADI_LIB_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    if (CASADI_LIB_DIR)
        file(TO_CMAKE_PATH ${CASADI_LIB_DIR} CASADI_LIB_DIR)
        message("Found Python CasADi library directory: ${CASADI_LIB_DIR}")
        target_link_directories(idaklu PRIVATE ${CASADI_LIB_DIR})

        # RPATH entries for `pybammsolvers/idaklu.<abi>.so`:
        #   @loader_path/../casadi / $ORIGIN/../casadi: finds libcasadi at
        #     runtime.  .idaklu/{lib,lib64}: vendored SUNDIALS/SuiteSparse
        #     (used at runtime and by auditwheel/delocate at repair time).
        #     Windows uses vcpkg static linking; no RPATH needed.
        if (APPLE)
            set(IDAKLU_INSTALL_RPATH
                "@loader_path/../casadi"
                "${SUNDIALS_ROOT}/lib"
                "${SuiteSparse_ROOT}/lib"
            )
        else()
            set(IDAKLU_INSTALL_RPATH
                "$ORIGIN/../casadi"
                "${SUNDIALS_ROOT}/lib"
                "${SUNDIALS_ROOT}/lib64"
                "${SuiteSparse_ROOT}/lib"
                "${SuiteSparse_ROOT}/lib64"
            )
        endif()
        set_target_properties(idaklu PROPERTIES
            INSTALL_RPATH "${IDAKLU_INSTALL_RPATH}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
        # casadi 3.7+ exports casadi::casadi; link by name to avoid stale
        # paths from pip's isolated build environments.
        if(TARGET casadi::casadi)
            target_link_libraries(idaklu PRIVATE casadi::casadi)
        else()
            target_link_libraries(idaklu PRIVATE casadi)
        endif()
    else ()
        message(FATAL_ERROR "Could not find CasADi library directory")
    endif ()
else ()
    message("Trying to link against any casadi package apart from the Python one")
    find_package(casadi CONFIG REQUIRED)
    # casadi 3.7+ exports the target as casadi::casadi instead of casadi
    if(TARGET casadi::casadi)
        target_link_libraries(idaklu PRIVATE casadi::casadi)
    else()
        target_link_libraries(idaklu PRIVATE casadi)
    endif()
endif ()

# Math flags for the IDAKLU target.
# -fno-math-errno: allows inlining fabs/fmax/sqrt without errno-setting overhead.
# -fno-trapping-math: allows reordering around potential FP exceptions.
if (NOT MSVC)
    target_compile_options(idaklu PRIVATE
        -fno-math-errno
        -fno-trapping-math
    )
endif()

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
if (MSVC)
    set(OpenMP_CXX_FLAGS "/openmp:experimental")
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
target_link_libraries(idaklu PRIVATE ${SUNDIALS_LIBRARIES})

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

install(TARGETS idaklu LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
