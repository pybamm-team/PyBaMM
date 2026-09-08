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

# Default to .idaklu (install_KLU_Sundials.py output). Skipped under conda
# (which sets CONDA_BUILD=1), where SUNDIALS/SuiteSparse come from the host
# environment and are found on CMAKE_PREFIX_PATH.
if(NOT WIN32 AND NOT DEFINED ENV{CONDA_BUILD} AND NOT DEFINED SuiteSparse_ROOT)
    set(SuiteSparse_ROOT "${CMAKE_SOURCE_DIR}/.idaklu")
endif()
if(NOT WIN32 AND NOT DEFINED ENV{CONDA_BUILD} AND NOT DEFINED SUNDIALS_ROOT)
    set(SUNDIALS_ROOT "${CMAKE_SOURCE_DIR}/.idaklu")
endif()

# Auto-build the vendored SUNDIALS/SuiteSparse on a from-source build, so a clean
# checkout with submodules needs no manual pre-step. Skipped on Windows (vcpkg),
# and skipped once the libraries already exist — under editable.rebuild CMake
# reconfigures on every import, and re-requiring the submodules (or re-running the
# bootstrap) for an already-built tree would needlessly fail or repeat work.
# Also skipped under conda (CONDA_BUILD=1), which supplies both libraries as
# host dependencies and ships no submodules in the sdist.
if(NOT WIN32 AND NOT DEFINED VCPKG_ROOT_DIR AND NOT DEFINED ENV{CONDA_BUILD})
    file(GLOB _sundials_prebuilt
        "${SUNDIALS_ROOT}/lib/libsundials_idas.*"
        "${SUNDIALS_ROOT}/lib64/libsundials_idas.*")
    file(GLOB _suitesparse_prebuilt
        "${SuiteSparse_ROOT}/lib/libklu.*"
        "${SuiteSparse_ROOT}/lib64/libklu.*")
    if(NOT _sundials_prebuilt OR NOT _suitesparse_prebuilt)
        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/sundials/CMakeLists.txt"
                OR NOT EXISTS "${CMAKE_SOURCE_DIR}/SuiteSparse/SuiteSparse_config/CMakeLists.txt")
            message(FATAL_ERROR
                "SUNDIALS/SuiteSparse are not built (no libraries under "
                "${SUNDIALS_ROOT}) and the submodule sources are missing. Either:\n"
                "  * fetch the sources: run `git submodule update --init` from the\n"
                "    repository root (or clone with --recurse-submodules), or\n"
                "  * use a system install: set SUNDIALS_ROOT and SuiteSparse_ROOT.")
        endif()
        execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" install_KLU_Sundials.py
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RESULT_VARIABLE _idaklu_rc)
        if(_idaklu_rc)
            message(FATAL_ERROR
                "install_KLU_Sundials.py failed (exit ${_idaklu_rc}); "
                "check that make, cmake, and a C/Fortran compiler are installed.")
        endif()
    endif()
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

# The libstdc++ std::string ABI (_GLIBCXX_USE_CXX11_ABI) is reconciled against
# the actual libcasadi build further down, once CasADi has been located.

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
        src/pybammsolvers/idaklu_source/brent.hpp
        src/pybammsolvers/idaklu_source/brent.cpp
        src/pybammsolvers/idaklu_source/brent_impl.hpp
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
    # Under conda (CONDA_BUILD=1) CasADi is a host dependency on every platform,
    # so use that build rather than one from a pip site-packages tree.
    if (WIN32 OR DEFINED ENV{CONDA_BUILD})
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
        # CasADi's internal headers include their public siblings unqualified
        target_include_directories(idaklu PRIVATE ${CASADI_INCLUDE_DIR}/casadi/core)
    else ()
        message(FATAL_ERROR "Could not find CasADi include directory")
    endif ()

    # fallback version, used only if casadi/config.h cannot be found below
    execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" -c "import casadi; print(casadi.__version__)"
            OUTPUT_VARIABLE CASADI_LINKED_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)

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
    # CasADi's internal headers include their public siblings unqualified
    if(TARGET casadi::casadi)
        get_target_property(_casadi_incs casadi::casadi INTERFACE_INCLUDE_DIRECTORIES)
        foreach(_casadi_inc IN LISTS _casadi_incs)
            target_include_directories(idaklu PRIVATE ${_casadi_inc}/casadi/core)
        endforeach()
    endif()
endif ()

# casadi::Rootfinder's headers are not installed and are LGPL-3.0-or-later, so stage
# them from the matching sdist rather than vendoring them into this BSD-3-Clause tree.
set(CASADI_INTERNAL_VERSION ""
        CACHE STRING "CasADi version to source the internal headers from; read from the linked CasADi when empty")
set(CASADI_INTERNAL_SHA256 ""
        CACHE STRING "SHA-256 of the CasADi source distribution, checked on download when set")
set(CASADI_SOURCE_DIR ""
        CACHE PATH "Local CasADi source tree; set this for offline builds to skip the download")

option(CASADI_ALLOW_UNVERIFIED_INTERNALS
        "Build the \"brent\" plugin even if the linked CasADi cannot be identified" OFF)

# Identify the linked CasADi from its own config.h: the system path (Windows/vcpkg,
# conda) has no interpreter to ask, and that is where a mismatch would go unnoticed.
set(_casadi_include_dirs "")
if (CASADI_INCLUDE_DIR)
    list(APPEND _casadi_include_dirs "${CASADI_INCLUDE_DIR}")
endif ()
if (TARGET casadi::casadi)
    get_target_property(_casadi_target_incs casadi::casadi INTERFACE_INCLUDE_DIRECTORIES)
    if (_casadi_target_incs)
        list(APPEND _casadi_include_dirs ${_casadi_target_incs})
    endif ()
endif ()

unset(CASADI_CONFIG_HEADER CACHE)
find_file(CASADI_CONFIG_HEADER casadi/config.h PATHS ${_casadi_include_dirs} NO_DEFAULT_PATH)

set(_casadi_config_flags "")
if (CASADI_CONFIG_HEADER)
    file(READ "${CASADI_CONFIG_HEADER}" _casadi_config_contents)
    string(REGEX MATCH "CASADI_VERSION_STRING[ \t]+\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
            _casadi_version_match "${_casadi_config_contents}")
    set(CASADI_LINKED_VERSION "${CMAKE_MATCH_1}")
    string(REGEX MATCH "CASADI_COMPILER_FLAGS[ \t]+\"([^\"]*)\""
            _casadi_flags_match "${_casadi_config_contents}")
    set(_casadi_config_flags "${CMAKE_MATCH_1}")
endif ()

# Follow the CasADi being linked rather than a version pinned here, so the pin in
# pyproject.toml stays the only one.
set(_casadi_requested_version "${CASADI_INTERNAL_VERSION}")
if (NOT _casadi_requested_version)
    set(CASADI_INTERNAL_VERSION "${CASADI_LINKED_VERSION}")
endif ()

if (NOT CASADI_INTERNAL_VERSION)
    message(FATAL_ERROR
            "Could not find casadi/config.h in any of: ${_casadi_include_dirs}\n"
            "The \"brent\" rootfinder plugin subclasses CasADi internals that are not "
            "ABI-stable, so the version to source them from has to be known. Pass "
            "-DCASADI_INTERNAL_VERSION=<x.y.z>, with "
            "-DCASADI_ALLOW_UNVERIFIED_INTERNALS=ON to skip the check that it matches.")
elseif (NOT CASADI_LINKED_VERSION)
    if (CASADI_ALLOW_UNVERIFIED_INTERNALS)
        message(WARNING
                "Could not read the linked CasADi version from casadi/config.h. The \"brent\" "
                "plugin is being built against CasADi ${CASADI_INTERNAL_VERSION} internals "
                "without checking that they match; an ABI mismatch will not be diagnosed.")
    else ()
        message(FATAL_ERROR
                "Could not find casadi/config.h in any of: ${_casadi_include_dirs}\n"
                "The \"brent\" rootfinder plugin subclasses CasADi internals that are not "
                "ABI-stable, so the linked version must be known before it can be built. "
                "Configure with -DCASADI_ALLOW_UNVERIFIED_INTERNALS=ON to build anyway.")
    endif ()
elseif (NOT CASADI_LINKED_VERSION STREQUAL CASADI_INTERNAL_VERSION)
    message(FATAL_ERROR
            "CasADi version mismatch: linking CasADi ${CASADI_LINKED_VERSION} but sourcing the "
            "internal headers for the \"brent\" plugin from the requested "
            "${CASADI_INTERNAL_VERSION}. These headers are not ABI-stable, so they must match; "
            "drop -DCASADI_INTERNAL_VERSION to follow the linked CasADi.")
else ()
    message(STATUS "idaklu: \"brent\" plugin verified against CasADi ${CASADI_LINKED_VERSION}")
endif ()

# options.hpp is public, but stage it too: the headers below include it unqualified, and
# MSVC then searches the include stack, where our own case-insensitively equal Options.hpp
# would answer for it.
set(_casadi_internal_headers
        casadi_os.hpp
        function_internal.hpp
        options.hpp
        oracle_function.hpp
        plugin_interface.hpp
        rootfinder_impl.hpp)
set(_casadi_internal_work "${CMAKE_CURRENT_BINARY_DIR}/casadi-internal")
set(_casadi_internal_include "${_casadi_internal_work}/${CASADI_INTERNAL_VERSION}/include")

# editable.rebuild reconfigures on every import; skip the work once staged.
set(_casadi_internal_staged TRUE)
foreach (_header IN LISTS _casadi_internal_headers)
    if (NOT EXISTS "${_casadi_internal_include}/casadi/core/${_header}")
        set(_casadi_internal_staged FALSE)
    endif ()
endforeach ()

if (NOT _casadi_internal_staged)
    if (NOT CASADI_SOURCE_DIR AND DEFINED ENV{CASADI_SOURCE_DIR})
        set(CASADI_SOURCE_DIR "$ENV{CASADI_SOURCE_DIR}")
    endif ()

    if (CASADI_SOURCE_DIR)
        file(TO_CMAKE_PATH "${CASADI_SOURCE_DIR}" _casadi_internal_src)
        message(STATUS "idaklu: taking CasADi internal headers from ${_casadi_internal_src}")
    else ()
        set(_casadi_sdist "${_casadi_internal_work}/casadi-${CASADI_INTERNAL_VERSION}.tar.gz")
        set(_casadi_sdist_url
                "https://pypi.org/packages/source/c/casadi/casadi-${CASADI_INTERNAL_VERSION}.tar.gz")
        message(STATUS "idaklu: downloading ${_casadi_sdist_url} for the CasADi internal headers")
        set(_casadi_expected_hash "")
        if (CASADI_INTERNAL_SHA256)
            set(_casadi_expected_hash EXPECTED_HASH SHA256=${CASADI_INTERNAL_SHA256})
        endif ()
        file(DOWNLOAD "${_casadi_sdist_url}" "${_casadi_sdist}"
                ${_casadi_expected_hash}
                STATUS _casadi_download_status)
        list(GET _casadi_download_status 0 _casadi_download_rc)
        if (_casadi_download_rc)
            list(GET _casadi_download_status 1 _casadi_download_message)
            file(REMOVE "${_casadi_sdist}")
            message(FATAL_ERROR
                    "Could not download the CasADi source distribution, which supplies the "
                    "internal headers the \"brent\" rootfinder plugin is built against:\n"
                    "  ${_casadi_sdist_url}\n"
                    "  ${_casadi_download_message}\n"
                    "For an offline build, unpack that archive yourself and configure with "
                    "-DCASADI_SOURCE_DIR=<dir> (or set the CASADI_SOURCE_DIR environment "
                    "variable), pointing at a CasADi ${CASADI_INTERNAL_VERSION} source tree.")
        endif ()
        set(_casadi_internal_members "")
        foreach (_header IN LISTS _casadi_internal_headers)
            list(APPEND _casadi_internal_members
                    "casadi-${CASADI_INTERNAL_VERSION}/casadi/core/${_header}")
        endforeach ()
        execute_process(
                COMMAND "${CMAKE_COMMAND}" -E tar xzf "${_casadi_sdist}" ${_casadi_internal_members}
                WORKING_DIRECTORY "${_casadi_internal_work}"
                RESULT_VARIABLE _casadi_extract_rc)
        if (_casadi_extract_rc)
            message(FATAL_ERROR
                    "Could not extract the CasADi internal headers from ${_casadi_sdist} "
                    "(exit ${_casadi_extract_rc}).")
        endif ()
        set(_casadi_internal_src "${_casadi_internal_work}/casadi-${CASADI_INTERNAL_VERSION}")
    endif ()

    # Stage only the listed headers, so the rest of a CasADi source tree can never
    # shadow the generated public headers of the CasADi actually being linked.
    foreach (_header IN LISTS _casadi_internal_headers)
        if (NOT EXISTS "${_casadi_internal_src}/casadi/core/${_header}")
            message(FATAL_ERROR
                    "${_casadi_internal_src} is not a CasADi source tree: "
                    "casadi/core/${_header} is missing.")
        endif ()
        configure_file(
                "${_casadi_internal_src}/casadi/core/${_header}"
                "${_casadi_internal_include}/casadi/core/${_header}"
                COPYONLY)
    endforeach ()
endif ()

target_include_directories(idaklu PRIVATE ${_casadi_internal_include})

# brent_impl.hpp is emitted verbatim into generated C, so stringify it rather than
# keeping a second copy.
set(BRENT_IMPL_HEADER
        "${CMAKE_CURRENT_SOURCE_DIR}/src/pybammsolvers/idaklu_source/brent_impl.hpp")
file(READ "${BRENT_IMPL_HEADER}" BRENT_IMPL_SOURCE)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/brent_impl_source.hpp"
        "// Generated by CMake from brent_impl.hpp. Do not edit.\n"
        "namespace casadi {\n"
        "static const char brent_impl_str[] = R\"PYBAMM_BRENT(\n"
        "${BRENT_IMPL_SOURCE}"
        ")PYBAMM_BRENT\";\n"
        "}  // namespace casadi\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${BRENT_IMPL_HEADER}")
target_include_directories(idaklu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
# Reuse CASADI_COMPILER_FLAGS from config.h: CASADI_WITH_THREADSAFE_SYMBOLICS adds a
# static mutex to Rootfinder, so guessing it wrong is an ABI break.
set(_casadi_internal_defs "")
if (_casadi_config_flags)
    separate_arguments(_casadi_config_flag_list UNIX_COMMAND "${_casadi_config_flags}")
    foreach (_flag IN LISTS _casadi_config_flag_list)
        if (_flag MATCHES "^-D(.+)$")
            list(APPEND _casadi_internal_defs "${CMAKE_MATCH_1}")
        endif ()
    endforeach ()
endif ()
if (NOT _casadi_internal_defs)
    # No config.h to read: fall back to the flags CasADi ${CASADI_INTERNAL_VERSION} uses.
    set(_casadi_internal_defs CASADI_VERSION=31 WITH_DL
            CASADI_WITH_THREADSAFE_SYMBOLICS CASADI_WITH_THREAD)
endif ()
message(STATUS "idaklu: CasADi internal header flags: ${_casadi_internal_defs}")
target_compile_definitions(idaklu PRIVATE ${_casadi_internal_defs})

# Match idaklu's libstdc++ std::string ABI to the linked libcasadi (GNU only;
# no-op for MSVC and macOS/libc++). CasADi's PyPI casadi-config.cmake forces
# `_GLIBCXX_USE_CXX11_ABI=0` on consumers: correct for the x86_64 wheel (legacy
# ABI) but wrong for the aarch64 wheel (C++11 ABI), leaving casadi's std::string
# symbols unresolved. Detect libcasadi's real ABI and override CasADi's guess.
# Only relevant for the PyPI wheel: a system CasADi (conda-forge, distro, vcpkg)
# is built against the same libstdc++ as this extension, and the probe below has
# no CASADI_LIB_DIR to look in, so it would wrongly force the legacy ABI.
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND USE_PYTHON_CASADI)
    # Drop CasADi's hard-coded (and, on aarch64, wrong) ABI interface definition.
    if (TARGET casadi::casadi)
        get_target_property(_casadi_defs casadi::casadi INTERFACE_COMPILE_DEFINITIONS)
        if (_casadi_defs)
            list(REMOVE_ITEM _casadi_defs
                "_GLIBCXX_USE_CXX11_ABI=0" "_GLIBCXX_USE_CXX11_ABI=1")
            set_target_properties(casadi::casadi PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS "${_casadi_defs}")
        endif ()
    endif ()

    # Key off casadi::Function::deserialize(const std::string&): the C++11-ABI
    # build exports the __cxx11 mangling, the legacy build the `Ss` mangling.
    # Default to the legacy ABI if the probe cannot run (never worse than before).
    set(_casadi_cxx11_abi 0)
    set(_nm "${CMAKE_NM}")
    if (NOT _nm)
        find_program(_nm NAMES nm gcc-nm)
    endif ()
    file(GLOB _libcasadi "${CASADI_LIB_DIR}/libcasadi.so*")
    if (_libcasadi AND _nm)
        list(GET _libcasadi 0 _libcasadi0)
        execute_process(
            COMMAND "${_nm}" -D --defined-only "${_libcasadi0}"
            OUTPUT_VARIABLE _casadi_syms
            ERROR_QUIET)
        if (_casadi_syms MATCHES "Function11deserializeERKNSt7__cxx11")
            set(_casadi_cxx11_abi 1)
        endif ()
    endif ()
    message(STATUS "idaklu: _GLIBCXX_USE_CXX11_ABI=${_casadi_cxx11_abi} (matched to libcasadi)")
    target_compile_definitions(idaklu PRIVATE _GLIBCXX_USE_CXX11_ABI=${_casadi_cxx11_abi})
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
# With vcpkg, SuiteSparse is split into per-component ports with no combined
# config; find KLU directly (its target pulls in AMD/COLAMD/BTF/config).
# Otherwise, use the FindSuiteSparse module.
if (DEFINED VCPKG_ROOT_DIR)
    find_package(KLU CONFIG REQUIRED)
    target_link_libraries(idaklu PRIVATE SuiteSparse::KLU)
else ()
    find_package(SuiteSparse REQUIRED)
    message("SuiteSparse found in ${SuiteSparse_INCLUDE_DIRS}: ${SuiteSparse_LIBRARIES}")
    target_include_directories(idaklu PRIVATE ${SuiteSparse_INCLUDE_DIRS})
    target_link_libraries(idaklu PRIVATE ${SuiteSparse_LIBRARIES})
endif ()

install(TARGETS idaklu LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
