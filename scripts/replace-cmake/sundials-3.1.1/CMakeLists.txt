# ---------------------------------------------------------------
# Programmer:  Radu Serban @ LLNL
# ---------------------------------------------------------------
# LLNS Copyright Start
# Copyright (c) 2014, Lawrence Livermore National Security
# This work was performed under the auspices of the U.S. Department
# of Energy by Lawrence Livermore National Laboratory in part under
# Contract W-7405-Eng-48 and in part under Contract DE-AC52-07NA27344.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see the LICENSE file.
# LLNS Copyright End
# ---------------------------------------------------------------
# Top level CMakeLists.txt for SUNDIALS (for cmake build system)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Initial commands
# ---------------------------------------------------------------

# Require a fairly recent cmake version
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.1)

# Set CMake policy to allow examples to build
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# Project SUNDIALS (initially only C supported)
# sets PROJECT_SOURCE_DIR and PROJECT_BINARY_DIR variables
PROJECT(sundials C)

# Set some variables with info on the SUNDIALS project
SET(PACKAGE_BUGREPORT "woodward6@llnl.gov")
SET(PACKAGE_NAME "SUNDIALS")
SET(PACKAGE_STRING "SUNDIALS 3.1.1")
SET(PACKAGE_TARNAME "sundials")

# set SUNDIALS version numbers 
# (use "" for the version label if none is needed)
SET(PACKAGE_VERSION_MAJOR "3")
SET(PACKAGE_VERSION_MINOR "1")
SET(PACKAGE_VERSION_PATCH "1")
SET(PACKAGE_VERSION_LABEL "")

IF(PACKAGE_VERSION_LABEL)
  SET(PACKAGE_VERSION "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}-${PACKAGE_VERSION_LABEL}")
ELSE()
  SET(PACKAGE_VERSION "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}")
ENDIF()

#
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# Prohibit in-source build
IF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  MESSAGE(FATAL_ERROR "In-source build prohibited.")
ENDIF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")

# Hide some cache variables
MARK_AS_ADVANCED(EXECUTABLE_OUTPUT_PATH LIBRARY_OUTPUT_PATH)

# Always show the C compiler and flags
MARK_AS_ADVANCED(CLEAR
  CMAKE_C_COMPILER
  CMAKE_C_FLAGS)

# Specify the VERSION and SOVERSION for shared libraries

SET(arkodelib_VERSION "2.1.1")
SET(arkodelib_SOVERSION "2")

SET(cvodelib_VERSION "3.1.1")
SET(cvodelib_SOVERSION "3")

SET(cvodeslib_VERSION "3.1.1")
SET(cvodeslib_SOVERSION "3")

SET(idalib_VERSION "3.1.1")
SET(idalib_SOVERSION "3")

SET(idaslib_VERSION "2.1.0")
SET(idaslib_SOVERSION "2")

SET(kinsollib_VERSION "3.1.1")
SET(kinsollib_SOVERSION "3")

SET(cpodeslib_VERSION "0.0.0")
SET(cpodeslib_SOVERSION "0")

SET(nveclib_VERSION "3.1.1")
SET(nveclib_SOVERSION "3")

SET(sunmatrixlib_VERSION "1.1.1")
SET(sunmatrixlib_SOVERSION "1")

SET(sunlinsollib_VERSION "1.1.1")
SET(sunlinsollib_SOVERSION "1")

# Specify the location of additional CMAKE modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/config)

# ---------------------------------------------------------------
# MACRO definitions
# ---------------------------------------------------------------
INCLUDE(SundialsCMakeMacros)

# ---------------------------------------------------------------
# Check for deprecated SUNDIALS CMake options/variables
# ---------------------------------------------------------------
INCLUDE(SundialsDeprecated)

# ---------------------------------------------------------------
# Which modules to build?
# ---------------------------------------------------------------

# For each SUNDIALS solver available (i.e. for which we have the
# sources), give the user the option of enabling/disabling it.

IF(IS_DIRECTORY "${sundials_SOURCE_DIR}/src/arkode")
  OPTION(BUILD_ARKODE "Build the ARKODE library" ON)
ELSE()
  SET(BUILD_ARKODE OFF)
ENDIF()

IF(IS_DIRECTORY "${sundials_SOURCE_DIR}/src/cvode")
  OPTION(BUILD_CVODE "Build the CVODE library" ON)
ELSE()
  SET(BUILD_CVODE OFF)
ENDIF()

IF(IS_DIRECTORY "${sundials_SOURCE_DIR}/src/cvodes")
  OPTION(BUILD_CVODES "Build the CVODES library" ON)
ELSE()
  SET(BUILD_CVODES OFF)
ENDIF()

IF(IS_DIRECTORY "${sundials_SOURCE_DIR}/src/ida")
  OPTION(BUILD_IDA "Build the IDA library" ON)
ELSE()
  SET(BUILD_IDA OFF)
ENDIF()

IF(IS_DIRECTORY "${sundials_SOURCE_DIR}/src/idas")
  OPTION(BUILD_IDAS "Build the IDAS library" ON)
ELSE()
  SET(BUILD_IDAS OFF)
ENDIF()

IF(IS_DIRECTORY "${sundials_SOURCE_DIR}/src/kinsol")
  OPTION(BUILD_KINSOL "Build the KINSOL library" ON)
ELSE()
  SET(BUILD_KINSOL OFF)
ENDIF()

# CPODES is always OFF for now.  (commented out for Release); ToDo: better way to do this?
#IF(IS_DIRECTORY "${sundials_SOURCE_DIR}/src/cpodes")
#  OPTION(BUILD_CPODES  "Build the CPODES library"  OFF)
#ELSE()
#  SET(BUILD_CPODES OFF)
#ENDIF()

# ---------------------------------------------------------------
# xSDK specific options
# ---------------------------------------------------------------
INCLUDE(SundialsXSDK)

# ---------------------------------------------------------------
# Build specific C flags
# ---------------------------------------------------------------

# Hide all build type specific flags
MARK_AS_ADVANCED(FORCE
  CMAKE_C_FLAGS_DEBUG
  CMAKE_C_FLAGS_MINSIZEREL
  CMAKE_C_FLAGS_RELEASE
  CMAKE_C_FLAGS_RELWITHDEBINFO)

# Only show flags for the current build type it is set
# NOTE: Build specific flags are appended those in CMAKE_C_FLAGS
IF(CMAKE_BUILD_TYPE)
  IF(CMAKE_BUILD_TYPE MATCHES "Debug")
    MESSAGE("Appending C debug flags")
    MARK_AS_ADVANCED(CLEAR CMAKE_C_FLAGS_DEBUG)
  ELSEIF(CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
    MESSAGE("Appending C min size release flags")
    MARK_AS_ADVANCED(CLEAR CMAKE_C_FLAGS_MINSIZEREL)
  ELSEIF(CMAKE_BUILD_TYPE MATCHES "Release")
    MESSAGE("Appending C release flags")
    MARK_AS_ADVANCED(CLEAR CMAKE_C_FLAGS_RELEASE)
  ELSEIF(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    MESSAGE("Appending C release with debug info flags")
    MARK_AS_ADVANCED(CLEAR CMAKE_C_FLAGS_RELWITHDEBINFO)
  ENDIF()
ENDIF()

# ---------------------------------------------------------------
# Option to specify precision (realtype)
# ---------------------------------------------------------------

SET(DOCSTR "single, double, or extended")
SHOW_VARIABLE(SUNDIALS_PRECISION STRING "${DOCSTR}" "double")

# prepare substitution variable PRECISION_LEVEL for sundials_config.h
STRING(TOUPPER ${SUNDIALS_PRECISION} SUNDIALS_PRECISION)
SET(PRECISION_LEVEL "#define SUNDIALS_${SUNDIALS_PRECISION}_PRECISION 1")

# prepare substitution variable FPRECISION_LEVEL for sundials_fconfig.h
IF(SUNDIALS_PRECISION MATCHES "SINGLE")
  SET(FPRECISION_LEVEL "4")
ENDIF(SUNDIALS_PRECISION MATCHES "SINGLE")
IF(SUNDIALS_PRECISION MATCHES "DOUBLE")
  SET(FPRECISION_LEVEL "8")
ENDIF(SUNDIALS_PRECISION MATCHES "DOUBLE")
IF(SUNDIALS_PRECISION MATCHES "EXTENDED")
  SET(FPRECISION_LEVEL "16")
ENDIF(SUNDIALS_PRECISION MATCHES "EXTENDED")

# ---------------------------------------------------------------
# Option to specify index type
# ---------------------------------------------------------------

SET(DOCSTR "Signed 64-bit (int64_t) or signed 32-bit (int32_t) integer")
SHOW_VARIABLE(SUNDIALS_INDEX_TYPE STRING "${DOCSTR}" "int64_t")

# prepare substitution variable INDEX_TYPE for sundials_config.h
STRING(TOUPPER ${SUNDIALS_INDEX_TYPE} SUNDIALS_INDEX_TYPE)
SET(INDEX_TYPE "#define SUNDIALS_${SUNDIALS_INDEX_TYPE} 1")

# prepare substitution variable FINDEX_TYPE for sundials_fconfig.h
IF(SUNDIALS_INDEX_TYPE MATCHES "INT32_T")
  SET(FINDEX_TYPE "4")
ENDIF(SUNDIALS_INDEX_TYPE MATCHES "INT32_T")
IF(SUNDIALS_INDEX_TYPE MATCHES "INT64_T")
  SET(FINDEX_TYPE "8")
ENDIF(SUNDIALS_INDEX_TYPE MATCHES "INT64_T")

# ---------------------------------------------------------------
# Enable Fortran interface?
# ---------------------------------------------------------------

# Fortran interface is disabled by default
SET(DOCSTR "Enable Fortran-C support")
SHOW_VARIABLE(FCMIX_ENABLE BOOL "${DOCSTR}" OFF)

# Check that at least one solver with a Fortran interface is built
IF(NOT BUILD_ARKODE AND NOT BUILD_CVODE AND NOT BUILD_IDA AND NOT BUILD_KINSOL)
  IF(FCMIX_ENABLE)
    PRINT_WARNING("Enabled packages do not support Fortran" "Disabling FCMIX")
    FORCE_VARIABLE(FCMIX_ENABLE BOOL "${DOCSTR}" OFF)
  ENDIF()
  HIDE_VARIABLE(FCMIX_ENABLE)
ENDIF()

# ---------------------------------------------------------------
# Options to build static and/or shared libraries
# ---------------------------------------------------------------

OPTION(BUILD_STATIC_LIBS "Build static libraries" ON)
OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Prepare substitution variable SUNDIALS_EXPORT for sundials_config.h
# When building shared SUNDIALS libraries under Windows, use
#      #define SUNDIALS_EXPORT __declspec(dllexport)
# When linking to shared SUNDIALS libraries under Windows, use
#      #define SUNDIALS_EXPORT __declspec(dllimport)
# In all other cases (other platforms or static libraries
# under Windows), the SUNDIALS_EXPORT macro is empty

IF(BUILD_SHARED_LIBS AND WIN32)
  SET(SUNDIALS_EXPORT
    "#ifdef BUILD_SUNDIALS_LIBRARY
#define SUNDIALS_EXPORT __declspec(dllexport)
#else
#define SUNDIALS_EXPORT __declspec(dllimport)
#endif")
ELSE(BUILD_SHARED_LIBS AND WIN32)
  SET(SUNDIALS_EXPORT "#define SUNDIALS_EXPORT")
ENDIF(BUILD_SHARED_LIBS AND WIN32)

# Make sure we build at least one type of libraries
IF(NOT BUILD_STATIC_LIBS AND NOT BUILD_SHARED_LIBS)
  PRINT_WARNING("Both static and shared library generation were disabled"
                "Building static libraries was re-enabled")
  FORCE_VARIABLE(BUILD_STATIC_LIBS BOOL "Build static libraries" ON)
ENDIF(NOT BUILD_STATIC_LIBS AND NOT BUILD_SHARED_LIBS)

# ---------------------------------------------------------------
# Option to use the generic math libraries (UNIX only)
# ---------------------------------------------------------------

IF(UNIX)
  OPTION(USE_GENERIC_MATH "Use generic (std-c) math libraries" ON)
  IF(USE_GENERIC_MATH)
    # executables will be linked against -lm
    SET(EXTRA_LINK_LIBS -lm)
    # prepare substitution variable for sundials_config.h
    SET(SUNDIALS_USE_GENERIC_MATH TRUE)
  ENDIF(USE_GENERIC_MATH)
ENDIF(UNIX)

## clock-monotonic, see if we need to link with rt
include(CheckSymbolExists)
set(CMAKE_REQUIRED_LIBRARIES_SAVE ${CMAKE_REQUIRED_LIBRARIES})
set(CMAKE_REQUIRED_LIBRARIES rt)
CHECK_SYMBOL_EXISTS(_POSIX_TIMERS "unistd.h;time.h" SUNDIALS_POSIX_TIMERS)
set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES_SAVE})
if(SUNDIALS_POSIX_TIMERS)
  find_library(SUNDIALS_RT_LIBRARY NAMES rt)
  mark_as_advanced(SUNDIALS_RT_LIBRARY)
  if(SUNDIALS_RT_LIBRARY)
    # sundials_config.h symbol
    SET(SUNDIALS_HAVE_POSIX_TIMERS TRUE)
    set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} ${SUNDIALS_RT_LIBRARY})
  endif()
endif()


# ===============================================================
# Options for Parallelism
# ===============================================================

# ---------------------------------------------------------------
# Enable MPI support?
# ---------------------------------------------------------------
OPTION(MPI_ENABLE "Enable MPI support" OFF)

# ---------------------------------------------------------------
# Enable OpenMP support?
# ---------------------------------------------------------------
OPTION(OPENMP_ENABLE "Enable OpenMP support" OFF)

# ---------------------------------------------------------------
# Enable Pthread support?
# ---------------------------------------------------------------
OPTION(PTHREAD_ENABLE "Enable Pthreads support" OFF)

# -------------------------------------------------------------
# Enable CUDA support?
# -------------------------------------------------------------
OPTION(CUDA_ENABLE "Enable CUDA support" OFF)

# -------------------------------------------------------------
# Enable RAJA support?
# -------------------------------------------------------------
OPTION(RAJA_ENABLE "Enable RAJA support" OFF)


# ===============================================================
# Options for external packages
# ===============================================================

# ---------------------------------------------------------------
# Enable BLAS support?
# ---------------------------------------------------------------
OPTION(BLAS_ENABLE "Enable BLAS support" OFF)

# ---------------------------------------------------------------
# Enable LAPACK/BLAS support?
# ---------------------------------------------------------------
OPTION(LAPACK_ENABLE "Enable Lapack support" OFF)

# LAPACK does not support extended precision
IF(LAPACK_ENABLE AND SUNDIALS_PRECISION MATCHES "EXTENDED")
  PRINT_WARNING("LAPACK is not compatible with ${SUNDIALS_PRECISION} precision"
                "Disabling LAPACK")
  FORCE_VARIABLE(LAPACK_ENABLE BOOL "LAPACK is disabled" OFF)
ENDIF()

# LAPACK does not support 64-bit integer index types
IF(LAPACK_ENABLE AND SUNDIALS_INDEX_TYPE MATCHES "INT64_T")
  PRINT_WARNING("LAPACK is not compatible with ${SUNDIALS_INDEX_TYPE} integers"
                "Disabling LAPACK")
  SET(LAPACK_ENABLE OFF CACHE BOOL "LAPACK is disabled" FORCE)
ENDIF()

# ---------------------------------------------------------------
# Enable SuperLU_MT support?
# ---------------------------------------------------------------
OPTION(SUPERLUMT_ENABLE "Enable SUPERLUMT support" OFF)

# SuperLU_MT does not support extended precision
IF(SUPERLUMT_ENABLE AND SUNDIALS_PRECISION MATCHES "EXTENDED")
  PRINT_WARNING("SuperLU_MT is not compatible with ${SUNDIALS_PRECISION} precision"
                "Disabling SuperLU_MT")
  FORCE_VARIABLE(SUPERLUMT_ENABLE BOOL "SuperLU_MT is disabled" OFF)
ENDIF()

# ---------------------------------------------------------------
# Enable KLU support?
# ---------------------------------------------------------------
OPTION(KLU_ENABLE "Enable KLU support" OFF)

# KLU does not support single or extended precision
IF(KLU_ENABLE AND
  (SUNDIALS_PRECISION MATCHES "SINGLE" OR SUNDIALS_PRECISION MATCHES "EXTENDED"))
  PRINT_WARNING("KLU is not compatible with ${SUNDIALS_PRECISION} precision"
                "Disabling KLU")
  FORCE_VARIABLE(KLU_ENABLE BOOL "KLU is disabled" OFF)
ENDIF()

# ---------------------------------------------------------------
# Enable hypre Vector support?
# ---------------------------------------------------------------
OPTION(HYPRE_ENABLE "Enable hypre support" OFF)

# Using hypre requres building with MPI enabled
IF(HYPRE_ENABLE AND NOT MPI_ENABLE)
  PRINT_WARNING("MPI not enabled - Disabling hypre" 
                "Set MPI_ENABLE to ON to use parhyp")
  FORCE_VARIABLE(HYPRE_ENABLE BOOL "Enable hypre support" OFF)
ENDIF()

# ---------------------------------------------------------------
# Enable PETSc support?
# ---------------------------------------------------------------
OPTION(PETSC_ENABLE "Enable PETSc support" OFF)

# Using PETSc requires building with MPI enabled
IF(PETSC_ENABLE AND NOT MPI_ENABLE)
  PRINT_WARNING("MPI not enabled - Disabling PETSc" 
                "Set MPI_ENABLE to ON to use PETSc")
  FORCE_VARIABLE(PETSC_ENABLE BOOL "Enable PETSc support" OFF)
ENDIF()


# ===============================================================
# Options for examples
# ===============================================================

# ---------------------------------------------------------------
# Enable examples?
# ---------------------------------------------------------------

# Enable C examples (on by default)
OPTION(EXAMPLES_ENABLE_C "Build SUNDIALS C examples" ON)

# F77 examples (on by default) are an option only if the Fortran
# interface is enabled
SET(DOCSTR "Build SUNDIALS Fortran examples")
IF(FCMIX_ENABLE)
  OPTION(EXAMPLES_ENABLE_F77 "${DOCSTR}" ON)
  # Fortran examples do not support single or extended precision
  IF(SUNDIALS_PRECISION MATCHES "EXTENDED" OR SUNDIALS_PRECISION MATCHES "SINGLE")
    PRINT_WARNING("F77 examples are not compatible with ${SUNDIALS_PRECISION} precision"
                  "EXAMPLES_ENABLE_F77")
    FORCE_VARIABLE(EXAMPLES_ENABLE_F77 BOOL "Fortran examples are disabled" OFF)
  ENDIF()
ELSE()
  # set back to OFF (in case was ON)
  IF(EXAMPLES_ENABLE_F77)
    PRINT_WARNING("EXAMPLES_ENABLE_F77 is ON but FCMIX is OFF"
                  "Disabling EXAMPLES_ENABLE_F77")
    FORCE_VARIABLE(EXAMPLES_ENABLE_F77 BOOL "${DOCSTR}" OFF)
  ENDIF()
  HIDE_VARIABLE(EXAMPLES_ENABLE_F77)
ENDIF()

# C++ examples (off by default) are an option only if ARKode is enabled
SET(DOCSTR "Build ARKode C++ examples")
IF(BUILD_ARKODE)
  SHOW_VARIABLE(EXAMPLES_ENABLE_CXX BOOL "${DOCSTR}" OFF)
ELSE()
  # set back to OFF (in case was ON)
  IF(EXAMPLES_ENABLE_CXX)
    PRINT_WARNING("EXAMPLES_ENABLE_CXX is ON but BUILD_ARKODE is OFF"
                  "Disabling EXAMPLES_ENABLE_CXX")
    FORCE_VARIABLE(EXAMPLES_ENABLE_CXX BOOL "${DOCSTR}" OFF)
  ENDIF()
  HIDE_VARIABLE(EXAMPLES_ENABLE_CXX)
ENDIF()

# F90 examples (off by default) are an option only if ARKode is
# built and the Fortran interface is enabled
SET(DOCSTR "Build ARKode F90 examples")
IF(FCMIX_ENABLE AND BUILD_ARKODE)
  SHOW_VARIABLE(EXAMPLES_ENABLE_F90 BOOL "${DOCSTR}" OFF)
  # Fortran90 examples do not support single or extended precision 
  # NOTE: This check can be removed after Fortran configure file is integrated into examples
  IF(SUNDIALS_PRECISION MATCHES "EXTENDED" OR SUNDIALS_PRECISION MATCHES "SINGLE")
    PRINT_WARNING("F90 examples are not compatible with ${SUNDIALS_PRECISION} precision"
                  "EXAMPLES_ENABLE_F90")
    FORCE_VARIABLE(EXAMPLES_ENABLE_F90 BOOL "Fortran90 examples are disabled" OFF)
  ENDIF()
ELSE()
  # set back to OFF (in case was ON)
  IF(EXAMPLES_ENABLE_F90)
    PRINT_WARNING("EXAMPLES_ENABLE_F90 is ON but FCMIX or BUILD_ARKODE is OFF"
                  "Disabling EXAMPLES_ENABLE_F90")
    FORCE_VARIABLE(EXAMPLES_ENABLE_F90 BOOL "${DOCSTR}" OFF)
  ENDIF()
  HIDE_VARIABLE(EXAMPLES_ENABLE_F90)
ENDIF()

# CUDA examples (off by default)
SET(DOCSTR "Build SUNDIALS CUDA examples")
IF(CUDA_ENABLE)
  SHOW_VARIABLE(EXAMPLES_ENABLE_CUDA BOOL "${DOCSTR}" OFF)
ELSE()
  IF(EXAMPLES_ENABLE_CUDA)
    PRINT_WARNING("EXAMPLES_ENABLE_CUDA is ON but CUDA_ENABLE is OFF"
                  "Disabling EXAMPLES_ENABLE_CUDA")
    FORCE_VARIABLE(EXAMPLES_ENABLE_CUDA BOOL "${DOCSTR}" OFF)
  ENDIF()
ENDIF()

# RAJA examples (off by default)
SET(DOCSTR "Build SUNDIALS RAJA examples")
IF(RAJA_ENABLE)
  SHOW_VARIABLE(EXAMPLES_ENABLE_RAJA BOOL "${DOCSTR}" OFF)
ELSE()
  IF(EXAMPLES_ENABLE_RAJA)
    PRINT_WARNING("EXAMPLES_ENABLE_RAJA is ON but RAJA_ENABLE is OFF"
                  "Disabling EXAMPLES_ENABLE_RAJA")
    FORCE_VARIABLE(EXAMPLES_ENABLE_RAJA BOOL "${DOCSTR}" OFF)
  ENDIF()
ENDIF()

# If any of the above examples are enabled set EXAMPLES_ENABLED to TRUE
IF(EXAMPLES_ENABLE_C OR
    EXAMPLES_ENABLE_F77 OR
    EXAMPLES_ENABLE_CXX OR
    EXAMPLES_ENABLE_F90 OR
    EXAMPLES_ENABLE_CUDA OR
    EXAMPLES_ENABLE_RAJA)
  SET(EXAMPLES_ENABLED TRUE)
ELSE()
  SET(EXAMPLES_ENABLED FALSE)
ENDIF()

# ---------------------------------------------------------------
# Install examples?
# ---------------------------------------------------------------

IF(EXAMPLES_ENABLED)

  # If examples are enabled, set different options

  # The examples will be linked with the library corresponding to the build type.
  # Whenever building shared libraries, use them to link the examples.
  IF(BUILD_SHARED_LIBS)
    SET(LINK_LIBRARY_TYPE "shared")
  ELSE(BUILD_SHARED_LIBS)
    SET(LINK_LIBRARY_TYPE "static")
  ENDIF(BUILD_SHARED_LIBS)

  # Enable installing examples by default
  SHOW_VARIABLE(EXAMPLES_INSTALL BOOL "Install example files" ON)

  # If examples are to be exported, check where we should install them.
  IF(EXAMPLES_INSTALL)

    SHOW_VARIABLE(EXAMPLES_INSTALL_PATH PATH
      "Output directory for installing example files" "${CMAKE_INSTALL_PREFIX}/examples")

    IF(NOT EXAMPLES_INSTALL_PATH)
      PRINT_WARNING("The example installation path is empty"
                    "Example installation path was reset to its default value")
      SET(EXAMPLES_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/examples" CACHE STRING
        "Output directory for installing example files" FORCE)
    ENDIF(NOT EXAMPLES_INSTALL_PATH)
    
    # create test_install target and directory for running smoke tests after
    # installation
    ADD_CUSTOM_TARGET(test_install)

    SET(TEST_INSTALL_DIR ${PROJECT_BINARY_DIR}/Testing_Install)

    IF(NOT EXISTS ${TEST_INSTALL_DIR})
      FILE(MAKE_DIRECTORY ${TEST_INSTALL_DIR})
    ENDIF()


  ELSE(EXAMPLES_INSTALL)

    HIDE_VARIABLE(EXAMPLES_INSTALL_PATH)

  ENDIF(EXAMPLES_INSTALL)

ELSE(EXAMPLES_ENABLED)

  # If examples are disabled, hide all options related to
  # building and installing the SUNDIALS examples

  HIDE_VARIABLE(EXAMPLES_INSTALL)
  HIDE_VARIABLE(EXAMPLES_INSTALL_PATH)
  
ENDIF(EXAMPLES_ENABLED)

# ---------------------------------------------------------------
# Include development examples in regression tests?
# ---------------------------------------------------------------
OPTION(SUNDIALS_DEVTESTS "Include development tests in make test" OFF)
MARK_AS_ADVANCED(FORCE SUNDIALS_DEVTESTS)

# ===============================================================
# Add any other necessary compiler flags & definitions
# ===============================================================

# Under Windows, add compiler directive to inhibit warnings
# about use of unsecure functions

IF(WIN32)
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
ENDIF(WIN32)

IF(APPLE)
  SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -undefined dynamic_lookup")
ENDIF(APPLE)

# ---------------------------------------------------------------
# A Fortran compiler is needed if:
# (a) FCMIX is enabled
# (b) BLAS is enabled (for the name-mangling scheme)
# (c) LAPACK is enabled (for the name-mangling scheme)
# ---------------------------------------------------------------

IF(FCMIX_ENABLE OR BLAS_ENABLE OR LAPACK_ENABLE)
  INCLUDE(SundialsFortran)
  IF(NOT F77_FOUND AND FCMIX_ENABLE)
    PRINT_WARNING("Fortran compiler not functional"
                  "FCMIX support will not be provided")
  ENDIF()
ENDIF()

# ---------------------------------------------------------------
# A Fortran90 compiler is needed if:
# (a) F90 ARKODE examples are enabled
# ---------------------------------------------------------------

IF(EXAMPLES_ENABLE_F90)
  INCLUDE(SundialsFortran90)
  IF(NOT F90_FOUND)
    PRINT_WARNING("Fortran90 compiler not functional"
                  "F90 support will not be provided")
  ENDIF()
ENDIF()

# ---------------------------------------------------------------
# A C++ compiler is needed if:
# (a) C++ ARKODE examples are enabled
# (b) CUDA is enabled
# (c) RAJA is enabled
# ---------------------------------------------------------------

IF(EXAMPLES_ENABLE_CXX OR CUDA_ENABLE OR RAJA_ENABLE)
  INCLUDE(SundialsCXX)
  IF(NOT CXX_FOUND)
    PRINT_WARNING("C++ compiler not functional"
                  "C++ support will not be provided")
  ENDIF()
ENDIF()

# ---------------------------------------------------------------
# Check if we need an alternate way of specifying the Fortran
# name-mangling scheme if we were unable to infer it using a
# compiler.
# Ask the user to specify the case and number of appended underscores
# corresponding to the Fortran name-mangling scheme of symbol names
# that do not themselves contain underscores (recall that this is all
# we really need for the interfaces to LAPACK).
# Note: the default scheme is lower case - one underscore
# ---------------------------------------------------------------

IF(BLAS_ENABLE OR LAPACK_ENABLE AND NOT F77SCHEME_FOUND)
  # Specify the case for the Fortran name-mangling scheme
  SHOW_VARIABLE(SUNDIALS_F77_FUNC_CASE STRING
    "case of Fortran function names (lower/upper)"
    "lower")
  # Specify the number of appended underscores for the Fortran name-mangling scheme
  SHOW_VARIABLE(SUNDIALS_F77_FUNC_UNDERSCORES STRING
    "number of underscores appended to Fortran function names"
    "one")
  # Based on the given case and number of underscores,
  # set the C preprocessor macro definition
  IF(${SUNDIALS_F77_FUNC_CASE} MATCHES "lower")
    IF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "none")
      SET(CMAKE_Fortran_SCHEME_NO_UNDERSCORES "mysub")
    ENDIF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "none")
    IF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "one")
      SET(CMAKE_Fortran_SCHEME_NO_UNDERSCORES "mysub_")
    ENDIF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "one")
    IF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "two")
      SET(CMAKE_Fortran_SCHEME_NO_UNDERSCORES "mysub__")
    ENDIF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "two")
  ELSE(${SUNDIALS_F77_FUNC_CASE} MATCHES "lower")
    IF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "none")
      SET(CMAKE_Fortran_SCHEME_NO_UNDERSCORES "MYSUB")
    ENDIF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "none")
    IF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "one")
      SET(CMAKE_Fortran_SCHEME_NO_UNDERSCORES "MYSUB_")
    ENDIF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "one")
    IF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "two")
      SET(CMAKE_Fortran_SCHEME_NO_UNDERSCORES "MYSUB__")
    ENDIF(${SUNDIALS_F77_FUNC_UNDERSCORES} MATCHES "two")
  ENDIF(${SUNDIALS_F77_FUNC_CASE} MATCHES "lower")
  # Since the SUNDIALS codes never use symbol names containing
  # underscores, set a default scheme (probably wrong) for symbols
  # with underscores.
  SET(CMAKE_Fortran_SCHEME_WITH_UNDERSCORES "my_sub_")
  # We now "have" a scheme.
  SET(F77SCHEME_FOUND TRUE)
ENDIF(BLAS_ENABLE OR LAPACK_ENABLE AND NOT F77SCHEME_FOUND)

# ---------------------------------------------------------------
# If we have a name-mangling scheme (either automatically
# inferred or provided by the user), set the SUNDIALS
# compiler preprocessor macro definitions.
# ---------------------------------------------------------------

SET(F77_MANGLE_MACRO1 "")
SET(F77_MANGLE_MACRO2 "")

IF(F77SCHEME_FOUND)
  # Symbols WITHOUT underscores
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "mysub")
    SET(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) name")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "mysub")
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "mysub_")
    SET(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) name ## _")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "mysub_")
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "mysub__")
    SET(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) name ## __")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "mysub__")
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MYSUB")
    SET(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) NAME")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MYSUB")
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MYSUB_")
    SET(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) NAME ## _")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MYSUB_")
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MYSUB__")
    SET(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) NAME ## __")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MYSUB__")
  # Symbols with underscores
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "my_sub")
    SET(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) name")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "my_sub")
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "my_sub_")
    SET(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) name ## _")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "my_sub_")
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "my_sub__")
    SET(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) name ## __")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "my_sub__")
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MY_SUB")
    SET(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) NAME")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MY_SUB")
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MY_SUB_")
    SET(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) NAME ## _")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MY_SUB_")
  IF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MY_SUB__")
    SET(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) NAME ## __")
  ENDIF(${CMAKE_Fortran_SCHEME_NO_UNDERSCORES} MATCHES "MY_SUB__")
ENDIF(F77SCHEME_FOUND)

# ---------------------------------------------------------------
# Decide how to compile MPI codes.
# ---------------------------------------------------------------

IF(MPI_ENABLE)
  # show command to run MPI codes (defaults to mpirun)
  SHOW_VARIABLE(MPI_RUN_COMMAND STRING "MPI run command" "mpirun")

  INCLUDE(SundialsMPIC)
  IF(MPIC_FOUND)
    IF(CXX_FOUND AND EXAMPLES_ENABLE_CXX)
      INCLUDE(SundialsMPICXX)
    ENDIF()
    IF(F77_FOUND AND EXAMPLES_ENABLE_F77)
      INCLUDE(SundialsMPIF)
    ENDIF()
    IF(F90_FOUND AND EXAMPLES_ENABLE_F90)
      INCLUDE(SundialsMPIF90)
    ENDIF()
  ELSE()
    PRINT_WARNING("MPI not functional"
                  "Parallel support will not be provided")
  ENDIF()

  IF(MPIC_MPI2)
    SET(F77_MPI_COMM_F2C "#define SUNDIALS_MPI_COMM_F2C 1")
  ELSE()
    SET(F77_MPI_COMM_F2C "#define SUNDIALS_MPI_COMM_F2C 0")
  ENDIF()

ELSE()

  HIDE_VARIABLE(MPI_INCLUDE_PATH)
  HIDE_VARIABLE(MPI_LIBRARIES)
  HIDE_VARIABLE(MPI_EXTRA_LIBRARIES)
  HIDE_VARIABLE(MPI_MPICC)
  HIDE_VARIABLE(MPI_MPICXX)
  HIDE_VARIABLE(MPI_MPIF77)
  HIDE_VARIABLE(MPI_MPIF90)

ENDIF(MPI_ENABLE)

# ---------------------------------------------------------------
# If using MPI with C++, disable C++ extensions (for known wrappers)
# ---------------------------------------------------------------

# IF(MPICXX_FOUND)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX -DLAM_BUILDING")
# ENDIF(MPICXX_FOUND)

# -------------------------------------------------------------
# Find OpenMP
# -------------------------------------------------------------

IF(OPENMP_ENABLE)
  FIND_PACKAGE(OpenMP)
  IF(NOT OPENMP_FOUND)
    message(STATUS "Disabling OpenMP support, could not determine compiler flags")
  ENDIF(NOT OPENMP_FOUND)
ENDIF(OPENMP_ENABLE)

# -------------------------------------------------------------
# Find PThreads
# -------------------------------------------------------------

IF(PTHREAD_ENABLE)
  FIND_PACKAGE(Threads)
  IF(CMAKE_USE_PTHREADS_INIT)
    message(STATUS "Using Pthreads")
    SET(PTHREADS_FOUND TRUE)
    # SGS    
  ELSE()
    message(STATUS "Disabling Pthreads support, could not determine compiler flags")
  endif()
ENDIF(PTHREAD_ENABLE)

# -------------------------------------------------------------
# Find CUDA
# -------------------------------------------------------------

# disable CUDA if a working C++ compiler is not found
IF(CUDA_ENABLE AND (NOT CXX_FOUND))
  PRINT_WARNING("C++ compiler required for CUDA support" "Disabling CUDA")
  FORCE_VARIABLE(CUDA_ENABLE BOOL "CUDA disabled" OFF)
ENDIF()

if(CUDA_ENABLE)
  find_package(CUDA)
  
  if (CUDA_FOUND)
    #message("CUDA found!")
    set(CUDA_NVCC_FLAGS "-lineinfo")
  else()
    message(STATUS "Disabling CUDA support, could not find CUDA.")
  endif()
endif(CUDA_ENABLE)

# -------------------------------------------------------------
# Find RAJA
# -------------------------------------------------------------

# disable RAJA if CUDA is not enabled/working
IF(RAJA_ENABLE AND (NOT CUDA_FOUND))
  PRINT_WARNING("CUDA is required for RAJA support" "Please enable CUDA and RAJA")
  FORCE_VARIABLE(RAJA_ENABLE BOOL "RAJA disabled" OFF)
ENDIF()

# Check if C++11 compiler is available
IF(RAJA_ENABLE)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

  IF(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_STANDARD 11)
  ELSE()
    PRINT_WARNING("C++11 compliant compiler required for RAJA support" "Disabling RAJA")
    FORCE_VARIABLE(RAJA_ENABLE BOOL "RAJA disabled" OFF)
  ENDIF()
ENDIF()

if(RAJA_ENABLE)
  # Look for CMake configuration file in RAJA installation
  find_package(RAJA CONFIGS)
  if (RAJA_FOUND)
    #message("RAJA found!")
    include_directories(${RAJA_INCLUDE_DIR})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${RAJA_NVCC_FLAGS})
  else()
    PRINT_WARNING("RAJA configuration not found" "Please set RAJA_DIR to provide path to RAJA CMake configuration file.")
  endif()
endif(RAJA_ENABLE)

# ===============================================================
# Find (and test) external packages
# ===============================================================

# ---------------------------------------------------------------
# Find (and test) the BLAS libraries
# ---------------------------------------------------------------

# If BLAS is needed, first try to find the appropriate
# libraries and linker flags needed to link against them.

IF(BLAS_ENABLE)

  # find BLAS
  INCLUDE(SundialsBlas)

  # show after include so FindBlas can locate BLAS_LIBRARIES if necessary
  SHOW_VARIABLE(BLAS_LIBRARIES STRING "Blas libraries" "${BLAS_LIBRARIES}")

  IF(BLAS_LIBRARIES AND NOT BLAS_FOUND)
    PRINT_WARNING("BLAS not functional"
                  "BLAS support will not be provided")
  ELSE()
    #set sundials_config.h symbol via sundials_config.in
    SET(SUNDIALS_BLAS TRUE)
  ENDIF()

ELSE()

  IF(NOT LAPACK_ENABLE)
    HIDE_VARIABLE(SUNDIALS_F77_FUNC_CASE)
    HIDE_VARIABLE(SUNDIALS_F77_FUNC_UNDERSCORES)
  ENDIF()
  HIDE_VARIABLE(BLAS_LIBRARIES)

ENDIF()

# ---------------------------------------------------------------
# Find (and test) the Lapack libraries
# ---------------------------------------------------------------

# If LAPACK is needed, first try to find the appropriate
# libraries and linker flags needed to link against them.

IF(LAPACK_ENABLE)

  # find LAPACK and BLAS Libraries
  INCLUDE(SundialsLapack)

  # show after include so FindLapack can locate LAPCK_LIBRARIES if necessary
  SHOW_VARIABLE(LAPACK_LIBRARIES STRING "Lapack and Blas libraries" "${LAPACK_LIBRARIES}")

  IF(LAPACK_LIBRARIES AND NOT LAPACK_FOUND)
    PRINT_WARNING("LAPACK not functional"
                  "Blas/Lapack support will not be provided")
  ELSE()
    #set sundials_config.h symbol via sundials_config.in
    SET(SUNDIALS_BLAS_LAPACK TRUE)
  ENDIF()

ELSE()

  IF(NOT BLAS_ENABLE)
    HIDE_VARIABLE(SUNDIALS_F77_FUNC_CASE)
    HIDE_VARIABLE(SUNDIALS_F77_FUNC_UNDERSCORES)
  ENDIF()
  HIDE_VARIABLE(LAPACK_LIBRARIES)

ENDIF()

# ---------------------------------------------------------------
# Find (and test) the SUPERLUMT libraries
# ---------------------------------------------------------------

# >>>>>>> NOTE: Need to add check for SuperLU_MT integer type

# If SUPERLUMT is needed, first try to find the appropriate
# libraries to link against them.

IF(SUPERLUMT_ENABLE)

  # Show SuperLU_MT options and set default thread type (Pthreads)
  SHOW_VARIABLE(SUPERLUMT_THREAD_TYPE STRING "SUPERLUMT threading type: OpenMP or Pthread" "Pthread")
  SHOW_VARIABLE(SUPERLUMT_INCLUDE_DIR PATH "SUPERLUMT include directory" "${SUPERLUMT_INCLUDE_DIR}")
  SHOW_VARIABLE(SUPERLUMT_LIBRARY_DIR PATH "SUPERLUMT library directory" "${SUPERLUMT_LIBRARY_DIR}")

  INCLUDE(SundialsSuperLUMT)

  IF(SUPERLUMT_FOUND)
    # sundials_config.h symbols
    SET(SUNDIALS_SUPERLUMT TRUE)
    SET(SUNDIALS_SUPERLUMT_THREAD_TYPE ${SUPERLUMT_THREAD_TYPE})
    INCLUDE_DIRECTORIES(${SUPERLUMT_INCLUDE_DIR})
  ENDIF()

  IF(SUPERLUMT_LIBRARIES AND NOT SUPERLUMT_FOUND)
    PRINT_WARNING("SUPERLUMT not functional - support will not be provided"
                  "Double check spelling specified libraries (search is case sensitive)")
  ENDIF(SUPERLUMT_LIBRARIES AND NOT SUPERLUMT_FOUND)

ELSE()

  HIDE_VARIABLE(SUPERLUMT_THREAD_TYPE)
  HIDE_VARIABLE(SUPERLUMT_LIBRARY_DIR)
  HIDE_VARIABLE(SUPERLUMT_INCLUDE_DIR)
  SET (SUPERLUMT_DISABLED TRUE CACHE INTERNAL "GUI - return when first set")

ENDIF()

# ---------------------------------------------------------------
# Find (and test) the KLU libraries
# ---------------------------------------------------------------

# If KLU is requested, first try to find the appropriate libraries to
# link against them.

IF(KLU_ENABLE)

  SHOW_VARIABLE(KLU_INCLUDE_DIR PATH "KLU include directory"
    "${KLU_INCLUDE_DIR}")
  SHOW_VARIABLE(KLU_LIBRARY_DIR PATH
    "Klu library directory" "${KLU_LIBRARY_DIR}")

  set(KLU_FOUND TRUE)
  get_filename_component(PYBAMM_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PYBAMM_DIR}) # use FindSuiteSparse.cmake that is in PyBaMM root
  set(SuiteSparse_ROOT ${PYBAMM_DIR}/SuiteSparse-5.6.0)
  find_package(SuiteSparse OPTIONAL_COMPONENTS KLU AMD COLAMD BTF)
  include_directories(${SuiteSparse_INCLUDE_DIRS})
  set(KLU_LIBRARIES ${SuiteSparse_LIBRARIES})

  IF(KLU_LIBRARIES AND NOT KLU_FOUND)
    PRINT_WARNING("KLU not functional - support will not be provided"
                  "Double check spelling of include path and specified libraries (search is case sensitive)")
  ENDIF(KLU_LIBRARIES AND NOT KLU_FOUND)

ELSE()

  HIDE_VARIABLE(KLU_LIBRARY_DIR)
  HIDE_VARIABLE(KLU_INCLUDE_DIR)
  SET (KLU_DISABLED TRUE CACHE INTERNAL "GUI - return when first set")

ENDIF(KLU_ENABLE)

# ---------------------------------------------------------------
# Find (and test) the hypre libraries
# ---------------------------------------------------------------

# >>>>>>> NOTE: Need to add check for hypre precision and integer type

IF(HYPRE_ENABLE)
  SHOW_VARIABLE(HYPRE_INCLUDE_DIR PATH "HYPRE include directory"
    "${HYPRE_INCLUDE_DIR}")
  SHOW_VARIABLE(HYPRE_LIBRARY_DIR PATH
    "HYPRE library directory" "${HYPRE_LIBRARY_DIR}")

  INCLUDE(SundialsHypre)

  IF(HYPRE_FOUND)
    # sundials_config.h symbol
    SET(SUNDIALS_HYPRE TRUE)
    INCLUDE_DIRECTORIES(${HYPRE_INCLUDE_DIR})
  ENDIF(HYPRE_FOUND)

  IF(HYPRE_LIBRARIES AND NOT HYPRE_FOUND)
    PRINT_WARNING("HYPRE not functional - support will not be provided"
                  "Found hypre library, test code does not work")
  ENDIF(HYPRE_LIBRARIES AND NOT HYPRE_FOUND)
  
ELSE()

  HIDE_VARIABLE(HYPRE_INCLUDE_DIR)
  HIDE_VARIABLE(HYPRE_LIBRARY_DIR)
  SET (HYPRE_DISABLED TRUE CACHE INTERNAL "GUI - return when first set")

ENDIF()

# ---------------------------------------------------------------
# Find (and test) the PETSc libraries
# ---------------------------------------------------------------

# >>>>>>> NOTE: Need to add check for PETSc precision and integer type

IF(PETSC_ENABLE)
  SHOW_VARIABLE(PETSC_INCLUDE_DIR PATH "PETSc include directory"
    "${PETSC_INCLUDE_DIR}")
  SHOW_VARIABLE(PETSC_LIBRARY_DIR PATH
    "PETSc library directory" "${PETSC_LIBRARY_DIR}")

  INCLUDE(SundialsPETSc)

  IF(PETSC_FOUND)
    # sundials_config.h symbol
    SET(SUNDIALS_PETSC TRUE)
    INCLUDE_DIRECTORIES(${PETSC_INCLUDE_DIR})
  ENDIF(PETSC_FOUND)

  IF(PETSC_LIBRARIES AND NOT PETSC_FOUND)
    PRINT_WARNING("PETSC not functional - support will not be provided"
                  "Double check spelling specified libraries (search is case sensitive)")
  ENDIF(PETSC_LIBRARIES AND NOT PETSC_FOUND)

ELSE()

  HIDE_VARIABLE(PETSC_LIBRARY_DIR)
  HIDE_VARIABLE(PETSC_INCLUDE_DIR)
  SET (PETSC_DISABLED TRUE CACHE INTERNAL "GUI - return when first set")

ENDIF()


# ===============================================================
# Add source and configuration files 
# ===============================================================

# ---------------------------------------------------------------
# Configure the header file sundials_config.h
# ---------------------------------------------------------------

# All required substitution variables should be available at this point.
# Generate the header file and place it in the binary dir.
CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/include/sundials/sundials_config.in
  ${PROJECT_BINARY_DIR}/include/sundials/sundials_config.h
  )
CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/include/sundials/sundials_fconfig.in
  ${PROJECT_BINARY_DIR}/include/sundials/sundials_fconfig.h
  )

# Add the include directory in the source tree and the one in
# the binary tree (for the header file sundials_config.h)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)

# ---------------------------------------------------------------
# Add selected modules to the build system
# ---------------------------------------------------------------

# Shared components

ADD_SUBDIRECTORY(src/sundials)
ADD_SUBDIRECTORY(src/nvec_ser)
ADD_SUBDIRECTORY(src/sunmat_dense)
ADD_SUBDIRECTORY(src/sunmat_band)
ADD_SUBDIRECTORY(src/sunmat_sparse)
ADD_SUBDIRECTORY(src/sunlinsol_band)
ADD_SUBDIRECTORY(src/sunlinsol_dense)
IF(KLU_FOUND)
  ADD_SUBDIRECTORY(src/sunlinsol_klu)
ENDIF(KLU_FOUND)
IF(SUPERLUMT_FOUND)
  ADD_SUBDIRECTORY(src/sunlinsol_superlumt)
ENDIF(SUPERLUMT_FOUND)
IF(LAPACK_FOUND)
  ADD_SUBDIRECTORY(src/sunlinsol_lapackband)
  ADD_SUBDIRECTORY(src/sunlinsol_lapackdense)
ENDIF(LAPACK_FOUND)
ADD_SUBDIRECTORY(src/sunlinsol_spgmr)
ADD_SUBDIRECTORY(src/sunlinsol_spfgmr)
ADD_SUBDIRECTORY(src/sunlinsol_spbcgs)
ADD_SUBDIRECTORY(src/sunlinsol_sptfqmr)
ADD_SUBDIRECTORY(src/sunlinsol_pcg)
IF(MPIC_FOUND)
  ADD_SUBDIRECTORY(src/nvec_par)
ENDIF(MPIC_FOUND)

IF(HYPRE_FOUND)
  ADD_SUBDIRECTORY(src/nvec_parhyp)
ENDIF(HYPRE_FOUND)

IF(OPENMP_FOUND)
  ADD_SUBDIRECTORY(src/nvec_openmp)
ENDIF(OPENMP_FOUND)

IF(PTHREADS_FOUND)
  ADD_SUBDIRECTORY(src/nvec_pthreads)
ENDIF(PTHREADS_FOUND)

IF(PETSC_FOUND)
  ADD_SUBDIRECTORY(src/nvec_petsc)
ENDIF(PETSC_FOUND)

IF(CUDA_FOUND)
  ADD_SUBDIRECTORY(src/nvec_cuda)   
ENDIF(CUDA_FOUND)

IF(RAJA_FOUND)
  ADD_SUBDIRECTORY(src/nvec_raja)   
ENDIF(RAJA_FOUND)

# ARKODE library

IF(BUILD_ARKODE)
  ADD_SUBDIRECTORY(src/arkode)
  IF(FCMIX_ENABLE AND F77_FOUND)
    ADD_SUBDIRECTORY(src/arkode/fcmix)
  ENDIF(FCMIX_ENABLE AND F77_FOUND)
ENDIF(BUILD_ARKODE)

# CVODE library

IF(BUILD_CVODE)
  ADD_SUBDIRECTORY(src/cvode)
  IF(FCMIX_ENABLE AND F77_FOUND)
    ADD_SUBDIRECTORY(src/cvode/fcmix)
  ENDIF(FCMIX_ENABLE AND F77_FOUND)
ENDIF(BUILD_CVODE)

# CVODES library

IF(BUILD_CVODES)
  ADD_SUBDIRECTORY(src/cvodes)
ENDIF(BUILD_CVODES)

# IDA library

IF(BUILD_IDA)
  ADD_SUBDIRECTORY(src/ida)
  IF(FCMIX_ENABLE AND F77_FOUND)
    ADD_SUBDIRECTORY(src/ida/fcmix)
  ENDIF(FCMIX_ENABLE AND F77_FOUND)
ENDIF(BUILD_IDA)

# IDAS library

IF(BUILD_IDAS)
  ADD_SUBDIRECTORY(src/idas)
ENDIF(BUILD_IDAS)

# KINSOL library

IF(BUILD_KINSOL)
  ADD_SUBDIRECTORY(src/kinsol)
  IF(FCMIX_ENABLE AND F77_FOUND)
    ADD_SUBDIRECTORY(src/kinsol/fcmix)
  ENDIF(FCMIX_ENABLE AND F77_FOUND)
ENDIF(BUILD_KINSOL)

# CPODES library

IF(BUILD_CPODES)
  ADD_SUBDIRECTORY(src/cpodes)
ENDIF(BUILD_CPODES)

# ---------------------------------------------------------------
# Include the subdirectories corresponding to various examples
# ---------------------------------------------------------------

# If building and installing the examples is enabled, include
# the subdirectories for those examples that will be built.
# Also, if we will generate exported example Makefiles, set
# variables needed in generating them from templates.

# For now, TestRunner is not being distributed.
# So:
#  - Don't show TESTRUNNER variable
#  - Don't enable testing if TestRunner if not found.
#  - There will be no 'make test' target

INCLUDE(SundialsAddTest)
HIDE_VARIABLE(TESTRUNNER)

IF(EXAMPLES_ENABLED)

  # enable regression testing with 'make test'
  IF(TESTRUNNER)
    ENABLE_TESTING()
  ENDIF()

  # set variables used in generating CMake and Makefiles for examples
  IF(EXAMPLES_INSTALL)

    SET(SHELL "sh")
    SET(prefix "${CMAKE_INSTALL_PREFIX}")
    SET(exec_prefix "${CMAKE_INSTALL_PREFIX}")
    SET(includedir "${prefix}/include")
    SET(libdir "${exec_prefix}/lib")
    SET(CPP "${CMAKE_C_COMPILER}")
    SET(CPPFLAGS "${CMAKE_C_FLAGS_RELEASE}")
    SET(CC "${CMAKE_C_COMPILER}")
    SET(CFLAGS "${CMAKE_C_FLAGS_RELEASE}")
    SET(LDFLAGS "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
    LIST2STRING(EXTRA_LINK_LIBS LIBS)
    
    IF(CXX_FOUND)
      SET(CXX "${CMAKE_CXX_COMPILER}")
      SET(CXX_LNKR "${CMAKE_CXX_COMPILER}")
      SET(CXXFLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
      SET(CXX_LDFLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
      LIST2STRING(EXTRA_LINK_LIBS CXX_LIBS)
    ENDIF(CXX_FOUND)

    IF(F77_FOUND)
      SET(F77 "${CMAKE_Fortran_COMPILER}")
      SET(F77_LNKR "${CMAKE_Fortran_COMPILER}")
      SET(FFLAGS "${CMAKE_Fortran_FLAGS_RELEASE}")
      SET(F77_LDFLAGS "${CMAKE_Fortran_FLAGS_RELEASE}")
      LIST2STRING(EXTRA_LINK_LIBS F77_LIBS)
    ENDIF(F77_FOUND)

    IF(F90_FOUND)
      SET(F90 "${CMAKE_Fortran_COMPILER}")
      SET(F90_LNKR "${CMAKE_Fortran_COMPILER}")
      SET(F90FLAGS "${CMAKE_Fortran_FLAGS_RELEASE}")
      SET(F90_LDFLAGS "${CMAKE_Fortran_FLAGS_RELEASE}")
      LIST2STRING(EXTRA_LINK_LIBS F90_LIBS)
    ENDIF(F90_FOUND)

    IF(SUPERLUMT_FOUND)
      LIST2STRING(SUPERLUMT_LIBRARIES SUPERLUMT_LIBS)
      SET(SUPERLUMT_LIBS "${SUPERLUMT_LINKER_FLAGS} ${SUPERLUMT_LIBS}")
    ENDIF(SUPERLUMT_FOUND)

    IF(KLU_FOUND)
      LIST2STRING(KLU_LIBRARIES KLU_LIBS)
      SET(KLU_LIBS "${KLU_LINKER_FLAGS} ${KLU_LIBS}")
    ENDIF(KLU_FOUND)

    IF(BLAS_FOUND)
      LIST2STRING(BLAS_LIBRARIES BLAS_LIBS)
    ENDIF(BLAS_FOUND)

    IF(LAPACK_FOUND)
      LIST2STRING(LAPACK_LIBRARIES LAPACK_LIBS)
    ENDIF(LAPACK_FOUND)

    IF(MPIC_FOUND)
      IF(MPI_MPICC)
        SET(MPICC "${MPI_MPICC}")
        SET(MPI_INC_DIR ".")
        SET(MPI_LIB_DIR ".")
        SET(MPI_LIBS "")
        SET(MPI_FLAGS "")
      ELSE(MPI_MPICC)
        SET(MPICC "${CMAKE_C_COMPILER}")
        SET(MPI_INC_DIR "${MPI_INCLUDE_PATH}")
        SET(MPI_LIB_DIR ".")
        LIST2STRING(MPI_LIBRARIES MPI_LIBS)
      ENDIF(MPI_MPICC)
      SET(HYPRE_INC_DIR "${HYPRE_INCLUDE_DIR}")
      SET(HYPRE_LIB_DIR "${HYPRE_LIBRARY_DIR}")
      SET(HYPRE_LIBS "${HYPRE_LIBRARIES}")
    ENDIF(MPIC_FOUND)

    IF(MPICXX_FOUND)
      IF(MPI_MPICXX)
        SET(MPICXX "${MPI_MPICXX}")
      ELSE(MPI_MPICXX)
        SET(MPICXX "${CMAKE_CXX_COMPILER}")
        LIST2STRING(MPI_LIBRARIES MPI_LIBS)
      ENDIF(MPI_MPICXX)
    ENDIF(MPICXX_FOUND)

    IF(MPIF_FOUND)
      IF(MPI_MPIF77)
        SET(MPIF77 "${MPI_MPIF77}")
        SET(MPIF77_LNKR "${MPI_MPIF77}")
      ELSE(MPI_MPIF77)
        SET(MPIF77 "${CMAKE_Fortran_COMPILER}")
        SET(MPIF77_LNKR "${CMAKE_Fortran_COMPILER}")
        SET(MPI_INC_DIR "${MPI_INCLUDE_PATH}")
        SET(MPI_LIB_DIR ".")
        LIST2STRING(MPI_LIBRARIES MPI_LIBS)
      ENDIF(MPI_MPIF77)
    ENDIF(MPIF_FOUND)

    IF(MPIF90_FOUND)
      IF(MPI_MPIF90)
        SET(MPIF90 "${MPI_MPIF90}")
        SET(MPIF90_LNKR "${MPI_MPIF90}")
      ELSE(MPI_MPIF90)
        SET(MPIF90 "${CMAKE_Fortran_COMPILER}")
        SET(MPIF90_LNKR "${CMAKE_Fortran_COMPILER}")
        LIST2STRING(MPI_LIBRARIES MPI_LIBS)
      ENDIF(MPI_MPIF90)
    ENDIF(MPIF90_FOUND)

  ENDIF(EXAMPLES_INSTALL)

  # add ARKode examples
  IF(BUILD_ARKODE)
    # C examples
    IF(EXAMPLES_ENABLE_C)
      ADD_SUBDIRECTORY(examples/arkode/C_serial)
      IF(OPENMP_FOUND)
        ADD_SUBDIRECTORY(examples/arkode/C_openmp)
      ENDIF()
      IF(MPIC_FOUND)
        ADD_SUBDIRECTORY(examples/arkode/C_parallel)
      ENDIF()
      IF(HYPRE_ENABLE AND HYPRE_FOUND)
        ADD_SUBDIRECTORY(examples/arkode/C_parhyp)
      ENDIF()
    ENDIF()
    # C++ examples
    IF(EXAMPLES_ENABLE_CXX)
      IF(CXX_FOUND)
        ADD_SUBDIRECTORY(examples/arkode/CXX_serial)
      ENDIF()
      IF(MPICXX_FOUND)
        ADD_SUBDIRECTORY(examples/arkode/CXX_parallel)
      ENDIF()
    ENDIF()
    # F77 examples
    IF(EXAMPLES_ENABLE_F77)
      IF(F77_FOUND)
        ADD_SUBDIRECTORY(examples/arkode/F77_serial)
      ENDIF()
      IF(MPIF_FOUND)
        ADD_SUBDIRECTORY(examples/arkode/F77_parallel)
      ENDIF()
    ENDIF()
    # F90 examples
    IF(EXAMPLES_ENABLE_F90)
      IF(F90_FOUND)
        ADD_SUBDIRECTORY(examples/arkode/F90_serial)
      ENDIF()
      IF(MPIF90_FOUND)
        ADD_SUBDIRECTORY(examples/arkode/F90_parallel)
      ENDIF()
    ENDIF()   
  ENDIF(BUILD_ARKODE)

  # add CVODE examples
  IF(BUILD_CVODE)
    # C examples
    IF(EXAMPLES_ENABLE_C)
      ADD_SUBDIRECTORY(examples/cvode/serial)
      IF(OPENMP_FOUND)
        ADD_SUBDIRECTORY(examples/cvode/C_openmp)
      ENDIF()
      IF(MPIC_FOUND)
        ADD_SUBDIRECTORY(examples/cvode/parallel)
      ENDIF()
      IF(HYPRE_ENABLE AND HYPRE_FOUND)
        ADD_SUBDIRECTORY(examples/cvode/parhyp)
      ENDIF()
    ENDIF()
    # Fortran examples
    IF(EXAMPLES_ENABLE_F77)
      IF(F77_FOUND)
        ADD_SUBDIRECTORY(examples/cvode/fcmix_serial)
      ENDIF()
      IF(MPIF_FOUND)
        ADD_SUBDIRECTORY(examples/cvode/fcmix_parallel)
      ENDIF()
    ENDIF()
    # cuda examples
    IF(EXAMPLES_ENABLE_CUDA)
      IF(CUDA_ENABLE AND CUDA_FOUND)
        ADD_SUBDIRECTORY(examples/cvode/cuda)
      ENDIF()
    ENDIF(EXAMPLES_ENABLE_CUDA)
    # raja examples 
    IF(EXAMPLES_ENABLE_RAJA)
      IF(RAJA_ENABLE AND RAJA_FOUND)
        ADD_SUBDIRECTORY(examples/cvode/raja)
      ENDIF()
    ENDIF(EXAMPLES_ENABLE_RAJA)
  ENDIF(BUILD_CVODE)

  # add CVODES Examples
  IF(BUILD_CVODES)
    # C examples
    IF(EXAMPLES_ENABLE_C)
      ADD_SUBDIRECTORY(examples/cvodes/serial)
      IF(MPIC_FOUND)
        ADD_SUBDIRECTORY(examples/cvodes/parallel)
      ENDIF()
      IF(OPENMP_FOUND)
        ADD_SUBDIRECTORY(examples/cvodes/C_openmp)
      ENDIF()
    ENDIF()
  ENDIF(BUILD_CVODES)

  # add IDA examples
  IF(BUILD_IDA)
    # C examples
    IF(EXAMPLES_ENABLE_C)
      ADD_SUBDIRECTORY(examples/ida/serial)
      IF(OPENMP_FOUND)
        ADD_SUBDIRECTORY(examples/ida/C_openmp)
      ENDIF()
      IF(MPIC_FOUND)
        ADD_SUBDIRECTORY(examples/ida/parallel)
      ENDIF()
      IF(PETSC_FOUND)
        ADD_SUBDIRECTORY(examples/ida/petsc)
      ENDIF()
    ENDIF()
    # Fortran examples
    IF(EXAMPLES_ENABLE_F77)
      IF(F77_FOUND)
        ADD_SUBDIRECTORY(examples/ida/fcmix_serial)
      ENDIF()
      IF(OPENMP_FOUND)
        ADD_SUBDIRECTORY(examples/ida/fcmix_openmp)
      ENDIF()
      IF(PTHREADS_FOUND)
	ADD_SUBDIRECTORY(examples/ida/fcmix_pthreads)
      ENDIF()
      IF(MPIF_FOUND)
        ADD_SUBDIRECTORY(examples/ida/fcmix_parallel)
      ENDIF()
    ENDIF()
  ENDIF(BUILD_IDA)

  # add IDAS examples
  IF(BUILD_IDAS)
    # C examples
    IF(EXAMPLES_ENABLE_C)
      ADD_SUBDIRECTORY(examples/idas/serial)
      IF(OPENMP_FOUND)
        ADD_SUBDIRECTORY(examples/idas/C_openmp)
      ENDIF()
      IF(MPIC_FOUND)
        ADD_SUBDIRECTORY(examples/idas/parallel)
      ENDIF()
    ENDIF()
  ENDIF(BUILD_IDAS)

  # add KINSOL examples
  IF(BUILD_KINSOL)
    # C examples
    IF(EXAMPLES_ENABLE_C)
      ADD_SUBDIRECTORY(examples/kinsol/serial)
      IF(OPENMP_FOUND)
        # the only example here need special handling from testrunner (not yet implemented)
        ADD_SUBDIRECTORY(examples/kinsol/C_openmp)
      ENDIF()
      IF(MPIC_FOUND)
        ADD_SUBDIRECTORY(examples/kinsol/parallel)
      ENDIF()
    ENDIF()
    # Fortran examples
    IF(EXAMPLES_ENABLE_F77)
      IF(F77_FOUND)
        ADD_SUBDIRECTORY(examples/kinsol/fcmix_serial)
      ENDIF()
      IF(MPIF_FOUND)
        ADD_SUBDIRECTORY(examples/kinsol/fcmix_parallel)
      ENDIF()
    ENDIF()
  ENDIF(BUILD_KINSOL)

  # add CPODES examples
  IF(BUILD_CPODES)
    IF(EXAMPLES_ENABLE_C)
      ADD_SUBDIRECTORY(examples/cpodes/serial)
      IF(MPIC_FOUND)
        ADD_SUBDIRECTORY(examples/cpodes/parallel)
      ENDIF()
    ENDIF()
  ENDIF(BUILD_CPODES)

  # Always add the nvector serial examples
  ADD_SUBDIRECTORY(examples/nvector/serial)

  # # Always add the serial sunmatrix dense/band/sparse examples
  ADD_SUBDIRECTORY(examples/sunmatrix/dense)
  ADD_SUBDIRECTORY(examples/sunmatrix/band)
  ADD_SUBDIRECTORY(examples/sunmatrix/sparse)

  # # Always add the serial sunlinearsolver dense/band/spils examples
  ADD_SUBDIRECTORY(examples/sunlinsol/band)
  ADD_SUBDIRECTORY(examples/sunlinsol/dense)
  IF(KLU_FOUND)
    ADD_SUBDIRECTORY(examples/sunlinsol/klu)
  ENDIF(KLU_FOUND)
  IF(SUPERLUMT_FOUND)
    ADD_SUBDIRECTORY(examples/sunlinsol/superlumt)
  ENDIF(SUPERLUMT_FOUND)
  IF(LAPACK_FOUND)
    ADD_SUBDIRECTORY(examples/sunlinsol/lapackband)
    ADD_SUBDIRECTORY(examples/sunlinsol/lapackdense)
  ENDIF(LAPACK_FOUND)
  ADD_SUBDIRECTORY(examples/sunlinsol/spgmr/serial)
  ADD_SUBDIRECTORY(examples/sunlinsol/spfgmr/serial)
  ADD_SUBDIRECTORY(examples/sunlinsol/spbcgs/serial)
  ADD_SUBDIRECTORY(examples/sunlinsol/sptfqmr/serial)
  ADD_SUBDIRECTORY(examples/sunlinsol/pcg/serial)

  IF(MPIC_FOUND)
      ADD_SUBDIRECTORY(examples/nvector/parallel)
      ADD_SUBDIRECTORY(examples/sunlinsol/spgmr/parallel)
      ADD_SUBDIRECTORY(examples/sunlinsol/spfgmr/parallel)
      ADD_SUBDIRECTORY(examples/sunlinsol/spbcgs/parallel)
      ADD_SUBDIRECTORY(examples/sunlinsol/sptfqmr/parallel)
      #ADD_SUBDIRECTORY(examples/sunlinsol/pcg/parallel)
  ENDIF(MPIC_FOUND)

  IF(HYPRE_FOUND)
      ADD_SUBDIRECTORY(examples/nvector/parhyp)
  ENDIF()

  IF(PTHREADS_FOUND)
      ADD_SUBDIRECTORY(examples/nvector/pthreads)
  ENDIF()

  IF(OPENMP_FOUND)
      ADD_SUBDIRECTORY(examples/nvector/C_openmp)
  ENDIF()

  IF(PETSC_FOUND)
      ADD_SUBDIRECTORY(examples/nvector/petsc)
  ENDIF()

  IF(CUDA_FOUND)
      ADD_SUBDIRECTORY(examples/nvector/cuda)
  ENDIF(CUDA_FOUND)
  
  IF(RAJA_FOUND)
      ADD_SUBDIRECTORY(examples/nvector/raja)
  ENDIF(RAJA_FOUND)
  
ENDIF(EXAMPLES_ENABLED)

# ---------------------------------------------------------------
# Install configuration header files and license file
# ---------------------------------------------------------------

# install configured header file
INSTALL(
  FILES ${PROJECT_BINARY_DIR}/include/sundials/sundials_config.h
  DESTINATION include/sundials
  )

# install configured header file for Fortran 90
INSTALL(
  FILES ${PROJECT_BINARY_DIR}/include/sundials/sundials_fconfig.h
  DESTINATION include/sundials
  )

# install license file
INSTALL(
  FILES ${PROJECT_SOURCE_DIR}/LICENSE
  DESTINATION .)
