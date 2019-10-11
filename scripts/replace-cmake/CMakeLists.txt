# ---------------------------------------------------------------
# Programmer: Radu Serban, David J. Gardner, Cody J. Balos,
#             and Slaven Peles @ LLNL
# ---------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2019, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ---------------------------------------------------------------
# Top level CMakeLists.txt for SUNDIALS (for cmake build system)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Initial commands
# ---------------------------------------------------------------

# Require a fairly recent cmake version
cmake_minimum_required(VERSION 3.1.3)

# Libraries linked via full path no longer produce linker search paths
# Allows examples to build
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# MACOSX_RPATH is enabled by default
# Fixes dynamic loading on OSX
if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # Added in CMake 3.0
else()
  if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
  endif()
endif()

# Project SUNDIALS (initially only C supported)
# sets PROJECT_SOURCE_DIR and PROJECT_BINARY_DIR variables
PROJECT(sundials C)

# Set some variables with info on the SUNDIALS project
SET(PACKAGE_BUGREPORT "woodward6@llnl.gov")
SET(PACKAGE_NAME "SUNDIALS")
SET(PACKAGE_STRING "SUNDIALS 4.1.0")
SET(PACKAGE_TARNAME "sundials")

# set SUNDIALS version numbers
# (use "" for the version label if none is needed)
SET(PACKAGE_VERSION_MAJOR "4")
SET(PACKAGE_VERSION_MINOR "1")
SET(PACKAGE_VERSION_PATCH "0")
SET(PACKAGE_VERSION_LABEL "")

IF(PACKAGE_VERSION_LABEL)
  SET(PACKAGE_VERSION "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}-${PACKAGE_VERSION_LABEL}")
ELSE()
  SET(PACKAGE_VERSION "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}")
ENDIF()

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

SET(arkodelib_VERSION "3.1.0")
SET(arkodelib_SOVERSION "3")

SET(cvodelib_VERSION "4.1.0")
SET(cvodelib_SOVERSION "4")

SET(cvodeslib_VERSION "4.1.0")
SET(cvodeslib_SOVERSION "4")

SET(idalib_VERSION "4.1.0")
SET(idalib_SOVERSION "4")

SET(idaslib_VERSION "3.1.0")
SET(idaslib_SOVERSION "3")

SET(kinsollib_VERSION "4.1.0")
SET(kinsollib_SOVERSION "4")

SET(cpodeslib_VERSION "0.0.0")
SET(cpodeslib_SOVERSION "0")

SET(nveclib_VERSION "4.1.0")
SET(nveclib_SOVERSION "4")

SET(sunmatrixlib_VERSION "2.1.0")
SET(sunmatrixlib_SOVERSION "2")

SET(sunlinsollib_VERSION "2.1.0")
SET(sunlinsollib_SOVERSION "2")

SET(sunnonlinsollib_VERSION "1.1.0")
SET(sunnonlinsollib_SOVERSION "1")

# Specify the location of additional CMAKE modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/config)

# Get correct build paths automatically, but expose CMAKE_INSTALL_LIBDIR
# as a regular cache variable so that a user can more easily see what
# the library dir was set to be by GNUInstallDirs.
INCLUDE(GNUInstallDirs)
MARK_AS_ADVANCED(CLEAR CMAKE_INSTALL_LIBDIR)

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
# MACRO definitions
# ---------------------------------------------------------------
INCLUDE(CMakeParseArguments) # can be removed when CMake 3.5+ is required
INCLUDE(SundialsCMakeMacros)
INCLUDE(SundialsAddF2003InterfaceLibrary)
INCLUDE(SundialsAddTest)
INCLUDE(SundialsAddTestInstall)

# ---------------------------------------------------------------
# Check for deprecated SUNDIALS CMake options/variables
# ---------------------------------------------------------------
INCLUDE(SundialsDeprecated)

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

# Only show flags for the current build type if it is set
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

SET(DOCSTR "Signed 64-bit (64) or signed 32-bit (32) integer")
SHOW_VARIABLE(SUNDIALS_INDEX_SIZE STRING "${DOCSTR}" "64")
SET(DOCSTR "Integer type to use for indices in SUNDIALS")
SHOW_VARIABLE(SUNDIALS_INDEX_TYPE STRING "${DOCSTR}" "")
MARK_AS_ADVANCED(SUNDIALS_INDEX_TYPE)
include(SundialsIndexSize)

# ---------------------------------------------------------------
# Enable Fortran interface?
# ---------------------------------------------------------------

# Fortran interface is disabled by default
SET(DOCSTR "Enable Fortran 77 interfaces")
OPTION(F77_INTERFACE_ENABLE "${DOCSTR}" OFF)

# Check that at least one solver with a Fortran 77 interface is built
IF(NOT BUILD_ARKODE AND NOT BUILD_CVODE AND NOT BUILD_IDA AND NOT BUILD_KINSOL)
  IF(F77_INTERFACE_ENABLE)
    PRINT_WARNING("Enabled packages do not support Fortran 77 interface" "Disabling F77 interface")
    FORCE_VARIABLE(F77_INTERFACE_ENABLE BOOL "${DOCSTR}" OFF)
  ENDIF()
  HIDE_VARIABLE(F77_INTERFACE_ENABLE)
ENDIF()

# Fortran 2003 interface is disabled by default
SET(DOCSTR "Enable Fortran 2003 interfaces")
OPTION(F2003_INTERFACE_ENABLE "${DOCSTR}" OFF)

# Check that at least one solver with a Fortran 2003 interface is built
IF(NOT BUILD_CVODE)
  IF(F2003_INTERFACE_ENABLE)
    PRINT_WARNING("Enabled packages do not support Fortran 2003 interface" "Disabling F2003 interface")
    FORCE_VARIABLE(F2003_INTERFACE_ENABLE BOOL "${DOCSTR}" OFF)
  ENDIF()
  HIDE_VARIABLE(F2003_INTERFACE_ENABLE)
ENDIF()

IF(F2003_INTERFACE_ENABLE)
  # F2003 interface only supports double precision
  IF(NOT (SUNDIALS_PRECISION MATCHES "DOUBLE"))
    PRINT_WARNING("F2003 interface is not compatible with ${SUNDIALS_PRECISION} precision"
                  "Disabling F2003 interface")
    FORCE_VARIABLE(F2003_INTERFACE_ENABLE BOOL "${DOCSTR}" OFF)
  ENDIF()

  # F2003 interface only supports 64-bit indices
  IF(NOT (SUNDIALS_INDEX_SIZE MATCHES "64"))
    PRINT_WARNING("F2003 interface is not compatible with ${SUNDIALS_INDEX_SIZE}-bit indicies"
                  "Disabling F2003 interface")
    FORCE_VARIABLE(F2003_INTERFACE_ENABLE BOOL "${DOCSTR}" OFF)
  ENDIF()

  # Put all F2003 modules into one build directory
  SET(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/fortran")

  # Allow a user to set where the Fortran modules will be installed
  SET(DOCSTR "Directory where Fortran module files are installed")
  SHOW_VARIABLE(Fortran_INSTALL_MODDIR DIRECTORY "${DOCSTR}" "fortran")
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

# ---------------------------------------------------------------
# Check for POSIX timers
# ---------------------------------------------------------------
INCLUDE(SundialsPOSIXTimers)

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

# provide OPENMP_DEVICE_ENABLE option
OPTION(OPENMP_DEVICE_ENABLE "Enable OpenMP device offloading support" OFF)

# Advanced option to skip OpenMP device offloading support check.
# This is needed for a specific compiler that doesn't correctly
# report its OpenMP spec date (with CMake >= 3.9).
OPTION(SKIP_OPENMP_DEVICE_CHECK "Skip the OpenMP device offloading support check" OFF)
MARK_AS_ADVANCED(FORCE SKIP_OPENMP_DEVICE_CHECK)

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
IF(LAPACK_ENABLE AND SUNDIALS_INDEX_SIZE MATCHES "64")
  PRINT_WARNING("LAPACK is not compatible with ${SUNDIALS_INDEX_SIZE} integers"
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

# ---------------------------------------------------------------
# Enable Trilinos support?
# ---------------------------------------------------------------
OPTION(Trilinos_ENABLE "Enable Trilinos support" OFF)


# ===============================================================
# Options for examples
# ===============================================================

# ---------------------------------------------------------------
# Enable examples?
# ---------------------------------------------------------------

# Enable C examples (on by default)
OPTION(EXAMPLES_ENABLE_C "Build SUNDIALS C examples" ON)

# C++ examples (off by default, unless Trilinos is enabled)
SET(DOCSTR "Build C++ examples")
OPTION(EXAMPLES_ENABLE_CXX "${DOCSTR}" ${Trilinos_ENABLE})

# F77 examples (on by default) are an option only if the Fortran
# interface is enabled
SET(DOCSTR "Build SUNDIALS Fortran examples")
IF(F77_INTERFACE_ENABLE)
  SHOW_VARIABLE(EXAMPLES_ENABLE_F77 BOOL "${DOCSTR}" ON)
  # Fortran 77 examples do not support single or extended precision
  IF(EXAMPLES_ENABLE_F77 AND (SUNDIALS_PRECISION MATCHES "EXTENDED" OR SUNDIALS_PRECISION MATCHES "SINGLE"))
    PRINT_WARNING("F77 examples are not compatible with ${SUNDIALS_PRECISION} precision"
                  "EXAMPLES_ENABLE_F77")
    FORCE_VARIABLE(EXAMPLES_ENABLE_F77 BOOL "${DOCSTR}" OFF)
  ENDIF()
ELSE()
  # set back to OFF (in case was ON)
  IF(EXAMPLES_ENABLE_F77)
    PRINT_WARNING("EXAMPLES_ENABLE_F77 is ON but F77_INTERFACE_ENABLE is OFF"
                  "Disabling EXAMPLES_ENABLE_F77")
    FORCE_VARIABLE(EXAMPLES_ENABLE_F77 BOOL "${DOCSTR}" OFF)
  ENDIF()
  HIDE_VARIABLE(EXAMPLES_ENABLE_F77)
ENDIF()

# F90 examples (on by default) are an option only if a Fortran interface is enabled.
SET(DOCSTR "Build SUNDIALS F90 examples")
IF(F77_INTERFACE_ENABLE OR F2003_INTERFACE_ENABLE)
  SHOW_VARIABLE(EXAMPLES_ENABLE_F90 BOOL "${DOCSTR}" ON)
  # Fortran 90 examples do not support extended precision
  IF(EXAMPLES_ENABLE_F90 AND (SUNDIALS_PRECISION MATCHES "EXTENDED"))
    PRINT_WARNING("F90 examples are not compatible with ${SUNDIALS_PRECISION} precision"
                  "Disabling EXAMPLES_ENABLE_F90")
    FORCE_VARIABLE(EXAMPLES_ENABLE_F90 BOOL "${DOCSTR}" OFF)
  ENDIF()
ELSE()
  # set back to OFF (in case was ON)
  IF(EXAMPLES_ENABLE_F90)
    PRINT_WARNING("EXAMPLES_ENABLE_F90 is ON but both F77 and F2003 interfaces are OFF"
                  "Disabling EXAMPLES_ENABLE_F90")
    FORCE_VARIABLE(EXAMPLES_ENABLE_F90 BOOL "${DOCSTR}" OFF)
  ENDIF()
  HIDE_VARIABLE(EXAMPLES_ENABLE_F90)
ENDIF()

# CUDA examples (off by default)
SET(DOCSTR "Build SUNDIALS CUDA examples")
IF(CUDA_ENABLE)
  OPTION(EXAMPLES_ENABLE_CUDA "${DOCSTR}" OFF)
ELSE()
  IF(EXAMPLES_ENABLE_CUDA)
    PRINT_WARNING("EXAMPLES_ENABLE_CUDA is ON but CUDA_ENABLE is OFF"
                  "Disabling EXAMPLES_ENABLE_CUDA")
    FORCE_VARIABLE(EXAMPLES_ENABLE_CUDA BOOL "${DOCSTR}" OFF)
  ENDIF()
ENDIF()

# If any of the above examples are enabled set EXAMPLES_ENABLED to TRUE
IF(EXAMPLES_ENABLE_C OR
    EXAMPLES_ENABLE_F77 OR
    EXAMPLES_ENABLE_CXX OR
    EXAMPLES_ENABLE_F90 OR
    EXAMPLES_ENABLE_CUDA)
  SET(EXAMPLES_ENABLED TRUE)
ELSE()
  SET(EXAMPLES_ENABLED FALSE)
ENDIF()

# ---------------------------------------------------------------
# Install examples?
# ---------------------------------------------------------------

# Enable installing examples by default
SET(DOCSTR "Install SUNDIALS examples")
IF(EXAMPLES_ENABLED)
  OPTION(EXAMPLES_INSTALL "${DOCSTR}" ON)
ELSE()
  FORCE_VARIABLE(EXAMPLES_INSTALL BOOL "${DOCSTR}" OFF)
  HIDE_VARIABLE(EXAMPLES_INSTALL)
ENDIF()

# If examples are to be exported, check where we should install them.
IF(EXAMPLES_INSTALL)

  SHOW_VARIABLE(EXAMPLES_INSTALL_PATH PATH
    "Output directory for installing example files"
    "${CMAKE_INSTALL_PREFIX}/examples")

  IF(NOT EXAMPLES_INSTALL_PATH)
    PRINT_WARNING("The example installation path is empty"
      "Example installation path was reset to its default value")
    SET(EXAMPLES_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/examples" CACHE STRING
      "Output directory for installing example files" FORCE)
  ENDIF()

ELSE()

  HIDE_VARIABLE(EXAMPLES_INSTALL_PATH)

ENDIF()


# ==============================================================================
# Advanced (hidden) options
# ==============================================================================

# ------------------------------------------------------------------------------
# Manually specify the Fortran name-mangling scheme
#
# The build system tries to infer the Fortran name-mangling scheme using a
# Fortran compiler and defaults to using lower case and one underscore if the
# scheme can not be determined. If a working Fortran compiler is not available
# or the user needs to override the inferred or default scheme, the following
# options specify the case and number of appended underscores corresponding to
# the Fortran name-mangling scheme of symbol names that do not themselves
# contain underscores. This is all we really need for the FCMIX and LAPACK
# interfaces. A working Fortran compiler is only necessary for building Fortran
# example programs.
# ------------------------------------------------------------------------------

# The case to use in the name-mangling scheme
show_variable(SUNDIALS_F77_FUNC_CASE STRING
  "case of Fortran function names (lower/upper)"
  "")

# The number of underscores of appended in the name-mangling scheme
show_variable(SUNDIALS_F77_FUNC_UNDERSCORES STRING
  "number of underscores appended to Fortran function names (none/one/two)"
  "")

# Hide the name-mangling varibales as advanced options
mark_as_advanced(FORCE SUNDIALS_F77_FUNC_CASE)
mark_as_advanced(FORCE SUNDIALS_F77_FUNC_UNDERSCORES)

# If used, both case and underscores must be set
if((NOT SUNDIALS_F77_FUNC_CASE) AND SUNDIALS_F77_FUNC_UNDERSCORES)
  message(FATAL_ERROR
    "If SUNDIALS_F77_FUNC_UNDERSCORES is set, SUNDIALS_F77_FUNC_CASE must also be set.")
endif()

if(SUNDIALS_F77_FUNC_CASE AND (NOT SUNDIALS_F77_FUNC_UNDERSCORES))
  message(FATAL_ERROR
    "If SUNDIALS_F77_FUNC_CASE is set, SUNDIALS_F77_FUNC_UNDERSCORES must also be set.")
endif()

# ------------------------------------------------------------------------------
# Include development examples in regression tests?
#
# NOTE: Development examples are currently used for internal testing and may
# produce erroneous failures when run on different systems as the pass/fail
# status is determined by comparing the output against a saved output file.
# ------------------------------------------------------------------------------
OPTION(SUNDIALS_DEVTESTS "Include development tests in make test" OFF)
MARK_AS_ADVANCED(FORCE SUNDIALS_DEVTESTS)

# ===============================================================
# Add any platform specifc settings
# ===============================================================

# Under Windows, add compiler directive to inhibit warnings
# about use of unsecure functions

IF(WIN32)
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
ENDIF(WIN32)

IF(APPLE)
  SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -undefined dynamic_lookup")
ENDIF(APPLE)

# ===============================================================
# Fortran and C++ settings
# ===============================================================

# ---------------------------------------------------------------
# A Fortran compiler is needed to:
# (a) Determine the name-mangling scheme if FCMIX, BLAS, or
#     LAPACK are enabled
# (b) Compile example programs if F77 or F90 examples are enabled
# ---------------------------------------------------------------

# Do we need a Fortran name-mangling scheme?
if(F77_INTERFACE_ENABLE OR BLAS_ENABLE OR LAPACK_ENABLE)
  set(NEED_FORTRAN_NAME_MANGLING TRUE)
endif()

# Did the user provide a name-mangling scheme?
if(SUNDIALS_F77_FUNC_CASE AND SUNDIALS_F77_FUNC_UNDERSCORES)

  STRING(TOUPPER ${SUNDIALS_F77_FUNC_CASE} SUNDIALS_F77_FUNC_CASE)
  STRING(TOUPPER ${SUNDIALS_F77_FUNC_UNDERSCORES} SUNDIALS_F77_FUNC_UNDERSCORES)

  # Based on the given case and number of underscores, set the C preprocessor
  # macro definitions. Since SUNDIALS never uses symbols names containing
  # underscores we set the name-mangling schemes to be the same. In general,
  # names of symbols with and without underscore may be mangled differently
  # (e.g. g77 mangles mysub to mysub_ and my_sub to my_sub__)
  if(SUNDIALS_F77_FUNC_CASE MATCHES "LOWER")
    if(SUNDIALS_F77_FUNC_UNDERSCORES MATCHES "NONE")
      set(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) name")
      set(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) name")
    elseif(SUNDIALS_F77_FUNC_UNDERSCORES MATCHES "ONE")
      set(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) name ## _")
      SET(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) name ## _")
    elseif(SUNDIALS_F77_FUNC_UNDERSCORES MATCHES "TWO")
      set(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) name ## __")
      set(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) name ## __")
    else()
      message(FATAL_ERROR "Invalid SUNDIALS_F77_FUNC_UNDERSCORES option.")
    endif()
  elseif(SUNDIALS_F77_FUNC_CASE MATCHES "UPPER")
    if(SUNDIALS_F77_FUNC_UNDERSCORES MATCHES "NONE")
      set(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) NAME")
      set(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) NAME")
    elseif(SUNDIALS_F77_FUNC_UNDERSCORES MATCHES "ONE")
      set(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) NAME ## _")
      set(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) NAME ## _")
    elseif(SUNDIALS_F77_FUNC_UNDERSCORES MATCHES "TWO")
      set(F77_MANGLE_MACRO1 "#define SUNDIALS_F77_FUNC(name,NAME) NAME ## __")
      set(F77_MANGLE_MACRO2 "#define SUNDIALS_F77_FUNC_(name,NAME) NAME ## __")
    else()
      message(FATAL_ERROR "Invalid SUNDIALS_F77_FUNC_UNDERSCORES option.")
    endif()
  else()
    message(FATAL_ERROR "Invalid SUNDIALS_F77_FUNC_CASE option.")
  endif()

  # name-mangling scheme has been manually set
  set(NEED_FORTRAN_NAME_MANGLING FALSE)

endif()

# Do we need a Fortran compiler?
if(F2003_INTERFACE_ENABLE OR EXAMPLES_ENABLE_F77 OR EXAMPLES_ENABLE_F90 OR NEED_FORTRAN_NAME_MANGLING)
  include(SundialsFortran)
endif()

# Ensure that F90 compiler is found if F90 examples are enabled
if (EXAMPLES_ENABLE_F90 AND (NOT F90_FOUND))
  PRINT_WARNING("Compiler with F90 support not found" "Disabling F90 Examples")
  SET(DOCSTR "Build F90 examples")
  FORCE_VARIABLE(EXAMPLES_ENABLE_F90 "${DOCSTR}" OFF)
endif()

# Ensure that F90 compiler found if F2003 interface is enabled
if (F2003_INTERFACE_ENABLE AND (NOT F90_FOUND))
  PRINT_WARNING("Compiler with F90 support not found" "Disabling F2003 Interface")
  SET(DOCSTR "Enable Fortran 2003 interfaces")
  FORCE_VARIABLE(F2003_INTERFACE_ENABLE BOOL "${DOCSTR}" OFF)
endif()

# F2003 interface requires ISO_C_BINDING
IF(F2003_INTERFACE_ENABLE AND (NOT Fortran_COMPILER_SUPPORTS_ISOCBINDING))
  PRINT_WARNING("Fortran compiler does not provide ISO_C_BINDING support"
                "Disabling F2003 interface")
  SET(DOCSTR "Enable Fortran 2003 interfaces")
  FORCE_VARIABLE(F2003_INTERFACE_ENABLE BOOL "${DOCSTR}" OFF)
ENDIF()


# ---------------------------------------------------------------
# A C++ compiler is needed if:
# (a) C++ examples are enabled
# (b) CUDA is enabled
# (c) RAJA is enabled
# (d) Trilinos is enabled
# ---------------------------------------------------------------

if(EXAMPLES_ENABLE_CXX OR CUDA_ENABLE OR RAJA_ENABLE OR Trilinos_ENABLE)
  include(SundialsCXX)
endif()

# ---------------------------------------------------------------
# Setup CUDA. Since CUDA is its own language we do this
# separate from the TPLs.
# ---------------------------------------------------------------

if(CUDA_ENABLE)
  find_package(CUDA)
  if (CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
  else()
    message(STATUS "Disabling CUDA support, could not find CUDA.")
    set(CUDA_ENABLE OFF)
  endif()
endif(CUDA_ENABLE)

# ---------------------------------------------------------------
# Now that all languages are setup, we can configure them more.
# ---------------------------------------------------------------

# C++11 is needed if:
#   (a) CUDA is enabled
# C++11 should not be enabled if
#   (a) RAJA is enabled (they provide a std flag)
if (CXX_FOUND AND CUDA_ENABLE AND CUDA_FOUND AND (NOT RAJA_ENABLE))
  USE_CXX_STD(11)
endif()

# ---------------------------------------------------------------
# Decide how to compile MPI codes. We must check for MPI if
# MPI is enabled or if Trilinos is enabled because the Trilinos
# examples may need MPI without us turning on the MPI SUNDIALS
# components.
# ---------------------------------------------------------------

if(MPI_ENABLE OR Trilinos_ENABLE)
  include(SundialsMPI)
endif()

if(MPI_ENABLE)
  if(NOT MPI_C_FOUND)
    print_warning("MPI not functional" "Parallel support will not be provided")
  else()
    set(IS_MPI_ENABLED "#ifndef SUNDIALS_MPI_ENABLED\n#define SUNDIALS_MPI_ENABLED 1\n#endif")
  endif()
endif()

# always define FMPI_COMM_F2C in sundials_fconfig.h file
if(MPIC_MPI2)
  set(F77_MPI_COMM_F2C "#define SUNDIALS_MPI_COMM_F2C 1")
  set(FMPI_COMM_F2C ".true.")
else()
  set(F77_MPI_COMM_F2C "#define SUNDIALS_MPI_COMM_F2C 0")
  set(FMPI_COMM_F2C ".false.")
endif()

# -------------------------------------------------------------
# Find OpenMP
# -------------------------------------------------------------

if(OPENMP_ENABLE OR OPENMP_DEVICE_ENABLE)

  include(SundialsOpenMP)

  # turn off OPENMP_ENABLE and OPENMP_DEVICE_ENABLE if OpenMP is not found
  if(NOT OPENMP_FOUND)
    print_warning("Could not determine OpenMP compiler flags" "Disabling OpenMP support")
    force_variable(OPENMP_ENABLE BOOL "Enable OpenMP support" OFF)
    force_variable(OPENMP_DEVICE_ENABLE BOOL "Enable OpenMP device offloading support" OFF)
  endif()

  # turn off OPENMP_DEVICE_ENABLE if offloading is not supported
  if(OPENMP_DEVICE_ENABLE AND (NOT OPENMP_SUPPORTS_DEVICE_OFFLOADING))
    print_warning("OpenMP found does not support device offloading"
                  "Disabling OpenMP device offloading support")
    force_variable(OPENMP_DEVICE_ENABLE BOOL "Enable OpenMP device offloading support" OFF)
  endif()

endif()

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
# Find RAJA
# -------------------------------------------------------------

# disable RAJA if CUDA is not enabled/working
if(RAJA_ENABLE AND (NOT CUDA_FOUND))
  PRINT_WARNING("CUDA is required for RAJA support" "Please enable CUDA and RAJA")
  FORCE_VARIABLE(RAJA_ENABLE BOOL "RAJA disabled" OFF)
endif()

if(RAJA_ENABLE)
  # Look for CMake configuration file in RAJA installation
  find_package(RAJA)
  if (RAJA_FOUND)
    include_directories(${RAJA_INCLUDE_DIR})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${RAJA_NVCC_FLAGS})
  else()
    PRINT_WARNING("RAJA configuration not found"
                  "Please set RAJA_DIR to provide path to RAJA CMake configuration file.")
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


  # setting the paths for KLU directly without testing - scott marquis
  # INCLUDE(SundialsKLU) 
  set(KLU_LIBRARY ${SUITESPARSE_DIR}/KLU/Lib/libklu.a)
  set(AMD_LIBRARY ${SUITESPARSE_DIR}/AMD/Lib/libamd.a)
  set(COLAMD_LIBRARY ${SUITESPARSE_DIR}/COLAMD/Lib/libcolamd.a)
  set(BTF_LIBRARY ${SUITESPARSE_DIR}/BTF/Lib/libbtf.a)
  set(SUITESPARSECONFIG_LIBRARY ${SUITESPARSE_DIR}/SuiteSparse_config/libsuitesparseconfig.a)
  set(KLU_LIBRARIES ${KLU_LIBRARY} ${AMD_LIBRARY} ${COLAMD_LIBRARY} ${BTF_LIBRARY} ${SUITESPARSECONFIG_LIBRARY})


  set(KLU_FOUND TRUE)

  IF(KLU_FOUND)
    # sundials_config.h symbol
    SET(SUNDIALS_KLU TRUE)
    set(KLU_INCLUDE_DIR ${SUITESPARSE_DIR}/KLU/Include)
    set(AMD_INCLUDE_DIR ${SUITESPARSE_DIR}/AMD/Include)
    set(COLAMD_INCLUDE_DIR ${SUITESPARSE_DIR}/COLAMD/Include)
    set(BTF_INCLUDE_DIR ${SUITESPARSE_DIR}/BTF/Include)
    set(SUITESPARSECONFIG_INCLUDE_DIR ${SUITESPARSE_DIR}/SuiteSparse_config)

    INCLUDE_DIRECTORIES(${AMD_INCLUDE_DIR} ${COLAMD_INCLUDE_DIR} ${BTF_INCLUDE_DIR} ${SUITESPARSECONFIG_INCLUDE_DIR} ${KLU_INCLUDE_DIR})
  ENDIF(KLU_FOUND)

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

# -------------------------------------------------------------
# Find Trilinos
# -------------------------------------------------------------

if(Trilinos_ENABLE)
  include(SundialsTrilinos)
  if(NOT Trilinos_FUNCTIONAL)
    PRINT_WARNING("Trilinos not functional" "Verify the path to Trilinos and check the Trilinos installation")
  endif()
endif(Trilinos_ENABLE)


# ===============================================================
# At this point all the configuration options are set.
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
# Enable testing and add source and example files to the build.
# ---------------------------------------------------------------

# Enable testing
IF(EXAMPLES_ENABLED)
  INCLUDE(SundialsTesting)
ENDIF()

# Add selected packages and modules to the build
ADD_SUBDIRECTORY(src)

# Add selected examples to the build
IF(EXAMPLES_ENABLED)
  ADD_SUBDIRECTORY(examples)
ENDIF()

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

# install shared Fortran 2003 modules
IF(F2003_INTERFACE_ENABLE)
  # While the .mod files get generated for static and shared
  # libraries, they are identical. So only install one set
  # of the .mod files.
  IF(BUILD_STATIC_LIBS)
    INSTALL(
      DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}_STATIC/
      DESTINATION ${Fortran_INSTALL_MODDIR}
      )
  ELSE()
    INSTALL(
      DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}_SHARED/
      DESTINATION ${Fortran_INSTALL_MODDIR}
      )
  ENDIF()
ENDIF()

# install license and notice files
INSTALL(
  FILES ${PROJECT_SOURCE_DIR}/LICENSE
  DESTINATION include/sundials
  )
INSTALL(
  FILES ${PROJECT_SOURCE_DIR}/NOTICE
  DESTINATION include/sundials
  )

