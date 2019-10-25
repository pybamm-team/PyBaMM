# This CMakeFile is adapted from that in dune-common: 

# .. cmake_module::
#
#    Find the SuiteSparse libraries like UMFPACK or SPQR.
#
#    Example which tries to find Suite Sparse's UMFPack component:
#
#    :code:`find_package(SuiteSparse OPTIONAL_COMPONENTS UMFPACK)`
#
#    `OPTIONAL_COMPONENTS`
#       A list of components. Components are:
#       AMD, BTF, CAMD, CCOLAMD, CHOLMOD, COLAMD, CXSPARSE,
#       KLU, LDL, RBIO, SPQR, UMFPACK
#
#    :ref:`SuiteSparse_ROOT`
#       Path list to search for SuiteSparse
#
#    Sets the following variables:
#
#    :code:`SuiteSparse_FOUND`
#       True if SuiteSparse was found.
#
#    :code:`SuiteSparse_INCLUDE_DIRS`
#       Path to the SuiteSparse include dirs.
#
#    :code:`SuiteSparse_LIBRARIES`
#       Name of the SuiteSparse libraries.
#
#    :code:`SuiteSparse_<COMPONENT>_FOUND`
#       Whether <COMPONENT> was found as part of SuiteSparse.
#
# .. cmake_variable:: SuiteSparse_ROOT
#
#   You may set this variable to have :ref:`FindSuiteSparse` look
#   for SuiteSparse in the given path before inspecting
#   system paths.
#

find_package(BLAS QUIET)

# look for desired componenents
set(SUITESPARSE_COMPONENTS ${SuiteSparse_FIND_COMPONENTS})

# resolve inter-component dependencies
list(FIND SUITESPARSE_COMPONENTS "UMFPACK" WILL_USE_UMFPACK)
if(NOT WILL_USE_UMFPACK EQUAL -1)
  list(APPEND SUITESPARSE_COMPONENTS AMD CHOLMOD)
endif()
list(FIND SUITESPARSE_COMPONENTS "CHOLMOD" WILL_USE_CHOLMOD)
if(NOT WILL_USE_CHOLMOD EQUAL -1)
  list(APPEND SUITESPARSE_COMPONENTS AMD CAMD COLAMD CCOLAMD)
endif()

if(SUITESPARSE_COMPONENTS)
  list(REMOVE_DUPLICATES SUITESPARSE_COMPONENTS)
endif()

# find SuiteSparse config:
# look for library at positions given by the user
find_library(SUITESPARSE_CONFIG_LIB
  NAMES "suitesparseconfig"
  PATHS ${SuiteSparse_ROOT}
  PATH_SUFFIXES "lib" "lib32" "lib64" "Lib"
  NO_DEFAULT_PATH
)
# now also include the deafult paths
find_library(SUITESPARSE_CONFIG_LIB
  NAMES "suitesparseconfig"
  PATH_SUFFIXES "lib" "lib32" "lib64" "Lib"
)

#look for header files at positions given by the user
find_path(SUITESPARSE_INCLUDE_DIR
  NAMES "SuiteSparse_config.h"
  PATHS ${SuiteSparse_ROOT}
  PATH_SUFFIXES "SuiteSparse_config" "SuiteSparse_config/include" "suitesparse" "include" "src" "SuiteSparse_config/Include"
  NO_DEFAULT_PATH
)
#now also look for default paths
find_path(SUITESPARSE_INCLUDE_DIR
  NAMES "SuiteSparse_config.h"
  PATH_SUFFIXES "SuiteSparse_config" "SuiteSparse_config/include" "suitesparse" "include" "src" "SuiteSparse_config/Include"
)

foreach(_component ${SUITESPARSE_COMPONENTS})
  string(TOLOWER ${_component} _componentLower)

  #look for library at positions given by the user
  find_library(${_component}_LIBRARY
    NAMES "${_componentLower}"
    PATHS ${SuiteSparse_ROOT}
    PATH_SUFFIXES "lib" "lib32" "lib64" "${_component}" "${_component}/Lib"
    NO_DEFAULT_PATH
  )
  #now  also include the deafult paths
  find_library(${_component}_LIBRARY
    NAMES "${_componentLower}"
    PATH_SUFFIXES "lib" "lib32" "lib64" "${_component}" "${_component}/Lib"
  )

  #look for header files at positions given by the user
  find_path(${_component}_INCLUDE_DIR
    NAMES "${_componentLower}.h"
    PATHS ${SuiteSparse_ROOT}
    PATH_SUFFIXES "${_componentLower}" "include/${_componentLower}" "suitesparse" "include" "src" "${_component}" "${_component}/Include"
    NO_DEFAULT_PATH
  )
  #now also look for default paths
  find_path(${_component}_INCLUDE_DIR
    NAMES "${_componentLower}.h"
    PATH_SUFFIXES "${_componentLower}" "include/${_componentLower}" "suitesparse" "include" "${_component}" "${_component}/Include"
  )
endforeach()

# SPQR has different header file name SuiteSparseQR.hpp
#look for header files at positions given by the user
find_path(SPQR_INCLUDE_DIR
  NAMES "SuiteSparseQR.hpp"
  PATHS ${SuiteSparse_ROOT}
  PATH_SUFFIXES "spqr" "include/spqr" "suitesparse" "include" "src" "SPQR" "SPQR/Include"
  NO_DEFAULT_PATH
)
#now also look for default paths
find_path(SPQR_INCLUDE_DIR
  NAMES "SuiteSparseQR.hpp"
  PATH_SUFFIXES "spqr" "include/spqr" "suitesparse" "include" "SPQR" "SPQR/Include"
)

# resolve inter-modular dependencies

# CHOLMOD requires AMD, COLAMD; CAMD and CCOLAMD are optional
if(CHOLMOD_LIBRARY)
  if(NOT (AMD_LIBRARY AND COLAMD_LIBRARY))
    message(WARNING "CHOLMOD requires AMD and COLAMD which were not found, skipping the test.")
    set(SuiteSparse_CHOLMOD_FOUND "CHOLMOD requires AMD and COLAMD-NOTFOUND")
  endif()

  list(APPEND CHOLMOD_LIBRARY ${AMD_LIBRARY} ${COLAMD_LIBRARY})
  if(CAMD_LIBRARY)
    list(APPEND CHOLMOD_LIBRARY ${CAMD_LIBRARY})
  endif()
  if(CCOLAMD_LIBRARY)
    list(APPEND CHOLMOD_LIBRARY ${CCOLAMD_LIBRARY})
  endif()
  list(REVERSE CHOLMOD_LIBRARY)
  # remove duplicates
  list(REMOVE_DUPLICATES CHOLMOD_LIBRARY)
  list(REVERSE CHOLMOD_LIBRARY)
endif()

# UMFPack requires AMD, can depend on CHOLMOD
if(UMFPACK_LIBRARY)
  # check wether cholmod was found
  if(CHOLMOD_LIBRARY)
    list(APPEND UMFPACK_LIBRARY ${CHOLMOD_LIBRARY})
  else()
    list(APPEND UMFPACK_LIBRARY ${AMD_LIBRARY})
  endif()
  list(REVERSE UMFPACK_LIBRARY)
  # remove duplicates
  list(REMOVE_DUPLICATES UMFPACK_LIBRARY)
  list(REVERSE UMFPACK_LIBRARY)
endif()

# check wether everything was found
foreach(_component ${SUITESPARSE_COMPONENTS})
  # variable used for component handling
  set(SuiteSparse_${_component}_FOUND (${_component}_LIBRARY AND ${_component}_INCLUDE_DIR))
  set(HAVE_SUITESPARSE_${_component} SuiteSparse_${_component}_FOUND)
  if(SuiteSparse_${_component}_FOUND)
    list(APPEND SUITESPARSE_INCLUDE_DIR "${${_component}_INCLUDE_DIR}")
    list(APPEND SUITESPARSE_LIBRARY "${${_component}_LIBRARY}")
  endif()

  mark_as_advanced(
    HAVE_SUITESPARSE_${_component}
    SuiteSparse_${_component}_FOUND
    ${_component}_INCLUDE_DIR
    ${_component}_LIBRARY)
endforeach()

list(APPEND SUITESPARSE_LIBRARY ${SUITESPARSE_CONFIG_LIB})

# make them unique
if(SUITESPARSE_INCLUDE_DIR)
  list(REMOVE_DUPLICATES SUITESPARSE_INCLUDE_DIR)
endif()
if(SUITESPARSE_LIBRARY)
  list(REVERSE SUITESPARSE_LIBRARY)
  list(REMOVE_DUPLICATES SUITESPARSE_LIBRARY)
  list(REVERSE SUITESPARSE_LIBRARY)
endif()

# behave like a CMake module is supposed to behave
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  "SuiteSparse"
  FOUND_VAR SuiteSparse_FOUND
  REQUIRED_VARS
  BLAS_FOUND
  SUITESPARSE_INCLUDE_DIR
  SUITESPARSE_LIBRARY
  HANDLE_COMPONENTS
)

mark_as_advanced(
  SUITESPARSE_INCLUDE_DIR
  SUITESPARSE_LIBRARY
  SUITESPARSE_CONFIG_LIB
  WILL_USE_CHOLMOD
  WILL_USE_UMFPACK)

# if both headers and library are found, store results
if(SuiteSparse_FOUND)
  set(SuiteSparse_LIBRARIES ${SUITESPARSE_LIBRARY})
  set(SuiteSparse_INCLUDE_DIRS ${SUITESPARSE_INCLUDE_DIR})
  # log result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
    "Determining location of SuiteSparse succeded:\n"
    "Include directory: ${SuiteSparse_INCLUDE_DIRS}\n"
    "Library directory: ${SuiteSparse_LIBRARIES}\n\n")
  set(SuiteSparse_COMPILER_FLAGS)
  foreach(dir ${SuiteSparse_INCLUDE_DIRS})
    set(SuiteSparse_COMPILER_FLAGS "${SuiteSparse_COMPILER_FLAGS} -I${dir}/")
  endforeach()
  set(SuiteSparse_DUNE_COMPILE_FLAGS ${SuiteSparse_COMPILER_FLAGS}
    CACHE STRING "Compile Flags used by DUNE when compiling with SuiteSparse programs")
  set(SuiteSparse_DUNE_LIBRARIES ${BLAS_LIBRARIES} ${SuiteSparse_LIBRARIES}
    CACHE STRING "Libraries used by DUNE when linking SuiteSparse programs")
else()
  # log errornous result
  file(APPEND ${CMAKE_BINARY_DIR}${CMAKES_FILES_DIRECTORY}/CMakeError.log
    "Determing location of SuiteSparse failed:\n"
    "Include directory: ${SuiteSparse_INCLUDE_DIRS}\n"
    "Library directory: ${SuiteSparse_LIBRARIES}\n\n")
endif()

#set HAVE_SUITESPARSE for config.h
set(HAVE_SUITESPARSE ${SuiteSparse_FOUND})
set(HAVE_UMFPACK ${SuiteSparse_UMFPACK_FOUND})

