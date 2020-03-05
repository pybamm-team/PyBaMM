# =============================================================================
#  CADET - The Chromatography Analysis and Design Toolkit
#  
#  Copyright © 2008-2020: The CADET Authors
#            Please see the AUTHORS and CONTRIBUTORS file.
#  
#  All rights reserved. This program and the accompanying materials
#  are made available under the terms of the GNU Public License v3.0 (or, at
#  your option, any later version) which accompanies this distribution, and
#  is available at http://www.gnu.org/licenses/gpl.html
# =============================================================================

# Find SUNDIALS, the SUite of Nonlinear and DIfferential/ALgebraic equation Solvers.
#
# The module will optionally accept the COMPONENTS argument. If no COMPONENTS
# are specified, then the find module will default to find all the SUNDIALS
# libraries. If one or more COMPONENTS are specified, the module will attempt to
# find the specified components.
#
# Valid components are
#   * sundials_cvode
#   * sundials_cvodes
#   * sundials_ida
#   * sundials_idas
#   * sundials_kinsol
#   * sundials_nvecserial
#   * sundials_nvecopenmp
#   * sundials_nvecpthreads
#
#
# On UNIX systems, this module will read the variable SUNDIALS_PREFER_STATIC_LIBRARIES
# to determine whether or not to prefer a static link to a dynamic link for SUNDIALS
# and all of it's dependencies.  To use this feature, make sure that the
# SUNDIALS_PREFER_STATIC_LIBRARIES variable is set before the call to find_package.
#
# To provide the module with a hint about where to find your SUNDIALS installation,
# you can set the environment variable SUNDIALS_ROOT. The FindSUNDIALS module will
# then look in this path when searching for SUNDIALS paths and libraries.
#
# This module will define the following variables:
#  SUNDIALS_FOUND - true if SUNDIALS was found on the system
#  SUNDIALS_INCLUDE_DIRS - Location of the SUNDIALS includes
#  SUNDIALS_LIBRARIES - Required libraries for all requested components
#  SUNDIALS_VERSION_MAJOR - Major version
#  SUNDIALS_VERSION_MINOR - Minro version
#  SUNDIALS_VERSION_PATCH - Patch level
#  SUNDIALS_VERSION - Full version string
#
# This module exports the target SUNDIALS::<component> if it was found.


# List of the valid SUNDIALS components

# find the SUNDIALS include directories
find_path(SUNDIALS_INCLUDE_DIR
  ida/ida.h
  sundials/sundials_math.h
  sundials/sundials_types.h
  sunlinsol/sunlinsol_klu.h
  sunmatrix/sunmatrix_sparse.h
  )

set(SUNDIALS_WANT_COMPONENTS
  sundials_ida
  sundials_sunlinsolklu
  sundials_sunmatrixsparse
  sundials_nvecserial
  )

# find the SUNDIALS libraries
foreach(LIB ${SUNDIALS_WANT_COMPONENTS})
    if (UNIX AND SUNDIALS_PREFER_STATIC_LIBRARIES)
        # According to bug 1643 on the CMake bug tracker, this is the
        # preferred method for searching for a static library.
        # See http://www.cmake.org/Bug/view.php?id=1643.  We search
        # first for the full static library name, but fall back to a
        # generic search on the name if the static search fails.
        set(THIS_LIBRARY_SEARCH lib${LIB}.a ${LIB})
    else()
        set(THIS_LIBRARY_SEARCH ${LIB})
    endif()

    find_library(SUNDIALS_${LIB}_LIBRARY
        NAMES ${THIS_LIBRARY_SEARCH}
        PATH_SUFFIXES
            lib
            Lib
    )

    set(SUNDIALS_${LIB}_FOUND FALSE)
    if (SUNDIALS_${LIB}_LIBRARY)
      list(APPEND SUNDIALS_LIBRARIES ${SUNDIALS_${LIB}_LIBRARY})
      set(SUNDIALS_${LIB}_FOUND TRUE)
    endif()
    mark_as_advanced(SUNDIALS_${LIB}_LIBRARY)
endforeach()

mark_as_advanced(
    SUNDIALS_LIBRARIES
    SUNDIALS_INCLUDE_DIR
    SUNDIALS_INCLUDE_DIRS
)
