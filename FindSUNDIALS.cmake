# This module is adapted from that in CADET (`<https://github.com/modsim/CADET)>`_):

# .. cmake_module::

#    Find SUNDIALS, the SUite of Nonlinear and DIfferential/ALgebraic equation Solvers.
#
#    The module looks for the following sundials components
#
#    * sundials_ida
#    * sundials_sunlinsolklu
#    * sundials_sunlinsoldense
#    * sundials_sunlinsollapackdense
#    * sundials_sunmatrix_sparse
#    * sundials_nvecserial
#
#    To provide the module with a hint about where to find your SUNDIALS installation,
#    you can set the environment variable :code:`SUNDIALS_ROOT`. The FindSUNDIALS module will
#    then look in this path when searching for SUNDIALS paths and libraries.
#    This behavior is defined in CMake >= 3.12, see policy CMP0074.
#    It is replicated for older versions by adding the :code:`SUNDIALS_ROOT` variable to the
#    :code:`PATHS` entry.
#
#    This module will define the following variables:
#    :code:`SUNDIALS_INCLUDE_DIRS` - Location of the SUNDIALS includes
#    :code:`SUNDIALS_LIBRARIES` - Required libraries for all requested components

# List of the valid SUNDIALS components

# find the SUNDIALS include directories
find_path(SUNDIALS_INCLUDE_DIR
  NAMES
    idas/idas.h
    sundials/sundials_math.h
    sundials/sundials_types.h
    sunlinsol/sunlinsol_klu.h
    sunlinsol/sunlinsol_dense.h
    sunlinsol/sunlinsol_spbcgs.h
    sunlinsol/sunlinsol_lapackdense.h
    sunmatrix/sunmatrix_sparse.h
  PATH_SUFFIXES
    include
  PATHS
    ${SUNDIALS_ROOT}
  )

set(SUNDIALS_WANT_COMPONENTS
  sundials_idas
  sundials_sunlinsolklu
  sundials_sunlinsoldense
  sundials_sunlinsolspbcgs
  sundials_sunlinsollapackdense
  sundials_sunmatrixsparse
  sundials_nvecserial
  sundials_nvecopenmp
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
	PATHS
	    ${SUNDIALS_ROOT}
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
)

# behave like a CMake module is supposed to behave
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  "SUNDIALS"
  FOUND_VAR SUNDIALS_FOUND
  REQUIRED_VARS SUNDIALS_INCLUDE_DIR SUNDIALS_LIBRARIES
  HANDLE_COMPONENTS
)
