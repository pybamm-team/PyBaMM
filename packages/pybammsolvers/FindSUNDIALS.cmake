# Adapted from CADET; finds SUNDIALS (IDA, SUNLINSOLKLU, sunlinsoldense, sunlinsollapackdense, sunmatrix_sparse, nvecserial)
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
  sundials_core
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
        # CMake bug 1643: search full static name first, fall back to generic name
        set(THIS_LIBRARY_SEARCH lib${LIB}.a ${LIB})
    elseif(WIN32)
        # Windows: try lib${LIB}_static/${LIB}_static/lib${LIB}/${LIB} (SUNDIALS 7.x appends _static on MSVC)
        set(THIS_LIBRARY_SEARCH lib${LIB}_static ${LIB}_static lib${LIB} ${LIB})
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
