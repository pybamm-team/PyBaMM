/* ========================================================================== */
/* === UMFPACK_defaults ===================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Sets default control parameters.  See umfpack_defaults.h
    for details.
*/

#include "umf_internal.h"

GLOBAL void UMFPACK_defaults
(
    double Control [UMFPACK_CONTROL]
)
{
    Int i ;

    if (!Control)
    {
	/* silently return if no Control array */
	return ;
    }

    for (i = 0 ; i < UMFPACK_CONTROL ; i++)
    {
	Control [i] = 0 ;
    }

    /* ---------------------------------------------------------------------- */
    /* default control settings: can be modified at run-time */
    /* ---------------------------------------------------------------------- */

    /* used in UMFPACK_report_* routines: */
    Control [UMFPACK_PRL] = UMFPACK_DEFAULT_PRL ;

    Control [UMFPACK_DENSE_ROW] = UMFPACK_DEFAULT_DENSE_ROW ;
    Control [UMFPACK_DENSE_COL] = UMFPACK_DEFAULT_DENSE_COL ;
    Control [UMFPACK_AMD_DENSE] = UMFPACK_DEFAULT_AMD_DENSE ;
    Control [UMFPACK_STRATEGY] = UMFPACK_DEFAULT_STRATEGY ;
    Control [UMFPACK_AGGRESSIVE] = UMFPACK_DEFAULT_AGGRESSIVE ;
    Control [UMFPACK_SINGLETONS] = UMFPACK_DEFAULT_SINGLETONS ;
    Control [UMFPACK_ORDERING] = UMFPACK_DEFAULT_ORDERING ;
    Control [UMFPACK_PIVOT_TOLERANCE] = UMFPACK_DEFAULT_PIVOT_TOLERANCE ;
    Control [UMFPACK_SYM_PIVOT_TOLERANCE] = UMFPACK_DEFAULT_SYM_PIVOT_TOLERANCE;
    Control [UMFPACK_BLOCK_SIZE] = UMFPACK_DEFAULT_BLOCK_SIZE ;
    Control [UMFPACK_ALLOC_INIT] = UMFPACK_DEFAULT_ALLOC_INIT ;
    Control [UMFPACK_FRONT_ALLOC_INIT] = UMFPACK_DEFAULT_FRONT_ALLOC_INIT ;
    Control [UMFPACK_SCALE] = UMFPACK_DEFAULT_SCALE ;

    /* used in UMFPACK_*solve: */
    Control [UMFPACK_IRSTEP] = UMFPACK_DEFAULT_IRSTEP ;

    /* ---------------------------------------------------------------------- */
    /* compile-time settings: cannot be modified at run-time */
    /* ---------------------------------------------------------------------- */

#ifdef NBLAS
    /* do not use the BLAS - use in-line C code instead */
    Control [UMFPACK_COMPILED_WITH_BLAS] = 0 ;
#else
    /* use externally-provided BLAS (dgemm, dger, dgemv, zgemm, zgeru, zgemv) */
    Control [UMFPACK_COMPILED_WITH_BLAS] = 1 ;
#endif
}
