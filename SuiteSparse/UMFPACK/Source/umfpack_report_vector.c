/* ========================================================================== */
/* === UMFPACK_report_vector ================================================ */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Prints a real or complex vector.
    See umfpack_report_vector.h for details.
*/

#include "umf_internal.h"
#include "umf_report_vector.h"

GLOBAL Int UMFPACK_report_vector
(
    Int n,
    const double Xx [ ],
#ifdef COMPLEX
    const double Xz [ ],
#endif
    const double Control [UMFPACK_CONTROL]
)
{
    Int prl ;

#ifndef COMPLEX
    double *Xz = (double *) NULL ;
#endif

    prl = GET_CONTROL (UMFPACK_PRL, UMFPACK_DEFAULT_PRL) ;

    if (prl <= 2)
    {
	return (UMFPACK_OK) ;
    }

    return (UMF_report_vector (n, Xx, Xz, prl, TRUE, FALSE)) ;
}
