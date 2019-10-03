/* ========================================================================== */
/* === UMFPACK_symbolic ===================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Performs a symbolic factorization.
    See umfpack_symbolic.h for details.
*/

#include "umf_internal.h"

GLOBAL Int UMFPACK_symbolic
(
    Int n_row,
    Int n_col,
    const Int Ap [ ],
    const Int Ai [ ],
    const double Ax [ ],
#ifdef COMPLEX
    const double Az [ ],
#endif
    void **SymbolicHandle,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
)
{
    return (UMFPACK_qsymbolic (n_row, n_col, Ap, Ai, Ax,
#ifdef COMPLEX
        Az,
#endif
        (Int *) NULL, SymbolicHandle, Control, Info)) ;
}
