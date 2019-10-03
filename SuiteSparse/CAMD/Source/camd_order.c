/* ========================================================================= */
/* === CAMD_order ========================================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,			     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/* User-callable CAMD minimum degree ordering routine.  See camd.h for
 * documentation.
 */

#include "camd_internal.h"

/* ========================================================================= */
/* === CAMD_order ========================================================== */
/* ========================================================================= */

GLOBAL Int CAMD_order
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],
    Int P [ ],
    double Control [ ],
    double Info [ ],
    const Int C [ ]
)
{
    Int *Len, *S, nz, i, *Pinv, info, status, *Rp, *Ri, *Cp, *Ci, ok ;
    size_t nzaat, slen ;
    double mem = 0 ;

#ifndef NDEBUG
    CAMD_debug_init ("camd") ;
#endif

    /* clear the Info array, if it exists */
    info = Info != (double *) NULL ;
    if (info)
    {
	for (i = 0 ; i < CAMD_INFO ; i++)
	{
	    Info [i] = EMPTY ;
	}
	Info [CAMD_N] = n ;
	Info [CAMD_STATUS] = CAMD_OK ;
    }

    /* make sure inputs exist and n is >= 0 */
    if (Ai == (Int *) NULL || Ap == (Int *) NULL || P == (Int *) NULL || n < 0)
    {
	if (info) Info [CAMD_STATUS] = CAMD_INVALID ;
	return (CAMD_INVALID) ;	    /* arguments are invalid */
    }

    if (n == 0)
    {
	return (CAMD_OK) ;	    /* n is 0 so there's nothing to do */
    }

    nz = Ap [n] ;
    if (info)
    {
	Info [CAMD_NZ] = nz ;
    }
    if (nz < 0)
    {
	if (info) Info [CAMD_STATUS] = CAMD_INVALID ;
	return (CAMD_INVALID) ;
    }

    /* check if n or nz will cause size_t overflow */
    if ((size_t) n >= SIZE_T_MAX / sizeof (Int)
     || (size_t) nz >= SIZE_T_MAX / sizeof (Int))
    {
	if (info) Info [CAMD_STATUS] = CAMD_OUT_OF_MEMORY ;
	return (CAMD_OUT_OF_MEMORY) ;	    /* problem too large */
    }

    /* check the input matrix:	CAMD_OK, CAMD_INVALID, or CAMD_OK_BUT_JUMBLED */
    status = CAMD_valid (n, n, Ap, Ai) ;

    if (status == CAMD_INVALID)
    {
	if (info) Info [CAMD_STATUS] = CAMD_INVALID ;
	return (CAMD_INVALID) ;	    /* matrix is invalid */
    }

    /* allocate two size-n integer workspaces */
    Len  = SuiteSparse_malloc (n, sizeof (Int)) ;
    Pinv = SuiteSparse_malloc (n, sizeof (Int)) ;
    mem += n ;
    mem += n ;
    if (!Len || !Pinv)
    {
	/* :: out of memory :: */
	SuiteSparse_free (Len) ;
	SuiteSparse_free (Pinv) ;
	if (info) Info [CAMD_STATUS] = CAMD_OUT_OF_MEMORY ;
	return (CAMD_OUT_OF_MEMORY) ;
    }

    if (status == CAMD_OK_BUT_JUMBLED)
    {
	/* sort the input matrix and remove duplicate entries */
	CAMD_DEBUG1 (("Matrix is jumbled\n")) ;
	Rp = SuiteSparse_malloc (n+1, sizeof (Int)) ;
	Ri = SuiteSparse_malloc (nz,  sizeof (Int)) ;
	mem += (n+1) ;
	mem += MAX (nz,1) ;
	if (!Rp || !Ri)
	{
	    /* :: out of memory :: */
	    SuiteSparse_free (Rp) ;
	    SuiteSparse_free (Ri) ;
	    SuiteSparse_free (Len) ;
	    SuiteSparse_free (Pinv) ;
	    if (info) Info [CAMD_STATUS] = CAMD_OUT_OF_MEMORY ;
	    return (CAMD_OUT_OF_MEMORY) ;
	}
	/* use Len and Pinv as workspace to create R = A' */
	CAMD_preprocess (n, Ap, Ai, Rp, Ri, Len, Pinv) ;
	Cp = Rp ;
	Ci = Ri ;
    }
    else
    {
	/* order the input matrix as-is.  No need to compute R = A' first */
	Rp = NULL ;
	Ri = NULL ;
	Cp = (Int *) Ap ;
	Ci = (Int *) Ai ;
    }

    /* --------------------------------------------------------------------- */
    /* determine the symmetry and count off-diagonal nonzeros in A+A' */
    /* --------------------------------------------------------------------- */

    nzaat = CAMD_aat (n, Cp, Ci, Len, P, Info) ;
    CAMD_DEBUG1 (("nzaat: %g\n", (double) nzaat)) ;
    ASSERT ((MAX (nz-n, 0) <= nzaat) && (nzaat <= 2 * (size_t) nz)) ;

    /* --------------------------------------------------------------------- */
    /* allocate workspace for matrix, elbow room, and 7 size-n vectors */
    /* --------------------------------------------------------------------- */

    S = NULL ;
    slen = nzaat ;			/* space for matrix */
    ok = ((slen + nzaat/5) >= slen) ;	/* check for size_t overflow */
    slen += nzaat/5 ;			/* add elbow room */
    for (i = 0 ; ok && i < 8 ; i++)
    {
	ok = ((slen + n+1) > slen) ;	/* check for size_t overflow */
	slen += (n+1) ;		/* size-n elbow room, 7 size-(n+1) workspace */
    }
    mem += slen ;
    ok = ok && (slen < SIZE_T_MAX / sizeof (Int)) ; /* check for overflow */
    ok = ok && (slen < Int_MAX) ;	/* S[i] for Int i must be OK */
    if (ok)
    {
	S = SuiteSparse_malloc (slen, sizeof (Int)) ;
    }
    CAMD_DEBUG1 (("slen %g\n", (double) slen)) ;
    if (!S)
    {
	/* :: out of memory :: (or problem too large) */
	SuiteSparse_free (Rp) ;
	SuiteSparse_free (Ri) ;
	SuiteSparse_free (Len) ;
	SuiteSparse_free (Pinv) ;
	if (info) Info [CAMD_STATUS] = CAMD_OUT_OF_MEMORY ;
	return (CAMD_OUT_OF_MEMORY) ;
    }
    if (info)
    {
	/* memory usage, in bytes. */
	Info [CAMD_MEMORY] = mem * sizeof (Int) ;
    }

    /* --------------------------------------------------------------------- */
    /* order the matrix */
    /* --------------------------------------------------------------------- */

    CAMD_1 (n, Cp, Ci, P, Pinv, Len, slen, S, Control, Info, C) ;

    /* --------------------------------------------------------------------- */
    /* free the workspace */
    /* --------------------------------------------------------------------- */

    SuiteSparse_free (Rp) ;
    SuiteSparse_free (Ri) ;
    SuiteSparse_free (Len) ;
    SuiteSparse_free (Pinv) ;
    SuiteSparse_free (S) ;
    if (info) Info [CAMD_STATUS] = status ;
    return (status) ;	    /* successful ordering */
}
