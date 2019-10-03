/* ========================================================================= */
/* === CAMD_preprocess ===================================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,			     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/* Sorts, removes duplicate entries, and transposes from the nonzero pattern of
 * a column-form matrix A, to obtain the matrix R.  The input matrix can have
 * duplicate entries and/or unsorted columns (CAMD_valid (n,Ap,Ai) must not be
 * CAMD_INVALID).
 *
 * This input condition is NOT checked.  This routine is not user-callable.
 */

#include "camd_internal.h"

/* ========================================================================= */
/* === CAMD_preprocess ===================================================== */
/* ========================================================================= */

/* CAMD_preprocess does not check its input for errors or allocate workspace.
 * On input, the condition (CAMD_valid (n,n,Ap,Ai) != CAMD_INVALID) must hold.
 */

GLOBAL void CAMD_preprocess
(
    Int n,		/* input matrix: A is n-by-n */
    const Int Ap [ ],	/* size n+1 */
    const Int Ai [ ],	/* size nz = Ap [n] */

    /* output matrix R: */
    Int Rp [ ],		/* size n+1 */
    Int Ri [ ],		/* size nz (or less, if duplicates present) */

    Int W [ ],		/* workspace of size n */
    Int Flag [ ]	/* workspace of size n */
)
{

    /* --------------------------------------------------------------------- */
    /* local variables */
    /* --------------------------------------------------------------------- */

    Int i, j, p, p2 ;

    ASSERT (CAMD_valid (n, n, Ap, Ai) != CAMD_INVALID) ;

    /* --------------------------------------------------------------------- */
    /* count the entries in each row of A (excluding duplicates) */
    /* --------------------------------------------------------------------- */

    for (i = 0 ; i < n ; i++)
    {
	W [i] = 0 ;		/* # of nonzeros in row i (excl duplicates) */
	Flag [i] = EMPTY ;	/* Flag [i] = j if i appears in column j */
    }
    for (j = 0 ; j < n ; j++)
    {
	p2 = Ap [j+1] ;
	for (p = Ap [j] ; p < p2 ; p++)
	{
	    i = Ai [p] ;
	    if (Flag [i] != j)
	    {
		/* row index i has not yet appeared in column j */
		W [i]++ ;	    /* one more entry in row i */
		Flag [i] = j ;	    /* flag row index i as appearing in col j*/
	    }
	}
    }

    /* --------------------------------------------------------------------- */
    /* compute the row pointers for R */
    /* --------------------------------------------------------------------- */

    Rp [0] = 0 ;
    for (i = 0 ; i < n ; i++)
    {
	Rp [i+1] = Rp [i] + W [i] ;
    }
    for (i = 0 ; i < n ; i++)
    {
	W [i] = Rp [i] ;
	Flag [i] = EMPTY ;
    }

    /* --------------------------------------------------------------------- */
    /* construct the row form matrix R */
    /* --------------------------------------------------------------------- */

    /* R = row form of pattern of A */
    for (j = 0 ; j < n ; j++)
    {
	p2 = Ap [j+1] ;
	for (p = Ap [j] ; p < p2 ; p++)
	{
	    i = Ai [p] ;
	    if (Flag [i] != j)
	    {
		/* row index i has not yet appeared in column j */
		Ri [W [i]++] = j ;  /* put col j in row i */
		Flag [i] = j ;	    /* flag row index i as appearing in col j*/
	    }
	}
    }

#ifndef NDEBUG
    ASSERT (CAMD_valid (n, n, Rp, Ri) == CAMD_OK) ;
    for (j = 0 ; j < n ; j++)
    {
	ASSERT (W [j] == Rp [j+1]) ;
    }
#endif
}
