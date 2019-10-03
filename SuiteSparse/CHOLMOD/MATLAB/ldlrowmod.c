/* ========================================================================== */
/* === CHOLMOD/MATLAB/ldlrowmod mexFunction ================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis.
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* Rank-1 row add/delete of a sparse LDL' factorization.
 *
 * 'Adding' or 'deleting' a row does not change the dimension of L.
 * Instead, deleting a row A(k,:) and the corresponding column A(:,k) means
 * replacing the kth row and column of A with the kth row and column of the
 * identity matrix.  Adding a row is the opposite.  This function then modifies
 * the LDL' factorization of A to reflect this change.
 *
 * To add a row k, the kth row of A is assumed to already be the kth row of
 * identity.  This condition is not checked.
 *
 * On input, LD contains the LDL' factorization of A.  See ldlchol for details.
 * If A has been permuted, then C must reflect that permutation.  In other
 * words, the caller must have already permuted C according the the fill-
 * reducing ordering found by ldlchol.
 *
 * Usage:
 *
 *	LD = ldlrowmod (LD,k,C)		add row k to an LDL' factorization
 *	LD = ldlrowmod (LD,k)	        delete row k from an LDL' factorization
 *
 * See ldlrowmod.m for details.  LD and C must be real and sparse.
 * C is a column vector that is the new column A(:,k).
 *
 * The bulk of the time is spent copying the input LD to the output LD.  This
 * mexFunction could be much faster if it could safely modify its input LD.
 */

#include "cholmod_matlab.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double dummy = 0 ;
    double *Lx, *Lx2 ;
    Long *Li, *Lp, *Li2, *Lp2, *Lnz2, *ColCount ;
    cholmod_sparse Cmatrix, *C, *Lsparse ;
    cholmod_factor *L ;
    cholmod_common Common, *cm ;
    Long j, k, s, rowadd, n, lnz, ok ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */ 
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (nargout > 1 || nargin < 2 || nargin > 3)
    {
	mexErrMsgTxt ("Usage: LD = ldlrowmod (LD,k,C) or ldlrowmod (LD,k)") ; 
    }

    n = mxGetN (pargin [0]) ;
    k = (Long) mxGetScalar (pargin [1]) ;
    k = k - 1 ;         /* change from 1-based to 0-based */

    if (!mxIsSparse (pargin [0])
	    || n != mxGetM (pargin [0])
	    || mxIsComplex (pargin [0]))
    {
	mexErrMsgTxt ("ldlrowmod: L must be real, square, and sparse") ;
    }

    /* ---------------------------------------------------------------------- */
    /* determine if we're doing an rowadd or rowdel */
    /* ---------------------------------------------------------------------- */

    rowadd = (nargin > 2) ;

    if (rowadd)
    {
        if (!mxIsSparse (pargin [2])
                || n != mxGetM (pargin [2])
                || 1 != mxGetN (pargin [2])
                || mxIsComplex (pargin [2]))
        {
            mexErrMsgTxt ("ldlrowmod: C must be a real sparse vector, "
                "with the same number of rows as LD") ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* get C: sparse vector of incoming/outgoing column */
    /* ---------------------------------------------------------------------- */

    C = (rowadd) ? sputil_get_sparse (pargin [2], &Cmatrix, &dummy, 0) : NULL ;

    /* ---------------------------------------------------------------------- */
    /* construct a copy of the input sparse matrix L */
    /* ---------------------------------------------------------------------- */

    /* get the MATLAB L */
    Lp = (Long *) mxGetJc (pargin [0]) ;
    Li = (Long *) mxGetIr (pargin [0]) ;
    Lx = mxGetPr (pargin [0]) ;

    /* allocate the CHOLMOD symbolic L */
    L = cholmod_l_allocate_factor (n, cm) ;
    L->ordering = CHOLMOD_NATURAL ;
    ColCount = L->ColCount ;
    for (j = 0 ; j < n ; j++)
    {
	ColCount [j] = Lp [j+1] - Lp [j] ;
    }

    /* allocate space for a CHOLMOD LDL' packed factor */
    cholmod_l_change_factor (CHOLMOD_REAL, FALSE, FALSE, TRUE, TRUE, L, cm) ;

    /* copy MATLAB L into CHOLMOD L */
    Lp2 = L->p ;
    Li2 = L->i ;
    Lx2 = L->x ;
    Lnz2 = L->nz ;
    lnz = L->nzmax ;
    for (j = 0 ; j <= n ; j++)
    {
	Lp2 [j] = Lp [j] ;
    }
    for (j = 0 ; j < n ; j++)
    {
	Lnz2 [j] = Lp [j+1] - Lp [j] ;
    }
    for (s = 0 ; s < lnz ; s++)
    {
	Li2 [s] = Li [s] ;
    }
    for (s = 0 ; s < lnz ; s++)
    {
	Lx2 [s] = Lx [s] ;
    }

    /* ---------------------------------------------------------------------- */
    /* rowadd/rowdel the LDL' factorization */
    /* ---------------------------------------------------------------------- */

    if (rowadd)
    {
        ok = cholmod_l_rowadd (k, C, L, cm) ;
    }
    else
    {
        ok = cholmod_l_rowdel (k, NULL, L, cm) ;
    }
    if (!ok) mexErrMsgTxt ("ldlrowmod failed\n") ;

    /* ---------------------------------------------------------------------- */
    /* copy the results back to MATLAB */
    /* ---------------------------------------------------------------------- */

    /* change L back to packed LDL' (it may have become unpacked if the
     * sparsity pattern changed).  This change takes O(n) time if the pattern
     * of L wasn't updated. */
    Lsparse = cholmod_l_factor_to_sparse (L, cm) ;

    /* return L as a sparse matrix */
    pargout [0] = sputil_put_sparse (&Lsparse, cm) ;

    /* ---------------------------------------------------------------------- */
    /* free workspace and the CHOLMOD L, except for what is copied to MATLAB */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free_factor (&L, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /*
    if (cm->malloc_count != 3 + mxIsComplex (pargout[0])) mexErrMsgTxt ("!") ;
    */
}
