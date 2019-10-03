/* ========================================================================== */
/* === CHOLMOD/MATLAB/resymbol mexFunction ================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* Usage:
 *	L = resymbol (L, A)
 *
 * Recompute the symbolic Cholesky factorization of the matrix A.  A must be
 * symmetric.  Only tril(A) is used.  Entries in L that are not in the Cholesky
 * factorization of A are removed from L.  L can be from an LL' or LDL'
 * factorization.  The numerical values of A are ignored; only its nonzero
 * pattern is used.
 */

/* ========================================================================== */

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
    double *Lx, *Lx2, *Lz, *Lz2 ;
    Long *Li, *Lp, *Lnz2, *Li2, *Lp2, *ColCount ;
    cholmod_sparse *A, Amatrix, *Lsparse, *S ;
    cholmod_factor *L ;
    cholmod_common Common, *cm ;
    Long j, s, n, lnz, is_complex ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */ 
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (nargout > 1 || nargin != 2)
    {
	mexErrMsgTxt ("usage: L = resymbol (L, A)\n") ;
    }

    n = mxGetN (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetM (pargin [0]))
    {
	mexErrMsgTxt ("resymbol: L must be sparse and square") ;
    }
    if (n != mxGetM (pargin [1]) || n != mxGetN (pargin [1]))
    {
	mexErrMsgTxt ("resymbol: A and L must have same dimensions") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the sparse matrix A */
    /* ---------------------------------------------------------------------- */

    A = sputil_get_sparse_pattern (pargin [1], &Amatrix, &dummy, cm) ;
    S = (A == &Amatrix) ? NULL : A ;

    A->stype = -1 ;

    /* A = sputil_get_sparse (pargin [1], &Amatrix, &dummy, -1) ; */

    /* ---------------------------------------------------------------------- */
    /* construct a copy of the input sparse matrix L */
    /* ---------------------------------------------------------------------- */

    /* get the MATLAB L */
    Lp = (Long *) mxGetJc (pargin [0]) ;
    Li = (Long *) mxGetIr (pargin [0]) ;
    Lx = mxGetPr (pargin [0]) ;
    Lz = mxGetPi (pargin [0]) ;
    is_complex = mxIsComplex (pargin [0]) ;

    /* allocate the CHOLMOD symbolic L */
    L = cholmod_l_allocate_factor (n, cm) ;
    L->ordering = CHOLMOD_NATURAL ;
    ColCount = L->ColCount ;
    for (j = 0 ; j < n ; j++)
    {
	ColCount [j] = Lp [j+1] - Lp [j] ;
    }

    /* allocate space for a CHOLMOD LDL' packed factor */
    /* (LL' and LDL' are treated identically) */
    cholmod_l_change_factor (is_complex ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL,
	    FALSE, FALSE, TRUE, TRUE, L, cm) ;

    /* copy MATLAB L into CHOLMOD L */
    Lp2 = L->p ;
    Li2 = L->i ;
    Lx2 = L->x ;
    Lz2 = L->z ;
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
    if (is_complex)
    {
	for (s = 0 ; s < lnz ; s++)
	{
	    Lz2 [s] = Lz [s] ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* resymbolic factorization */
    /* ---------------------------------------------------------------------- */

    cholmod_l_resymbol (A, NULL, 0, TRUE, L, cm) ;

    /* ---------------------------------------------------------------------- */
    /* copy the results back to MATLAB */
    /* ---------------------------------------------------------------------- */

    Lsparse = cholmod_l_factor_to_sparse (L, cm) ;

    /* return L as a sparse matrix */
    pargout [0] = sputil_put_sparse (&Lsparse, cm) ;

    /* ---------------------------------------------------------------------- */
    /* free workspace and the CHOLMOD L, except for what is copied to MATLAB */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free_factor (&L, cm) ;
    cholmod_l_free_sparse (&S, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /*
    if (cm->malloc_count != 3 + mxIsComplex (pargout[0])) mexErrMsgTxt ("!") ;
    */
}
