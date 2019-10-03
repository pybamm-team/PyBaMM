/* ========================================================================== */
/* === CHOLMOD/MATLAB/ldlsolve mexFunction ================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* Solve LDL'x=b given an LDL' factorization computed by ldlchol.
 *
 * Usage:
 *
 *	x = ldlsolve (LD,b)
 *
 * b can be dense or sparse.
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
    double dummy = 0, rcond ;
    Long *Lp, *Lnz, *Lprev, *Lnext ;
    cholmod_sparse *Bs, Bspmatrix, *Xs ;
    cholmod_dense *B, Bmatrix, *X ;
    cholmod_factor *L ;
    cholmod_common Common, *cm ;
    Long j, k, n, B_is_sparse, head, tail ;

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
	mexErrMsgTxt ("Usage: x = ldlsolve (LD, b)") ; 
    }

    n = mxGetN (pargin [0]) ;
    k = mxGetN (pargin [1]) ;

    if (!mxIsSparse (pargin [0]) || n != mxGetM (pargin [0]))
    {
	mexErrMsgTxt ("ldlsolve: LD must be sparse and square") ;
    }
    if (n != mxGetM (pargin [1]))
    {
	mexErrMsgTxt ("ldlsolve: b wrong dimension") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get b */
    /* ---------------------------------------------------------------------- */

    /* get sparse or dense matrix B */
    B = NULL ;
    Bs = NULL ;
    B_is_sparse = mxIsSparse (pargin [1]) ;
    if (B_is_sparse)
    {
	/* get sparse matrix B (unsymmetric) */
	Bs = sputil_get_sparse (pargin [1], &Bspmatrix, &dummy, 0) ;
    }
    else
    {
	/* get dense matrix B */
	B = sputil_get_dense (pargin [1], &Bmatrix, &dummy) ;
    }

    /* ---------------------------------------------------------------------- */
    /* construct a shallow copy of the input sparse matrix L */
    /* ---------------------------------------------------------------------- */

    /* the construction of the CHOLMOD takes O(n) time and memory */

    /* allocate the CHOLMOD symbolic L */
    L = cholmod_l_allocate_factor (n, cm) ;
    L->ordering = CHOLMOD_NATURAL ;

    /* get the MATLAB L */
    L->p = mxGetJc (pargin [0]) ;
    L->i = mxGetIr (pargin [0]) ;
    L->x = mxGetPr (pargin [0]) ;
    L->z = mxGetPi (pargin [0]) ;

    /* allocate and initialize the rest of L */
    L->nz = cholmod_l_malloc (n, sizeof (Long), cm) ;
    Lp = L->p ;
    Lnz = L->nz ;
    for (j = 0 ; j < n ; j++)
    {
	Lnz [j] = Lp [j+1] - Lp [j] ;
    }
    L->prev = cholmod_l_malloc (n+2, sizeof (Long), cm) ;
    L->next = cholmod_l_malloc (n+2, sizeof (Long), cm) ;
    Lprev = L->prev ;
    Lnext = L->next ;

    head = n+1 ;
    tail = n ;
    Lnext [head] = 0 ;
    Lprev [head] = -1 ;
    Lnext [tail] = -1 ;
    Lprev [tail] = n-1 ;
    for (j = 0 ; j < n ; j++)
    {
	Lnext [j] = j+1 ;
	Lprev [j] = j-1 ;
    }
    Lprev [0] = head ;

    L->xtype = (mxIsComplex (pargin [0])) ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL ;
    L->nzmax = Lp [n] ;

    /* ---------------------------------------------------------------------- */
    /* solve and return solution to MATLAB */
    /* ---------------------------------------------------------------------- */

    if (B_is_sparse)
    {
	/* solve LDL'X=B with sparse X and B; return sparse X to MATLAB */
	Xs = cholmod_l_spsolve (CHOLMOD_LDLt, L, Bs, cm) ;
	pargout [0] = sputil_put_sparse (&Xs, cm) ;
    }
    else
    {
	/* solve AX=B with dense X and B; return dense X to MATLAB */
	X = cholmod_l_solve (CHOLMOD_LDLt, L, B, cm) ;
	pargout [0] = sputil_put_dense (&X, cm) ;
    }

    rcond = cholmod_l_rcond (L, cm) ;
    if (rcond == 0)
    {
	mexWarnMsgTxt ("Matrix is indefinite or singular to working precision");
    }
    else if (rcond < DBL_EPSILON)
    {
	mexWarnMsgTxt ("Matrix is close to singular or badly scaled.") ;
	mexPrintf ("         Results may be inaccurate. RCOND = %g.\n", rcond) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace and the CHOLMOD L, except for what is copied to MATLAB */
    /* ---------------------------------------------------------------------- */

    L->p = NULL ;
    L->i = NULL ;
    L->x = NULL ;
    L->z = NULL ;
    cholmod_l_free_factor (&L, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /*
    if (cm->malloc_count !=
	(mxIsComplex (pargout [0]) + (mxIsSparse (pargout[0]) ? 3:1)))
	mexErrMsgTxt ("!") ;
    */
}
