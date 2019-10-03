/* ========================================================================== */
/* === CHOLMOD/MATLAB/Test/lxbpattern mexFunction =========================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2013, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* Find the nonzero pattern of x=L\b for a sparse vector b.  If numerical
 * cancellation has caused entries to drop in L, then this function may
 * give an incorrect result.
 *
 * xpattern = lxbpattern (L,b), same as xpattern = find (L\b),
 * assuming no numerical cancellation, except that xpattern will not
 * appear in sorted ordering (it appears in topological ordering).
 *
 * The core cholmod_lsolve_pattern function takes O(length (xpattern)) time,
 * except that the initialzations in this mexFunction interface add O(n) time.
 *
 * This function is for testing cholmod_lsolve_pattern only.
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
    Long *Lp, *Lnz, *Xp, *Xi, xnz ;
    cholmod_sparse *B, Bmatrix, *X ;
    cholmod_factor *L ;
    cholmod_common Common, *cm ;
    Long j, n ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */ 
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin != 2 || nargout > 1)
    {
	mexErrMsgTxt ("usage: xpattern = lxbpattern (L,b)") ;
    }

    n = mxGetN (pargin [0]) ;

    if (!mxIsSparse (pargin [0]) || n != mxGetM (pargin [0]))
    {
	mexErrMsgTxt ("lxbpattern: L must be sparse and square") ;
    }
    if (n != mxGetM (pargin [1]) || mxGetN (pargin [1]) != 1)
    {
	mexErrMsgTxt ("lxbpattern: b wrong dimension") ;
    }
    if (!mxIsSparse (pargin [1]))
    {
	mexErrMsgTxt ("lxbpattern: b must be sparse") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the sparse b */
    /* ---------------------------------------------------------------------- */

    /* get sparse matrix B (unsymmetric) */
    B = sputil_get_sparse (pargin [1], &Bmatrix, &dummy, 0) ;

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

    /* L is not truly a valid CHOLMOD sparse factor, since L->prev and next are
        NULL.  But these pointers are not accessed in cholmod_lsolve_pattern */
    L->prev = NULL ;
    L->next = NULL ;

    L->xtype = (mxIsComplex (pargin [0])) ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL ;
    L->nzmax = Lp [n] ;

    /* ---------------------------------------------------------------------- */
    /* allocate size-n space for the result X */
    /* ---------------------------------------------------------------------- */

    X = cholmod_l_allocate_sparse (n, 1, n, FALSE, TRUE, 0, 0, cm) ;

    /* ---------------------------------------------------------------------- */
    /* find the pattern of X=L\B */
    /* ---------------------------------------------------------------------- */

    cholmod_l_lsolve_pattern (B, L, X, cm) ;

    /* ---------------------------------------------------------------------- */
    /* return result, converting to 1-based */ 
    /* ---------------------------------------------------------------------- */

    Xp = (Long *) X->p ;
    Xi = (Long *) X->i ;
    xnz = Xp [1] ;
    pargout [0] = sputil_put_int (Xi, xnz, 1) ;

    /* ---------------------------------------------------------------------- */
    /* free workspace and the CHOLMOD L, except for what is copied to MATLAB */
    /* ---------------------------------------------------------------------- */

    L->p = NULL ;
    L->i = NULL ;
    L->x = NULL ;
    L->z = NULL ;
    cholmod_l_free_factor (&L, cm) ;
    cholmod_l_free_sparse (&X, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
}
