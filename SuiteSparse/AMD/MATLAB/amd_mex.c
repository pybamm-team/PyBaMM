/* ========================================================================= */
/* === AMD mexFunction ===================================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* AMD, Copyright (c) Timothy A. Davis,					     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/*
 * Usage:
 *	p = amd (A)
 *	p = amd (A, Control)
 *	[p, Info] = amd (A)
 *	[p, Info] = amd (A, Control)
 *	Control = amd ;	    % return the default Control settings for AMD
 *	amd ;		    % print the default Control settings for AMD
 *
 * Given a square matrix A, compute a permutation P suitable for a Cholesky
 * factorization of the matrix B (P,P), where B = spones (A) + spones (A').
 * The method used is the approximate minimum degree ordering method.  See
 * amd.m and amd.h for more information.
 *
 * The input matrix need not have sorted columns, and can have duplicate
 * entries.
 */

#include "amd.h"
#include "mex.h"
#include "matrix.h"
#define Long SuiteSparse_long

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    Long i, m, n, *Ap, *Ai, *P, nc, result, spumoni, full ;
    double *Pout, *InfoOut, Control [AMD_CONTROL], Info [AMD_INFO], *ControlIn ;
    mxArray *A ;

    /* --------------------------------------------------------------------- */
    /* get control parameters */
    /* --------------------------------------------------------------------- */

    spumoni = 0 ;
    if (nargin == 0)
    {
	/* get the default control parameters, and return */
	pargout [0] = mxCreateDoubleMatrix (AMD_CONTROL, 1, mxREAL) ;
	amd_l_defaults (mxGetPr (pargout [0])) ;
	if (nargout == 0)
	{
	    amd_l_control (mxGetPr (pargout [0])) ;
	}
	return ;
    }

    amd_l_defaults (Control) ;
    if (nargin > 1)
    {
	ControlIn = mxGetPr (pargin [1]) ;
	nc = mxGetM (pargin [1]) * mxGetN (pargin [1]) ;
	Control [AMD_DENSE]
	    = (nc > 0) ? ControlIn [AMD_DENSE] : AMD_DEFAULT_DENSE ;
	Control [AMD_AGGRESSIVE]
	    = (nc > 1) ? ControlIn [AMD_AGGRESSIVE] : AMD_DEFAULT_AGGRESSIVE ;
	spumoni = (nc > 2) ? (ControlIn [2] != 0) : 0 ;
    }

    if (spumoni > 0)
    {
	amd_l_control (Control) ;
    }

    /* --------------------------------------------------------------------- */
    /* get inputs */
    /* --------------------------------------------------------------------- */

    if (nargout > 2 || nargin > 2)
    {
	mexErrMsgTxt ("Usage: p = amd (A)\nor [p, Info] = amd (A, Control)") ;
    }

    A = (mxArray *) pargin [0] ;
    n = mxGetN (A) ;
    m = mxGetM (A) ;
    if (spumoni > 0)
    {
	mexPrintf ("    input matrix A is %d-by-%d\n", m, n) ;
    }
    if (mxGetNumberOfDimensions (A) != 2)
    {
	mexErrMsgTxt ("amd: A must be 2-dimensional") ;
    }
    if (m != n)
    {
    	mexErrMsgTxt ("amd: A must be square") ;
    }

    /* --------------------------------------------------------------------- */
    /* allocate workspace for output permutation */
    /* --------------------------------------------------------------------- */

    P = mxMalloc ((n+1) * sizeof (Long)) ;

    /* --------------------------------------------------------------------- */
    /* if A is full, convert to a sparse matrix */
    /* --------------------------------------------------------------------- */

    full = !mxIsSparse (A) ;
    if (full)
    {
	if (spumoni > 0)
	{
	    mexPrintf (
	    "    input matrix A is full (sparse copy of A will be created)\n");
	}
	mexCallMATLAB (1, &A, 1, (mxArray **) pargin, "sparse") ;
    }
    Ap = (Long *) mxGetJc (A) ;
    Ai = (Long *) mxGetIr (A) ;
    if (spumoni > 0)
    {
	mexPrintf ("    input matrix A has %d nonzero entries\n", Ap [n]) ;
    }

    /* --------------------------------------------------------------------- */
    /* order the matrix */
    /* --------------------------------------------------------------------- */

    result = amd_l_order (n, Ap, Ai, P, Control, Info) ;

    /* --------------------------------------------------------------------- */
    /* if A is full, free the sparse copy of A */
    /* --------------------------------------------------------------------- */

    if (full)
    {
	mxDestroyArray (A) ;
    }

    /* --------------------------------------------------------------------- */
    /* print results (including return value) */
    /* --------------------------------------------------------------------- */

    if (spumoni > 0)
    {
	amd_l_info (Info) ;
    }

    /* --------------------------------------------------------------------- */
    /* check error conditions */
    /* --------------------------------------------------------------------- */

    if (result == AMD_OUT_OF_MEMORY)
    {
	mexErrMsgTxt ("amd: out of memory") ;
    }
    else if (result == AMD_INVALID)
    {
	mexErrMsgTxt ("amd: input matrix A is corrupted") ;
    }

    /* --------------------------------------------------------------------- */
    /* copy the outputs to MATLAB */
    /* --------------------------------------------------------------------- */

    /* output permutation, P */
    pargout [0] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    Pout = mxGetPr (pargout [0])  ;
    for (i = 0 ; i < n ; i++)
    {
	Pout [i] = P [i] + 1 ;	    /* change to 1-based indexing for MATLAB */
    }
    mxFree (P) ;

    /* Info */
    if (nargout > 1)
    {
	pargout [1] = mxCreateDoubleMatrix (AMD_INFO, 1, mxREAL) ;
	InfoOut = mxGetPr (pargout [1]) ;
	for (i = 0 ; i < AMD_INFO ; i++)
	{
	    InfoOut [i] = Info [i] ;
	}
    }
}
