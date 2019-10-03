/* ========================================================================= */
/* === CAMD mexFunction ==================================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,			     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/*
 * Usage:
 *	p = camd (A)
 *	p = camd (A, Control)
 *	[p, Info] = camd (A)
 *	[p, Info] = camd (A, Control, C)
 *	Control = camd ;    % return the default Control settings for CAMD
 *	camd ;		    % print the default Control settings for CAMD
 *
 * Given a square matrix A, compute a permutation P suitable for a Cholesky
 * factorization of the matrix B (P,P), where B = spones (A) + spones (A').
 * The method used is the approximate minimum degree ordering method.  See
 * camd.m and camd.h for more information.
 *
 * The input matrix need not have sorted columns, and can have duplicate
 * entries.
 */

#include "camd.h"
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
    Long i, m, n, *Ap, *Ai, *P, nc, result, spumoni, full, *C, Clen ;
    double *Pout, *InfoOut, Control [CAMD_CONTROL], Info [CAMD_INFO],
	*ControlIn, *Cin ;
    mxArray *A ;

    /* --------------------------------------------------------------------- */
    /* get control parameters */
    /* --------------------------------------------------------------------- */

    spumoni = 0 ;
    if (nargin == 0)
    {
	/* get the default control parameters, and return */
	pargout [0] = mxCreateDoubleMatrix (CAMD_CONTROL, 1, mxREAL) ;
	camd_l_defaults (mxGetPr (pargout [0])) ;
	if (nargout == 0)
	{
	    camd_l_control (mxGetPr (pargout [0])) ;
	}
	return ;
    }

    camd_l_defaults (Control) ;
    if (nargin > 1)
    {
	ControlIn = mxGetPr (pargin [1]) ;
	nc = mxGetM (pargin [1]) * mxGetN (pargin [1]) ;
	Control [CAMD_DENSE]
	    = (nc > 0) ? ControlIn [CAMD_DENSE] : CAMD_DEFAULT_DENSE ;
	Control [CAMD_AGGRESSIVE]
	    = (nc > 1) ? ControlIn [CAMD_AGGRESSIVE] : CAMD_DEFAULT_AGGRESSIVE ;
	spumoni = (nc > 2) ? (ControlIn [2] != 0) : 0 ;
    }

    if (spumoni > 0)
    {
	camd_l_control (Control) ;
    }

    /* --------------------------------------------------------------------- */
    /* get inputs */
    /* --------------------------------------------------------------------- */

    if (nargout > 2 || nargin > 3)
    {
	mexErrMsgTxt ("Usage: p = camd (A)\n"
	    "or [p, Info] = camd (A, Control, C)") ;
    }

    Clen = 0 ;
    C = NULL ;
    if (nargin > 2)
    {
	Cin = mxGetPr (pargin [2]) ;
	Clen = mxGetNumberOfElements (pargin [2]) ;
	if (Clen != 0)
	{
	    C = (Long *) mxCalloc (Clen, sizeof (Long)) ;
	    for (i = 0 ; i < Clen ; i++)
	    {
		/* convert c from 1-based to 0-based */
		C [i] = (Long) Cin [i] - 1 ;
	    }
	}
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
	mexErrMsgTxt ("camd: A must be 2-dimensional") ;
    }
    if (m != n)
    {
    	mexErrMsgTxt ("camd: A must be square") ;
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

    result = camd_l_order (n, Ap, Ai, P, Control, Info, C) ;

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
	camd_l_info (Info) ;
    }

    /* --------------------------------------------------------------------- */
    /* check error conditions */
    /* --------------------------------------------------------------------- */

    if (result == CAMD_OUT_OF_MEMORY)
    {
	mexErrMsgTxt ("camd: out of memory") ;
    }
    else if (result == CAMD_INVALID)
    {
	mexErrMsgTxt ("camd: input matrix A is corrupted") ;
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
    if (nargin > 2) mxFree (C) ;

    /* Info */
    if (nargout > 1)
    {
	pargout [1] = mxCreateDoubleMatrix (CAMD_INFO, 1, mxREAL) ;
	InfoOut = mxGetPr (pargout [1]) ;
	for (i = 0 ; i < CAMD_INFO ; i++)
	{
	    InfoOut [i] = Info [i] ;
	}
    }
}
