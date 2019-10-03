/* ========================================================================== */
/* === ldlmex.c:  LDL mexFunction =========================================== */
/* ========================================================================== */

/* MATLAB interface for numerical LDL' factorization using the LDL sparse matrix
 * package.
 *
 * MATLAB calling syntax is:
 *
 *       [L, D, Parent, flops] = ldl (A)
 *       [L, D, Parent, flops] = ldl (A, P)
 *       [x, flops] = ldl (A, [ ], b)
 *       [x, flops] = ldl (A, P,   b)
 *
 * The factorization is L*D*L' = A or L*D*L' = A(P,P).   A must be sparse,
 * square, and real.  L is lower triangular with unit diagonal, but the diagonal
 * is not returned.  D is diagonal sparse matrix.  Let n = size (A,1).   If P is
 * not present or empty, the factorization is:
 *
 *	(L + speye (n)) * D * (L + speye (n))' = A
 *
 * otherwise, the factorization is
 *
 *	(L + speye (n)) * D * (L + speye (n))' = A(P,P)
 *
 * P is a permutation of 1:n, an output of AMD, SYMAMD, or SYMRCM, for example.
 * Only the diagonal and upper triangular part of A or A(P,P) is accessed; the
 * lower triangular part is ignored.
 *
 * The elimination tree is returned in the Parent array.
 *
 * In the x = ldl (A, P, b) usage, the LDL' factorization is not returned.
 * Instead, the system A*x=b is solved for x, where b is a dense n-by-m matrix,
 * using P as the fill-reducing ordering for the LDL' factorization of A(P,P).
 * If P is not present or equal to [ ], it is assumed to be the identity
 * permutation.
 *
 * If no zero entry on the diagonal of D is encountered, then the flops argument
 * is the floating point count.
 *
 * If any entry on the diagonal of D is zero, then the LDL' factorization is
 * terminated at that point.  If there is no flops output argument, an error
 * message is printed and no outputs are returned.  Otherwise, flops is
 * negative, d = -flops, and D (d,d) is the first zero entry on the diagonal of
 * D.  A partial factorization is returned.  Let B = A if P is not present or
 * empty, or B = A(P,P) otherwise.  Then the factorization is
 *
 *	LDL = (L + speye (n)) * D * (L + speye (n))'
 *	LDL (1:d, 1:d) = B (1:d,1:d)
 *
 * That is, the LDL' factorization of B (1:d,1:d) is in the first d rows and
 * columns of L and D.  The rest of L and D are zero.
 *
 * Copyright (c) by Timothy A Davis, http://www.suitesparse.com.
 * All Rights Reserved.  See LDL/Doc/License.txt for the License.
 */

#ifndef LDL_LONG
#define LDL_LONG
#endif

#include "ldl.h"
#include "mex.h"
#define Long SuiteSparse_long

/* ========================================================================== */
/* === LDL mexFunction ====================================================== */
/* ========================================================================== */

void mexFunction
(
    int	nargout,
    mxArray *pargout[ ],
    int	nargin,
    const mxArray *pargin[ ]
)
{
    Long i, n, *Pattern, *Flag, *Li, *Lp, *Ap, *Ai, *Lnz, *Parent, do_chol,
	nrhs = 0, lnz, do_solve, *P, *Pinv, nn, k, j, permute, *Dp = NULL, *Di,
	d, do_flops, psrc, pdst ;
    double *Y, *D, *Lx, *Ax, flops, *X = NULL, *B = NULL, *p ;

    /* ---------------------------------------------------------------------- */
    /* get inputs and allocate workspace */
    /* ---------------------------------------------------------------------- */

    do_chol  = (nargin > 0) && (nargin <= 2) && (nargout <= 4) ;
    do_solve = (nargin == 3) && (nargout <= 2) ;
    if (!(do_chol || do_solve))
    {
	mexErrMsgTxt ("Usage:\n"
	    "  [L, D, etree, flopcount] = ldl (A) ;\n"
	    "  [L, D, etree, flopcount] = ldl (A, P) ;\n"
	    "  [x, flopcount] = ldl (A, [ ], b) ;\n"
	    "  [x, flopcount] = ldl (A, P, b) ;\n"
	    "The etree and flopcount arguments are optional.") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetN (pargin [0])
	    || mxIsComplex (pargin [0]))
    {
    	mexErrMsgTxt ("ldl: A must be sparse, square, and real") ;
    }
    if (do_solve)
    {
	if (mxIsSparse (pargin [2]) || n != mxGetM (pargin [2])
	    || !mxIsDouble (pargin [2]) || mxIsComplex (pargin [2]))
	{
	    mexErrMsgTxt (
		"ldl: b must be dense, real, and with proper dimension") ;
	}
    }
    nn = (n == 0) ? 1 : n ;

    /* get sparse matrix A */
    Ap = (Long *) mxGetJc (pargin [0]) ;
    Ai = (Long *) mxGetIr (pargin [0]) ;
    Ax = mxGetPr (pargin [0]) ;

    /* get fill-reducing ordering, if present */
    permute = ((nargin > 1) && !mxIsEmpty (pargin [1])) ;
    if (permute)
    {
	if (mxGetM (pargin [1]) * mxGetN (pargin [1]) != n ||
		mxIsSparse (pargin [1]))
	{
	    mexErrMsgTxt ("ldl: invalid input permutation\n") ;
	}
	P    = (Long *) mxMalloc (nn * sizeof (Long)) ;
	Pinv = (Long *) mxMalloc (nn * sizeof (Long)) ;
	p = mxGetPr (pargin [1]) ;
	for (k = 0 ; k < n ; k++)
	{
	    P [k] = p [k] - 1 ;	/* convert to 0-based */
	}
    }
    else
    {
	P    = (Long *) NULL ;
	Pinv = (Long *) NULL ;
    }

    /* allocate first part of L */
    Lp      = (Long *) mxMalloc ((n+1) * sizeof (Long)) ;
    Parent  = (Long *) mxMalloc (nn * sizeof (Long)) ;

    /* get workspace */
    Y       = (double *)  mxMalloc (nn * sizeof (double)) ;
    Flag    = (Long *) mxMalloc (nn * sizeof (Long)) ;
    Pattern = (Long *) mxMalloc (nn * sizeof (Long)) ;
    Lnz     = (Long *) mxMalloc (nn * sizeof (Long)) ;

    /* make sure the input P is valid */
    if (permute && !ldl_l_valid_perm (n, P, Flag))
    {
	mexErrMsgTxt ("ldl: invalid input permutation\n") ;
    }

    /* note that we assume that the input matrix is valid */

    /* ---------------------------------------------------------------------- */
    /* symbolic factorization to get Lp, Parent, Lnz, and optionally Pinv */
    /* ---------------------------------------------------------------------- */

    ldl_l_symbolic (n, Ap, Ai, Lp, Parent, Lnz, Flag, P, Pinv) ;
    lnz = Lp [n] ;

    /* ---------------------------------------------------------------------- */
    /* create outputs */
    /* ---------------------------------------------------------------------- */

    if (do_chol)
    {
	/* create the output matrix L, using the Lp array from ldl_l_symbolic */
	pargout [0] = mxCreateSparse (n, n, lnz+1, mxREAL) ;
	mxFree (mxGetJc (pargout [0])) ;
	mxSetJc (pargout [0], (void *) Lp) ;	/* Lp is not mxFree'd */
	Li = (Long *) mxGetIr (pargout [0]) ;
	Lx = mxGetPr (pargout [0]) ;

	/* create sparse diagonal matrix D */
	if (nargout > 1)
	{
	    pargout [1] = mxCreateSparse (n, n, nn, mxREAL) ;
	    Dp = (Long *) mxGetJc (pargout [1]) ;
	    Di = (Long *) mxGetIr (pargout [1]) ;
	    for (j = 0 ; j < n ; j++)
	    {
		Dp [j] = j ;
		Di [j] = j ;
	    }
	    Dp [n] = n ;
	    D = mxGetPr (pargout [1])  ;
	}
	else
	{
	    D  = (double *) mxMalloc (nn * sizeof (double)) ;
	}

	/* return elimination tree (add 1 to change from 0-based to 1-based) */
	if (nargout > 2)
	{
	    pargout [2] = mxCreateDoubleMatrix (1, n, mxREAL) ;
	    p = mxGetPr (pargout [2]) ;
	    for (i = 0 ; i < n ; i++)
	    {
		p [i] = Parent [i] + 1 ;
	    }
	}

	do_flops = (nargout == 4) ? (3) : (-1) ;
    }
    else
    {
	/* create L and D as temporary matrices */
	Li = (Long *)    mxMalloc ((lnz+1) * sizeof (Long)) ;
	Lx = (double *) mxMalloc ((lnz+1) * sizeof (double)) ;
	D  = (double *) mxMalloc (nn * sizeof (double)) ;

	/* create solution x */
	nrhs = mxGetN (pargin [2]) ;
	pargout [0] = mxCreateDoubleMatrix (n, nrhs, mxREAL) ;
	X = mxGetPr (pargout [0]) ;
	B = mxGetPr (pargin [2]) ;

	do_flops = (nargout == 2) ? (1) : (-1) ;
    }

    if (do_flops >= 0)
    {
	/* find flop count for ldl_l_numeric */
	flops = 0 ;
	for (k = 0 ; k < n ; k++)
	{
	    flops += ((double) Lnz [k]) * (Lnz [k] + 2) ;
	}
	if (do_solve)
	{
	    /* add flop count for solve */
	    for (k = 0 ; k < n ; k++)
	    {
		flops += 4 * ((double) Lnz [k]) + 1 ;
	    }
	}
	pargout [do_flops] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
	p = mxGetPr (pargout [do_flops]) ;
	p [0] = flops ;
    }

    /* ---------------------------------------------------------------------- */
    /* numeric factorization to get Li, Lx, and D */
    /* ---------------------------------------------------------------------- */

    d = ldl_l_numeric (n, Ap, Ai, Ax, Lp, Parent, Lnz, Li, Lx, D, Y, Flag,
	Pattern, P, Pinv) ;

    /* ---------------------------------------------------------------------- */
    /* singular case : truncate the factorization */
    /* ---------------------------------------------------------------------- */

    if (d != n)
    {
	/* D [d] is zero:  report error, or clean up */
	if (do_chol && do_flops < 0)
	{
	    mexErrMsgTxt ("ldl: zero pivot encountered\n") ;
	}
	else
	{
	    /* L and D are incomplete, compact them */
	    if (do_chol)
	    {
		for (k = d ; k < n ; k++)
		{
		    Dp [k] = d ;
		}
		Dp [n] = d ;
	    }
	    for (k = d ; k < n ; k++)
	    {
		D [k] = 0 ;
	    }
	    pdst = 0 ;
	    for (k = 0 ; k < d ; k++)
	    {
		for (psrc = Lp [k] ; psrc < Lp [k] + Lnz [k] ; psrc++)
		{
		    Li [pdst] = Li [psrc] ;
		    Lx [pdst] = Lx [psrc] ;
		    pdst++ ;
		}
	    }
	    for (k = 0 ; k < d  ; k++)
	    {
		Lp [k+1] = Lp [k] + Lnz [k] ;
	    }
	    for (k = d ; k <= n ; k++)
	    {
		Lp [k] = pdst ;
	    }
	    if (do_flops >= 0)
	    {
		/* return -d instead of the flop count (convert to 1-based) */
		p = mxGetPr (pargout [do_flops]) ;
		p [0] = -(1+d) ;
	    }
	}
    }

    /* ---------------------------------------------------------------------- */
    /* solve Ax=b, if requested */
    /* ---------------------------------------------------------------------- */

    if (do_solve)
    {
	if (permute)
	{
	    for (j = 0 ; j < nrhs ; j++)
	    {
		ldl_l_perm (n, Y, B, P) ;		    /* y = Pb */
		ldl_l_lsolve (n, Y, Lp, Li, Lx) ;	    /* y = L\y */
		ldl_l_dsolve (n, Y, D) ;		    /* y = D\y */
		ldl_l_ltsolve (n, Y, Lp, Li, Lx) ;	    /* y = L'\y */
		ldl_l_permt (n, X, Y, P) ;		    /* x = P'y */
		X += n ;
		B += n ;
	    }
	}
	else
	{
	    for (j = 0 ; j < nrhs ; j++)
	    {
		for (k = 0 ; k < n ; k++)		    /* x = b */
		{
		    X [k] = B [k] ;
		}
		ldl_l_lsolve (n, X, Lp, Li, Lx) ;	    /* x = L\x */
		ldl_l_dsolve (n, X, D) ;		    /* x = D\x */
		ldl_l_ltsolve (n, X, Lp, Li, Lx) ;	    /* x = L'\x */
		X += n ;
		B += n ;
	    }
	}
	/* free the matrix L */
	mxFree (Lp) ;
	mxFree (Li) ;
	mxFree (Lx) ;
	mxFree (D) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    if (do_chol && nargout < 2)
    {
	mxFree (D) ;
    }
    if (permute)
    {
	mxFree (P) ;
	mxFree (Pinv) ;
    }
    mxFree (Parent) ;
    mxFree (Y) ;
    mxFree (Flag) ;
    mxFree (Pattern) ;
    mxFree (Lnz) ;
}
