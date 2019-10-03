/* ========================================================================== */
/* === btf mexFunction ====================================================== */
/* ========================================================================== */

/* BTF: Permute a square matrix to upper block triangular form with a zero-free
 * diagonal, or with a maximum number of nonzeros along the diagonal if a
 * zero-free permutation does not exist.
 *
 * Usage:
 *
 *      [p,q,r] = btf (A) ;
 *      [p,q,r] = btf (A, maxwork) ;
 *
 * If the matrix has structural full rank, this is essentially identical to
 *
 *      [p,q,r] = dmperm (A)
 *
 * except that p, q, and r will differ in trivial ways.  Both return an upper
 * block triangular form with a zero-free diagonal, if the matrix is
 * structurally non-singular.  The number and sizes of the blocks will be
 * identical, but the order of the blocks, and the ordering within the blocks,
 * can be different.
 * 
 * If the matrix is structurally singular, q will contain negative entries.
 * The permuted matrix is C = A(p,abs(q)), and find(q<0) gives a list of
 * indices of the diagonal of C that are equal to zero.  This differs from
 * dmperm, which does not place the maximum matching along the main diagonal
 * of C=A(p,q), but places it above the diagonal instead.
 *
 * See maxtrans, or btf.m, for a description of maxwork.
 *
 * An optional fourth output [p,q,r,work] = btf (...) returns the amount of
 * work performed, or -1 if the maximum work limit is reached (in which case
 * the maximum matching might not have been found).
 *
 * By Tim Davis.  Copyright (c) 2004-2007, University of Florida.
 * with support from Sandia National Laboratories.  All Rights Reserved.
 *
 * See also maxtrans, strongcomp, dmperm
 */

/* ========================================================================== */

#include "mex.h"
#include "btf.h"
#define Long SuiteSparse_long

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double work, maxwork ;
    Long b, n, k, *Ap, *Ai, *P, *R, nblocks, *Work, *Q, nmatch ;
    double *Px, *Rx, *Qx, *w ;

    /* ---------------------------------------------------------------------- */
    /* get inputs and allocate workspace */
    /* ---------------------------------------------------------------------- */

    if (nargin < 1 || nargin > 2 || nargout > 4)
    {
        mexErrMsgTxt ("Usage: [p,q,r] = btf (A)") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetN (pargin [0]))
    {
        mexErrMsgTxt ("btf: A must be sparse, square, and non-empty") ;
    }

    /* get sparse matrix A */
    Ap = (Long *) mxGetJc (pargin [0]) ;
    Ai = (Long *) mxGetIr (pargin [0]) ;

    /* get output arrays */
    Q = mxMalloc (n * sizeof (Long)) ;
    P = mxMalloc (n * sizeof (Long)) ;
    R = mxMalloc ((n+1) * sizeof (Long)) ;

    /* get workspace */
    Work = mxMalloc (5*n * sizeof (Long)) ;

    maxwork = 0 ;
    if (nargin > 1)
    {
        maxwork = mxGetScalar (pargin [1]) ;
    }
    work = 0 ;

    /* ---------------------------------------------------------------------- */
    /* find the permutation to BTF */
    /* ---------------------------------------------------------------------- */

    nblocks = btf_l_order (n, Ap, Ai, maxwork, &work, P, Q, R, &nmatch, Work) ;

    /* ---------------------------------------------------------------------- */
    /* create outputs and free workspace */
    /* ---------------------------------------------------------------------- */

    /* create P */
    pargout [0] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    Px = mxGetPr (pargout [0]) ;
    for (k = 0 ; k < n ; k++)
    {
        Px [k] = P [k] + 1 ;    /* convert to 1-based */
    }

    /* create Q */
    if (nargout > 1)
    {
        pargout [1] = mxCreateDoubleMatrix (1, n, mxREAL) ;
        Qx = mxGetPr (pargout [1]) ;
        for (k = 0 ; k < n ; k++)
        {
            Qx [k] = Q [k] + 1 ;        /* convert to 1-based */
        }
    }

    /* create R */
    if (nargout > 2)
    {
        pargout [2] = mxCreateDoubleMatrix (1, nblocks+1, mxREAL) ;
        Rx = mxGetPr (pargout [2]) ;
        for (b = 0 ; b <= nblocks ; b++)
        {
            Rx [b] = R [b] + 1 ;        /* convert to 1-based */
        }
    }

    /* create work output */
    if (nargout > 3)
    {
        pargout [3] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
        w = mxGetPr (pargout [3]) ;
        w [0] = work ;
    }

    mxFree (P) ;
    mxFree (R) ;
    mxFree (Work) ;
    mxFree (Q) ;
}
