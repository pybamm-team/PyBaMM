/* ========================================================================== */
/* === stongcomp mexFunction ================================================ */
/* ========================================================================== */

/* STRONGCOMP: Find a symmetric permutation to upper block triangular form of
 * a sparse square matrix.
 *
 * Usage:
 *
 *      [p,r] = strongcomp (A) ;
 *
 *      [p,q,r] = strongcomp (A,qin) ;
 *
 * In the first usage, the permuted matrix is C = A (p,p).  In the second usage,
 * the matrix A (:,qin) is symmetrically permuted to upper block triangular
 * form, where qin is an input column permutation, and the final permuted
 * matrix is C = A (p,q).  This second usage is equivalent to
 *
 *      [p,r] = strongcomp (A (:,qin)) ;
 *      q = qin (p) ;
 *
 * That is, if qin is not present it is assumed to be qin = 1:n.
 *
 * C is the permuted matrix, with a number of blocks equal to length(r)-1.
 * The kth block is from row/col r(k) to row/col r(k+1)-1 of C.
 * r(1) is one and the last entry in r is equal to n+1.
 * The diagonal of A (or A (:,qin)) is ignored.
 *
 * strongcomp is normally proceeded by a maximum transversal:
 *
 *      [p,q,r] = strongcomp (A, maxtrans (A))
 *
 * if the matrix has full structural rank.  This is identical to
 *
 *      [p,q,r] = btf (A)
 *
 * (except that btf handles the case when A is structurally rank-deficient).
 * It essentially the same as
 *
 *      [p,q,r] = dmperm (A)
 *
 * except that p, q, and r will differ between btf and dmperm.  Both return an
 * upper block triangular form with a zero-free diagonal.  The number and sizes
 * of the blocks will be identical, but the order of the blocks, and the
 * ordering within the blocks, can be different.  For structurally rank
 * deficient matrices, dmpmerm returns the maximum matching as a zero-free 
 * diagonal that is above the main diagonal; btf always returns the matching as
 * the main diagonal (which will thus contain zeros).
 *
 * By Tim Davis.  Copyright (c) 2004-2007, University of Florida.
 * with support from Sandia National Laboratories.  All Rights Reserved.
 *
 * See also maxtrans, btf, dmperm
 */

/* ========================================================================== */

#include "mex.h"
#include "btf.h"
#define Long SuiteSparse_long

void mexFunction
(
    int nargout,
    mxArray *pargout[],
    int nargin,
    const mxArray *pargin[]
)
{
    Long b, n, i, k, j, *Ap, *Ai, *P, *R, nblocks, *Work, *Q, jj ;
    double *Px, *Rx, *Qx ;

    /* ---------------------------------------------------------------------- */
    /* get inputs and allocate workspace */
    /* ---------------------------------------------------------------------- */

    if (!((nargin == 1 && nargout <= 2) || (nargin == 2 && nargout <= 3)))
    {
        mexErrMsgTxt ("Usage: [p,r] = strongcomp (A)"
                      " or [p,q,r] = strongcomp (A,qin)") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetN (pargin [0]))
    {
        mexErrMsgTxt ("strongcomp: A must be sparse, square, and non-empty") ;
    }

    /* get sparse matrix A */
    Ap = (Long *) mxGetJc (pargin [0]) ;
    Ai = (Long *) mxGetIr (pargin [0]) ;

    /* get output arrays */
    P = mxMalloc (n * sizeof (Long)) ;
    R = mxMalloc ((n+1) * sizeof (Long)) ;

    /* get workspace of size 4n (recursive code only needs 2n) */
    Work = mxMalloc (4*n * sizeof (Long)) ;

    /* get the input column permutation Q */
    if (nargin == 2)
    {
        if (mxGetNumberOfElements (pargin [1]) != n)
        {
            mexErrMsgTxt
                ("strongcomp: qin must be a permutation vector of size n") ;
        }
        Qx = mxGetPr (pargin [1]) ;
        Q = mxMalloc (n * sizeof (Long)) ;
        /* connvert Qin to 0-based and check validity */
        for (i = 0 ; i < n ; i++)
        {
            Work [i] = 0 ;
        }
        for (k = 0 ; k < n ; k++)
        {
            j = Qx [k] - 1 ;    /* convert to 0-based */
            jj = BTF_UNFLIP (j) ;
            if (jj < 0 || jj >= n || Work [jj] == 1)
            {
                mexErrMsgTxt
                    ("strongcomp: qin must be a permutation vector of size n") ;
            }
            Work [jj] = 1 ;
            Q [k] = j ;
        }
    }
    else
    {
        /* no input column permutation */
        Q = (Long *) NULL ;
    }

    /* ---------------------------------------------------------------------- */
    /* find the strongly-connected components of A */
    /* ---------------------------------------------------------------------- */

    nblocks = btf_l_strongcomp (n, Ap, Ai, Q, P, R, Work) ;

    /* ---------------------------------------------------------------------- */
    /* create outputs and free workspace */
    /* ---------------------------------------------------------------------- */

    /* create P */
    pargout [0] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    Px = mxGetPr (pargout [0]) ;
    for (k = 0 ; k < n ; k++)
    {
        Px [k] = P [k] + 1 ;            /* convert to 1-based */
    }

    /* create Q */
    if (nargin == 2 && nargout > 1)
    {
        pargout [1] = mxCreateDoubleMatrix (1, n, mxREAL) ;
        Qx = mxGetPr (pargout [1]) ;
        for (k = 0 ; k < n ; k++)
        {
            Qx [k] = Q [k] + 1 ;        /* convert to 1-based */
        }
    }

    /* create R */
    if (nargout == nargin + 1)
    {
        pargout [nargin] = mxCreateDoubleMatrix (1, nblocks+1, mxREAL) ;
        Rx = mxGetPr (pargout [nargin]) ;
        for (b = 0 ; b <= nblocks ; b++)
        {
            Rx [b] = R [b] + 1 ;                /* convert to 1-based */
        }
    }

    mxFree (P) ;
    mxFree (R) ;
    mxFree (Work) ;
    if (nargin == 2)
    {
        mxFree (Q) ;
    }
}
