/* ========================================================================== */
/* === maxtrans mexFunction ================================================= */
/* ========================================================================== */

#define MIN(a,b) (((a) < (b)) ?  (a) : (b))

/* MAXTRANS: Find a column permutation for a zero-free diagonal.
 *
 * Usage:
 *
 * q = maxtrans (A) ;
 * q = maxtrans (A,maxwork) ;
 *
 * A (:,q) has a zero-free diagonal if sprank(A) == n.
 * If the matrix is structurally singular, q will contain zeros.  Similar
 * to p = dmperm (A), except that dmperm returns a row permutation.
 *
 * An optional second output [q,work] = maxtrans (...) returns the amount of
 * work performed, or -1 if the maximum work limit is reached (in which case
 * the maximum matching might not have been found).
 *
 * By Tim Davis.  Copyright (c) 2004-2007, University of Florida.
 * with support from Sandia National Laboratories.  All Rights Reserved.
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
    double maxwork, work ;
    Long nrow, ncol, i, *Ap, *Ai, *Match, nmatch, *Work ;
    double *Matchx, *w ;

    /* ---------------------------------------------------------------------- */
    /* get inputs and allocate workspace */
    /* ---------------------------------------------------------------------- */

    if (nargin < 1 || nargin > 2 || nargout > 2)
    {
        mexErrMsgTxt ("Usage: q = maxtrans (A)") ;
    }
    nrow = mxGetM (pargin [0]) ;
    ncol = mxGetN (pargin [0]) ;
    if (!mxIsSparse (pargin [0]))
    {
        mexErrMsgTxt ("maxtrans: A must be sparse, and non-empty") ;
    }

    /* get sparse matrix A */
    Ap = (Long *) mxGetJc (pargin [0]) ;
    Ai = (Long *) mxGetIr (pargin [0]) ;

    /* get output array */
    Match = mxMalloc (nrow * sizeof (Long)) ;

    /* get workspace of size 5n (recursive version needs only 2n) */
    Work = mxMalloc (5*ncol * sizeof (Long)) ;

    maxwork = 0 ;
    if (nargin > 1)
    {
        maxwork = mxGetScalar (pargin [1]) ;
    }
    work = 0 ;

    /* ---------------------------------------------------------------------- */
    /* perform the maximum transversal */
    /* ---------------------------------------------------------------------- */

    nmatch = btf_l_maxtrans (nrow, ncol, Ap, Ai, maxwork, &work, Match, Work) ;

    /* ---------------------------------------------------------------------- */
    /* create outputs and free workspace */
    /* ---------------------------------------------------------------------- */

    pargout [0] = mxCreateDoubleMatrix (1, nrow, mxREAL) ;
    Matchx = mxGetPr (pargout [0]) ;
    for (i = 0 ; i < nrow ; i++)
    {
        Matchx [i] = Match [i] + 1 ;    /* convert to 1-based */
    }

    if (nargout > 1)
    {
        pargout [1] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
        w = mxGetPr (pargout [1]) ;
        w [0] = work ;
    }

    mxFree (Work) ;
    mxFree (Match) ;
}
