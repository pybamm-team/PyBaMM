/* ========================================================================== */
/* === RBio/RBio/RBtype.c: MATLAB mexFunction to find matrix type =========== */
/* ========================================================================== */

/* Copyright 2009, Timothy A. Davis, All Rights Reserved.
   Refer to RBio/Doc/license.txt for the RBio license. */

/*
-----------------------------------------------------------------------
 RBtype mexFunction:
-----------------------------------------------------------------------

   [mtype mkind skind] = RBtype (A)

   A: a sparse matrix.   Determines the Rutherford/Boeing type of the
   matrix.  Very little memory is used (just size(A,2) integer
   workspace), so this can succeed where a test such as nnz(A-A')==0
   will fail.

       mkind:  r: (0), A is real, and not binary
               p: (1), A is binary
               c: (2), A is complex
               i: (3), A is integer

       skind:  r: (-1), A is rectangular
               u: (0), A is unsymmetric (not S, H, Z, below)
               s: (1), A is symmetric (nnz(A-A.') is 0)
               h: (2), A is Hermitian (nnz(A-A') is 0)
               z: (3), A is skew symmetric (nnz(A+A.') is 0)

   mtype is a 3-character string, where mtype(1) is the mkind
   ('r', 'p', 'c', or 'i').  mtype(2) is the skind ('r', 'u', 's', 'h',
   or 'z'), and mtype(3) is always 'a'.
-----------------------------------------------------------------------
*/

#include "RBio.h"
#define TRUE (1)
#define Long SuiteSparse_long

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double xmin, xmax ;
    Long *Ap, *Ai ;
    double *Ax, *Az ;
    Long nrow, ncol, nnz, mkind, skind, mkind_in ;
    char mtype [4] ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin != 1 || nargout > 3)
    {
        mexErrMsgTxt ("Usage: [mtype mkind skind] = RBtype (A)") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get A */
    /* ---------------------------------------------------------------------- */

    if (!mxIsClass (pargin [0], "double") || !mxIsSparse (pargin [0]))
    {
        mexErrMsgTxt ("A must be sparse and double") ;
    }

    Ap = (Long *) mxGetJc (pargin [0]) ;
    Ai = (Long *) mxGetIr (pargin [0]) ;
    Ax = mxGetPr (pargin [0]) ;
    Az = mxGetPi (pargin [0]) ;
    nrow = mxGetM (pargin [0]) ;
    ncol = mxGetN (pargin [0]) ;

    /* ---------------------------------------------------------------------- */
    /* determine the mtype of A */
    /* ---------------------------------------------------------------------- */

    mkind_in = mxIsComplex (pargin [0]) ? 2 : 0 ;

    RBkind (nrow, ncol, Ap, Ai, Ax, Az, mkind_in, &mkind, &skind, mtype,
        &xmin, &xmax, NULL) ;

    /* ---------------------------------------------------------------------- */
    /* return the result */
    /* ---------------------------------------------------------------------- */

    pargout [0] = mxCreateString (mtype) ;
    if (nargout >= 2)
    {
        pargout [1] = mxCreateDoubleScalar ((double) mkind) ;
    }
    if (nargout >= 3)
    {
        pargout [2] = mxCreateDoubleScalar ((double) skind) ;
    }
}
