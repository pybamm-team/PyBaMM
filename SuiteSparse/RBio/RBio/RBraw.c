/* ========================================================================== */
/* === RBio/RBio/RBraw.c: MATLAB mexFunction to read raw contents of RB file  */
/* ========================================================================== */

/* Copyright 2009, Timothy A. Davis, All Rights Reserved.
   Refer to RBio/Doc/license.txt for the RBio license. */

/*
c-----------------------------------------------------------------------
c RBraw mexFunction: read the raw contents of a Rutherford/Boeing file
c-----------------------------------------------------------------------
c
c   [mtype Ap Ai Ax title key nrow] = RBraw (filename)
c
c   mtype: Rutherford/Boeing matrix type (psa, rua, rsa, rse, ...)
c   Ap: column pointers (1-based)
c   Ai: row indices (1-based)
c   Ax: numerical values (real, complex, or integer).  Empty for p*a
c       matrices.  A complex matrix is read in as a single double array
c       Ax, where the kth entry has real value Ax(2*k-1) and imaginary
c       value Ax(2*k).
c   title: a string containing the title from the first line of the file
c   key: a string containing the 8-char key, from 1st line of the file
c   nrow: number of rows in the matrix
c
c This function works for both assembled and unassembled (finite-
c element) matrices.  It is also useful for checking the contents of a
c Rutherford/Boeing file in detail, in case the file has invalid column
c pointers, unsorted columns, duplicate entries, entries in the upper
c triangular part of the file for a symmetric matrix, etc.
c
c Example:
c
c   load west0479
c   RBwrite ('mywest', west0479, [ ], 'My west0479 file', 'west0479') ;
c   [mtype Ap Ai Ax title key nrow] = RBraw ('mywest') ;
c
c See also RBfix, RBread, RBreade.
c-----------------------------------------------------------------------
*/

#include "RBio.h"
#define LEN 1024
#define Long SuiteSparse_long

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    Long *Ap, *Ai ;
    double *Ax ;
    Long p, j, nrow, ncol, mkind, skind, xsize, status, nelnz, fem,
        iclass, nnz ;
    mwSize dims [2] = { 0, 1 } ;
    char filename [LEN+1], title [73], key [9], mtype [4] ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin != 1 || nargout > 7 || !mxIsChar (pargin [0]))
    {
        mexErrMsgTxt ("Usage:  = RBread (filename)") ;
        mexErrMsgTxt
            ("Usage: [mtype Ap Ai Ax title key nrow] = RBraw (filename)") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get filename */
    /* ---------------------------------------------------------------------- */

    if (mxGetString (pargin [0], filename, LEN) != 0)
    {
        mexErrMsgTxt ("filename too long") ;
    }

    /* ---------------------------------------------------------------------- */
    /* read the raw contents of the file */
    /* ---------------------------------------------------------------------- */

    status = RBreadraw (filename, title, key, mtype, &nrow, &ncol, &nnz, &nelnz,
        &mkind, &skind, &fem, &xsize, &Ap, &Ai, &Ax) ;

    if (status != RBIO_OK)
    {
        mexErrMsgTxt ("error reading file") ;
    }

    /* ---------------------------------------------------------------------- */
    /* convert back to 1-based */
    /* ---------------------------------------------------------------------- */

    for (j = 0 ; j <= ncol ; j++) Ap [j]++ ;
    for (p = 0 ; p <  nnz  ; p++) Ai [p]++ ;

    /* ---------------------------------------------------------------------- */
    /* return results to MATLAB */
    /* ---------------------------------------------------------------------- */

    iclass = (sizeof (Long) == 4) ? mxINT32_CLASS : mxINT64_CLASS ;

    /* return mtype */
    pargout [0] = mxCreateString (mtype) ;

    if (nargout > 1)
    {
        /* return Ap, of size ncol+1 */
        pargout [1] = mxCreateNumericArray (2, dims, iclass, mxREAL) ;
        mxFree (mxGetData (pargout [1])) ;
        mxSetData (pargout [1], Ap) ;
        mxSetM (pargout [1], ncol + 1) ;
    }

    if (nargout > 2)
    {
        /* return Ai, of size nnz */
        pargout [2] = mxCreateNumericArray (2, dims, iclass, mxREAL) ;
        mxFree (mxGetData (pargout [2])) ;
        mxSetData (pargout [2], Ai) ;
        mxSetM (pargout [2], nnz) ;
    }

    if (nargout > 3)
    {
        /* return Ax, of size xsize */
        pargout [3] = mxCreateNumericArray (2, dims, mxDOUBLE_CLASS, mxREAL) ;
        mxFree (mxGetData (pargout [3])) ;
        mxSetData (pargout [3], Ax) ;
        mxSetM (pargout [3], xsize) ;
    }

    if (nargout > 4)
    {
        /* return title */
        pargout [4] = mxCreateString (title) ;
    }

    if (nargout > 5)
    {
        /* return key */
        pargout [5] = mxCreateString (key) ;
    }

    if (nargout > 6)
    {
        /* return nrow */
        pargout [6] = mxCreateDoubleScalar ((double) nrow) ;
    }
}
