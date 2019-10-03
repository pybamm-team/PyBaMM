/* ========================================================================== */
/* === RBio/RBio/RBread.c: MATLAB mexFunction for reading R/B file ========== */
/* ========================================================================== */

/* Copyright 2009, Timothy A. Davis, All Rights Reserved.
   Refer to RBio/Doc/license.txt for the RBio license. */

/*
   [A Z title key mtype] = RBread (filename)

   A: a sparse matrix (no explicit zero entries)
   Z: binary pattern of explicit zero entries in Rutherford/Boeing file.
        This always has the same size as A, and is always sparse.

   title: the 72-character title string in the file
   key: the 8-character matrix name in the file
   mtype: see RBwrite.m for a description.
*/

#include "RBio.h"
#define LEN 1024
#define TRUE (1)
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define Long SuiteSparse_long

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    Long *Ap, *Ai, *Zp, *Zi ;
    double *Ax, *Az, *Zx ;
    Long p, j, build_upper, zero_handling, nrow, ncol, mkind, skind, asize, znz,
        status ;
    char filename [LEN+1], title [73], key [9], mtype [4] ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin != 1 || nargout > 5 || !mxIsChar (pargin [0]))
    {
        mexErrMsgTxt ("Usage: [A Z title key mtype] = RBread (filename)") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get filename */
    /* ---------------------------------------------------------------------- */

    if (mxGetString (pargin [0], filename, LEN) != 0)
    {
        mexErrMsgTxt ("filename too long") ;
    }

    /* ---------------------------------------------------------------------- */
    /* read the matrix */
    /* ---------------------------------------------------------------------- */

    build_upper = TRUE ;                    /* always build upper tri. part */
    zero_handling = (nargout > 1) ? 2 : 1 ; /* prune or extract zeros */

    status = RBread (filename, build_upper, zero_handling, title, key, mtype,
        &nrow, &ncol, &mkind, &skind, &asize, &znz,
        &Ap, &Ai, &Ax, &Az, &Zp, &Zi) ;

    if (status != RBIO_OK)
    {
        RBerror (status) ;
        mexErrMsgTxt ("error reading file") ;
    }

    /* ---------------------------------------------------------------------- */
    /* return A to MATLAB */
    /* ---------------------------------------------------------------------- */

    pargout [0] = mxCreateSparse (0, 0, 0, (mkind == 2) ? mxCOMPLEX : mxREAL) ;
    mxFree (mxGetJc (pargout [0])) ;
    mxFree (mxGetIr (pargout [0])) ;
    mxFree (mxGetPr (pargout [0])) ;
    if (mkind == 2) mxFree (mxGetPi (pargout [0])) ;
    mxSetM (pargout [0], nrow) ;
    mxSetN (pargout [0], ncol) ;
    mxSetNzmax (pargout [0], asize) ;
    mxSetJc (pargout [0], (mwIndex *) Ap) ;
    mxSetIr (pargout [0], (mwIndex *) Ai) ;
    mxSetPr (pargout [0], Ax) ;
    if (mkind == 2) mxSetPi (pargout [0], Az) ;

    /* ---------------------------------------------------------------------- */
    /* return Z to MATLAB */
    /* ---------------------------------------------------------------------- */

    if (nargout > 1)
    {
        Zx = (double *) SuiteSparse_malloc (znz, sizeof (double)) ;
        for (p = 0 ; p < znz ; p++)
        {
            Zx [p] = 1 ;
        }
        pargout [1] = mxCreateSparse (0, 0, 0, mxREAL) ;
        mxFree (mxGetJc (pargout [1])) ;
        mxFree (mxGetIr (pargout [1])) ;
        mxFree (mxGetPr (pargout [1])) ;
        mxSetM (pargout [1], nrow) ;
        mxSetN (pargout [1], ncol) ;
        mxSetNzmax (pargout [1], MAX (znz,1)) ;
        mxSetJc (pargout [1], (mwIndex *) Zp) ;
        mxSetIr (pargout [1], (mwIndex *) Zi) ;
        mxSetPr (pargout [1], Zx) ;
    }

    /* ---------------------------------------------------------------------- */
    /* return title */
    /* ---------------------------------------------------------------------- */

    if (nargout > 2)
    {
        pargout [2] = mxCreateString (title) ;
    }

    /* ---------------------------------------------------------------------- */
    /* return key */
    /* ---------------------------------------------------------------------- */

    if (nargout > 3)
    {
        pargout [3] = mxCreateString (key) ;
    }

    /* ---------------------------------------------------------------------- */
    /* return mtype */
    /* ---------------------------------------------------------------------- */

    if (nargout > 4)
    {
        pargout [4] = mxCreateString (mtype) ;
    }
}
