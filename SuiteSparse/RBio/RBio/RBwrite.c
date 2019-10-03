/* ========================================================================== */
/* === RBio/RBio/RBwrite.c: MATLAB mexFunction to write R/B file ============ */
/* ========================================================================== */

/* Copyright 2009, Timothy A. Davis, All Rights Reserved.
   Refer to RBio/Doc/license.txt for the RBio license. */

/*
function mtype = RBwrite (filename, A, Z, title, key)                       %#ok
%RBWRITE write a sparse matrix to a Rutherford/Boeing file
% Usage:
%   mtype = RBwrite (filename, A, Z, title, key)
%
%   filename: name of the file to create
%   A: a sparse matrix
%   Z: binary pattern of explicit zero entries to include in the
%       Rutherford/Boeing file.  This always has the same size as A, and is
%       always sparse.  Not used if empty ([ ]), or if nnz(Z) is 0.
%   title: title for 1st line of  Rutherford/Boeing file, up to 72 characters
%   key: matrix key, up to 8 characters, for 1st line of the file
%
% Z is optional.  RBwrite (filename, A) uses a default title and key, and does
% not include any explicit zeros.  RBwrite (filname, A, 'title...', 'key') uses
% the given title and key.  A must be sparse.  Z must be empty, or sparse.
%
% mtype is a 3-character string with the Rutherford/Boeing type used:
%   mtype(1):  r: real, p: pattern, c: complex, i: integer
%   mtype(2):  r: rectangular, u: unsymmetric, s: symmetric,
%              h: Hermitian, Z: skew symmetric
%   mtype(3):  a: assembled matrix, e: finite-element (not used by RBwrite)
%
% Example:
%   load west0479
%   C = west0479 ;
%   RBwrite ('west0479', C, 'WEST0479 chemical eng. problem', 'west0479')
%   A = RBread ('west0479') ;
%   norm (A-C,1)
%
% See also RBread, RBtype.
*/

#include "RBio.h"
#define LEN 1024
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
    Long nrow, ncol, ititle, zrow, zcol, i, mkind ;
    Long *Ap, *Ai, *Zp, *Zi, *w, *cp ;
    double *Ax, *Az ;
    char filename [LEN+1], title [73], key [9], mtype [4] ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 2 || nargin > 5 || nargout > 2 || !mxIsChar (pargin [0]))
    {
        mexErrMsgTxt ("[m s] = RBwrite (filename, A, Z, title, key)") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get filename */
    /* ---------------------------------------------------------------------- */

    if (mxGetString (pargin [0], filename, LEN) != 0)
    {
        mexErrMsgTxt ("filename too long") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get A */
    /* ---------------------------------------------------------------------- */

    if (!mxIsClass (pargin [1], "double") || !mxIsSparse (pargin [1]))
    {
        mexErrMsgTxt ("A must be sparse and double") ;
    }

    Ap = (Long *) mxGetJc (pargin [1]) ;
    Ai = (Long *) mxGetIr (pargin [1]) ;
    Ax = mxGetPr (pargin [1]) ;
    nrow = mxGetM (pargin [1]) ;
    ncol = mxGetN (pargin [1]) ;

    if (mxIsComplex (pargin [1]))
    {
        mkind = 2 ;
        Az = mxGetPi (pargin [1]) ;
    }
    else
    {
        mkind = 0 ;
        Az = NULL ;
    }

    /* ---------------------------------------------------------------------- */
    /* get title and key */
    /* ---------------------------------------------------------------------- */

    title [0] = '\0' ;
    key [0] = '\0' ;
    ititle = 99 ;
    for (i = 2 ; i < nargin ; i++)
    {
        if (mxIsChar (pargin [i]))
        {
            if (ititle == 99)
            {
                /* get the title, up to 72 characters long */
                mxGetString (pargin [i], title, 72) ;
                ititle = i ;
            }
            else
            {
                /* get the key, up to 8 characters long */
                mxGetString (pargin [i], key, 8) ;
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* get Z, if present */
    /* ---------------------------------------------------------------------- */

    Zp = NULL ;
    Zi = NULL ;

    if (nargin >= 3 && ititle > 2)
    {
        zrow = mxGetM (pargin [2]) ;
        zcol = mxGetN (pargin [2]) ;
        if (zrow > 0 && zcol > 0)
        {
            if (!mxIsClass (pargin [2], "double") || !mxIsSparse (pargin [2]) ||
                mxIsComplex (pargin [2]) || zrow != nrow || zcol != ncol)
            {
                mexErrMsgTxt
                    ("Z must be sparse, double, real, and same size as A") ;
            }
            Zp = (Long *) mxGetJc (pargin [2]) ;
            Zi = (Long *) mxGetIr (pargin [2]) ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* write the matrix to the file */
    /* ---------------------------------------------------------------------- */

    RBwrite (filename, title, key, nrow, ncol, Ap, Ai, Ax, Az, Zp, Zi,
        mkind, mtype) ;

    /* ---------------------------------------------------------------------- */
    /* return mtype */
    /* ---------------------------------------------------------------------- */

    if (nargout > 0)
    {
        pargout [0] = mxCreateString (mtype) ;
    }
}
