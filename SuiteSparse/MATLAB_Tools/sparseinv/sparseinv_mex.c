#include "sparseinv.h"

/*
    Z = sparseinv_mex (L, d, UT, Zpattern)

    Given (L+I)*D*(UT+I)' = A, and the symbolic Cholesky factorization of A+A',
    compute the sparse inverse subset, Z.  UT is stored by column, so U = UT'
    is implicitly stored by row, and is implicitly unit-diagonal.  The diagonal
    is not present.  L is stored by column, and is also unit-diagonal.  The
    diagonal is not present in L, either.  d is a full vector of size n.

    This mexFunction is only meant to be called from the sparsinv m-file.
    An optional 2nd output argument returns the flop count.

    Copyright 2011, Timothy A. Davis, http://www.suitesparse.com
*/

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    Int *Zp, *Zi, *Lp, *Li, *Up, *Uj, *Zpatp, *Zpati, n, *Zdiagp, *Lmunch,
        znz, j, p, flops ;
    double *Zx, *Lx, *Ux, *z, *d ;

    /* check inputs */
    if (nargin != 4 || nargout > 2)
    {
        mexErrMsgTxt ("Usage: [Z flops] = sparseinv_mex (L, d, UT, Zpattern)") ;
    }
    n = mxGetN (pargin [0]) ;
    for (j = 0 ; j < 4 ; j++)
    {
        if (j == 1) continue ;
        if (!mxIsSparse (pargin [j]) || mxIsComplex (pargin [j]) ||
            (mxGetM (pargin [j]) != n) || (mxGetN (pargin [j]) != n))
        {
            mexErrMsgTxt ("Matrices must be sparse, real, square, & same size");
        }
    }
    if (mxIsSparse (pargin [1]) || mxIsComplex (pargin [1]) ||
        (mxGetM (pargin [1]) != n) || (mxGetN (pargin [1]) != 1))
    {
        mexErrMsgTxt ("Input d must be a real dense vector of right size") ;
    }

    /* get inputs */
    Lp = (Int *) mxGetJc (pargin [0]) ;
    Li = (Int *) mxGetIr (pargin [0]) ;
    Lx = mxGetPr (pargin [0]) ;

    d = mxGetPr (pargin [1]) ;

    Up = (Int *) mxGetJc (pargin [2]) ;
    Uj = (Int *) mxGetIr (pargin [2]) ;
    Ux = mxGetPr (pargin [2]) ;

    Zpatp = (Int *) mxGetJc (pargin [3]) ;
    Zpati = (Int *) mxGetIr (pargin [3]) ;
    znz = Zpatp [n] ;

    /* create output */
    pargout [0] = mxCreateSparse (n, n, znz, mxREAL) ;
    Zx = mxGetPr (pargout [0]) ;

    /* get workspace */
    z = mxCalloc (n, sizeof (double)) ;
    Zdiagp = mxMalloc (n * sizeof (Int)) ;
    Lmunch = mxMalloc (n * sizeof (Int)) ;

    /* do the work */
    flops = sparseinv (n, Lp, Li, Lx, d, Up, Uj, Ux, Zpatp, Zpati, Zx,
        z, Zdiagp, Lmunch) ;

    /* free workspace */
    mxFree (z) ;
    mxFree (Zdiagp) ;
    mxFree (Lmunch) ;

    /* return results to MATLAB */
    Zp = (Int *) mxGetJc (pargout [0]) ;
    Zi = (Int *) mxGetIr (pargout [0]) ;
    for (j = 0 ; j <= n ; j++)
    {
        Zp [j] = Zpatp [j] ;
    }
    for (p = 0 ; p < znz ; p++)
    {
        Zi [p] = Zpati [p] ;
    }

    /* drop explicit zeros from the output Z matrix */
    znz = 0 ;
    for (j = 0 ; j < n ; j++)
    {
        p = Zp [j] ;                        /* get current location of col j */
        Zp [j] = znz ;                      /* record new location of col j */
        for ( ; p < Zp [j+1] ; p++)
        {
            if (Zx [p] != 0)
            {
                Zx [znz] = Zx [p] ;         /* keep Z(i,j) */
                Zi [znz++] = Zi [p] ;
            }
        }
    }
    Zp [n] = znz ;                          /* finalize Z */

    if (nargout > 1)
    {
        pargout [1] = mxCreateDoubleScalar ((double) flops) ;
    }
}
