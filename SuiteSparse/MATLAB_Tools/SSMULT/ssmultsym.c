/* ========================================================================== */
/* === ssmultsym ============================================================ */
/* ========================================================================== */

/* s = ssmultsym (A,B) computes nnz(A*B), and the flops and memory required to
 * compute it, where A and B are both sparse.  Either A or B, or both, can be
 * complex.  Memory usage includes C itself, and workspace.  If C is m-by-n,
 * ssmultsym requires only 4*m bytes for 32-bit MATLAB, 8*m for 64-bit MATLAB.
 *
 * Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com
 */

#include "ssmult.h"

/* -------------------------------------------------------------------------- */
/* ssmultsym mexFunction */
/* -------------------------------------------------------------------------- */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double cnz, flops, mem, e, multadds ;
    Int *Ap, *Ai, *Bp, *Bi, *Flag ;
    Int Anrow, Ancol, Bnrow, Bncol, i, j, k, pb, pa, pbend, paend, mark,
        A_is_complex, B_is_complex, C_is_complex, pbstart, cjnz ;
    static const char *snames [ ] =
    {
        "nz",           /* # of nonzeros in C=A*B */
        "flops",        /* flop count required to compute C=A*B */
        "memory"        /* memory requirement in bytes */
    } ;

    /* ---------------------------------------------------------------------- */
    /* get inputs and workspace */
    /* ---------------------------------------------------------------------- */

    if (nargin != 2 || nargout > 1)
    {
        mexErrMsgTxt ("Usage: s = ssmultsym (A,B)") ;
    }

    Ap = (Int *) mxGetJc (pargin [0]) ;
    Ai = (Int *) mxGetIr (pargin [0]) ;
    Anrow = mxGetM (pargin [0]) ;
    Ancol = mxGetN (pargin [0]) ;
    A_is_complex = mxIsComplex (pargin [0]) ;

    Bp = (Int *) mxGetJc (pargin [1]) ;
    Bi = (Int *) mxGetIr (pargin [1]) ;
    Bnrow = mxGetM (pargin [1]) ;
    Bncol = mxGetN (pargin [1]) ;
    B_is_complex = mxIsComplex (pargin [1]) ;

    if (Ancol != Bnrow || !mxIsSparse (pargin [0]) || !mxIsSparse (pargin [1]))
    {
        mexErrMsgTxt ("wrong dimensions, or A and B not sparse") ;
    }

    Flag = mxCalloc (Anrow, sizeof (Int)) ;     /* workspace */

    /* ---------------------------------------------------------------------- */
    /* compute # of nonzeros in result, flop count, and memory */
    /* ---------------------------------------------------------------------- */

    pb = 0 ;
    cnz = 0 ;
    multadds = 0 ;
    for (j = 0 ; j < Bncol ; j++)       /* compute C(:,j) */
    {
        mark = j+1 ;
        pbstart = Bp [j] ;
        pbend = Bp [j+1] ;
        cjnz = 0 ;
        for ( ; pb < pbend ; pb++)
        {
            k = Bi [pb] ;               /* nonzero entry B(k,j) */
            pa = Ap [k] ;
            paend = Ap [k+1] ;
            multadds += (paend - pa) ;  /* one mult-add per entry in A(:,k) */
            if (cjnz == Anrow)
            {
                /* C(:,j) is already completely dense; no need to scan A.
                 * Continue scanning B(:,j) to compute flop count. */
                continue ;
            }
            for ( ; pa < paend ; pa++)
            {
                i = Ai [pa] ;           /* nonzero entry A(i,k) */
                if (Flag [i] != mark)
                {
                    /* C(i,j) is a new nonzero */
                    Flag [i] = mark ;   /* mark i as appearing in C(:,j) */
                    cjnz++ ;
                }
            }
        }
        cnz += cjnz ;
    }

    C_is_complex = A_is_complex || B_is_complex ;
    e = (C_is_complex ? 2 : 1) ;

    mem =
        Anrow * sizeof (Int)            /* Flag */
        + e * Anrow * sizeof (double)   /* W */
        + (Bncol+1) * sizeof (Int)      /* Cp */
        + cnz * sizeof (Int)            /* Ci */
        + e * cnz * sizeof (double) ;   /* Cx and Cx */

    if (A_is_complex)
    {
        if (B_is_complex)
        {
            /* all of C, A, and B are complex */
            flops = 8 * multadds - 2 * cnz ;
        }
        else
        {
            /* C and A are complex, B is real */
            flops = 4 * multadds - 2 * cnz ;
        }
    }
    else
    {
        if (B_is_complex)
        {
            /* C and B are complex, A is real */
            flops = 4 * multadds - 2 * cnz ;
        }
        else
        {
            /* all of C, A, and B are real */
            flops = 2 * multadds - cnz ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* return result */
    /* ---------------------------------------------------------------------- */

    pargout [0] = mxCreateStructMatrix (1, 1, 3, snames) ;
    mxSetFieldByNumber (pargout [0], 0, 0, mxCreateDoubleScalar (cnz)) ;
    mxSetFieldByNumber (pargout [0], 0, 1, mxCreateDoubleScalar (flops)) ;
    mxSetFieldByNumber (pargout [0], 0, 2, mxCreateDoubleScalar (mem)) ;
}
