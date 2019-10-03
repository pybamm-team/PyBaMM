#include "sparseinv.h"

/* sparsinv: computes the sparse inverse subset, using Takahashi's equations.

   On input, the pattern of Z must be equal to the symbolic Cholesky
   factorization of A+A', where A=(L+I)*(U+I).  The pattern of L+U must be a
   subset of Z.  Z must have zero-free diagonal.  These conditions are
   difficult to check, so they are assumed to hold.  Results will be completely
   wrong if the conditions do not hold.

   This function performs the same amount of work as the initial LU
   factorization, assuming that the pattern of P*A*Q is symmetric.  For large
   matrices, this function can take a lot more time than LU in MATLAB, even if
   P*A*Q is symmetric.  This is because LU is a multifrontal method, whereas
   this sparseinv function is based on gather/scatter operations.

   The basic integer type is an Int, or ptrdiff_t, which is 32 bits on a 32
   bits and 64 bits on a 64 bit system.  The function returns the flop count as
   an Int.  This will not overflow on a 64 bit system but might on a 32 bit.
   The total work is flops + O(n + nnz(Z)).  Since flops > n and flops > nnz(Z),
   this is O(flops).

   Copyright 2011, Timothy A. Davis, http://www.suitesparse.com
*/

Int sparseinv       /* returns -1 on error, or flop count if OK */
(
    /* inputs, not modified on output: */
    Int n,          /* L, U, D, and Z are n-by-n */

    Int *Lp,        /* L is sparse, lower triangular, stored by column */
    Int *Li,        /* the row indices of L must be sorted */
    double *Lx,     /* diagonal of L, if present, is ignored */

    double *d,      /* diagonal of D, of size n */

    Int *Up,        /* U is sparse, upper triangular, stored by row */
    Int *Uj,        /* the column indices of U need not be sorted */
    double *Ux,     /* diagonal of U, if present, is ignored */

    Int *Zp,        /* Z is sparse, stored by column */
    Int *Zi,        /* the row indices of Z must be sorted */

    /* output, not defined on input: */ 
    double *Zx,

    /* workspace: */
    double *z,      /* size n, zero on input, restored as such on output */
    Int *Zdiagp,    /* size n */
    Int *Lmunch     /* size n */
)
{
    double ljk, zkj ;
    Int j, i, k, p, znz, pdiag, up, zp, flops = n ;

    /* ---------------------------------------------------------------------- */
    /* initializations */
    /* ---------------------------------------------------------------------- */

    /* clear the numerical values of Z */
    znz = Zp [n] ;
    for (p = 0 ; p < znz ; p++)
    {
        Zx [p] = 0 ;
    }

    /* find the diagonal of Z and initialize it */
    for (j = 0 ; j < n ; j++)
    {
        pdiag = -1 ;
        for (p = Zp [j] ; p < Zp [j+1] && pdiag == -1 ; p++)
        {
            if (Zi [p] == j)
            {
                pdiag = p ;
                Zx [p] = 1 / d [j] ;
            }
        }
        Zdiagp [j] = pdiag ;
        if (pdiag == -1) return (-1) ;  /* Z must have a zero-free diagonal */
    }

    /* Lmunch [k] points to the last entry in column k of L */
    for (k = 0 ; k < n ; k++)
    {
        Lmunch [k] = Lp [k+1] - 1 ;
    }

    /* ---------------------------------------------------------------------- */
    /* compute the sparse inverse subset */
    /* ---------------------------------------------------------------------- */

    for (j = n-1 ; j >= 0 ; j--)
    {

        /* ------------------------------------------------------------------ */
        /* scatter Z (:,j) into z workspace */
        /* ------------------------------------------------------------------ */

        /* only the lower triangular part is needed, since the upper triangular
           part is all zero */
        for (p = Zdiagp [j] ; p < Zp [j+1] ; p++)
        {
            z [Zi [p]] = Zx [p] ;
        }

        /* ------------------------------------------------------------------ */
        /* compute the strictly upper triangular part of Z (:,j) */
        /* ------------------------------------------------------------------ */

        /* for k = (j-1):-1:1 but only for the entries Z(k,j) */
        for (p = Zdiagp [j]-1 ; p >= Zp [j] ; p--)
        {
            /* Z (k,j) = - U (k,k+1:n) * Z (k+1:n,j) */
            k = Zi [p] ;
            zkj = 0 ;
            flops += (Up [k+1] - Up [k]) ;
            for (up = Up [k] ; up < Up [k+1] ; up++)
            {
                /* skip the diagonal of U, if present */
                i = Uj [up] ;
                if (i > k)
                {
                    zkj -= Ux [up] * z [i] ;
                }
            }
            z [k] = zkj ;
        }

        /* ------------------------------------------------------------------ */
        /* left-looking update to lower triangular part of Z */
        /* ------------------------------------------------------------------ */

        /* for k = (j-1):-1:1 but only for the entries Z(k,j) */
        for (p = Zdiagp [j]-1 ; p >= Zp [j] ; p--)
        {
            k = Zi [p] ;

            /* ljk = L (j,k) */
            if (Lmunch [k] < Lp [k] || Li [Lmunch [k]] != j)
            {
                /* L (j,k) is zero, so there is no work to do */
                continue ;
            }
            ljk = Lx [Lmunch [k]--] ;

            /* Z (k+1:n,k) = Z (k+1:n,k) - Z (k+1:n,j) * L (j,k) */
            flops += (Zp [k+1] - Zdiagp [k]) ;
            for (zp = Zdiagp [k] ; zp < Zp [k+1] ; zp++)
            {
                Zx [zp] -= z [Zi [zp]] * ljk ;
            }
        }

        /* ------------------------------------------------------------------ */
        /* gather Z (:,j) back from z workspace */
        /* ------------------------------------------------------------------ */

        for (p = Zp [j] ; p < Zp [j+1] ; p++)
        {
            i = Zi [p] ;
            Zx [p] = z [i] ;
            z [i] = 0 ;
        }
    }
    return (flops) ;
}
