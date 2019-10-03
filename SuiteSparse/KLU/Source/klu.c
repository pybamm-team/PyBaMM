/* ========================================================================== */
/* === klu ================================================================== */
/* ========================================================================== */

/* KLU: factorizes P*A into L*U, using the Gilbert-Peierls algorithm [1], with
 * optional symmetric pruning by Eisenstat and Liu [2].  The code is by Tim
 * Davis.  This algorithm is what appears as the default sparse LU routine in
 * MATLAB version 6.0, and still appears in MATLAB 6.5 as [L,U,P] = lu (A).
 * Note that no column ordering is provided (see COLAMD or AMD for suitable
 * orderings).  SuperLU is based on this algorithm, except that it adds the
 * use of dense matrix operations on "supernodes" (adjacent columns with
 * identical).  This code doesn't use supernodes, thus its name ("Kent" LU,
 * as in "Clark Kent", in contrast with Super-LU...).  This algorithm is slower
 * than SuperLU and UMFPACK for large matrices with lots of nonzeros in their
 * factors (such as for most finite-element problems).  However, for matrices
 * with very sparse LU factors, this algorithm is typically faster than both
 * SuperLU and UMFPACK, since in this case there is little chance to exploit
 * dense matrix kernels (the BLAS).
 *
 * Only one block of A is factorized, in the BTF form.  The input n is the
 * size of the block; k1 is the first row and column in the block.
 *
 * NOTE: no error checking is done on the inputs.  This version is not meant to
 * be called directly by the user.  Use klu_factor instead.
 *
 * No fill-reducing ordering is provided.  The ordering quality of
 * klu_kernel_factor is the responsibility of the caller.  The input A must
 * pre-permuted to reduce fill-in, or fill-reducing input permutation Q must
 * be provided.
 *
 * The input matrix A must be in compressed-column form, with either sorted
 * or unsorted row indices.  Row indices for column j of A is in
 * Ai [Ap [j] ... Ap [j+1]-1] and the same range of indices in Ax holds the
 * numerical values.  No duplicate entries are allowed.
 *
 * Copyright 2004-2009, Tim Davis.  All rights reserved.  See the README
 * file for details on permitted use.  Note that no code from The MathWorks,
 * Inc, or from SuperLU, or from any other source appears here.  The code is
 * written from scratch, from the algorithmic description in Gilbert & Peierls'
 * and Eisenstat & Liu's journal papers [1,2].
 *
 * If an input permutation Q is provided, the factorization L*U = A (P,Q)
 * is computed, where P is determined by partial pivoting, and Q is the input
 * ordering.  If the pivot tolerance is less than 1, the "diagonal" entry that
 * KLU attempts to choose is the diagonal of A (Q,Q).  In other words, the
 * input permutation is applied symmetrically to the input matrix.  The output
 * permutation P includes both the partial pivoting ordering and the input
 * permutation.  If Q is NULL, then it is assumed to be the identity
 * permutation.  Q is not modified.
 *
 * [1] Gilbert, J. R. and Peierls, T., "Sparse Partial Pivoting in Time
 *      Proportional to Arithmetic Operations," SIAM J. Sci. Stat. Comp.,
 *      vol 9, pp.  862-874, 1988.
 * [2] Eisenstat, S. C. and Liu, J. W. H., "Exploiting Structural Symmetry in
 *      Unsymmetric Sparse Symbolic Factorization," SIAM J. Matrix Analysis &
 *      Applic., vol 13, pp.  202-211, 1992.
 */

/* ========================================================================== */

#include "klu_internal.h"

size_t KLU_kernel_factor            /* 0 if failure, size of LU if OK */
(
    /* inputs, not modified */
    Int n,          /* A is n-by-n. n must be > 0. */
    Int Ap [ ],     /* size n+1, column pointers for A */
    Int Ai [ ],     /* size nz = Ap [n], row indices for A */
    Entry Ax [ ],   /* size nz, values of A */
    Int Q [ ],      /* size n, optional column permutation */
    double Lsize,   /* estimate of number of nonzeros in L */

    /* outputs, not defined on input */
    Unit **p_LU,        /* row indices and values of L and U */
    Entry Udiag [ ],    /* size n, diagonal of U */
    Int Llen [ ],       /* size n, column length of L */
    Int Ulen [ ],       /* size n, column length of U */
    Int Lip [ ],        /* size n, column pointers for L */
    Int Uip [ ],        /* size n, column pointers for U */
    Int P [ ],          /* row permutation, size n */
    Int *lnz,           /* size of L */
    Int *unz,           /* size of U */

    /* workspace, undefined on input */
    Entry *X,       /* size n double's, zero on output */
    Int *Work,      /* size 5n Int's */

    /* inputs, not modified on output */
    Int k1,             /* the block of A is from k1 to k2-1 */
    Int PSinv [ ],      /* inverse of P from symbolic factorization */
    double Rs [ ],      /* scale factors for A */

    /* inputs, modified on output */
    Int Offp [ ],   /* off-diagonal matrix (modified by this routine) */
    Int Offi [ ],
    Entry Offx [ ],
    /* --------------- */
    KLU_common *Common
)
{
    double maxlnz, dunits ;
    Unit *LU ;
    Int *Pinv, *Lpend, *Stack, *Flag, *Ap_pos, *W ;
    Int lsize, usize, anz, ok ;
    size_t lusize ;
    ASSERT (Common != NULL) ;

    /* ---------------------------------------------------------------------- */
    /* get control parameters, or use defaults */
    /* ---------------------------------------------------------------------- */

    n = MAX (1, n) ;
    anz = Ap [n+k1] - Ap [k1] ;

    if (Lsize <= 0)
    {
        Lsize = -Lsize ;
        Lsize = MAX (Lsize, 1.0) ;
        lsize = Lsize * anz + n ;
    }
    else
    {
        lsize = Lsize ;
    }

    usize = lsize ;

    lsize  = MAX (n+1, lsize) ;
    usize  = MAX (n+1, usize) ;

    maxlnz = (((double) n) * ((double) n) + ((double) n)) / 2. ;
    maxlnz = MIN (maxlnz, ((double) Int_MAX)) ;
    lsize  = MIN (maxlnz, lsize) ;
    usize  = MIN (maxlnz, usize) ;

    PRINTF (("Welcome to klu: n %d anz %d k1 %d lsize %d usize %d maxlnz %g\n",
        n, anz, k1, lsize, usize, maxlnz)) ;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace and outputs */
    /* ---------------------------------------------------------------------- */

    /* return arguments are not yet assigned */
    *p_LU = (Unit *) NULL ;

    /* these computations are safe from size_t overflow */
    W = Work ;
    Pinv = (Int *) W ;      W += n ;
    Stack = (Int *) W ;     W += n ;
    Flag = (Int *) W ;      W += n ;
    Lpend = (Int *) W ;     W += n ;
    Ap_pos = (Int *) W ;    W += n ;

    dunits = DUNITS (Int, lsize) + DUNITS (Entry, lsize) +
             DUNITS (Int, usize) + DUNITS (Entry, usize) ;
    lusize = (size_t) dunits ;
    ok = !INT_OVERFLOW (dunits) ; 
    LU = ok ? KLU_malloc (lusize, sizeof (Unit), Common) : NULL ;
    if (LU == NULL)
    {
        /* out of memory, or problem too large */
        Common->status = KLU_OUT_OF_MEMORY ;
        lusize = 0 ;
        return (lusize) ;
    }

    /* ---------------------------------------------------------------------- */
    /* factorize */
    /* ---------------------------------------------------------------------- */

    /* with pruning, and non-recursive depth-first-search */
    lusize = KLU_kernel (n, Ap, Ai, Ax, Q, lusize,
            Pinv, P, &LU, Udiag, Llen, Ulen, Lip, Uip, lnz, unz,
            X, Stack, Flag, Ap_pos, Lpend,
            k1, PSinv, Rs, Offp, Offi, Offx, Common) ;

    /* ---------------------------------------------------------------------- */
    /* return LU factors, or return nothing if an error occurred */
    /* ---------------------------------------------------------------------- */

    if (Common->status < KLU_OK)
    {
        LU = KLU_free (LU, lusize, sizeof (Unit), Common) ;
        lusize = 0 ;
    }
    *p_LU = LU ;
    PRINTF ((" in klu noffdiag %d\n", Common->noffdiag)) ;
    return (lusize) ;
}


/* ========================================================================== */
/* === KLU_lsolve =========================================================== */
/* ========================================================================== */

/* Solve Lx=b.  Assumes L is unit lower triangular and where the unit diagonal
 * entry is NOT stored.  Overwrites B  with the solution X.  B is n-by-nrhs
 * and is stored in ROW form with row dimension nrhs.  nrhs must be in the
 * range 1 to 4. */

void KLU_lsolve
(
    /* inputs, not modified: */
    Int n,
    Int Lip [ ],
    Int Llen [ ],
    Unit LU [ ],
    Int nrhs,
    /* right-hand-side on input, solution to Lx=b on output */
    Entry X [ ]
)
{
    Entry x [4], lik ;
    Int *Li ;
    Entry *Lx ;
    Int k, p, len, i ;

    switch (nrhs)
    {

        case 1:
            for (k = 0 ; k < n ; k++)
            {
                x [0] = X [k] ;
                GET_POINTER (LU, Lip, Llen, Li, Lx, k, len) ;
                /* unit diagonal of L is not stored*/
                for (p = 0 ; p < len ; p++)
                {
                    /* X [Li [p]] -= Lx [p] * x [0] ; */
                    MULT_SUB (X [Li [p]], Lx [p], x [0]) ;
                }
            }
            break ;

        case 2:

            for (k = 0 ; k < n ; k++)
            {
                x [0] = X [2*k    ] ;
                x [1] = X [2*k + 1] ;
                GET_POINTER (LU, Lip, Llen, Li, Lx, k, len) ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Li [p] ;
                    lik = Lx [p] ;
                    MULT_SUB (X [2*i], lik, x [0]) ;
                    MULT_SUB (X [2*i + 1], lik, x [1]) ;
                }
            }
            break ;

        case 3:

            for (k = 0 ; k < n ; k++)
            {
                x [0] = X [3*k    ] ;
                x [1] = X [3*k + 1] ;
                x [2] = X [3*k + 2] ;
                GET_POINTER (LU, Lip, Llen, Li, Lx, k, len) ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Li [p] ;
                    lik = Lx [p] ;
                    MULT_SUB (X [3*i], lik, x [0]) ;
                    MULT_SUB (X [3*i + 1], lik, x [1]) ;
                    MULT_SUB (X [3*i + 2], lik, x [2]) ;
                }
            }
            break ;

        case 4:

            for (k = 0 ; k < n ; k++)
            {
                x [0] = X [4*k    ] ;
                x [1] = X [4*k + 1] ;
                x [2] = X [4*k + 2] ;
                x [3] = X [4*k + 3] ;
                GET_POINTER (LU, Lip, Llen, Li, Lx, k, len) ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Li [p] ;
                    lik = Lx [p] ;
                    MULT_SUB (X [4*i], lik, x [0]) ;
                    MULT_SUB (X [4*i + 1], lik, x [1]) ;
                    MULT_SUB (X [4*i + 2], lik, x [2]) ;
                    MULT_SUB (X [4*i + 3], lik, x [3]) ;
                }
            }
            break ;

    }
}

/* ========================================================================== */
/* === KLU_usolve =========================================================== */
/* ========================================================================== */

/* Solve Ux=b.  Assumes U is non-unit upper triangular and where the diagonal
 * entry is NOT stored.  Overwrites B with the solution X.  B is n-by-nrhs
 * and is stored in ROW form with row dimension nrhs.  nrhs must be in the
 * range 1 to 4. */

void KLU_usolve
(
    /* inputs, not modified: */
    Int n,
    Int Uip [ ],
    Int Ulen [ ],
    Unit LU [ ],
    Entry Udiag [ ],
    Int nrhs,
    /* right-hand-side on input, solution to Ux=b on output */
    Entry X [ ]
)
{
    Entry x [4], uik, ukk ;
    Int *Ui ;
    Entry *Ux ;
    Int k, p, len, i ;

    switch (nrhs)
    {

        case 1:

            for (k = n-1 ; k >= 0 ; k--)
            {
                GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, len) ;
                /* x [0] = X [k] / Udiag [k] ; */
                DIV (x [0], X [k], Udiag [k]) ;
                X [k] = x [0] ;
                for (p = 0 ; p < len ; p++)
                {
                    /* X [Ui [p]] -= Ux [p] * x [0] ; */
                    MULT_SUB (X [Ui [p]], Ux [p], x [0]) ;

                }
            }

            break ;

        case 2:

            for (k = n-1 ; k >= 0 ; k--)
            {
                GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, len) ;
                ukk = Udiag [k] ;
                /* x [0] = X [2*k    ] / ukk ;
                x [1] = X [2*k + 1] / ukk ; */
                DIV (x [0], X [2*k], ukk) ;
                DIV (x [1], X [2*k + 1], ukk) ;

                X [2*k    ] = x [0] ;
                X [2*k + 1] = x [1] ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Ui [p] ;
                    uik = Ux [p] ;
                    /* X [2*i    ] -= uik * x [0] ;
                    X [2*i + 1] -= uik * x [1] ; */
                    MULT_SUB (X [2*i], uik, x [0]) ;
                    MULT_SUB (X [2*i + 1], uik, x [1]) ;
                }
            }

            break ;

        case 3:

            for (k = n-1 ; k >= 0 ; k--)
            {
                GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, len) ;
                ukk = Udiag [k] ;

                DIV (x [0], X [3*k], ukk) ;
                DIV (x [1], X [3*k + 1], ukk) ;
                DIV (x [2], X [3*k + 2], ukk) ;

                X [3*k    ] = x [0] ;
                X [3*k + 1] = x [1] ;
                X [3*k + 2] = x [2] ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Ui [p] ;
                    uik = Ux [p] ;
                    MULT_SUB (X [3*i], uik, x [0]) ;
                    MULT_SUB (X [3*i + 1], uik, x [1]) ;
                    MULT_SUB (X [3*i + 2], uik, x [2]) ;
                }
            }

            break ;

        case 4:

            for (k = n-1 ; k >= 0 ; k--)
            {
                GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, len) ;
                ukk = Udiag [k] ;

                DIV (x [0], X [4*k], ukk) ;
                DIV (x [1], X [4*k + 1], ukk) ;
                DIV (x [2], X [4*k + 2], ukk) ;
                DIV (x [3], X [4*k + 3], ukk) ;

                X [4*k    ] = x [0] ;
                X [4*k + 1] = x [1] ;
                X [4*k + 2] = x [2] ;
                X [4*k + 3] = x [3] ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Ui [p] ;
                    uik = Ux [p] ;

                    MULT_SUB (X [4*i], uik, x [0]) ;
                    MULT_SUB (X [4*i + 1], uik, x [1]) ;
                    MULT_SUB (X [4*i + 2], uik, x [2]) ;
                    MULT_SUB (X [4*i + 3], uik, x [3]) ;
                }
            }

            break ;

    }
}


/* ========================================================================== */
/* === KLU_ltsolve ========================================================== */
/* ========================================================================== */

/* Solve L'x=b.  Assumes L is unit lower triangular and where the unit diagonal
 * entry is NOT stored.  Overwrites B with the solution X.  B is n-by-nrhs
 * and is stored in ROW form with row dimension nrhs.  nrhs must in the
 * range 1 to 4. */

void KLU_ltsolve
(
    /* inputs, not modified: */
    Int n,
    Int Lip [ ],
    Int Llen [ ],
    Unit LU [ ],
    Int nrhs,
#ifdef COMPLEX
    Int conj_solve,
#endif
    /* right-hand-side on input, solution to L'x=b on output */
    Entry X [ ]
)
{
    Entry x [4], lik ;
    Int *Li ;
    Entry *Lx ;
    Int k, p, len, i ;

    switch (nrhs)
    {

        case 1:

            for (k = n-1 ; k >= 0 ; k--)
            {
                GET_POINTER (LU, Lip, Llen, Li, Lx, k, len) ;
                x [0] = X [k] ;
                for (p = 0 ; p < len ; p++)
                {
#ifdef COMPLEX
                    if (conj_solve)
                    {
                        /* x [0] -= CONJ (Lx [p]) * X [Li [p]] ; */
                        MULT_SUB_CONJ (x [0], X [Li [p]], Lx [p]) ;
                    }
                    else
#endif
                    {
                        /*x [0] -= Lx [p] * X [Li [p]] ;*/
                        MULT_SUB (x [0], Lx [p], X [Li [p]]) ;
                    }
                }
                X [k] = x [0] ;
            }
            break ;

        case 2:

            for (k = n-1 ; k >= 0 ; k--)
            {
                x [0] = X [2*k    ] ;
                x [1] = X [2*k + 1] ;
                GET_POINTER (LU, Lip, Llen, Li, Lx, k, len) ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Li [p] ;
#ifdef COMPLEX
                    if (conj_solve)
                    {
                        CONJ (lik, Lx [p]) ;
                    }
                    else
#endif
                    {
                        lik = Lx [p] ;
                    }
                    MULT_SUB (x [0], lik, X [2*i]) ;
                    MULT_SUB (x [1], lik, X [2*i + 1]) ;
                }
                X [2*k    ] = x [0] ;
                X [2*k + 1] = x [1] ;
            }
            break ;

        case 3:

            for (k = n-1 ; k >= 0 ; k--)
            {
                x [0] = X [3*k    ] ;
                x [1] = X [3*k + 1] ;
                x [2] = X [3*k + 2] ;
                GET_POINTER (LU, Lip, Llen, Li, Lx, k, len) ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Li [p] ;
#ifdef COMPLEX
                    if (conj_solve)
                    {
                        CONJ (lik, Lx [p]) ;
                    }
                    else
#endif
                    {
                        lik = Lx [p] ;
                    }
                    MULT_SUB (x [0], lik, X [3*i]) ;
                    MULT_SUB (x [1], lik, X [3*i + 1]) ;
                    MULT_SUB (x [2], lik, X [3*i + 2]) ;
                }
                X [3*k    ] = x [0] ;
                X [3*k + 1] = x [1] ;
                X [3*k + 2] = x [2] ;
            }
            break ;

        case 4:

            for (k = n-1 ; k >= 0 ; k--)
            {
                x [0] = X [4*k    ] ;
                x [1] = X [4*k + 1] ;
                x [2] = X [4*k + 2] ;
                x [3] = X [4*k + 3] ;
                GET_POINTER (LU, Lip, Llen, Li, Lx, k, len) ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Li [p] ;
#ifdef COMPLEX
                    if (conj_solve)
                    {
                        CONJ (lik, Lx [p]) ;
                    }
                    else
#endif
                    {
                        lik = Lx [p] ;
                    }
                    MULT_SUB (x [0], lik, X [4*i]) ;
                    MULT_SUB (x [1], lik, X [4*i + 1]) ;
                    MULT_SUB (x [2], lik, X [4*i + 2]) ;
                    MULT_SUB (x [3], lik, X [4*i + 3]) ;
                }
                X [4*k    ] = x [0] ;
                X [4*k + 1] = x [1] ;
                X [4*k + 2] = x [2] ;
                X [4*k + 3] = x [3] ;
            }
            break ;
    }
}


/* ========================================================================== */
/* === KLU_utsolve ========================================================== */
/* ========================================================================== */

/* Solve U'x=b.  Assumes U is non-unit upper triangular and where the diagonal
 * entry is stored (and appears last in each column of U).  Overwrites B
 * with the solution X.  B is n-by-nrhs and is stored in ROW form with row
 * dimension nrhs.  nrhs must be in the range 1 to 4. */

void KLU_utsolve
(
    /* inputs, not modified: */
    Int n,
    Int Uip [ ],
    Int Ulen [ ],
    Unit LU [ ],
    Entry Udiag [ ],
    Int nrhs,
#ifdef COMPLEX
    Int conj_solve,
#endif
    /* right-hand-side on input, solution to Ux=b on output */
    Entry X [ ]
)
{
    Entry x [4], uik, ukk ;
    Int k, p, len, i ;
    Int *Ui ;
    Entry *Ux ;

    switch (nrhs)
    {

        case 1:

            for (k = 0 ; k < n ; k++)
            {
                GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, len) ;
                x [0] = X [k] ;
                for (p = 0 ; p < len ; p++)
                {
#ifdef COMPLEX
                    if (conj_solve)
                    {
                        /* x [0] -= CONJ (Ux [p]) * X [Ui [p]] ; */
                        MULT_SUB_CONJ (x [0], X [Ui [p]], Ux [p]) ;
                    }
                    else
#endif
                    {
                        /* x [0] -= Ux [p] * X [Ui [p]] ; */
                        MULT_SUB (x [0], Ux [p], X [Ui [p]]) ;
                    }
                }
#ifdef COMPLEX
                if (conj_solve)
                {
                    CONJ (ukk, Udiag [k]) ;
                }
                else
#endif
                {
                    ukk = Udiag [k] ;
                }
                DIV (X [k], x [0], ukk) ;
            }
            break ;

        case 2:

            for (k = 0 ; k < n ; k++)
            {
                GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, len) ;
                x [0] = X [2*k    ] ;
                x [1] = X [2*k + 1] ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Ui [p] ;
#ifdef COMPLEX
                    if (conj_solve)
                    {
                        CONJ (uik, Ux [p]) ;
                    }
                    else
#endif
                    {
                        uik = Ux [p] ;
                    }
                    MULT_SUB (x [0], uik, X [2*i]) ;
                    MULT_SUB (x [1], uik, X [2*i + 1]) ;
                }
#ifdef COMPLEX
                if (conj_solve)
                {
                    CONJ (ukk, Udiag [k]) ;
                }
                else
#endif
                {
                    ukk = Udiag [k] ;
                }
                DIV (X [2*k], x [0], ukk) ;
                DIV (X [2*k + 1], x [1], ukk) ;
            }
            break ;

        case 3:

            for (k = 0 ; k < n ; k++)
            {
                GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, len) ;
                x [0] = X [3*k    ] ;
                x [1] = X [3*k + 1] ;
                x [2] = X [3*k + 2] ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Ui [p] ;
#ifdef COMPLEX
                    if (conj_solve)
                    {
                        CONJ (uik, Ux [p]) ;
                    }
                    else
#endif
                    {
                        uik = Ux [p] ;
                    }
                    MULT_SUB (x [0], uik, X [3*i]) ;
                    MULT_SUB (x [1], uik, X [3*i + 1]) ;
                    MULT_SUB (x [2], uik, X [3*i + 2]) ;
                }
#ifdef COMPLEX
                if (conj_solve)
                {
                    CONJ (ukk, Udiag [k]) ;
                }
                else
#endif
                {
                    ukk = Udiag [k] ;
                }
                DIV (X [3*k], x [0], ukk) ;
                DIV (X [3*k + 1], x [1], ukk) ;
                DIV (X [3*k + 2], x [2], ukk) ;
            }
            break ;

        case 4:

            for (k = 0 ; k < n ; k++)
            {
                GET_POINTER (LU, Uip, Ulen, Ui, Ux, k, len) ;
                x [0] = X [4*k    ] ;
                x [1] = X [4*k + 1] ;
                x [2] = X [4*k + 2] ;
                x [3] = X [4*k + 3] ;
                for (p = 0 ; p < len ; p++)
                {
                    i = Ui [p] ;
#ifdef COMPLEX
                    if (conj_solve)
                    {
                        CONJ (uik, Ux [p]) ;
                    }
                    else
#endif
                    {
                        uik = Ux [p] ;
                    }
                    MULT_SUB (x [0], uik, X [4*i]) ;
                    MULT_SUB (x [1], uik, X [4*i + 1]) ;
                    MULT_SUB (x [2], uik, X [4*i + 2]) ;
                    MULT_SUB (x [3], uik, X [4*i + 3]) ;
                }
#ifdef COMPLEX
                if (conj_solve)
                {
                    CONJ (ukk, Udiag [k]) ;
                }
                else
#endif
                {
                    ukk = Udiag [k] ;
                }
                DIV (X [4*k], x [0], ukk) ;
                DIV (X [4*k + 1], x [1], ukk) ;
                DIV (X [4*k + 2], x [2], ukk) ;
                DIV (X [4*k + 3], x [3], ukk) ;
            }
            break ;
    }
}
