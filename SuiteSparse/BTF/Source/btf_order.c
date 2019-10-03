/* ========================================================================== */
/* === BTF_ORDER ============================================================ */
/* ========================================================================== */

/* Find a permutation P and Q to permute a square sparse matrix into upper block
 * triangular form.  A(P,Q) will contain a zero-free diagonal if A has
 * structural full-rank.  Otherwise, the number of nonzeros on the diagonal of
 * A(P,Q) will be maximized, and will equal the structural rank of A.
 *
 * Q[k] will be "flipped" if a zero-free diagonal was not found.  Q[k] will be
 * negative, and j = BTF_UNFLIP (Q [k]) gives the corresponding permutation.
 *
 * R defines the block boundaries of A(P,Q).  The kth block consists of rows
 * and columns R[k] to R[k+1]-1.
 *
 * If maxwork > 0 on input, then the work performed in btf_maxtrans is limited
 * to maxwork*nnz(A) (excluding the "cheap match" phase, which can take another
 * nnz(A) work).  On output, the work parameter gives the actual work performed,
 * or -1 if the limit was reached.  In the latter case, the diagonal of A(P,Q)
 * might not be zero-free, and the number of nonzeros on the diagonal of A(P,Q)
 * might not be equal to the structural rank.
 *
 * See btf.h for more details.
 *
 * By Tim Davis.  Copyright (c) 2004-2007, University of Florida.
 * with support from Sandia National Laboratories.  All Rights Reserved.
 */

#include "btf.h"
#include "btf_internal.h"

/* This function only operates on square matrices (either structurally full-
 * rank, or structurally rank deficient). */

Int BTF(order)      /* returns number of blocks found */
(
    /* input, not modified: */
    Int n,          /* A is n-by-n in compressed column form */
    Int Ap [ ],     /* size n+1 */
    Int Ai [ ],     /* size nz = Ap [n] */
    double maxwork, /* do at most maxwork*nnz(A) work in the maximum
                     * transversal; no limit if <= 0 */

    /* output, not defined on input */
    double *work,   /* work performed in maxtrans, or -1 if limit reached */
    Int P [ ],      /* size n, row permutation */
    Int Q [ ],      /* size n, column permutation */
    Int R [ ],      /* size n+1.  block b is in rows/cols R[b] ... R[b+1]-1 */
    Int *nmatch,    /* # nonzeros on diagonal of P*A*Q */

    /* workspace, not defined on input or output */
    Int Work [ ]    /* size 5n */
)
{
    Int *Flag ;
    Int nblocks, i, j, nbadcol ;

    /* ---------------------------------------------------------------------- */
    /* compute the maximum matching */
    /* ---------------------------------------------------------------------- */

    /* if maxwork > 0, then a maximum matching might not be found */

    *nmatch = BTF(maxtrans) (n, n, Ap, Ai, maxwork, work, Q, Work) ;

    /* ---------------------------------------------------------------------- */
    /* complete permutation if the matrix is structurally singular */
    /* ---------------------------------------------------------------------- */

    /* Since the matrix is square, ensure BTF_UNFLIP(Q[0..n-1]) is a
     * permutation of the columns of A so that A has as many nonzeros on the
     * diagonal as possible.
     */

    if (*nmatch < n)
    {
        /* get a size-n work array */
        Flag = Work + n ;
        for (j = 0 ; j < n ; j++)
        {
            Flag [j] = 0 ;
        }

        /* flag all matched columns */
        for (i = 0 ; i < n ; i++)
        {
            j = Q [i] ;
            if (j != EMPTY)
            {
                /* row i and column j are matched to each other */
                Flag [j] = 1 ;
            }
        }

        /* make a list of all unmatched columns, in Work [0..nbadcol-1]  */
        nbadcol = 0 ;
        for (j = n-1 ; j >= 0 ; j--)
        {
            if (!Flag [j])
            {
                /* j is matched to nobody */
                Work [nbadcol++] = j ;
            }
        }
        ASSERT (*nmatch + nbadcol == n) ;

        /* make an assignment for each unmatched row */
        for (i = 0 ; i < n ; i++)
        {
            if (Q [i] == EMPTY && nbadcol > 0)
            {
                /* get an unmatched column j */
                j = Work [--nbadcol] ;
                /* assign j to row i and flag the entry by "flipping" it */
                Q [i] = BTF_FLIP (j) ;
            }
        }
    }

    /* The permutation of a square matrix can be recovered as follows: Row i is
     * matched with column j, where j = BTF_UNFLIP (Q [i]) and where j
     * will always be in the valid range 0 to n-1.  The entry A(i,j) is zero
     * if BTF_ISFLIPPED (Q [i]) is true, and nonzero otherwise.  nmatch
     * is the number of entries in the Q array that are non-negative. */

    /* ---------------------------------------------------------------------- */
    /* find the strongly connected components */
    /* ---------------------------------------------------------------------- */

    nblocks = BTF(strongcomp) (n, Ap, Ai, Q, P, R, Work) ;
    return (nblocks) ;
}
