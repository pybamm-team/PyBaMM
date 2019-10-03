/* ========================================================================== */
/* === BTF package ========================================================== */
/* ========================================================================== */

/* BTF_MAXTRANS:  find a column permutation Q to give A*Q a zero-free diagonal
 * BTF_STRONGCOMP:  find a symmetric permutation P to put P*A*P' into block
 *      upper triangular form.
 * BTF_ORDER: do both of the above (btf_maxtrans then btf_strongcomp).
 *
 * By Tim Davis.  Copyright (c) 2004-2007, University of Florida.
 * with support from Sandia National Laboratories.  All Rights Reserved.
 */


/* ========================================================================== */
/* === BTF_MAXTRANS ========================================================= */
/* ========================================================================== */

/* BTF_MAXTRANS: finds a permutation of the columns of a matrix so that it has a
 * zero-free diagonal.  The input is an m-by-n sparse matrix in compressed
 * column form.  The array Ap of size n+1 gives the starting and ending
 * positions of the columns in the array Ai.  Ap[0] must be zero. The array Ai
 * contains the row indices of the nonzeros of the matrix A, and is of size
 * Ap[n].  The row indices of column j are located in Ai[Ap[j] ... Ap[j+1]-1].
 * Row indices must be in the range 0 to m-1.  Duplicate entries may be present
 * in any given column.  The input matrix  is not checked for validity (row
 * indices out of the range 0 to m-1 will lead to an undeterminate result -
 * possibly a core dump, for example).  Row indices in any given column need
 * not be in sorted order.  However, if they are sorted and the matrix already
 * has a zero-free diagonal, then the identity permutation is returned.
 *
 * The output of btf_maxtrans is an array Match of size n.  If row i is matched
 * with column j, then A(i,j) is nonzero, and then Match[i] = j.  If the matrix
 * is structurally nonsingular, all entries in the Match array are unique, and
 * Match can be viewed as a column permutation if A is square.  That is, column
 * k of the original matrix becomes column Match[k] of the permuted matrix.  In
 * MATLAB, this can be expressed as (for non-structurally singular matrices):
 *
 *      Match = maxtrans (A) ;
 *      B = A (:, Match) ;
 *
 * except of course here the A matrix and Match vector are all 0-based (rows
 * and columns in the range 0 to n-1), not 1-based (rows/cols in range 1 to n).
 * The MATLAB dmperm routine returns a row permutation.  See the maxtrans
 * mexFunction for more details.
 *
 * If row i is not matched to any column, then Match[i] is == -1.  The
 * btf_maxtrans routine returns the number of nonzeros on diagonal of the
 * permuted matrix.
 *
 * In the MATLAB mexFunction interface to btf_maxtrans, 1 is added to the Match
 * array to obtain a 1-based permutation.  Thus, in MATLAB where A is m-by-n:
 *
 *      q = maxtrans (A) ;      % has entries in the range 0:n
 *      q                       % a column permutation (only if sprank(A)==n)
 *      B = A (:, q) ;          % permuted matrix (only if sprank(A)==n)
 *      sum (q > 0) ;           % same as "sprank (A)"
 *
 * This behaviour differs from p = dmperm (A) in MATLAB, which returns the
 * matching as p(j)=i if row i and column j are matched, and p(j)=0 if column j
 * is unmatched.
 *
 *      p = dmperm (A) ;        % has entries in the range 0:m
 *      p                       % a row permutation (only if sprank(A)==m)
 *      B = A (p, :) ;          % permuted matrix (only if sprank(A)==m)
 *      sum (p > 0) ;           % definition of sprank (A)
 *
 * This algorithm is based on the paper "On Algorithms for obtaining a maximum
 * transversal" by Iain Duff, ACM Trans. Mathematical Software, vol 7, no. 1,
 * pp. 315-330, and "Algorithm 575: Permutations for a zero-free diagonal",
 * same issue, pp. 387-390.  Algorithm 575 is MC21A in the Harwell Subroutine
 * Library.  This code is not merely a translation of the Fortran code into C.
 * It is a completely new implementation of the basic underlying method (depth
 * first search over a subgraph with nodes corresponding to columns matched so
 * far, and cheap matching).  This code was written with minimal observation of
 * the MC21A/B code itself.  See comments below for a comparison between the
 * maxtrans and MC21A/B codes.
 *
 * This routine operates on a column-form matrix and produces a column
 * permutation.  MC21A uses a row-form matrix and produces a row permutation.
 * The difference is merely one of convention in the comments and interpretation
 * of the inputs and outputs.  If you want a row permutation, simply pass a
 * compressed-row sparse matrix to this routine and you will get a row
 * permutation (just like MC21A).  Similarly, you can pass a column-oriented
 * matrix to MC21A and it will happily return a column permutation.
 */

#ifndef _BTF_H
#define _BTF_H

/* make it easy for C++ programs to include BTF */
#ifdef __cplusplus
extern "C" {
#endif

#include "SuiteSparse_config.h"

int btf_maxtrans    /* returns # of columns matched */
(
    /* --- input, not modified: --- */
    int nrow,       /* A is nrow-by-ncol in compressed column form */
    int ncol,
    int Ap [ ],     /* size ncol+1 */
    int Ai [ ],     /* size nz = Ap [ncol] */
    double maxwork, /* maximum amount of work to do is maxwork*nnz(A); no limit
                     * if <= 0 */

    /* --- output, not defined on input --- */
    double *work,   /* work = -1 if maxwork > 0 and the total work performed
                     * reached the maximum of maxwork*nnz(A).
                     * Otherwise, work = the total work performed. */

    int Match [ ],  /* size nrow.  Match [i] = j if column j matched to row i
                     * (see above for the singular-matrix case) */

    /* --- workspace, not defined on input or output --- */
    int Work [ ]    /* size 5*ncol */
) ;

/* long integer version (all "int" parameters become "SuiteSparse_long") */
SuiteSparse_long btf_l_maxtrans (SuiteSparse_long, SuiteSparse_long,
    SuiteSparse_long *, SuiteSparse_long *, double, double *,
    SuiteSparse_long *, SuiteSparse_long *) ;


/* ========================================================================== */
/* === BTF_STRONGCOMP ======================================================= */
/* ========================================================================== */

/* BTF_STRONGCOMP finds the strongly connected components of a graph, returning
 * a symmetric permutation.  The matrix A must be square, and is provided on
 * input in compressed-column form (see BTF_MAXTRANS, above).  The diagonal of
 * the input matrix A (or A*Q if Q is provided on input) is ignored.
 *
 * If Q is not NULL on input, then the strongly connected components of A*Q are
 * found.  Q may be flagged on input, where Q[k] < 0 denotes a flagged column k.
 * The permutation is j = BTF_UNFLIP (Q [k]).  On output, Q is modified (the
 * flags are preserved) so that P*A*Q is in block upper triangular form.
 *
 * If Q is NULL, then the permutation P is returned so that P*A*P' is in upper
 * block triangular form.
 *
 * The vector R gives the block boundaries, where block b is in rows/columns
 * R[b] to R[b+1]-1 of the permuted matrix, and where b ranges from 1 to the
 * number of strongly connected components found.
 */

int btf_strongcomp  /* return # of strongly connected components */
(
    /* input, not modified: */
    int n,          /* A is n-by-n in compressed column form */
    int Ap [ ],     /* size n+1 */
    int Ai [ ],     /* size nz = Ap [n] */

    /* optional input, modified (if present) on output: */
    int Q [ ],      /* size n, input column permutation */

    /* output, not defined on input */
    int P [ ],      /* size n.  P [k] = j if row and column j are kth row/col
                     * in permuted matrix. */

    int R [ ],      /* size n+1.  block b is in rows/cols R[b] ... R[b+1]-1 */

    /* workspace, not defined on input or output */
    int Work [ ]    /* size 4n */
) ;

SuiteSparse_long btf_l_strongcomp (SuiteSparse_long, SuiteSparse_long *,
    SuiteSparse_long *, SuiteSparse_long *, SuiteSparse_long *,
    SuiteSparse_long *, SuiteSparse_long *) ;


/* ========================================================================== */
/* === BTF_ORDER ============================================================ */
/* ========================================================================== */

/* BTF_ORDER permutes a square matrix into upper block triangular form.  It
 * does this by first finding a maximum matching (or perhaps a limited matching
 * if the work is limited), via the btf_maxtrans function.  If a complete
 * matching is not found, BTF_ORDER completes the permutation, but flags the
 * columns of P*A*Q to denote which columns are not matched.  If the matrix is
 * structurally rank deficient, some of the entries on the diagonal of the
 * permuted matrix will be zero.  BTF_ORDER then calls btf_strongcomp to find
 * the strongly-connected components.
 *
 * On output, P and Q are the row and column permutations, where i = P[k] if
 * row i of A is the kth row of P*A*Q, and j = BTF_UNFLIP(Q[k]) if column j of
 * A is the kth column of P*A*Q.  If Q[k] < 0, then the (k,k)th entry in P*A*Q
 * is structurally zero.
 *
 * The vector R gives the block boundaries, where block b is in rows/columns
 * R[b] to R[b+1]-1 of the permuted matrix, and where b ranges from 1 to the
 * number of strongly connected components found.
 */

int btf_order       /* returns number of blocks found */
(
    /* --- input, not modified: --- */
    int n,          /* A is n-by-n in compressed column form */
    int Ap [ ],     /* size n+1 */
    int Ai [ ],     /* size nz = Ap [n] */
    double maxwork, /* do at most maxwork*nnz(A) work in the maximum
                     * transversal; no limit if <= 0 */

    /* --- output, not defined on input --- */
    double *work,   /* return value from btf_maxtrans */
    int P [ ],      /* size n, row permutation */
    int Q [ ],      /* size n, column permutation */
    int R [ ],      /* size n+1.  block b is in rows/cols R[b] ... R[b+1]-1 */
    int *nmatch,    /* # nonzeros on diagonal of P*A*Q */

    /* --- workspace, not defined on input or output --- */
    int Work [ ]    /* size 5n */
) ;

SuiteSparse_long btf_l_order (SuiteSparse_long, SuiteSparse_long *,
    SuiteSparse_long *, double , double *, SuiteSparse_long *,
    SuiteSparse_long *, SuiteSparse_long *, SuiteSparse_long *,
    SuiteSparse_long *) ;


/* ========================================================================== */
/* === BTF marking of singular columns ====================================== */
/* ========================================================================== */

/* BTF_FLIP is a "negation about -1", and is used to mark an integer j
 * that is normally non-negative.  BTF_FLIP (-1) is -1.  BTF_FLIP of
 * a number > -1 is negative, and BTF_FLIP of a number < -1 is positive.
 * BTF_FLIP (BTF_FLIP (j)) = j for all integers j.  UNFLIP (j) acts
 * like an "absolute value" operation, and is always >= -1.  You can test
 * whether or not an integer j is "flipped" with the BTF_ISFLIPPED (j)
 * macro.
 */

#define BTF_FLIP(j) (-(j)-2)
#define BTF_ISFLIPPED(j) ((j) < -1)
#define BTF_UNFLIP(j) ((BTF_ISFLIPPED (j)) ? BTF_FLIP (j) : (j))

/* ========================================================================== */
/* === BTF version ========================================================== */
/* ========================================================================== */

/* All versions of BTF include these definitions.
 * As an example, to test if the version you are using is 1.2 or later:
 *
 *      if (BTF_VERSION >= BTF_VERSION_CODE (1,2)) ...
 *
 * This also works during compile-time:
 *
 *      #if (BTF >= BTF_VERSION_CODE (1,2))
 *          printf ("This is version 1.2 or later\n") ;
 *      #else
 *          printf ("This is an early version\n") ;
 *      #endif
 */

#define BTF_DATE "May 4, 2016"
#define BTF_VERSION_CODE(main,sub) ((main) * 1000 + (sub))
#define BTF_MAIN_VERSION 1
#define BTF_SUB_VERSION 2
#define BTF_SUBSUB_VERSION 6
#define BTF_VERSION BTF_VERSION_CODE(BTF_MAIN_VERSION,BTF_SUB_VERSION)

#ifdef __cplusplus
}
#endif
#endif
