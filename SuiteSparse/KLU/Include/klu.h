/* ========================================================================== */
/* === klu include file ===================================================== */
/* ========================================================================== */

/* Include file for user programs that call klu_* routines */

#ifndef _KLU_H
#define _KLU_H

/* make it easy for C++ programs to include KLU */
#ifdef __cplusplus
extern "C" {
#endif

#include "amd.h"
#include "colamd.h"
#include "btf.h"

/* -------------------------------------------------------------------------- */
/* Symbolic object - contains the pre-ordering computed by klu_analyze */
/* -------------------------------------------------------------------------- */

typedef struct
{
    /* A (P,Q) is in upper block triangular form.  The kth block goes from
     * row/col index R [k] to R [k+1]-1.  The estimated number of nonzeros
     * in the L factor of the kth block is Lnz [k]. 
     */

    /* only computed if the AMD ordering is chosen: */
    double symmetry ;   /* symmetry of largest block */
    double est_flops ;  /* est. factorization flop count */
    double lnz, unz ;   /* estimated nz in L and U, including diagonals */
    double *Lnz ;       /* size n, but only Lnz [0..nblocks-1] is used */

    /* computed for all orderings: */
    int
        n,              /* input matrix A is n-by-n */
        nz,             /* # entries in input matrix */
        *P,             /* size n */
        *Q,             /* size n */
        *R,             /* size n+1, but only R [0..nblocks] is used */
        nzoff,          /* nz in off-diagonal blocks */
        nblocks,        /* number of blocks */
        maxblock,       /* size of largest block */
        ordering,       /* ordering used (AMD, COLAMD, or GIVEN) */
        do_btf ;        /* whether or not BTF preordering was requested */

    /* only computed if BTF preordering requested */
    int structural_rank ;   /* 0 to n-1 if the matrix is structurally rank
                        * deficient.  -1 if not computed.  n if the matrix has
                        * full structural rank */

} klu_symbolic ;

typedef struct          /* 64-bit version (otherwise same as above) */
{
    double symmetry, est_flops, lnz, unz ;
    double *Lnz ;
    SuiteSparse_long n, nz, *P, *Q, *R, nzoff, nblocks, maxblock, ordering,
        do_btf, structural_rank ;

} klu_l_symbolic ;

/* -------------------------------------------------------------------------- */
/* Numeric object - contains the factors computed by klu_factor */
/* -------------------------------------------------------------------------- */

typedef struct
{
    /* LU factors of each block, the pivot row permutation, and the
     * entries in the off-diagonal blocks */

    int n ;             /* A is n-by-n */
    int nblocks ;       /* number of diagonal blocks */
    int lnz ;           /* actual nz in L, including diagonal */
    int unz ;           /* actual nz in U, including diagonal */
    int max_lnz_block ; /* max actual nz in L in any one block, incl. diag */
    int max_unz_block ; /* max actual nz in U in any one block, incl. diag */
    int *Pnum ;         /* size n. final pivot permutation */
    int *Pinv ;         /* size n. inverse of final pivot permutation */

    /* LU factors of each block */
    int *Lip ;          /* size n. pointers into LUbx[block] for L */
    int *Uip ;          /* size n. pointers into LUbx[block] for U */
    int *Llen ;         /* size n. Llen [k] = # of entries in kth column of L */
    int *Ulen ;         /* size n. Ulen [k] = # of entries in kth column of U */
    void **LUbx ;       /* L and U indices and entries (excl. diagonal of U) */
    size_t *LUsize ;    /* size of each LUbx [block], in sizeof (Unit) */
    void *Udiag ;       /* diagonal of U */

    /* scale factors; can be NULL if no scaling */
    double *Rs ;        /* size n. Rs [i] is scale factor for row i */

    /* permanent workspace for factorization and solve */
    size_t worksize ;   /* size (in bytes) of Work */
    void *Work ;        /* workspace */
    void *Xwork ;       /* alias into Numeric->Work */
    int *Iwork ;        /* alias into Numeric->Work */

    /* off-diagonal entries in a conventional compressed-column sparse matrix */
    int *Offp ;         /* size n+1, column pointers */
    int *Offi ;         /* size nzoff, row indices */
    void *Offx ;        /* size nzoff, numerical values */
    int nzoff ;

} klu_numeric ;

typedef struct          /* 64-bit version (otherwise same as above) */
{
    SuiteSparse_long n, nblocks, lnz, unz, max_lnz_block, max_unz_block, *Pnum,
        *Pinv, *Lip, *Uip, *Llen, *Ulen ;
    void **LUbx ;
    size_t *LUsize ;
    void *Udiag ;
    double *Rs ;
    size_t worksize ;
    void *Work, *Xwork ;
    SuiteSparse_long *Iwork ;
    SuiteSparse_long *Offp, *Offi ;
    void *Offx ;
    SuiteSparse_long nzoff ;

} klu_l_numeric ;

/* -------------------------------------------------------------------------- */
/* KLU control parameters and statistics */
/* -------------------------------------------------------------------------- */

/* Common->status values */
#define KLU_OK 0
#define KLU_SINGULAR (1)            /* status > 0 is a warning, not an error */
#define KLU_OUT_OF_MEMORY (-2)
#define KLU_INVALID (-3)
#define KLU_TOO_LARGE (-4)          /* integer overflow has occured */

typedef struct klu_common_struct
{

    /* ---------------------------------------------------------------------- */
    /* parameters */
    /* ---------------------------------------------------------------------- */

    double tol ;            /* pivot tolerance for diagonal preference */
    double memgrow ;        /* realloc memory growth size for LU factors */
    double initmem_amd ;    /* init. memory size with AMD: c*nnz(L) + n */
    double initmem ;        /* init. memory size: c*nnz(A) + n */
    double maxwork ;        /* maxwork for BTF, <= 0 if no limit */

    int btf ;               /* use BTF pre-ordering, or not */
    int ordering ;          /* 0: AMD, 1: COLAMD, 2: user P and Q,
                             * 3: user function */
    int scale ;             /* row scaling: -1: none (and no error check),
                             * 0: none, 1: sum, 2: max */

    /* pointer to user ordering function */
    int (*user_order) (int, int *, int *, int *, struct klu_common_struct *) ;

    /* pointer to user data, passed unchanged as the last parameter to the
     * user ordering function (optional, the user function need not use this
     * information). */
    void *user_data ;

    int halt_if_singular ;      /* how to handle a singular matrix:
        * FALSE: keep going.  Return a Numeric object with a zero U(k,k).  A
        *   divide-by-zero may occur when computing L(:,k).  The Numeric object
        *   can be passed to klu_solve (a divide-by-zero will occur).  It can
        *   also be safely passed to klu_refactor.
        * TRUE: stop quickly.  klu_factor will free the partially-constructed
        *   Numeric object.  klu_refactor will not free it, but will leave the
        *   numerical values only partially defined.  This is the default. */

    /* ---------------------------------------------------------------------- */
    /* statistics */
    /* ---------------------------------------------------------------------- */

    int status ;                /* KLU_OK if OK, < 0 if error */
    int nrealloc ;              /* # of reallocations of L and U */

    int structural_rank ;       /* 0 to n-1 if the matrix is structurally rank
        * deficient (as determined by maxtrans).  -1 if not computed.  n if the
        * matrix has full structural rank.  This is computed by klu_analyze
        * if a BTF preordering is requested. */

    int numerical_rank ;        /* First k for which a zero U(k,k) was found,
        * if the matrix was singular (in the range 0 to n-1).  n if the matrix
        * has full rank. This is not a true rank-estimation.  It just reports
        * where the first zero pivot was found.  -1 if not computed.
        * Computed by klu_factor and klu_refactor. */

    int singular_col ;          /* n if the matrix is not singular.  If in the
        * range 0 to n-1, this is the column index of the original matrix A that
        * corresponds to the column of U that contains a zero diagonal entry.
        * -1 if not computed.  Computed by klu_factor and klu_refactor. */

    int noffdiag ;      /* # of off-diagonal pivots, -1 if not computed */

    double flops ;      /* actual factorization flop count, from klu_flops */
    double rcond ;      /* crude reciprocal condition est., from klu_rcond */
    double condest ;    /* accurate condition est., from klu_condest */
    double rgrowth ;    /* reciprocal pivot rgrowth, from klu_rgrowth */
    double work ;       /* actual work done in BTF, in klu_analyze */

    size_t memusage ;   /* current memory usage, in bytes */
    size_t mempeak ;    /* peak memory usage, in bytes */

} klu_common ;

typedef struct klu_l_common_struct /* 64-bit version (otherwise same as above)*/
{

    double tol, memgrow, initmem_amd, initmem, maxwork ;
    SuiteSparse_long btf, ordering, scale ;
    SuiteSparse_long (*user_order) (SuiteSparse_long, SuiteSparse_long *,
        SuiteSparse_long *, SuiteSparse_long *,
        struct klu_l_common_struct *) ;
    void *user_data ;
    SuiteSparse_long halt_if_singular ;
    SuiteSparse_long status, nrealloc, structural_rank, numerical_rank,
        singular_col, noffdiag ;
    double flops, rcond, condest, rgrowth, work ;
    size_t memusage, mempeak ;

} klu_l_common ;

/* -------------------------------------------------------------------------- */
/* klu_defaults: sets default control parameters */
/* -------------------------------------------------------------------------- */

int klu_defaults
(
    klu_common *Common
) ;

SuiteSparse_long klu_l_defaults (klu_l_common *Common) ;

/* -------------------------------------------------------------------------- */
/* klu_analyze:  orders and analyzes a matrix */
/* -------------------------------------------------------------------------- */

/* Order the matrix with BTF (or not), then order each block with AMD, COLAMD,
 * a natural ordering, or with a user-provided ordering function */

klu_symbolic *klu_analyze
(
    /* inputs, not modified */
    int n,              /* A is n-by-n */
    int Ap [ ],         /* size n+1, column pointers */
    int Ai [ ],         /* size nz, row indices */
    klu_common *Common
) ;

klu_l_symbolic *klu_l_analyze (SuiteSparse_long, SuiteSparse_long *,
    SuiteSparse_long *, klu_l_common *Common) ;


/* -------------------------------------------------------------------------- */
/* klu_analyze_given: analyzes a matrix using given P and Q */
/* -------------------------------------------------------------------------- */

/* Order the matrix with BTF (or not), then use natural or given ordering
 * P and Q on the blocks.  P and Q are interpretted as identity
 * if NULL. */

klu_symbolic *klu_analyze_given
(
    /* inputs, not modified */
    int n,              /* A is n-by-n */
    int Ap [ ],         /* size n+1, column pointers */
    int Ai [ ],         /* size nz, row indices */
    int P [ ],          /* size n, user's row permutation (may be NULL) */
    int Q [ ],          /* size n, user's column permutation (may be NULL) */
    klu_common *Common
) ;

klu_l_symbolic *klu_l_analyze_given (SuiteSparse_long, SuiteSparse_long *,
    SuiteSparse_long *, SuiteSparse_long *, SuiteSparse_long *,
    klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_factor:  factors a matrix using the klu_analyze results */
/* -------------------------------------------------------------------------- */

klu_numeric *klu_factor /* returns KLU_OK if OK, < 0 if error */
(
    /* inputs, not modified */
    int Ap [ ],         /* size n+1, column pointers */
    int Ai [ ],         /* size nz, row indices */
    double Ax [ ],      /* size nz, numerical values */
    klu_symbolic *Symbolic,
    klu_common *Common
) ;

klu_numeric *klu_z_factor      /* returns KLU_OK if OK, < 0 if error */
(
     /* inputs, not modified */
     int Ap [ ],        /* size n+1, column pointers */
     int Ai [ ],        /* size nz, row indices */
     double Ax [ ],     /* size 2*nz, numerical values (real,imag pairs) */
     klu_symbolic *Symbolic,
     klu_common *Common
) ;

/* long / real version */
klu_l_numeric *klu_l_factor (SuiteSparse_long *, SuiteSparse_long *, double *,
    klu_l_symbolic *, klu_l_common *) ;

/* long / complex version */
klu_l_numeric *klu_zl_factor (SuiteSparse_long *, SuiteSparse_long *, double *,
    klu_l_symbolic *, klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_solve: solves Ax=b using the Symbolic and Numeric objects */
/* -------------------------------------------------------------------------- */

int klu_solve
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    int ldim,               /* leading dimension of B */
    int nrhs,               /* number of right-hand-sides */

    /* right-hand-side on input, overwritten with solution to Ax=b on output */
    double B [ ],           /* size ldim*nrhs */
    klu_common *Common
) ;

int klu_z_solve
(
     /* inputs, not modified */
     klu_symbolic *Symbolic,
     klu_numeric *Numeric,
     int ldim,               /* leading dimension of B */
     int nrhs,               /* number of right-hand-sides */

     /* right-hand-side on input, overwritten with solution to Ax=b on output */
     double B [ ],          /* size 2*ldim*nrhs */
     klu_common *Common
) ;

SuiteSparse_long klu_l_solve (klu_l_symbolic *, klu_l_numeric *,
    SuiteSparse_long, SuiteSparse_long, double *, klu_l_common *) ;

SuiteSparse_long klu_zl_solve (klu_l_symbolic *, klu_l_numeric *,
    SuiteSparse_long, SuiteSparse_long, double *, klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_tsolve: solves A'x=b using the Symbolic and Numeric objects */
/* -------------------------------------------------------------------------- */

int klu_tsolve
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    int ldim,               /* leading dimension of B */
    int nrhs,               /* number of right-hand-sides */

    /* right-hand-side on input, overwritten with solution to Ax=b on output */
    double B [ ],           /* size ldim*nrhs */
    klu_common *Common
) ;

int klu_z_tsolve
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    int ldim,               /* leading dimension of B */
    int nrhs,               /* number of right-hand-sides */

    /* right-hand-side on input, overwritten with solution to Ax=b on output */
    double B [ ],           /* size 2*ldim*nrhs */
    int conj_solve,         /* TRUE: conjugate solve, FALSE: solve A.'x=b */
    klu_common *Common
     
) ;

SuiteSparse_long klu_l_tsolve (klu_l_symbolic *, klu_l_numeric *,
    SuiteSparse_long, SuiteSparse_long, double *, klu_l_common *) ;

SuiteSparse_long klu_zl_tsolve (klu_l_symbolic *, klu_l_numeric *,
    SuiteSparse_long, SuiteSparse_long, double *, SuiteSparse_long,
    klu_l_common * ) ;


/* -------------------------------------------------------------------------- */
/* klu_refactor: refactorizes matrix with same ordering as klu_factor */
/* -------------------------------------------------------------------------- */

int klu_refactor            /* return TRUE if successful, FALSE otherwise */
(
    /* inputs, not modified */
    int Ap [ ],         /* size n+1, column pointers */
    int Ai [ ],         /* size nz, row indices */
    double Ax [ ],      /* size nz, numerical values */
    klu_symbolic *Symbolic,
    /* input, and numerical values modified on output */
    klu_numeric *Numeric,
    klu_common *Common
) ;

int klu_z_refactor          /* return TRUE if successful, FALSE otherwise */
(
     /* inputs, not modified */
     int Ap [ ],        /* size n+1, column pointers */
     int Ai [ ],        /* size nz, row indices */
     double Ax [ ],     /* size 2*nz, numerical values */
     klu_symbolic *Symbolic,
     /* input, and numerical values modified on output */
     klu_numeric *Numeric,
     klu_common *Common
) ;

SuiteSparse_long klu_l_refactor (SuiteSparse_long *, SuiteSparse_long *,
    double *, klu_l_symbolic *, klu_l_numeric *, klu_l_common *) ;

SuiteSparse_long klu_zl_refactor (SuiteSparse_long *, SuiteSparse_long *,
    double *, klu_l_symbolic *, klu_l_numeric *, klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_free_symbolic: destroys the Symbolic object */
/* -------------------------------------------------------------------------- */

int klu_free_symbolic
(
    klu_symbolic **Symbolic,
    klu_common *Common
) ;

SuiteSparse_long klu_l_free_symbolic (klu_l_symbolic **, klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_free_numeric: destroys the Numeric object */
/* -------------------------------------------------------------------------- */

/* Note that klu_free_numeric and klu_z_free_numeric are identical; each can
 * free both kinds of Numeric objects (real and complex) */

int klu_free_numeric
(
    klu_numeric **Numeric,
    klu_common *Common
) ;

int klu_z_free_numeric
(
     klu_numeric **Numeric,
     klu_common *Common
) ;

SuiteSparse_long klu_l_free_numeric (klu_l_numeric **, klu_l_common *) ;
SuiteSparse_long klu_zl_free_numeric (klu_l_numeric **, klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_sort: sorts the columns of the LU factorization */
/* -------------------------------------------------------------------------- */

/* this is not needed except for the MATLAB interface */

int klu_sort
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    /* input/output */
    klu_numeric *Numeric,
    klu_common *Common
) ;

int klu_z_sort
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    /* input/output */
    klu_numeric *Numeric,
    klu_common *Common
) ;

SuiteSparse_long klu_l_sort (klu_l_symbolic *, klu_l_numeric *,
    klu_l_common *) ;
SuiteSparse_long klu_zl_sort (klu_l_symbolic *, klu_l_numeric *,
    klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_flops: determines # of flops performed in numeric factorzation */
/* -------------------------------------------------------------------------- */

int klu_flops
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    /* input/output */
    klu_common *Common
) ;

int klu_z_flops
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    /* input/output */
    klu_common *Common
) ;

SuiteSparse_long klu_l_flops (klu_l_symbolic *, klu_l_numeric *,
    klu_l_common *) ;
SuiteSparse_long klu_zl_flops (klu_l_symbolic *, klu_l_numeric *,
    klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_rgrowth : compute the reciprocal pivot growth */
/* -------------------------------------------------------------------------- */

/* Pivot growth is computed after the input matrix is permuted, scaled, and
 * off-diagonal entries pruned.  This is because the LU factorization of each
 * block takes as input the scaled diagonal blocks of the BTF form.  The
 * reciprocal pivot growth in column j of an LU factorization of a matrix C
 * is the largest entry in C divided by the largest entry in U; then the overall
 * reciprocal pivot growth is the smallest such value for all columns j.  Note
 * that the off-diagonal entries are not scaled, since they do not take part in
 * the LU factorization of the diagonal blocks.
 *
 * In MATLAB notation:
 *
 * rgrowth = min (max (abs ((R \ A(p,q)) - F)) ./ max (abs (U))) */

int klu_rgrowth
(
    int Ap [ ],
    int Ai [ ],
    double Ax [ ],
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    klu_common *Common          /* Common->rgrowth = reciprocal pivot growth */
) ;

int klu_z_rgrowth
(
    int Ap [ ],
    int Ai [ ],
    double Ax [ ],
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    klu_common *Common          /* Common->rgrowth = reciprocal pivot growth */
) ;

SuiteSparse_long klu_l_rgrowth (SuiteSparse_long *, SuiteSparse_long *,
    double *, klu_l_symbolic *, klu_l_numeric *, klu_l_common *) ;

SuiteSparse_long klu_zl_rgrowth (SuiteSparse_long *, SuiteSparse_long *,
    double *, klu_l_symbolic *, klu_l_numeric *, klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_condest */
/* -------------------------------------------------------------------------- */

/* Computes a reasonably accurate estimate of the 1-norm condition number, using
 * Hager's method, as modified by Higham and Tisseur (same method as used in
 * MATLAB's condest */

int klu_condest
(
    int Ap [ ],             /* size n+1, column pointers, not modified */
    double Ax [ ],          /* size nz = Ap[n], numerical values, not modified*/
    klu_symbolic *Symbolic, /* symbolic analysis, not modified */
    klu_numeric *Numeric,   /* numeric factorization, not modified */
    klu_common *Common      /* result returned in Common->condest */
) ;

int klu_z_condest
(
    int Ap [ ],
    double Ax [ ],          /* size 2*nz */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    klu_common *Common      /* result returned in Common->condest */
) ;

SuiteSparse_long klu_l_condest (SuiteSparse_long *, double *, klu_l_symbolic *,
    klu_l_numeric *, klu_l_common *) ;

SuiteSparse_long klu_zl_condest (SuiteSparse_long *, double *, klu_l_symbolic *,
    klu_l_numeric *, klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_rcond: compute min(abs(diag(U))) / max(abs(diag(U))) */
/* -------------------------------------------------------------------------- */

int klu_rcond
(
    klu_symbolic *Symbolic,         /* input, not modified */
    klu_numeric *Numeric,           /* input, not modified */
    klu_common *Common              /* result in Common->rcond */
) ;

int klu_z_rcond
(
    klu_symbolic *Symbolic,         /* input, not modified */
    klu_numeric *Numeric,           /* input, not modified */
    klu_common *Common              /* result in Common->rcond */
) ;

SuiteSparse_long klu_l_rcond (klu_l_symbolic *, klu_l_numeric *,
    klu_l_common *) ;

SuiteSparse_long klu_zl_rcond (klu_l_symbolic *, klu_l_numeric *,
    klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_scale */
/* -------------------------------------------------------------------------- */

int klu_scale           /* return TRUE if successful, FALSE otherwise */
(
    /* inputs, not modified */
    int scale,          /* <0: none, no error check; 0: none, 1: sum, 2: max */
    int n,
    int Ap [ ],         /* size n+1, column pointers */
    int Ai [ ],         /* size nz, row indices */
    double Ax [ ],
    /* outputs, not defined on input */
    double Rs [ ],
    /* workspace, not defined on input or output */
    int W [ ],          /* size n, can be NULL */
    klu_common *Common
) ;

int klu_z_scale         /* return TRUE if successful, FALSE otherwise */
(
    /* inputs, not modified */
    int scale,          /* <0: none, no error check; 0: none, 1: sum, 2: max */
    int n,
    int Ap [ ],         /* size n+1, column pointers */
    int Ai [ ],         /* size nz, row indices */
    double Ax [ ],
    /* outputs, not defined on input */
    double Rs [ ],
    /* workspace, not defined on input or output */
    int W [ ],          /* size n, can be NULL */
    klu_common *Common
) ;

SuiteSparse_long klu_l_scale (SuiteSparse_long, SuiteSparse_long,
    SuiteSparse_long *, SuiteSparse_long *, double *,
    double *, SuiteSparse_long *, klu_l_common *) ;

SuiteSparse_long klu_zl_scale (SuiteSparse_long, SuiteSparse_long,
    SuiteSparse_long *, SuiteSparse_long *, double *,
    double *, SuiteSparse_long *, klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* klu_extract  */
/* -------------------------------------------------------------------------- */

int klu_extract     /* returns TRUE if successful, FALSE otherwise */
(
    /* inputs: */
    klu_numeric *Numeric,
    klu_symbolic *Symbolic,

    /* outputs, either allocated on input, or ignored otherwise */

    /* L */
    int *Lp,        /* size n+1 */
    int *Li,        /* size Numeric->lnz */
    double *Lx,     /* size Numeric->lnz */

    /* U */
    int *Up,        /* size n+1 */
    int *Ui,        /* size Numeric->unz */
    double *Ux,     /* size Numeric->unz */

    /* F */
    int *Fp,        /* size n+1 */
    int *Fi,        /* size Numeric->nzoff */
    double *Fx,     /* size Numeric->nzoff */

    /* P, row permutation */
    int *P,         /* size n */

    /* Q, column permutation */
    int *Q,         /* size n */

    /* Rs, scale factors */
    double *Rs,     /* size n */

    /* R, block boundaries */
    int *R,         /* size Symbolic->nblocks+1 (nblocks is at most n) */

    klu_common *Common
) ;


int klu_z_extract           /* returns TRUE if successful, FALSE otherwise */
(
    /* inputs: */
    klu_numeric *Numeric,
    klu_symbolic *Symbolic,

    /* outputs, all of which must be allocated on input */

    /* L */
    int *Lp,        /* size n+1 */
    int *Li,        /* size nnz(L) */
    double *Lx,     /* size nnz(L) */
    double *Lz,     /* size nnz(L) for the complex case, ignored if real */

    /* U */
    int *Up,        /* size n+1 */
    int *Ui,        /* size nnz(U) */
    double *Ux,     /* size nnz(U) */
    double *Uz,     /* size nnz(U) for the complex case, ignored if real */

    /* F */
    int *Fp,        /* size n+1 */
    int *Fi,        /* size nnz(F) */
    double *Fx,     /* size nnz(F) */
    double *Fz,     /* size nnz(F) for the complex case, ignored if real */

    /* P, row permutation */
    int *P,         /* size n */

    /* Q, column permutation */
    int *Q,         /* size n */

    /* Rs, scale factors */
    double *Rs,     /* size n */

    /* R, block boundaries */
    int *R,         /* size Symbolic->nblocks+1 (nblocks is at most n) */

    klu_common *Common
) ;

SuiteSparse_long klu_l_extract (klu_l_numeric *, klu_l_symbolic *,
    SuiteSparse_long *, SuiteSparse_long *, double *,
    SuiteSparse_long *, SuiteSparse_long *, double *,
    SuiteSparse_long *, SuiteSparse_long *, double *,
    SuiteSparse_long *, SuiteSparse_long *, double *,
    SuiteSparse_long *, klu_l_common *) ;

SuiteSparse_long klu_zl_extract (klu_l_numeric *, klu_l_symbolic *,
    SuiteSparse_long *, SuiteSparse_long *, double *, double *,
    SuiteSparse_long *, SuiteSparse_long *, double *, double *,
    SuiteSparse_long *, SuiteSparse_long *, double *, double *,
    SuiteSparse_long *, SuiteSparse_long *, double *,
    SuiteSparse_long *, klu_l_common *) ;


/* -------------------------------------------------------------------------- */
/* KLU memory management routines */
/* -------------------------------------------------------------------------- */

void *klu_malloc        /* returns pointer to the newly malloc'd block */
(
    /* ---- input ---- */
    size_t n,           /* number of items */
    size_t size,        /* size of each item */
    /* --------------- */
    klu_common *Common
) ;

void *klu_free          /* always returns NULL */
(
    /* ---- in/out --- */
    void *p,            /* block of memory to free */
    size_t n,           /* number of items */
    size_t size,        /* size of each item */
    /* --------------- */
    klu_common *Common
) ;

void *klu_realloc       /* returns pointer to reallocated block */
(
    /* ---- input ---- */
    size_t nnew,        /* requested # of items in reallocated block */
    size_t nold,        /* current size of block, in # of items */
    size_t size,        /* size of each item */
    /* ---- in/out --- */
    void *p,            /* block of memory to realloc */
    /* --------------- */
    klu_common *Common
) ;

void *klu_l_malloc (size_t, size_t, klu_l_common *) ;
void *klu_l_free (void *, size_t, size_t, klu_l_common *) ;
void *klu_l_realloc (size_t, size_t, size_t, void *, klu_l_common *) ;


/* ========================================================================== */
/* === KLU version ========================================================== */
/* ========================================================================== */

/* All versions of KLU include these definitions.
 * As an example, to test if the version you are using is 1.2 or later:
 *
 *      if (KLU_VERSION >= KLU_VERSION_CODE (1,2)) ...
 *
 * This also works during compile-time:
 *
 *      #if (KLU >= KLU_VERSION_CODE (1,2))
 *          printf ("This is version 1.2 or later\n") ;
 *      #else
 *          printf ("This is an early version\n") ;
 *      #endif
 */

#define KLU_DATE "Mar 12, 2018"
#define KLU_VERSION_CODE(main,sub) ((main) * 1000 + (sub))
#define KLU_MAIN_VERSION 1
#define KLU_SUB_VERSION 3
#define KLU_SUBSUB_VERSION 9
#define KLU_VERSION KLU_VERSION_CODE(KLU_MAIN_VERSION,KLU_SUB_VERSION)

#ifdef __cplusplus
}
#endif
#endif
