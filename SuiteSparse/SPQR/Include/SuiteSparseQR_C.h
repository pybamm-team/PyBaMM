/* ========================================================================== */
/* === SuiteSparseQR_C.h ==================================================== */
/* ========================================================================== */

/* For inclusion in a C or C++ program. */

#ifndef SUITESPARSEQR_C_H
#define SUITESPARSEQR_C_H

#ifdef __cplusplus
/* If included by a C++ program, the Complex type is std::complex<double> */
#include <complex>
#define Complex std::complex<double>
extern "C" {
#endif

#include "cholmod.h"
#include "SuiteSparseQR_definitions.h"

#ifndef __cplusplus
/* The C++ functions will return a pointer to a std::complex<double> array of
   size n, which the C code must then interpret as double array of size 2*n,
   with real and imaginary parts interleaved. */
#define Complex double
#endif

/* ========================================================================== */
/* === SuiteSparseQR_C ====================================================== */
/* ========================================================================== */

SuiteSparse_long SuiteSparseQR_C /* returns rank(A) estimate, (-1) if failure */
(
    /* inputs: */
    int ordering,               /* all, except 3:given treated as 0:fixed */
    double tol,                 /* columns with 2-norm <= tol treated as 0 */
    SuiteSparse_long econ,      /* e = max(min(m,econ),rank(A)) */
    int getCTX,                 /* 0: Z=C (e-by-k), 1: Z=C', 2: Z=X (e-by-k) */
    cholmod_sparse *A,          /* m-by-n sparse matrix to factorize */
    cholmod_sparse *Bsparse,    /* sparse m-by-k B */
    cholmod_dense  *Bdense,     /* dense  m-by-k B */
    /* outputs: */
    cholmod_sparse **Zsparse,   /* sparse Z */
    cholmod_dense  **Zdense,    /* dense Z */
    cholmod_sparse **R,         /* e-by-n sparse matrix */
    SuiteSparse_long **E,       /* size n column perm, NULL if identity */
    cholmod_sparse **H,         /* m-by-nh Householder vectors */
    SuiteSparse_long **HPinv,   /* size m row permutation */
    cholmod_dense **HTau,       /* 1-by-nh Householder coefficients */
    cholmod_common *cc          /* workspace and parameters */
) ;

/* ========================================================================== */
/* === SuiteSparseQR_C_QR =================================================== */
/* ========================================================================== */

/* [Q,R,E] = qr(A), returning Q as a sparse matrix */
SuiteSparse_long SuiteSparseQR_C_QR /* returns rank(A) est., (-1) if failure */
(
    /* inputs: */
    int ordering,               /* all, except 3:given treated as 0:fixed */
    double tol,                 /* columns with 2-norm <= tol treated as 0 */
    SuiteSparse_long econ,      /* e = max(min(m,econ),rank(A)) */
    cholmod_sparse *A,          /* m-by-n sparse matrix to factorize */
    /* outputs: */
    cholmod_sparse **Q,         /* m-by-e sparse matrix */
    cholmod_sparse **R,         /* e-by-n sparse matrix */
    SuiteSparse_long **E,       /* size n column perm, NULL if identity */
    cholmod_common *cc          /* workspace and parameters */
) ;

/* ========================================================================== */
/* === SuiteSparseQR_C_backslash ============================================ */
/* ========================================================================== */

/* X = A\B where B is dense */
cholmod_dense *SuiteSparseQR_C_backslash    /* returns X, NULL if failure */
(
    int ordering,               /* all, except 3:given treated as 0:fixed */
    double tol,                 /* columns with 2-norm <= tol treated as 0 */
    cholmod_sparse *A,          /* m-by-n sparse matrix */
    cholmod_dense  *B,          /* m-by-k */
    cholmod_common *cc          /* workspace and parameters */
) ;

/* ========================================================================== */
/* === SuiteSparseQR_C_backslash_default ==================================== */
/* ========================================================================== */

/* X = A\B where B is dense, using default ordering and tol */
cholmod_dense *SuiteSparseQR_C_backslash_default /* returns X, NULL if failure*/
(
    cholmod_sparse *A,          /* m-by-n sparse matrix */
    cholmod_dense  *B,          /* m-by-k */
    cholmod_common *cc          /* workspace and parameters */
) ;

/* ========================================================================== */
/* === SuiteSparseQR_C_backslash_sparse ===================================== */
/* ========================================================================== */

/* X = A\B where B is sparse */
cholmod_sparse *SuiteSparseQR_C_backslash_sparse   /* returns X, or NULL */
(
    /* inputs: */
    int ordering,               /* all, except 3:given treated as 0:fixed */
    double tol,                 /* columns with 2-norm <= tol treated as 0 */
    cholmod_sparse *A,          /* m-by-n sparse matrix */
    cholmod_sparse *B,          /* m-by-k */
    cholmod_common *cc          /* workspace and parameters */
) ;

#ifndef NEXPERT

/* ========================================================================== */
/* === SuiteSparseQR_C_factorization ======================================== */
/* ========================================================================== */

/* A real or complex QR factorization, computed by SuiteSparseQR_C_factorize */
typedef struct SuiteSparseQR_C_factorization_struct
{
    int xtype ;                 /* CHOLMOD_REAL or CHOLMOD_COMPLEX */
    void *factors ;             /* from SuiteSparseQR_factorize <double> or
                                        SuiteSparseQR_factorize <Complex> */

} SuiteSparseQR_C_factorization ;

/* ========================================================================== */
/* === SuiteSparseQR_C_factorize ============================================ */
/* ========================================================================== */

SuiteSparseQR_C_factorization *SuiteSparseQR_C_factorize
(
    /* inputs: */
    int ordering,               /* all, except 3:given treated as 0:fixed */
    double tol,                 /* columns with 2-norm <= tol treated as 0 */
    cholmod_sparse *A,          /* m-by-n sparse matrix */
    cholmod_common *cc          /* workspace and parameters */
) ;

/* ========================================================================== */
/* === SuiteSparseQR_C_symbolic ============================================= */
/* ========================================================================== */

SuiteSparseQR_C_factorization *SuiteSparseQR_C_symbolic
(
    /* inputs: */
    int ordering,               /* all, except 3:given treated as 0:fixed */
    int allow_tol,              /* if TRUE allow tol for rank detection */
    cholmod_sparse *A,          /* m-by-n sparse matrix, A->x ignored */
    cholmod_common *cc          /* workspace and parameters */
) ;

/* ========================================================================== */
/* === SuiteSparseQR_C_numeric ============================================== */
/* ========================================================================== */

int SuiteSparseQR_C_numeric
(
    /* inputs: */
    double tol,                 /* treat columns with 2-norm <= tol as zero */
    cholmod_sparse *A,          /* sparse matrix to factorize */
    /* input/output: */
    SuiteSparseQR_C_factorization *QR,
    cholmod_common *cc          /* workspace and parameters */
) ;

/* ========================================================================== */
/* === SuiteSparseQR_C_free ================================================= */
/* ========================================================================== */

/* Free the QR factors computed by SuiteSparseQR_C_factorize */
int SuiteSparseQR_C_free        /* returns TRUE (1) if OK, FALSE (0) otherwise*/
(
    SuiteSparseQR_C_factorization **QR,
    cholmod_common *cc          /* workspace and parameters */
) ;

/* ========================================================================== */
/* === SuiteSparseQR_C_solve ================================================ */
/* ========================================================================== */

cholmod_dense* SuiteSparseQR_C_solve    /* returnx X, or NULL if failure */
(
    int system,                 /* which system to solve */
    SuiteSparseQR_C_factorization *QR,  /* of an m-by-n sparse matrix A */
    cholmod_dense *B,           /* right-hand-side, m-by-k or n-by-k */
    cholmod_common *cc          /* workspace and parameters */
) ;

/* ========================================================================== */
/* === SuiteSparseQR_C_qmult ================================================ */
/* ========================================================================== */

/*
    Applies Q in Householder form (as stored in the QR factorization object
    returned by SuiteSparseQR_C_factorize) to a dense matrix X.

    method SPQR_QTX (0): Y = Q'*X
    method SPQR_QX  (1): Y = Q*X
    method SPQR_XQT (2): Y = X*Q'
    method SPQR_XQ  (3): Y = X*Q
*/

cholmod_dense *SuiteSparseQR_C_qmult /* returns Y, or NULL on failure */
(
    /* inputs: */
    int method,                 /* 0,1,2,3 */
    SuiteSparseQR_C_factorization *QR,  /* of an m-by-n sparse matrix A */
    cholmod_dense *X,           /* size m-by-n with leading dimension ldx */
    cholmod_common *cc          /* workspace and parameters */
) ;

#endif

/* ========================================================================== */

#ifdef __cplusplus
}
#endif
#endif
