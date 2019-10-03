/* ========================================================================== */
/* === qrdemo.c ============================================================= */
/* ========================================================================== */

/* A simple C demo of SuiteSparseQR.  The comments give the MATLAB equivalent
   statements.  See also qrdemo.m
 */

#include "SuiteSparseQR_C.h"

/* SuiteSparseQR uses an integer defined in SuiteSparse_config.h called
 * SuiteSparse_long.  It is a 32-bit integer on a 32-bit platform, and a 64-bit
 * integer on a 64-bit platform.  For most platforms (except Windows),
 * SuiteSparse_long is just "long". */

#define Long SuiteSparse_long

int main (int argc, char **argv)
{
    cholmod_common Common, *cc ;
    cholmod_sparse *A ;
    cholmod_dense *X, *B, *Residual ;
    double anorm, xnorm, rnorm, one [2] = {1,0}, minusone [2] = {-1,0} ;
    int mtype ;
    Long m, n, rnk ;

    /* start CHOLMOD */
    cc = &Common ;
    cholmod_l_start (cc) ;

    /* A = mread (stdin) ; read in the sparse matrix A */
    A = (cholmod_sparse *) cholmod_l_read_matrix (stdin, 1, &mtype, cc) ;
    if (mtype != CHOLMOD_SPARSE)
    {
        printf ("input matrix must be sparse\n") ;
        exit (1) ;
    }

    /* [m n] = size (A) ; */
    m = A->nrow ;
    n = A->ncol ;

    /* anorm = norm (A,1) ; */
    anorm = cholmod_l_norm_sparse (A, 1, cc) ;

    printf ("Matrix %6ld-by-%-6ld nnz: %6ld ", m, n, cholmod_l_nnz (A, cc)) ;

    /* B = ones (m,1), a dense right-hand-side of the same type as A */
    B = cholmod_l_ones (m, 1, A->xtype, cc) ;

    /* X = A\B ; with default ordering and default column 2-norm tolerance */
    X = SuiteSparseQR_C_backslash
        (SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, A, B, cc) ;

    /* get the rank(A) estimate */
    rnk = cc->SPQR_istat [4] ;

    /* rnorm = norm (A*X-B) */
    Residual = cholmod_l_copy_dense (B, cc) ;
    cholmod_l_sdmult (A, 0, one, minusone, X, Residual, cc) ;
    rnorm = cholmod_l_norm_dense (Residual, 2, cc) ;

    /* xnorm = norm (X) */
    xnorm = cholmod_l_norm_dense (X, 2, cc) ;

    if (m <= n && anorm > 0 && xnorm > 0)
    {
        /* find the relative residual, except for least-squares systems */
        rnorm /= (anorm * xnorm) ;
    }
    printf ("residual: %8.1e rank: %6ld\n", rnorm, rnk) ;

    /* free everything */
    cholmod_l_free_dense (&Residual, cc) ;
    cholmod_l_free_sparse (&A, cc) ;
    cholmod_l_free_dense (&X, cc) ;
    cholmod_l_free_dense (&B, cc) ;
    cholmod_l_finish (cc) ;
    return (0) ;
}
