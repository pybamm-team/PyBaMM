// =============================================================================
// === qrdemo.cpp ==============================================================
// =============================================================================

// A simple C++ demo of SuiteSparseQR.  The comments give the MATLAB equivalent
// statements.  See also qrdemo.m

#include "SuiteSparseQR.hpp"
#include <complex>

// SuiteSparseQR uses an integer defined in SuiteSparse_config.h called
// SuiteSparse_long.  It is a 32-bit integer on a 32-bit platform, and a 64-bit
// integer on a 64-bit platform.  For most platforms (except Windows),
// SuiteSparse_long is just "long".

#define Long SuiteSparse_long

// =============================================================================
// check_residual:  print the relative residual, norm (A*x-b)/norm(x)
// =============================================================================

void check_residual
(
    cholmod_sparse *A,
    cholmod_dense *X,
    cholmod_dense *B,
    cholmod_common *cc
)
{
    Long m = A->nrow ;
    Long n = A->ncol ;
    Long rnk ;
    double rnorm, anorm, xnorm, atrnorm ;
    double one [2] = {1,0}, minusone [2] = {-1,0}, zero [2] = {0,0} ;
    cholmod_dense *r, *atr ;

    // get the rank(A) estimate
    rnk = cc->SPQR_istat [4] ;

#ifndef NMATRIXOPS
    // anorm = norm (A,1) ;
    anorm = cholmod_l_norm_sparse (A, 1, cc) ;

    // rnorm = norm (A*X-B)
    r = cholmod_l_copy_dense (B, cc) ;
    cholmod_l_sdmult (A, 0, one, minusone, X, r, cc) ;
    rnorm = cholmod_l_norm_dense (r, 2, cc) ;

    // xnorm = norm (X)
    xnorm = cholmod_l_norm_dense (X, 2, cc) ;

    // atrnorm = norm (A'*r)
    atr = cholmod_l_zeros (n, 1, r->xtype, cc) ;        // atr = zeros (n,1)
    cholmod_l_sdmult (A, 1, one, zero, r, atr, cc) ;    // atr = A'*r
    atrnorm = cholmod_l_norm_dense (atr, 2, cc) ;       // atrnorm = norm (atr)
    if (anorm > 0) atrnorm /= anorm ;

    if (m <= n && anorm > 0 && xnorm > 0)
    {
        // find the relative residual, except for least-squares systems
        rnorm /= (anorm * xnorm) ;
    }
    printf ("relative norm(Ax-b): %8.1e rank: %6ld  "
        "rel. norm(A'(Ax-b)) %8.1e\n", rnorm, rnk, atrnorm) ;
    cholmod_l_free_dense (&r, cc) ;
    cholmod_l_free_dense (&atr, cc) ;
#else
    printf ("relative norm(Ax-b): not computed (requires CHOLMOD/MatrixOps)\n");
    printf ("rank: %6ld\n", rnk) ;
#endif
}

// =============================================================================

int main (int argc, char **argv)
{
    cholmod_common Common, *cc ;
    cholmod_sparse *A ;
    cholmod_dense *X, *B ;
    int mtype ;
    Long m, n ;

    // start CHOLMOD
    cc = &Common ;
    cholmod_l_start (cc) ;

    // A = mread (stdin) ; read in the sparse matrix A
    A = (cholmod_sparse *) cholmod_l_read_matrix (stdin, 1, &mtype, cc) ;
    if (mtype != CHOLMOD_SPARSE)
    {
        printf ("input matrix must be sparse\n") ;
        exit (1) ;
    }

    // [m n] = size (A) ;
    m = A->nrow ;
    n = A->ncol ;

    printf ("Matrix %6ld-by-%-6ld nnz: %6ld\n", m, n, cholmod_l_nnz (A, cc)) ;

    // B = ones (m,1), a dense right-hand-side of the same type as A
    B = cholmod_l_ones (m, 1, A->xtype, cc) ;

    // X = A\B ; with default ordering and default column 2-norm tolerance
    if (A->xtype == CHOLMOD_REAL)
    {
        // A, X, and B are all real
        X = SuiteSparseQR <double>
            (SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, A, B, cc) ;
    }
    else
    {
        // A, X, and B are all complex
        X = SuiteSparseQR < std::complex<double> >
            (SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, A, B, cc) ;
    }

    check_residual (A, X, B, cc) ;
    cholmod_l_free_dense (&X, cc) ;

    // -------------------------------------------------------------------------
    // factorizing once then solving twice with different right-hand-sides
    // -------------------------------------------------------------------------

    // Just the real case.  Complex case is essentially identical
    if (A->xtype == CHOLMOD_REAL)
    {
        SuiteSparseQR_factorization <double> *QR ;
        cholmod_dense *Y ;
        Long i ;
        double *Bx ;

        // factorize once
        QR = SuiteSparseQR_factorize <double>
            (SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, A, cc) ;

        // solve Ax=b, using the same B as before

        // Y = Q'*B
        Y = SuiteSparseQR_qmult (SPQR_QTX, QR, B, cc) ;
        // X = R\(E*Y)
        X = SuiteSparseQR_solve (SPQR_RETX_EQUALS_B, QR, Y, cc) ;
        // check the results
        check_residual (A, X, B, cc) ;
        // free X and Y
        cholmod_l_free_dense (&Y, cc) ;
        cholmod_l_free_dense (&X, cc) ;

        // repeat with a different B
        Bx = (double *) (B->x) ;
        for (i = 0 ; i < m ; i++)
        {
            Bx [i] = i ;
        }

        // Y = Q'*B
        Y = SuiteSparseQR_qmult (SPQR_QTX, QR, B, cc) ;
        // X = R\(E*Y)
        X = SuiteSparseQR_solve (SPQR_RETX_EQUALS_B, QR, Y, cc) ;
        // check the results
        check_residual (A, X, B, cc) ;
        // free X and Y
        cholmod_l_free_dense (&Y, cc) ;
        cholmod_l_free_dense (&X, cc) ;

        // free QR
        SuiteSparseQR_free (&QR, cc) ;
    }

    // -------------------------------------------------------------------------
    // free everything that remains
    // -------------------------------------------------------------------------

    cholmod_l_free_sparse (&A, cc) ;
    cholmod_l_free_dense (&B, cc) ;
    cholmod_l_finish (cc) ;
    return (0) ;
}
