// =============================================================================
// === qrdemo_gpu2.cpp =========================================================
// =============================================================================

// A simple C++ demo of SuiteSparseQR.  The comments give the MATLAB equivalent
// statements.  See also qrdemo.m
//
// usage:
// qrdemo_gpu2 matrixfile orderingoption

#include "SuiteSparseQR.hpp"
#include "SuiteSparseGPU_Runtime.hpp"
#include <complex>

int main (int argc, char **argv)
{
    cholmod_sparse *A, *R ;
    cholmod_dense *B, *C ;
    SuiteSparse_long *E ;
    int mtype ;
    long m, n, rnk ;
    size_t total_mem, available_mem ;
    double t ;

    printf ("\nqrdemo_gpu2: Testing SPQR on the GPU:\n") ;

    // start CHOLMOD
    cholmod_common *cc, Common ;
    cc = &Common ;
    cholmod_l_start (cc) ;

    // warmup the GPU.  This can take some time, but only needs
    // to be done once
    cc->useGPU = true ;
    t = SuiteSparse_time ( ) ;
    cholmod_l_gpu_memorysize (&total_mem, &available_mem, cc) ;
    cc->gpuMemorySize = available_mem ;
    t = SuiteSparse_time ( ) - t ;
    if (cc->gpuMemorySize <= 1)
    {
        printf ("no GPU available\n") ;
    }
    printf ("available GPU memory: %g MB, warmup time: %g\n",
        (double) (cc->gpuMemorySize) / (1024 * 1024), t) ;

    // A = mread (stdin) ; read in the sparse matrix A
    const char *filename = argv[1];
    FILE *file = fopen(filename, "r");
    A = (cholmod_sparse *) cholmod_l_read_matrix (file, 1, &mtype, cc) ;
    fclose(file);
    if (mtype != CHOLMOD_SPARSE)
    {
        printf ("input matrix must be sparse\n") ;
        exit (1) ;
    }

    // [m n] = size (A) ;
    m = A->nrow ;
    n = A->ncol ;

    long ordering = (argc < 3 ? SPQR_ORDERING_DEFAULT : atoi(argv[2]));

    printf ("Matrix %6ld-by-%-6ld nnz: %6ld\n",
        m, n, cholmod_l_nnz (A, cc)) ;

    // B = ones (m,1), a dense right-hand-side of the same type as A
    B = cholmod_l_ones (m, 1, A->xtype, cc) ;

    double tol = SPQR_NO_TOL ;
    long econ = 0 ;

    // [Q,R,E] = qr (A), but discard Q
    // SuiteSparseQR <double> (ordering, tol, econ, A, &R, &E, cc) ;

    // [C,R,E] = qr (A,b), but discard Q
    SuiteSparseQR <double> (ordering, tol, econ, A, B, &C, &R, &E, cc) ;

    // now R'*R-A(:,E)'*A(:,E) should be epsilon
    // and C = Q'*b.  The solution to the least-squares problem
    // should be x=R\C.

    // write out R to a file
    FILE *f = fopen ("R.mtx", "w") ;
    cholmod_l_write_sparse (f, R, NULL, NULL, cc) ;
    fclose (f) ;

    // write out C to a file
    f = fopen ("C.mtx", "w") ;
    cholmod_l_write_dense (f, C, NULL, cc) ;
    fclose (f) ;

    // write out E to a file
    f = fopen ("E.txt", "w") ;
    for (long i = 0 ; i < n ; i++)
    {
        fprintf (f, "%ld\n", 1 + E [i]) ;
    }
    fclose (f) ;

    // free everything
    cholmod_l_free_sparse (&A, cc) ;
    cholmod_l_free_sparse (&R, cc) ;
    cholmod_l_free_dense  (&C, cc) ;
    // cholmod_l_free (&E, cc) ;
    cholmod_l_finish (cc) ;

    return (0) ;
}
