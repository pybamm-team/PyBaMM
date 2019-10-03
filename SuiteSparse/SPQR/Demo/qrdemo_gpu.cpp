// =============================================================================
// === qrdemo_gpu.cpp ==========================================================
// =============================================================================

// A simple C++ demo of SuiteSparseQR.  The comments give the MATLAB equivalent
// statements.  See also qrdemo.m

#include "SuiteSparseQR.hpp"
#include "SuiteSparseGPU_Runtime.hpp"
#include <complex>

int main (int argc, char **argv)
{
    cholmod_sparse *A ;
    cholmod_dense *X, *B, *r, *atr ;
    double anorm, xnorm, rnorm, one [2] = {1,0}, minusone [2] = {-1,0}, t ;
    double zero [2] = {0,0}, atrnorm ;
    int mtype ;
    long m, n, rnk ;
    size_t total_mem, available_mem ;

    printf ("\nqrdemo_gpu: Testing SPQR on the GPU:\n") ;

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
    const char *filename = (argc < 2 ? "Problems/2.mtx" : argv[1]);
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

#if 1
    printf ("Matrix %6ld-by-%-6ld nnz: %6ld\n",
        m, n, cholmod_l_nnz (A, cc)) ;
#endif

    // anorm = norm (A,1) ;
    anorm = cholmod_l_norm_sparse (A, 1, cc) ;

    // B = ones (m,1), a dense right-hand-side of the same type as A
    B = cholmod_l_ones (m, 1, A->xtype, cc) ;

    // X = A\B ; with default ordering and default column 2-norm tolerance
    if (A->xtype == CHOLMOD_REAL)
    {
        // A, X, and B are all real
        X = SuiteSparseQR <double>(ordering, SPQR_NO_TOL, A, B, cc) ;
    }
    else
    {
#if SUPPORTS_COMPLEX
        // A, X, and B are all complex
        X = SuiteSparseQR < std::complex<double> >
            (SPQR_ORDERING_DEFAULT, SPQR_NO_TOL, A, B, cc) ;
#else
        printf("Code doesn't support std::complex<?> types.\n");
#endif
    }

    // get the rank(A) estimate
    rnk = cc->SPQR_istat [4] ;

    // compute the residual r, and A'*r, and their norms
    r = cholmod_l_copy_dense (B, cc) ;                  // r = B
    cholmod_l_sdmult (A, 0, one, minusone, X, r, cc) ;  // r = A*X-r = A*x-b
    rnorm = cholmod_l_norm_dense (r, 2, cc) ;           // rnorm = norm (r)
    atr = cholmod_l_zeros (n, 1, CHOLMOD_REAL, cc) ;    // atr = zeros (n,1)
    cholmod_l_sdmult (A, 1, one, zero, r, atr, cc) ;    // atr = A'*r
    atrnorm = cholmod_l_norm_dense (atr, 2, cc) ;       // atrnorm = norm (atr)

    // xnorm = norm (X)
    xnorm = cholmod_l_norm_dense (X, 2, cc) ;

    // write out X to a file
    FILE *f = fopen ("X.mtx", "w") ;
    cholmod_l_write_dense (f, X, NULL, cc) ;
    fclose (f) ;

    if (m <= n && anorm > 0 && xnorm > 0)
    {
        // find the relative residual, except for least-squares systems
        rnorm /= (anorm * xnorm) ;
    }
    printf ("\nnorm(Ax-b): %8.1e\n", rnorm) ;
    printf ("norm(A'(Ax-b))         %8.1e rank: %ld of %ld\n", 
        atrnorm, rnk, (m < n) ? m:n) ;

    /* Write an info file. */
    FILE *info = fopen("gpu_results.txt", "w");
    fprintf(info, "%ld\n", cc->SPQR_istat[7]);        // ordering method
    fprintf(info, "%ld\n", cc->memory_usage);         // memory usage (bytes)
    fprintf(info, "%30.16e\n", cc->SPQR_flopcount);   // flop count
    fprintf(info, "%lf\n", cc->SPQR_analyze_time);    // analyze time
    fprintf(info, "%lf\n", cc->SPQR_factorize_time);  // factorize time
    fprintf(info, "-1\n") ;                           // cpu memory (bytes)
    fprintf(info, "-1\n") ;                           // gpu memory (bytes)
    fprintf(info, "%32.16e\n", rnorm);                // residual
    fprintf(info, "%ld\n", cholmod_l_nnz (A, cc));    // nnz(A)
    fprintf(info, "%ld\n", cc->SPQR_istat [0]);       // nnz(R)
    fprintf(info, "%ld\n", cc->SPQR_istat [2]);       // # of frontal matrices
    fprintf(info, "%ld\n", cc->SPQR_istat [3]);       // ntasks, for now
    fprintf(info, "%lf\n", cc->gpuKernelTime);        // kernel time (ms)
    fprintf(info, "%ld\n", cc->gpuFlops);             // "actual" gpu flops
    fprintf(info, "%d\n", cc->gpuNumKernelLaunches);  // # of kernel launches
    fprintf(info, "%32.16e\n", atrnorm) ;             // norm (A'*(Ax-b))

    fclose(info);

    // free everything
    cholmod_l_free_dense (&r, cc) ;
    cholmod_l_free_dense (&atr, cc) ;
    cholmod_l_free_sparse (&A, cc) ;
    cholmod_l_free_dense (&X, cc) ;
    cholmod_l_free_dense (&B, cc) ;
    cholmod_l_finish (cc) ;

    return (0) ;
}
