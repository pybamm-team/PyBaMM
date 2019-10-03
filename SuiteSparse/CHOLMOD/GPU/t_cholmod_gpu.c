/* ========================================================================== */
/* === GPU/t_cholmod_gpu ==================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/GPU Module.  Copyright (C) 2005-2012, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* GPU BLAS template routine for cholmod_super_numeric. */

/* ========================================================================== */
/* === include files and definitions ======================================== */
/* ========================================================================== */

#ifdef GPU_BLAS

#include <string.h>
#include "cholmod_template.h"
#include "cholmod_gpu_kernels.h"
#include <fenv.h>
#include <cuda.h>
#include <cuda_runtime.h>

#undef L_ENTRY
#ifdef REAL
#define L_ENTRY 1
#else
#define L_ENTRY 2
#endif


/* ========================================================================== */
/* === gpu_clear_memory ===================================================== */
/* ========================================================================== */
/*
 * Ensure the Lx is zeroed before forming factor.  This is a significant cost
 * in the GPU case - so using this parallel memset code for efficiency.
 */

void TEMPLATE2 (CHOLMOD (gpu_clear_memory))
(
    double* buff,
    size_t size,
    int num_threads
)
{
    int chunk_multiplier = 5;
    int num_chunks = chunk_multiplier * num_threads;
    size_t chunksize = size / num_chunks;
    size_t i;

#pragma omp parallel for num_threads(num_threads) private(i) schedule(dynamic)
    for(i = 0; i < num_chunks; i++) {
        size_t chunkoffset = i * chunksize;
        if(i == num_chunks - 1) {
            memset(buff + chunkoffset, 0, (size - chunksize*(num_chunks - 1)) *
                   sizeof(double));
        }
        else {
            memset(buff + chunkoffset, 0, chunksize * sizeof(double));
        }
    }
}

/* ========================================================================== */
/* === gpu_init ============================================================= */
/* ========================================================================== */
/*
 * Performs required initialization for GPU computing.
 *
 * Returns 0 if there is an error, so the intended use is
 *
 *     useGPU = CHOLMOD(gpu_init)
 *
 * which would locally turn off gpu processing if the initialization failed.
 */

int TEMPLATE2 (CHOLMOD (gpu_init))
(
    void *Cwork,
    cholmod_factor *L,
    cholmod_common *Common,
    Int nsuper,
    Int n,
    Int nls,
    cholmod_gpu_pointers *gpu_p
)
{
    Int i, k, maxSize ;
    cublasStatus_t cublasError ;
    cudaError_t cudaErr ;
    size_t maxBytesSize, HostPinnedSize ;

    feenableexcept (FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );

    maxSize = L->maxcsize;

    /* #define PAGE_SIZE (4*1024) */
    CHOLMOD_GPU_PRINTF (("gpu_init : %p\n",
        (void *) ((size_t) Cwork & ~(4*1024-1)))) ;

    /* make sure the assumed buffer sizes are large enough */
    if ( (nls+2*n+4)*sizeof(Int) > Common->devBuffSize ) {
        ERROR (CHOLMOD_GPU_PROBLEM,"\n\n"
               "GPU Memory allocation error.  Ls, Map and RelativeMap exceed\n"
               "devBuffSize.  It is not clear if this is due to insufficient\n"
               "device or host memory or both.  You can try:\n"
               "     1) increasing the amount of GPU memory requested\n"
               "     2) reducing CHOLMOD_NUM_HOST_BUFFERS\n"
               "     3) using a GPU & host with more memory\n"
               "This issue is a known limitation and should be fixed in a \n"
               "future release of CHOLMOD.\n") ;
        return (0) ;
    }

    /* divvy up the memory in dev_mempool */
    gpu_p->d_Lx[0] = Common->dev_mempool;
    gpu_p->d_Lx[1] = Common->dev_mempool + Common->devBuffSize;
    gpu_p->d_C = Common->dev_mempool + 2*Common->devBuffSize;
    gpu_p->d_A[0] = Common->dev_mempool + 3*Common->devBuffSize;
    gpu_p->d_A[1] = Common->dev_mempool + 4*Common->devBuffSize;
    gpu_p->d_Ls = Common->dev_mempool + 5*Common->devBuffSize;
    gpu_p->d_Map = gpu_p->d_Ls + (nls+1)*sizeof(Int) ;
    gpu_p->d_RelativeMap = gpu_p->d_Map + (n+1)*sizeof(Int) ;

    /* Copy all of the Ls and Lpi data to the device.  If any supernodes are
     * to be computed on the device then this will be needed, so might as
     * well do it now.   */

    cudaErr = cudaMemcpy ( gpu_p->d_Ls, L->s, nls*sizeof(Int),
                           cudaMemcpyHostToDevice );
    CHOLMOD_HANDLE_CUDA_ERROR(cudaErr,"cudaMemcpy(d_Ls)");

    if (!(Common->gpuStream[0])) {

        /* ------------------------------------------------------------------ */
        /* create each CUDA stream */
        /* ------------------------------------------------------------------ */

        for ( i=0; i<CHOLMOD_HOST_SUPERNODE_BUFFERS; i++ ) {
            cudaErr = cudaStreamCreate ( &(Common->gpuStream[i]) );
            if (cudaErr != cudaSuccess) {
                ERROR (CHOLMOD_GPU_PROBLEM, "CUDA stream") ;
                return (0) ;
            }
        }

        /* ------------------------------------------------------------------ */
        /* create each CUDA event */
        /* ------------------------------------------------------------------ */

        for (i = 0 ; i < 3 ; i++) {
            cudaErr = cudaEventCreateWithFlags
                (&(Common->cublasEventPotrf [i]), cudaEventDisableTiming) ;
            if (cudaErr != cudaSuccess) {
                ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event") ;
                return (0) ;
            }
        }

        for (i = 0 ; i < CHOLMOD_HOST_SUPERNODE_BUFFERS ; i++) {
            cudaErr = cudaEventCreateWithFlags
                (&(Common->updateCBuffersFree[i]), cudaEventDisableTiming) ;
            if (cudaErr != cudaSuccess) {
                ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event") ;
                return (0) ;
            }
        }

        cudaErr = cudaEventCreateWithFlags ( &(Common->updateCKernelsComplete),
                                             cudaEventDisableTiming );
        if (cudaErr != cudaSuccess) {
            ERROR (CHOLMOD_GPU_PROBLEM, "CUDA updateCKernelsComplete event") ;
            return (0) ;
        }

    }

    gpu_p->h_Lx[0] = (double*)(Common->host_pinned_mempool);
    for ( k=1; k<CHOLMOD_HOST_SUPERNODE_BUFFERS; k++ ) {
        gpu_p->h_Lx[k] = (double*)((char *)(Common->host_pinned_mempool) +
                                   k*Common->devBuffSize);
    }

    return (1);  /* initialization successfull, useGPU = 1 */

}


/* ========================================================================== */
/* === gpu_reorder_descendants ============================================== */
/* ========================================================================== */
/* Reorder the descendant supernodes as:
 *    1st - descendant supernodes eligible for processing on the GPU
 *          in increasing (by flops) order
 *    2nd - supernodes whose processing is to remain on the CPU
 *          in any order
 *
 * All of the GPU-eligible supernodes will be scheduled first.  All
 * CPU-eligible descendants will overlap with the last (largest)
 * CHOLMOD_HOST_SUPERNODE_BUFFERS GPU-eligible descendants.
 */

void TEMPLATE2 (CHOLMOD (gpu_reorder_descendants))
(
    cholmod_common *Common,
    Int *Super,
    Int *locals,
    Int *Lpi,
    Int *Lpos,
    Int *Head,
    Int *Next,
    Int *Previous,
    Int *ndescendants,
    Int *tail,
    Int *mapCreatedOnGpu,
    cholmod_gpu_pointers *gpu_p
)
{

    Int prevd, nextd, firstcpu, d, k, kd1, kd2, ndcol, pdi, pdend, pdi1;
    Int dnext, ndrow2, p;
    Int n_descendant = 0;
    double score;

    /* use h_Lx[0] to buffer the GPU-eligible descendants */
    struct cholmod_descendant_score_t* scores =
        (struct cholmod_descendant_score_t*) gpu_p->h_Lx[0];

    double cpuref = 0.0;

    int nreverse = 1;
    int previousd;


    d = Head[*locals];
    prevd = -1;
    firstcpu = -1;
    *mapCreatedOnGpu = 0;

    while ( d != EMPTY )
    {

        /* Get the parameters for the current descendant supernode */
        kd1 = Super [d] ;       /* d contains cols kd1 to kd2-1 of L */
        kd2 = Super [d+1] ;
        ndcol = kd2 - kd1 ;     /* # of columns in all of d */
        pdi = Lpi [d] ;         /* pointer to first row of d in Ls */
        pdend = Lpi [d+1] ;     /* pointer just past last row of d in Ls */
        p = Lpos [d] ;          /* offset of 1st row of d affecting s */
        pdi1 = pdi + p ;        /* ptr to 1st row of d affecting s in Ls */
        ndrow2 = pdend - pdi1;

        nextd = Next[d];

        /* compute a rough flops 'score' for this descendant supernode */
        score = ndrow2 * ndcol;
        if ( ndrow2*L_ENTRY >= CHOLMOD_ND_ROW_LIMIT &&
             ndcol*L_ENTRY >= CHOLMOD_ND_COL_LIMIT ) {
            score += Common->devBuffSize;
        }

        /* place in sort buffer */
        scores[n_descendant].score = score;
        scores[n_descendant].d = d;
        n_descendant++;

        d = nextd;

    }

    /* Sort the GPU-eligible supernodes */
    qsort ( scores, n_descendant, sizeof(struct cholmod_descendant_score_t),
            (__compar_fn_t) CHOLMOD(score_comp) );

    /* Place sorted data back in descendant supernode linked list*/
    if ( n_descendant > 0 ) {
        Head[*locals] = scores[0].d;
        if ( n_descendant > 1 ) {

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
    if (n_descendant > 64)

            for ( k=1; k<n_descendant; k++ ) {
                Next[scores[k-1].d] = scores[k].d;
            }
        }
        Next[scores[n_descendant-1].d] = firstcpu;
    }

    /* reverse the first CHOLMOD_HOST_SUPERNODE_BUFFERS to better hide PCIe
       communications */

    if ( Head[*locals] != EMPTY && Next[Head[*locals]] != EMPTY ) {
        previousd = Head[*locals];
        d = Next[Head[*locals]];
        while ( d!=EMPTY && nreverse < CHOLMOD_HOST_SUPERNODE_BUFFERS ) {

            kd1 = Super [d] ;       /* d contains cols kd1 to kd2-1 of L */
            kd2 = Super [d+1] ;
            ndcol = kd2 - kd1 ;     /* # of columns in all of d */
            pdi = Lpi [d] ;         /* pointer to first row of d in Ls */
            pdend = Lpi [d+1] ;     /* pointer just past last row of d in Ls */
            p = Lpos [d] ;          /* offset of 1st row of d affecting s */
            pdi1 = pdi + p ;        /* ptr to 1st row of d affecting s in Ls */
            ndrow2 = pdend - pdi1;

            nextd = Next[d];

            nreverse++;

            if ( ndrow2*L_ENTRY >= CHOLMOD_ND_ROW_LIMIT && ndcol*L_ENTRY >=
                 CHOLMOD_ND_COL_LIMIT ) {
                /* place this supernode at the front of the list */
                Next[previousd] = Next[d];
                Next[d] = Head[*locals];
                Head[*locals] = d;
            }
            else {
                previousd = d;
            }
            d = nextd;
        }
    }

    /* create a 'previous' list so we can traverse backwards */
    *ndescendants = 0;
    if ( Head[*locals] != EMPTY ) {
        Previous[Head[*locals]] = EMPTY;
        for (d = Head [*locals] ; d != EMPTY ; d = dnext) {
            (*ndescendants)++;
            dnext = Next[d];
            if ( dnext != EMPTY ) {
                Previous[dnext] = d;
            }
            else {
                *tail = d;
            }
        }
    }

    return;

}

/* ========================================================================== */
/* === gpu_initialize_supernode ============================================= */
/* ========================================================================== */

/* C = L (k1:n-1, kd1:kd2-1) * L (k1:k2-1, kd1:kd2-1)', except that k1:n-1
 */

void TEMPLATE2 (CHOLMOD (gpu_initialize_supernode))
(
    cholmod_common *Common,
    Int nscol,
    Int nsrow,
    Int psi,
    cholmod_gpu_pointers *gpu_p
)
{

    cudaError_t cuErr;

    /* initialize the device supernode assemby memory to zero */
    cuErr = cudaMemset ( gpu_p->d_A[0], 0, nscol*nsrow*L_ENTRY*sizeof(double) );
    CHOLMOD_HANDLE_CUDA_ERROR(cuErr,"cudaMemset(d_A)");

    /* Create the Map on the device */
    createMapOnDevice ( (Int *)(gpu_p->d_Map),
                        (Int *)(gpu_p->d_Ls), psi, nsrow );

    return;

}


/* ========================================================================== */
/* === gpu_updateC ========================================================== */
/* ========================================================================== */

/* C = L (k1:n-1, kd1:kd2-1) * L (k1:k2-1, kd1:kd2-1)', except that k1:n-1
 * refers to all of the rows in L, but many of the rows are all zero.
 * Supernode d holds columns kd1 to kd2-1 of L.  Nonzero rows in the range
 * k1:k2-1 are in the list Ls [pdi1 ... pdi2-1], of size ndrow1.  Nonzero rows
 * in the range k2:n-1 are in the list Ls [pdi2 ... pdend], of size ndrow2.
 * Let L1 = L (Ls [pdi1 ... pdi2-1], kd1:kd2-1), and let L2 = L (Ls [pdi2 ...
 * pdend],  kd1:kd2-1).  C is ndrow2-by-ndrow1.  Let C1 be the first ndrow1
 * rows of C and let C2 be the last ndrow2-ndrow1 rows of C.  Only the lower
 * triangular part of C1 needs to be computed since C1 is symmetric.
 *
 * UpdateC is completely asynchronous w.r.t. the GPU.  Once the input buffer
 * d_Lx[] has been filled, all of the device operations are issues, and the
 * host can continue with filling the next input buffer / or start processing
 * all of the descendant supernodes which are not eligible for processing on
 * the device (since they are too small - will not fill the device).
 */

int TEMPLATE2 (CHOLMOD (gpu_updateC))
(
    Int ndrow1,         /* C is ndrow2-by-ndrow2 */
    Int ndrow2,
    Int ndrow,          /* leading dimension of Lx */
    Int ndcol,          /* L1 is ndrow1-by-ndcol */
    Int nsrow,
    Int pdx1,           /* L1 starts at Lx + L_ENTRY*pdx1 */
    /* L2 starts at Lx + L_ENTRY*(pdx1 + ndrow1) */
    Int pdi1,
    double *Lx,
    double *C,
    cholmod_common *Common,
    cholmod_gpu_pointers *gpu_p
)
{
    double *devPtrLx, *devPtrC ;
    double alpha, beta ;
    cublasStatus_t cublasStatus ;
    cudaError_t cudaStat [2] ;
    Int ndrow3 ;
    int icol, irow;
    int iHostBuff, iDevBuff ;

#ifndef NTIMER
    double tstart = 0;
#endif

    if ((ndrow2*L_ENTRY < CHOLMOD_ND_ROW_LIMIT) ||
        (ndcol*L_ENTRY <  CHOLMOD_ND_COL_LIMIT))
    {
        /* too small for the CUDA BLAS; use the CPU instead */
        return (0) ;
    }

    ndrow3 = ndrow2 - ndrow1 ;

#ifndef NTIMER
    Common->syrkStart = SuiteSparse_time ( ) ;
    Common->CHOLMOD_GPU_SYRK_CALLS++ ;
#endif

    /* ---------------------------------------------------------------------- */
    /* allocate workspace on the GPU */
    /* ---------------------------------------------------------------------- */

    iHostBuff = (Common->ibuffer)%CHOLMOD_HOST_SUPERNODE_BUFFERS;
    iDevBuff = (Common->ibuffer)%CHOLMOD_DEVICE_STREAMS;

    /* cycle the device Lx buffer, d_Lx, through CHOLMOD_DEVICE_STREAMS,
       usually 2, so we can overlap the copy of this descendent supernode
       with the compute of the previous descendant supernode */
    devPtrLx = (double *)(gpu_p->d_Lx[iDevBuff]);
    /* very little overlap between kernels for difference descendant supernodes
       (since we enforce the supernodes must be large enough to fill the
       device) so we only need one C buffer */
    devPtrC = (double *)(gpu_p->d_C);

    /* ---------------------------------------------------------------------- */
    /* copy Lx to the GPU */
    /* ---------------------------------------------------------------------- */

    /* copy host data to pinned buffer first for better H2D bandwidth */
#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS) if (ndcol > 32)
    for ( icol=0; icol<ndcol; icol++ ) {
        for ( irow=0; irow<ndrow2*L_ENTRY; irow++ ) {
            gpu_p->h_Lx[iHostBuff][icol*ndrow2*L_ENTRY+irow] =
                Lx[pdx1*L_ENTRY+icol*ndrow*L_ENTRY + irow];
        }
    }

    cudaStat[0] = cudaMemcpyAsync ( devPtrLx,
        gpu_p->h_Lx[iHostBuff],
        ndrow2*ndcol*L_ENTRY*sizeof(devPtrLx[0]),
        cudaMemcpyHostToDevice,
        Common->gpuStream[iDevBuff] );

    if ( cudaStat[0] ) {
        CHOLMOD_GPU_PRINTF ((" ERROR cudaMemcpyAsync = %d \n", cudaStat[0]));
        return (0);
    }

    /* make the current stream wait for kernels in previous streams */
    cudaStreamWaitEvent ( Common->gpuStream[iDevBuff],
                          Common->updateCKernelsComplete, 0 ) ;

    /* ---------------------------------------------------------------------- */
    /* create the relative map for this descendant supernode */
    /* ---------------------------------------------------------------------- */

    createRelativeMapOnDevice ( (Int *)(gpu_p->d_Map),
                                (Int *)(gpu_p->d_Ls),
                                (Int *)(gpu_p->d_RelativeMap),
                                pdi1, ndrow2,
                                &(Common->gpuStream[iDevBuff]) );

    /* ---------------------------------------------------------------------- */
    /* do the CUDA SYRK */
    /* ---------------------------------------------------------------------- */

    cublasStatus = cublasSetStream (Common->cublasHandle,
                                    Common->gpuStream[iDevBuff]) ;
    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS stream") ;
    }

    alpha  = 1.0 ;
    beta   = 0.0 ;

#ifdef REAL
    cublasStatus = cublasDsyrk (Common->cublasHandle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        (int) ndrow1,
        (int) ndcol,    /* N, K: L1 is ndrow1-by-ndcol */
        &alpha,         /* ALPHA:  1 */
        devPtrLx,
        ndrow2,         /* A, LDA: L1, ndrow2 */
        &beta,          /* BETA:   0 */
        devPtrC,
        ndrow2) ;       /* C, LDC: C1 */
#else
    cublasStatus = cublasZherk (Common->cublasHandle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        (int) ndrow1,
        (int) ndcol,    /* N, K: L1 is ndrow1-by-ndcol*/
        &alpha,         /* ALPHA:  1 */
        (const cuDoubleComplex *) devPtrLx,
        ndrow2,         /* A, LDA: L1, ndrow2 */
        &beta,          /* BETA:   0 */
        (cuDoubleComplex *) devPtrC,
        ndrow2) ;       /* C, LDC: C1 */
#endif

    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
    {
        ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS routine failure") ;
    }

#ifndef NTIMER
    Common->CHOLMOD_GPU_SYRK_TIME += SuiteSparse_time() - Common->syrkStart;
#endif

    /* ---------------------------------------------------------------------- */
    /* compute remaining (ndrow2-ndrow1)-by-ndrow1 block of C, C2 = L2*L1'    */
    /* ---------------------------------------------------------------------- */

#ifndef NTIMER
    Common->CHOLMOD_GPU_GEMM_CALLS++ ;
    tstart = SuiteSparse_time();
#endif

    if (ndrow3 > 0)
    {
#ifndef REAL
        cuDoubleComplex calpha  = {1.0,0.0} ;
        cuDoubleComplex cbeta   = {0.0,0.0} ;
#endif

        /* ------------------------------------------------------------------ */
        /* do the CUDA BLAS dgemm */
        /* ------------------------------------------------------------------ */

#ifdef REAL
        alpha  = 1.0 ;
        beta   = 0.0 ;
        cublasStatus = cublasDgemm (Common->cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            ndrow3, ndrow1, ndcol,          /* M, N, K */
            &alpha,                         /* ALPHA:  1 */
            devPtrLx + L_ENTRY*(ndrow1),    /* A, LDA: L2*/
            ndrow2,                         /* ndrow */
            devPtrLx,                       /* B, LDB: L1 */
            ndrow2,                         /* ndrow */
            &beta,                          /* BETA:   0 */
            devPtrC + L_ENTRY*ndrow1,       /* C, LDC: C2 */
            ndrow2) ;
#else
        cublasStatus = cublasZgemm (Common->cublasHandle,
            CUBLAS_OP_N, CUBLAS_OP_C,
            ndrow3, ndrow1, ndcol,          /* M, N, K */
            &calpha,                        /* ALPHA:  1 */
            (const cuDoubleComplex*) devPtrLx + ndrow1,
            ndrow2,                         /* ndrow */
            (const cuDoubleComplex *) devPtrLx,
            ndrow2,                         /* ndrow */
            &cbeta,                         /* BETA:   0 */
            (cuDoubleComplex *)devPtrC + ndrow1,
            ndrow2) ;
#endif

        if (cublasStatus != CUBLAS_STATUS_SUCCESS)
        {
            ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS routine failure") ;
        }

    }

#ifndef NTIMER
    Common->CHOLMOD_GPU_GEMM_TIME += SuiteSparse_time() - tstart;
#endif

    /* ------------------------------------------------------------------ */
    /* Assemble the update C on the device using the d_RelativeMap */
    /* ------------------------------------------------------------------ */

#ifdef REAL
    addUpdateOnDevice ( gpu_p->d_A[0], devPtrC,
        gpu_p->d_RelativeMap, ndrow1, ndrow2, nsrow,
        &(Common->gpuStream[iDevBuff]) );
#else
    addComplexUpdateOnDevice ( gpu_p->d_A[0], devPtrC,
        gpu_p->d_RelativeMap, ndrow1, ndrow2, nsrow,
        &(Common->gpuStream[iDevBuff]) );
#endif

    /* Record an event indicating that kernels for
       this descendant are complete */
    cudaEventRecord ( Common->updateCKernelsComplete,
                      Common->gpuStream[iDevBuff]);
    cudaEventRecord ( Common->updateCBuffersFree[iHostBuff],
                      Common->gpuStream[iDevBuff]);

    return (1) ;
}

/* ========================================================================== */
/* === gpu_final_assembly =================================================== */
/* ========================================================================== */

/* If the supernode was assembled on both the CPU and the GPU, this will
 * complete the supernode assembly on both the GPU and CPU.
 */

void TEMPLATE2 (CHOLMOD (gpu_final_assembly))
(
    cholmod_common *Common,
    double *Lx,
    Int psx,
    Int nscol,
    Int nsrow,
    int supernodeUsedGPU,
    int *iHostBuff,
    int *iDevBuff,
    cholmod_gpu_pointers *gpu_p
)
{
    Int iidx, i, j;
    Int iHostBuff2 ;
    Int iDevBuff2 ;

    if ( supernodeUsedGPU ) {

        /* ------------------------------------------------------------------ */
        /* Apply all of the Shur-complement updates, computed on the gpu, to */
        /* the supernode. */
        /* ------------------------------------------------------------------ */

        *iHostBuff = (Common->ibuffer)%CHOLMOD_HOST_SUPERNODE_BUFFERS;
        *iDevBuff = (Common->ibuffer)%CHOLMOD_DEVICE_STREAMS;

        if ( nscol * L_ENTRY >= CHOLMOD_POTRF_LIMIT ) {

            /* If this supernode is going to be factored using the GPU (potrf)
             * then it will need the portion of the update assembled ont the
             * CPU.  So copy that to a pinned buffer an H2D copy to device. */

            /* wait until a buffer is free */
            cudaEventSynchronize ( Common->updateCBuffersFree[*iHostBuff] );

            /* copy update assembled on CPU to a pinned buffer */

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
    private(iidx) if (nscol>32)

            for ( j=0; j<nscol; j++ ) {
                for ( i=j; i<nsrow*L_ENTRY; i++ ) {
                    iidx = j*nsrow*L_ENTRY+i;
                    gpu_p->h_Lx[*iHostBuff][iidx] = Lx[psx*L_ENTRY+iidx];
                }
            }

            /* H2D transfer of update assembled on CPU */
            cudaMemcpyAsync ( gpu_p->d_A[1], gpu_p->h_Lx[*iHostBuff],
                              nscol*nsrow*L_ENTRY*sizeof(double),
                              cudaMemcpyHostToDevice,
                              Common->gpuStream[*iDevBuff] );
        }

        Common->ibuffer++;

        iHostBuff2 = (Common->ibuffer)%CHOLMOD_HOST_SUPERNODE_BUFFERS;
        iDevBuff2 = (Common->ibuffer)%CHOLMOD_DEVICE_STREAMS;

        /* wait for all kernels to complete */
        cudaEventSynchronize( Common->updateCKernelsComplete );

        /* copy assembled Schur-complement updates computed on GPU */
        cudaMemcpyAsync ( gpu_p->h_Lx[iHostBuff2], gpu_p->d_A[0],
                          nscol*nsrow*L_ENTRY*sizeof(double),
                          cudaMemcpyDeviceToHost,
                          Common->gpuStream[iDevBuff2] );

        if ( nscol * L_ENTRY >= CHOLMOD_POTRF_LIMIT ) {

            /* with the current implementation, potrf still uses data from the
             * CPU - so put the fully assembled supernode in a pinned buffer for
             * fastest access */

            /* need both H2D and D2H copies to be complete */
            cudaDeviceSynchronize();

            /* sum updates from cpu and device on device */
#ifdef REAL
            sumAOnDevice ( gpu_p->d_A[1], gpu_p->d_A[0], -1.0, nsrow, nscol );
#else
            sumComplexAOnDevice ( gpu_p->d_A[1], gpu_p->d_A[0],
                                  -1.0, nsrow, nscol );
#endif

            /* place final assembled supernode in pinned buffer */

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
    private(iidx) if (nscol>32)

            for ( j=0; j<nscol; j++ ) {
                for ( i=j*L_ENTRY; i<nscol*L_ENTRY; i++ ) {
                    iidx = j*nsrow*L_ENTRY+i;
                    gpu_p->h_Lx[*iHostBuff][iidx] -=
                        gpu_p->h_Lx[iHostBuff2][iidx];
                }
            }

        }
        else
        {

            /* assemble with CPU updates */
            cudaDeviceSynchronize();

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
    private(iidx) if (nscol>32)

            for ( j=0; j<nscol; j++ ) {
                for ( i=j*L_ENTRY; i<nsrow*L_ENTRY; i++ ) {
                    iidx = j*nsrow*L_ENTRY+i;
                    Lx[psx*L_ENTRY+iidx] -= gpu_p->h_Lx[iHostBuff2][iidx];
                }
            }
        }
    }
    return;
}


/* ========================================================================== */
/* === gpu_lower_potrf ====================================================== */
/* ========================================================================== */

/* Cholesky factorzation (dpotrf) of a matrix S, operating on the lower
 * triangular part only.   S is nscol2-by-nscol2 with leading dimension nsrow.
 *
 * S is the top part of the supernode (the lower triangular matrx).
 * This function also copies the bottom rectangular part of the supernode (B)
 * onto the GPU, in preparation for gpu_triangular_solve.
 */

/*
 * On entry, d_A[1] contains the fully assembled supernode
 */

int TEMPLATE2 (CHOLMOD (gpu_lower_potrf))
(
    Int nscol2,     /* S is nscol2-by-nscol2 */
    Int nsrow,      /* leading dimension of S */
    Int psx,        /* S is located at Lx + L_ENTRY*psx */
    double *Lx,     /* contains S; overwritten with Cholesky factor */
    Int *info,      /* BLAS info return value */
    cholmod_common *Common,
    cholmod_gpu_pointers *gpu_p
)
{
    double *devPtrA, *devPtrB, *A ;
    double alpha, beta ;
    cudaError_t cudaStat ;
    cublasStatus_t cublasStatus ;
    Int j, nsrow2, nb, n, gpu_lda, lda, gpu_ldb ;
    int ilda, ijb, iinfo ;
#ifndef NTIMER
    double tstart ;
#endif

    if (nscol2 * L_ENTRY < CHOLMOD_POTRF_LIMIT)
    {
        /* too small for the CUDA BLAS; use the CPU instead */
        return (0) ;
    }

#ifndef NTIMER
    tstart = SuiteSparse_time ( ) ;
    Common->CHOLMOD_GPU_POTRF_CALLS++ ;
#endif

    nsrow2 = nsrow - nscol2 ;

    /* ---------------------------------------------------------------------- */
    /* heuristic to get the block size depending of the problem size */
    /* ---------------------------------------------------------------------- */

    nb = 128 ;
    if (nscol2 > 4096) nb = 256 ;
    if (nscol2 > 8192) nb = 384 ;
    n  = nscol2 ;
    gpu_lda = ((nscol2+31)/32)*32 ;
    lda = nsrow ;

    A = gpu_p->h_Lx[(Common->ibuffer+CHOLMOD_HOST_SUPERNODE_BUFFERS-1)%
                    CHOLMOD_HOST_SUPERNODE_BUFFERS];

    /* ---------------------------------------------------------------------- */
    /* determine the GPU leading dimension of B */
    /* ---------------------------------------------------------------------- */

    gpu_ldb = 0 ;
    if (nsrow2 > 0)
    {
        gpu_ldb = ((nsrow2+31)/32)*32 ;
    }

    /* ---------------------------------------------------------------------- */
    /* remember where device memory is, to be used by triangular solve later */
    /* ---------------------------------------------------------------------- */

    devPtrA = gpu_p->d_Lx[0];
    devPtrB = gpu_p->d_Lx[1];

    /* ---------------------------------------------------------------------- */
    /* copy A from device to device */
    /* ---------------------------------------------------------------------- */

    cudaStat = cudaMemcpy2DAsync ( devPtrA,
       gpu_lda * L_ENTRY * sizeof (devPtrA[0]),
       gpu_p->d_A[1],
       nsrow * L_ENTRY * sizeof (Lx[0]),
       nscol2 * L_ENTRY * sizeof (devPtrA[0]),
       nscol2,
       cudaMemcpyDeviceToDevice,
       Common->gpuStream[0] );

    if ( cudaStat ) {
        ERROR ( CHOLMOD_GPU_PROBLEM, "GPU memcopy device to device");
    }

    /* ---------------------------------------------------------------------- */
    /* copy B in advance, for gpu_triangular_solve */
    /* ---------------------------------------------------------------------- */

    if (nsrow2 > 0)
    {
        cudaStat = cudaMemcpy2DAsync (devPtrB,
            gpu_ldb * L_ENTRY * sizeof (devPtrB [0]),
            gpu_p->d_A[1] + L_ENTRY*nscol2,
            nsrow * L_ENTRY * sizeof (Lx [0]),
            nsrow2 * L_ENTRY * sizeof (devPtrB [0]),
            nscol2,
            cudaMemcpyDeviceToDevice,
            Common->gpuStream[0]) ;
        if (cudaStat)
        {
            ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy to device") ;
        }
    }

    /* ------------------------------------------------------------------ */
    /* define the dpotrf stream */
    /* ------------------------------------------------------------------ */

    cublasStatus = cublasSetStream (Common->cublasHandle,
                                    Common->gpuStream [0]) ;
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS stream") ;
    }

    /* ---------------------------------------------------------------------- */
    /* block Cholesky factorization of S */
    /* ---------------------------------------------------------------------- */

    for (j = 0 ; j < n ; j += nb)
    {
        Int jb = nb < (n-j) ? nb : (n-j) ;

        /* ------------------------------------------------------------------ */
        /* do the CUDA BLAS dsyrk */
        /* ------------------------------------------------------------------ */

        alpha = -1.0 ;
        beta  = 1.0 ;
#ifdef REAL
        cublasStatus = cublasDsyrk (Common->cublasHandle,
            CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, jb, j,
            &alpha, devPtrA + j, gpu_lda,
            &beta,  devPtrA + j + j*gpu_lda, gpu_lda) ;

#else
        cublasStatus = cublasZherk (Common->cublasHandle,
            CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, jb, j,
            &alpha, (cuDoubleComplex*)devPtrA + j,
            gpu_lda,
            &beta,
            (cuDoubleComplex*)devPtrA + j + j*gpu_lda,
            gpu_lda) ;
#endif

        if (cublasStatus != CUBLAS_STATUS_SUCCESS)
        {
            ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS routine failure") ;
        }

        /* ------------------------------------------------------------------ */

        cudaStat = cudaEventRecord (Common->cublasEventPotrf [0],
                                    Common->gpuStream [0]) ;
        if (cudaStat)
        {
            ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event failure") ;
        }

        cudaStat = cudaStreamWaitEvent (Common->gpuStream [1],
                                        Common->cublasEventPotrf [0], 0) ;
        if (cudaStat)
        {
            ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event failure") ;
        }

        /* ------------------------------------------------------------------ */
        /* copy back the jb columns on two different streams */
        /* ------------------------------------------------------------------ */

        cudaStat = cudaMemcpy2DAsync (A + L_ENTRY*(j + j*lda),
            lda * L_ENTRY * sizeof (double),
            devPtrA + L_ENTRY*(j + j*gpu_lda),
            gpu_lda * L_ENTRY * sizeof (double),
            L_ENTRY * sizeof (double)*jb,
            jb,
            cudaMemcpyDeviceToHost,
            Common->gpuStream [1]) ;

        if (cudaStat)
        {
            ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy from device") ;
        }

        /* ------------------------------------------------------------------ */
        /* do the CUDA BLAS dgemm */
        /* ------------------------------------------------------------------ */

        if ((j+jb) < n)
        {

#ifdef REAL
            alpha = -1.0 ;
            beta  = 1.0 ;
            cublasStatus = cublasDgemm (Common->cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                (n-j-jb), jb, j,
                &alpha,
                devPtrA + (j+jb), gpu_lda,
                devPtrA + (j)  , gpu_lda,
                &beta,
                devPtrA + (j+jb + j*gpu_lda), gpu_lda) ;

#else
            cuDoubleComplex calpha = {-1.0,0.0} ;
            cuDoubleComplex cbeta  = { 1.0,0.0} ;
            cublasStatus = cublasZgemm (Common->cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_C,
                (n-j-jb), jb, j,
                &calpha,
                (cuDoubleComplex*)devPtrA + (j+jb),
                gpu_lda,
                (cuDoubleComplex*)devPtrA + (j),
                gpu_lda,
                &cbeta,
                (cuDoubleComplex*)devPtrA +
                (j+jb + j*gpu_lda),
                gpu_lda ) ;
#endif

            if (cublasStatus != CUBLAS_STATUS_SUCCESS)
            {
                ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS routine failure") ;
            }
        }

        cudaStat = cudaStreamSynchronize (Common->gpuStream [1]) ;
        if (cudaStat)
        {
            ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy to device") ;
        }

        /* ------------------------------------------------------------------ */
        /* compute the Cholesky factorization of the jbxjb block on the CPU */
        /* ------------------------------------------------------------------ */

        ilda = (int) lda ;
        ijb  = jb ;
#ifdef REAL
        LAPACK_DPOTRF ("L", &ijb, A + L_ENTRY * (j + j*lda), &ilda, &iinfo) ;
#else
        LAPACK_ZPOTRF ("L", &ijb, A + L_ENTRY * (j + j*lda), &ilda, &iinfo) ;
#endif
        *info = iinfo ;

        if (*info != 0)
        {
            *info = *info + j ;
            break ;
        }

        /* ------------------------------------------------------------------ */
        /* copy the result back to the GPU */
        /* ------------------------------------------------------------------ */

        cudaStat = cudaMemcpy2DAsync (devPtrA + L_ENTRY*(j + j*gpu_lda),
            gpu_lda * L_ENTRY * sizeof (double),
            A + L_ENTRY * (j + j*lda),
            lda * L_ENTRY * sizeof (double),
            L_ENTRY * sizeof (double) * jb,
            jb,
            cudaMemcpyHostToDevice,
            Common->gpuStream [0]) ;

        if (cudaStat)
        {
            ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy to device") ;
        }

        /* ------------------------------------------------------------------ */
        /* do the CUDA BLAS dtrsm */
        /* ------------------------------------------------------------------ */

        if ((j+jb) < n)
        {

#ifdef REAL
            alpha  = 1.0 ;
            cublasStatus = cublasDtrsm (Common->cublasHandle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                (n-j-jb), jb,
                &alpha,
                devPtrA + (j + j*gpu_lda), gpu_lda,
                devPtrA + (j+jb + j*gpu_lda), gpu_lda) ;
#else
            cuDoubleComplex calpha  = {1.0,0.0};
            cublasStatus = cublasZtrsm (Common->cublasHandle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT,
                (n-j-jb), jb,
                &calpha,
                (cuDoubleComplex *)devPtrA +
                (j + j*gpu_lda),
                gpu_lda,
                (cuDoubleComplex *)devPtrA +
                (j+jb + j*gpu_lda),
                gpu_lda) ;
#endif

            if (cublasStatus != CUBLAS_STATUS_SUCCESS)
            {
                ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS routine failure") ;
            }

            /* -------------------------------------------------------------- */
            /* Copy factored column back to host.                             */
            /* -------------------------------------------------------------- */

            cudaStat = cudaEventRecord (Common->cublasEventPotrf[2],
                                        Common->gpuStream[0]) ;
            if (cudaStat)
            {
                ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event failure") ;
            }

            cudaStat = cudaStreamWaitEvent (Common->gpuStream[1],
                                            Common->cublasEventPotrf[2], 0) ;
            if (cudaStat)
            {
                ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event failure") ;
            }

            cudaStat = cudaMemcpy2DAsync (A + L_ENTRY*(j + jb + j * lda),
                  lda * L_ENTRY * sizeof (double),
                  devPtrA + L_ENTRY*
                  (j + jb + j * gpu_lda),
                  gpu_lda * L_ENTRY * sizeof (double),
                  L_ENTRY * sizeof (double)*
                  (n - j - jb), jb,
                  cudaMemcpyDeviceToHost,
                  Common->gpuStream[1]) ;

            if (cudaStat)
            {
                ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy to device") ;
            }
        }
    }

#ifndef NTIMER
    Common->CHOLMOD_GPU_POTRF_TIME += SuiteSparse_time ( ) - tstart ;
#endif

    return (1) ;
}


/* ========================================================================== */
/* === gpu_triangular_solve ================================================= */
/* ========================================================================== */

/* The current supernode is columns k1 to k2-1 of L.  Let L1 be the diagonal
 * block (factorized by dpotrf/zpotrf above; rows/cols k1:k2-1), and L2 be rows
 * k2:n-1 and columns k1:k2-1 of L.  The triangular system to solve is L2*L1' =
 * S2, where S2 is overwritten with L2.  More precisely, L2 = S2 / L1' in
 * MATLAB notation.
 */

/* Version with pre-allocation in POTRF */

int TEMPLATE2 (CHOLMOD (gpu_triangular_solve))
(
    Int nsrow2,     /* L1 and S2 are nsrow2-by-nscol2 */
    Int nscol2,     /* L1 is nscol2-by-nscol2 */
    Int nsrow,      /* leading dimension of L1, L2, and S2 */
    Int psx,        /* L1 is at Lx+L_ENTRY*psx;
                     * L2 at Lx+L_ENTRY*(psx+nscol2)*/
    double *Lx,     /* holds L1, L2, and S2 */
    cholmod_common *Common,
    cholmod_gpu_pointers *gpu_p
)
{
    double *devPtrA, *devPtrB ;
    cudaError_t cudaStat ;
    cublasStatus_t cublasStatus ;
    Int gpu_lda, gpu_ldb, gpu_rowstep ;

    Int gpu_row_start = 0 ;
    Int gpu_row_max_chunk, gpu_row_chunk;
    int ibuf = 0;
    int iblock = 0;
    int iHostBuff = (Common->ibuffer+CHOLMOD_HOST_SUPERNODE_BUFFERS-1) %
        CHOLMOD_HOST_SUPERNODE_BUFFERS;
    int i, j;
    Int iidx;
    int iwrap;

#ifndef NTIMER
    double tstart ;
#endif

#ifdef REAL
    double alpha  = 1.0 ;
    gpu_row_max_chunk = 768;
#else
    cuDoubleComplex calpha  = {1.0,0.0} ;
    gpu_row_max_chunk = 256;
#endif

    if ( nsrow2 <= 0 )
    {
        return (0) ;
    }

#ifndef NTIMER
    tstart = SuiteSparse_time ( ) ;
    Common->CHOLMOD_GPU_TRSM_CALLS++ ;
#endif

    gpu_lda = ((nscol2+31)/32)*32 ;
    gpu_ldb = ((nsrow2+31)/32)*32 ;

    devPtrA = gpu_p->d_Lx[0];
    devPtrB = gpu_p->d_Lx[1];

    /* make sure the copy of B has completed */
    cudaStreamSynchronize( Common->gpuStream[0] );

    /* ---------------------------------------------------------------------- */
    /* do the CUDA BLAS dtrsm */
    /* ---------------------------------------------------------------------- */

    while ( gpu_row_start < nsrow2 )
    {

        gpu_row_chunk = nsrow2 - gpu_row_start;
        if ( gpu_row_chunk  > gpu_row_max_chunk ) {
            gpu_row_chunk = gpu_row_max_chunk;
        }

        cublasStatus = cublasSetStream ( Common->cublasHandle,
                                         Common->gpuStream[ibuf] );

        if ( cublasStatus != CUBLAS_STATUS_SUCCESS )
        {
            ERROR ( CHOLMOD_GPU_PROBLEM, "GPU CUBLAS stream");
        }

#ifdef REAL
        cublasStatus = cublasDtrsm (Common->cublasHandle,
                                    CUBLAS_SIDE_RIGHT,
                                    CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_T,
                                    CUBLAS_DIAG_NON_UNIT,
                                    gpu_row_chunk,
                                    nscol2,
                                    &alpha,
                                    devPtrA,
                                    gpu_lda,
                                    devPtrB + gpu_row_start,
                                    gpu_ldb) ;
#else
        cublasStatus = cublasZtrsm (Common->cublasHandle,
                                    CUBLAS_SIDE_RIGHT,
                                    CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_C,
                                    CUBLAS_DIAG_NON_UNIT,
                                    gpu_row_chunk,
                                    nscol2,
                                    &calpha,
                                    (const cuDoubleComplex *) devPtrA,
                                    gpu_lda,
                                    (cuDoubleComplex *)devPtrB + gpu_row_start ,
                                    gpu_ldb) ;
#endif
        if (cublasStatus != CUBLAS_STATUS_SUCCESS)
        {
            ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS routine failure") ;
        }

        /* ------------------------------------------------------------------ */
        /* copy result back to the CPU */
        /* ------------------------------------------------------------------ */

        cudaStat = cudaMemcpy2DAsync (
            gpu_p->h_Lx[iHostBuff] +
            L_ENTRY*(nscol2+gpu_row_start),
            nsrow * L_ENTRY * sizeof (Lx [0]),
            devPtrB + L_ENTRY*gpu_row_start,
            gpu_ldb * L_ENTRY * sizeof (devPtrB [0]),
            gpu_row_chunk * L_ENTRY *
            sizeof (devPtrB [0]),
            nscol2,
            cudaMemcpyDeviceToHost,
            Common->gpuStream[ibuf]);

        if (cudaStat)
        {
            ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy from device") ;
        }

        cudaEventRecord ( Common->updateCBuffersFree[ibuf],
                          Common->gpuStream[ibuf] );

        gpu_row_start += gpu_row_chunk;
        ibuf++;
        ibuf = ibuf % CHOLMOD_HOST_SUPERNODE_BUFFERS;

        iblock ++;

        if ( iblock >= CHOLMOD_HOST_SUPERNODE_BUFFERS )
        {
            Int gpu_row_start2 ;
            Int gpu_row_end ;

            /* then CHOLMOD_HOST_SUPERNODE_BUFFERS worth of work has been
             *  scheduled, so check for completed events and copy result into
             *  Lx before continuing. */
            cudaEventSynchronize ( Common->updateCBuffersFree
                                   [iblock%CHOLMOD_HOST_SUPERNODE_BUFFERS] );

            /* copy into Lx */
            gpu_row_start2 = nscol2 +
                (iblock-CHOLMOD_HOST_SUPERNODE_BUFFERS)
                *gpu_row_max_chunk;
            gpu_row_end = gpu_row_start2+gpu_row_max_chunk;

            if ( gpu_row_end > nsrow ) gpu_row_end = nsrow;

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
    private(iidx) if ( nscol2 > 32 )

            for ( j=0; j<nscol2; j++ ) {
                for ( i=gpu_row_start2*L_ENTRY; i<gpu_row_end*L_ENTRY; i++ ) {
                    iidx = j*nsrow*L_ENTRY+i;
                    Lx[psx*L_ENTRY+iidx] = gpu_p->h_Lx[iHostBuff][iidx];
                }
            }
        }
    }

    /* Convenient to copy the L1 block here */

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
private ( iidx ) if ( nscol2 > 32 )

    for ( j=0; j<nscol2; j++ ) {
        for ( i=j*L_ENTRY; i<nscol2*L_ENTRY; i++ ) {
            iidx = j*nsrow*L_ENTRY + i;
            Lx[psx*L_ENTRY+iidx] = gpu_p->h_Lx[iHostBuff][iidx];
        }
    }

    /* now account for the last HSTREAMS buffers */
    for ( iwrap=0; iwrap<CHOLMOD_HOST_SUPERNODE_BUFFERS; iwrap++ )
    {
        int i, j;
        Int gpu_row_start2 = nscol2 + (iblock-CHOLMOD_HOST_SUPERNODE_BUFFERS)
            *gpu_row_max_chunk;
        if (iblock-CHOLMOD_HOST_SUPERNODE_BUFFERS >= 0 &&
            gpu_row_start2 < nsrow )
        {
            Int iidx;
            Int gpu_row_end = gpu_row_start2+gpu_row_max_chunk;
            if ( gpu_row_end > nsrow ) gpu_row_end = nsrow;
            cudaEventSynchronize ( Common->updateCBuffersFree
                                   [iblock%CHOLMOD_HOST_SUPERNODE_BUFFERS] );
            /* copy into Lx */

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
    private(iidx) if ( nscol2 > 32 )

            for ( j=0; j<nscol2; j++ ) {
                for ( i=gpu_row_start2*L_ENTRY; i<gpu_row_end*L_ENTRY; i++ ) {
                    iidx = j*nsrow*L_ENTRY+i;
                    Lx[psx*L_ENTRY+iidx] = gpu_p->h_Lx[iHostBuff][iidx];
                }
            }
        }
        iblock++;
    }

    /* ---------------------------------------------------------------------- */
    /* return */
    /* ---------------------------------------------------------------------- */

#ifndef NTIMER
    Common->CHOLMOD_GPU_TRSM_TIME += SuiteSparse_time ( ) - tstart ;
#endif

    return (1) ;
}

/* ========================================================================== */
/* === gpu_copy_supernode =================================================== */
/* ========================================================================== */
/*
 * In the event gpu_triangular_sovle is not needed / called, this routine
 * copies the factored diagonal block from the GPU to the CPU.
 */

void TEMPLATE2 (CHOLMOD (gpu_copy_supernode))
(
    cholmod_common *Common,
    double *Lx,
    Int psx,
    Int nscol,
    Int nscol2,
    Int nsrow,
    int supernodeUsedGPU,
    int iHostBuff,
    cholmod_gpu_pointers *gpu_p
)
{
    Int iidx, i, j;
    if ( supernodeUsedGPU && nscol2 * L_ENTRY >= CHOLMOD_POTRF_LIMIT ) {
        cudaDeviceSynchronize();

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
    private(iidx,i,j) if (nscol>32)

        for ( j=0; j<nscol; j++ ) {
            for ( i=j*L_ENTRY; i<nscol*L_ENTRY; i++ ) {
                iidx = j*nsrow*L_ENTRY+i;
                Lx[psx*L_ENTRY+iidx] = gpu_p->h_Lx[iHostBuff][iidx];
            }
        }
    }
    return;
}

#endif

#undef REAL
#undef COMPLEX
#undef ZOMPLEX
