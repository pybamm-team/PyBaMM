// =============================================================================
// === GPUQREngine/Include/Kernel/Factorize/factorize_vt.cu ====================
// =============================================================================

//  Constraints:
//      MCHUNK = M / BITTYROWS must be an integer
//      MCHUNK * N must be <= NUMTHREADS

#ifdef FACTORIZE

// TODO allow EPSILON to be defined by the caller
#define EPSILON (1e-90)

__device__ void FACTORIZE ( )
{

    //--------------------------------------------------------------------------
    // The bitty block
    //--------------------------------------------------------------------------

    // The matrix A is M by N, T is N by N (if present)

    // Each thread owns an r-by-1 bitty column of A.  If its first entry is
    // A(ia,ja), and then it owns entries in every mchunk rows after that
    // [A(ia,ja), A(ia+mchunk,ja), ... ].
    // Each column is operated on by mchunk threads.

    #define MCHUNK          (M / BITTYROWS)
    #define MYBITTYROW(ii)  (ii*MCHUNK + (threadIdx.x % MCHUNK))
    #define MYBITTYCOL      (threadIdx.x / MCHUNK)
    #define ATHREADS        (MCHUNK * N)
    #define WORKER          (ATHREADS == NUMTHREADS || threadIdx.x < ATHREADS)

    double rbitA [BITTYROWS] ;     // bitty block for A
    double rbitV [BITTYROWS] ;     // bitty block for V
    double sigma ;                 // only used by thread zero

    //--------------------------------------------------------------------------
    // shared memory usage
    //--------------------------------------------------------------------------

    #define shA         shMemory.factorize.A
    #define shZ         shMemory.factorize.Z
    #define shRdiag     shMemory.factorize.A1
    #define RSIGMA(i)   shMemory.factorize.V1 [i]

    #ifdef WHOLE_FRONT
        // T is not computed, and there is no list of tiles
        #define TAU     shMemory.factorize.tau
    #else
        // T is computed and saved in the VT tile, work on a set of tiles
        #define TAU     shT [k][k] 
        #define shT     shMemory.factorize.T
        #define shV1    shMemory.factorize.V1
    #endif


    //--------------------------------------------------------------------------
    // Grab my task from the queue.
    //--------------------------------------------------------------------------

    int fn = myTask.fn ;

    #ifdef WHOLE_FRONT

        int fm = myTask.fm ;
        int nv = MIN (fm, fn) ;
        // If nv is a constant, it allows the outer loop to be unrolled:
        // #define nv N
        #define j1 0
        // The whole front always considers the edge case
        #ifndef EDGE_CASE
        #define EDGE_CASE
        #endif

    #else

        int j1 = myTask.extra[4] ;
        #ifdef EDGE_CASE
        int fm = myTask.fm ;
        int nv = MIN (fm, fn) - j1 ;
        nv = MIN (nv, N) ;
        nv = MIN (nv, M) ;
        #else
        #define nv N
        #endif
        double (*glVT)[TILESIZE] = (double (*)[TILESIZE]) myTask.AuxAddress[0] ;

    #endif

    #ifdef EDGE_CASE
        // Check if an entry is inside the front.
        #define INSIDE(test) (test)
    #else
        // The entry is guaranteed to reside inside the frontal matrix.
        #define INSIDE(test) (1)
    #endif

    // bool is_false = (fn < 0) ;

    #define glA(i,j)        (myTask.F[((i)*fn + (j))])

    //--------------------------------------------------------------------------
    // Load A into shared memory
    //--------------------------------------------------------------------------

    // ACHUNKSIZE must be an integer
    #define it              (threadIdx.x / N)
    #define jt              (threadIdx.x % N)
    #define ACHUNKSIZE      (NUMTHREADS / N)

    #ifdef WHOLE_FRONT

        // all threads load the entire front (no tiles).
        // always check the edge case.
        // M / ACHUNKSIZE must be an integer.
        #define NACHUNKS    (M / ACHUNKSIZE)
        for (int ii = 0 ; ii < NACHUNKS ; ii++)
        {
            int i = ii * ACHUNKSIZE + it ;
            shA [i][jt] = (i < fm && jt < fn) ?  glA (i, jt) : 0 ;
        }

    #else

        // when all threads work on a tile.
        // (N*N / NUMTHREADS) does not have to be an integer.  With a tile
        // size of N=32, and NUMTHREADS=384, it isn't.  So compute the ceiling,
        // and handle the clean up by testing i < N below.
        #define NACHUNKS    CEIL (N*N, NUMTHREADS)

        /* If we're not coming from an apply-factorize, load from F. */
        if(IsApplyFactorize == 0)
        {
            // Load tiles from the frontal matrix
            // accounts for 25% of the total time on Kepler, 13% on Fermi
            for (int t = 0 ; t < ROW_PANELSIZE ; t++)
            {
                int rowTile = myTask.extra[t];
                if (INSIDE (rowTile != EMPTY))
                {
                    /* load the tile of A from global memory */
                    for (int ii = 0 ; ii < NACHUNKS ; ii++)
                    {
                        int i = ii * ACHUNKSIZE + it ;
                        if (ii < NACHUNKS-1 || i < N)
                        {
                            shA [i + t*TILESIZE][jt] =
                              (INSIDE (i+rowTile < fm) && INSIDE (jt+j1 < fn)) ?
                              glA (i+rowTile, jt+j1) : 0 ;
                        }
                    }
                }
                else
                {
                    /* clear the tile of A */
                    for (int ii = 0 ; ii < NACHUNKS ; ii++)
                    {
                        int i = ii * ACHUNKSIZE + it ;
                        if (ii < NACHUNKS-1 || i < N)
                        {
                            shA [i + t*TILESIZE][jt] = 0 ;
                        }
                    }
                }
            }
        }

        // clear the tile T
        for (int ii = 0 ; ii < NACHUNKS ; ii++)
        {
            int i = ii * ACHUNKSIZE + it ;
            if (ii < NACHUNKS-1 || i < N)
            {
                shT [i][jt] = 0 ;
            }
        }
    #endif

    /* We need all of A to be loaded and T to be cleared before proceeding. */
    __syncthreads();

    //--------------------------------------------------------------------------
    // load A into the bitty block
    //--------------------------------------------------------------------------

    if (WORKER)
    {
        #pragma unroll
        for (int ii = 0 ; ii < BITTYROWS ; ii++)
        {
            int i = MYBITTYROW (ii) ;
            rbitA [ii] = shA [i][MYBITTYCOL] ;
        }
    }

    //--------------------------------------------------------------------------
    // compute the first sigma = sum (A (1:m,1).^2)
    //--------------------------------------------------------------------------

    if (WORKER && MYBITTYCOL == 0)
    {
        // each thread that owns column 0 computes sigma for its
        // own bitty block
        double s = 0 ;
        #pragma unroll
        for (int ii = 0 ; ii < BITTYROWS ; ii++)
        {
            int i = MYBITTYROW (ii) ;
            if (i >= 1)
            {
                s += rbitA [ii] * rbitA [ii] ;
            }
        }
        RSIGMA (threadIdx.x) = s ;
    }

    // thread zero must wait for RSIGMA
    __syncthreads ( ) ;

    if (threadIdx.x == 0)
    {
        sigma = 0 ;
        #pragma unroll
        for (int ii = 0 ; ii < MCHUNK ; ii++)
        {
            sigma += RSIGMA (ii) ;
        }
    }

    //--------------------------------------------------------------------------
    // Do the block householder factorization
    //--------------------------------------------------------------------------

    // loop unrolling has no effect on the edge case (it is not unrolled),
    // but greatly speeds up the non-edge case.
    #pragma unroll
    for (int k = 0 ; k < nv ; k++)
    {

        //----------------------------------------------------------------------
        // write the kth column of A back into shared memory
        //----------------------------------------------------------------------

        if (WORKER && MYBITTYCOL == k && k > 0)
        {
            // the bitty block for threads that own column k contains
            // the kth column of R and the kth column of v.
            #pragma unroll
            for (int ii = 0 ; ii < BITTYROWS ; ii++)
            {
                int i = MYBITTYROW (ii) ;
                shA [i][k] = rbitA [ii] ;
            }
        }

        __syncthreads ( ) ;

        // At this point, A (:,k) is held in both shared memory, and in the
        // threads that own that column.  A (:,k) is not yet the kth
        // Householder vector, except for the diagnal (which is computed
        // below).  A (0:k-1,k) is now the kth column of R (above the
        // diagonal).

        //----------------------------------------------------------------------
        // compute the Householder coefficients
        //----------------------------------------------------------------------

        // This is costly, accounting for about 25% of the total time on
        // Kepler, and 22% on Fermi, when A is loaded from global memory.  This
        // means the work here is even a higher fraction when A is in shared.
        if (threadIdx.x == 0)
        {
            double x1 = shA [k][k] ;            // the diagonal A (k,k)
            double s, v1, tau ;

            if (sigma <= EPSILON)
            {
                // printf ("hit eps %g\n", sigma) ;
                s = x1 ;
                v1 = 0 ;
                tau = 0 ;
            }
            else
            {
                s = sqrt (x1*x1 + sigma) ;
                v1 = (x1 <= 0) ? (x1 - s) : (-sigma / (x1 + s)) ;
                tau = -1 / (s * v1) ;
            }
            shRdiag [k] = s ;       // the diagonal entry of R
            shA [k][k] = v1 ;       // the topmost entry of the vector v
            TAU = tau ;             // tile case: T (k,k) = tau
        }

        // All threads need v1, and later on they need tau
        __syncthreads ( ) ;

        // A (0:k-1,k) now holds the kth column of R (excluding the diagonal).
        // A (k:m-1,k) holds the kth Householder vector (incl. the diagonal).

        //----------------------------------------------------------------------
        // z = (-tau) * v' * A (k:m-1,:), where v is A (k:m-1,k)
        //----------------------------------------------------------------------

        if (WORKER) // && (COMPUTE_T || MYBITTYCOL > k))
        {
            // Load the vector v from A (k:m-1,k) into the V bitty block.
            // If T is not computed and MYBITTYCOL <= k, then this code can
            // be skipped, but the code is slower when that test is made.
            #pragma unroll
            for (int ii = 0 ; ii < BITTYROWS ; ii++)
            {
                int i = MYBITTYROW (ii) ;
                // only i >= k is needed, but it's faster to load it all
                rbitV [ii] = shA [i][k] ;
            }

            // compute v' * A (k:m-1,:), each thread in its own column
            {
                double z = 0.0 ;
                #pragma unroll
                for (int ii = 0 ; ii < BITTYROWS ; ii++)
                {
                    int i = MYBITTYROW (ii) ;
                    if (i >= k)
                    {
                        z += rbitV [ii] * rbitA [ii] ;
                    }
                }
                // store z into the reduction space in shared memory
                shZ [MYBITTYROW(0)][MYBITTYCOL] = z ;
            }
        }

        // All threads need to see the reduction space for Z
        __syncthreads ( ) ;

        // Reduce Z into a single row vector z, using the first warp only
        if (threadIdx.x < N) // && (COMPUTE_T || threadIdx.x > k))
        {
            double z = 0 ;
            #pragma unroll
            for (int ii = 0 ; ii < MCHUNK ; ii++)
            {
                z += shZ [ii][threadIdx.x] ;
            }
            shZ [0][threadIdx.x] = - z * TAU ;
        }

        // All threads need to see the z vector
        __syncthreads ( ) ;

        //----------------------------------------------------------------------
        // update A (in register) and compute the next sigma
        //----------------------------------------------------------------------

        if (WORKER && MYBITTYCOL > k)
        {
            // A (k:m,k+1:n) = A (k:,k+1:n) + v * z (k+1:n) ;
            // only threads that own a column MYBITTYCOL > k do any work
            {
                double z = shZ [0][MYBITTYCOL] ;
                #pragma unroll
                for (int ii = 0 ; ii < BITTYROWS ; ii++)
                {
                    int i = MYBITTYROW (ii) ;
                    if (i >= k)
                    {
                        rbitA [ii] += rbitV [ii] * z ;
                    }
                }
            }

            // sigma = sum (A ((k+2):m,k+1).^2), except for the reduction
            if (MYBITTYCOL == k+1)
            {
                // each thread that owns column k+1 computes sigma for its
                // own bitty block
                double s = 0 ;
                #pragma unroll
                for (int ii = 0 ; ii < BITTYROWS ; ii++)
                {
                    int i = MYBITTYROW (ii) ;
                    if (i >= k+2)
                    {
                        s += rbitA [ii] * rbitA [ii] ;
                    }
                }
                RSIGMA (MYBITTYROW(0)) = s ;
            }
        }

        //----------------------------------------------------------------------
        // construct the kth column of T
        //----------------------------------------------------------------------

        #ifndef WHOLE_FRONT

            // T (0:k-1,k) = T (0:k-1,0:k-1) * z (0:k-1)'
            if (threadIdx.x < k)
            {
                double t_ik = 0 ;
                for (int jj = 0 ; jj < k ; jj++)
                {
                    t_ik += shT [threadIdx.x][jj] * shZ [0][jj] ;
                }
                shT [threadIdx.x][k] = t_ik ;
            }

        #endif

        //----------------------------------------------------------------------
        // reduce sigma into a single scalar for the next iteration
        //----------------------------------------------------------------------

        // Thread zero must wait for RSIGMA
        __syncthreads ( ) ;

        if (threadIdx.x == 0)
        {
            sigma = 0 ;
            #pragma unroll
            for (int ii = 0 ; ii < MCHUNK ; ii++)
            {
                sigma += RSIGMA (ii) ;
            }
        }
    }

    // tril (A) now holds all the Householder vectors, including the diagonal.
    // triu (A,1) now holds R, excluding the diagonal.
    // shRdiag holds the diagonal of R.

    //--------------------------------------------------------------------------
    // write out the remaining columns of R, if any
    //--------------------------------------------------------------------------

    if (WORKER && MYBITTYCOL >= nv)
    {
        for (int ii = 0 ; ii < BITTYROWS ; ii++)
        {
            int i = MYBITTYROW (ii) ;
            shA [i][MYBITTYCOL] = rbitA [ii] ;
        }
    }

    //--------------------------------------------------------------------------

    /* Have a warp shuffle memory around. */
    if (threadIdx.x < N)
    {
        #ifndef WHOLE_FRONT
        shV1 [threadIdx.x] = shA [threadIdx.x][threadIdx.x];
        #endif
        shA [threadIdx.x][threadIdx.x] = shRdiag [threadIdx.x];
    }

    // Wait for the memory shuffle to finish before saving A to global memory
    __syncthreads();

    //--------------------------------------------------------------------------
    // save results back to global memory
    //--------------------------------------------------------------------------

    #ifdef WHOLE_FRONT

        if (jt < fn)
        {
            for (int ii = 0 ; ii < NACHUNKS ; ii++)
            {
                int i = ii * ACHUNKSIZE + it ;
                if (i < fm) glA (i, jt) = shA [i][jt] ;
            }
        }

    #else

        // Save VT back to global memory & clear out
        // lower-triangular part of the first tile (leaving R).
        for (int th=threadIdx.x; th<TILESIZE*TILESIZE; th+=blockDim.x)
        {
            int i = th / 32;
            int j = th % 32;

            /* The upper triangular part (including diagonal) is T. */
            if(i <= j)
            {
                glVT[i][j] = shT[i][j];
            }
            /* The lower triangular part is V.
             * Note we clear the tril part leaving only R in this tile. */
            else
            {
                glVT[i+1][j] = shA[i][j];
                shA[i][j] = 0.0;
            }
        }

        // Save the diagonal
        if (threadIdx.x < N)
        {
            glVT[threadIdx.x+1][threadIdx.x] = shV1[threadIdx.x];
        }

        // Wait for this operation to complete before saving A back to global
        // memory
        __syncthreads();

        // save the tiles in A back into the front in global memory
        for (int t = 0 ; t < ROW_PANELSIZE ; t++)
        {
            int rowTile = myTask.extra[t];
            if (INSIDE (rowTile != EMPTY))
            {
                for (int ii = 0 ; ii < NACHUNKS ; ii++)
                {
                    int i = ii * ACHUNKSIZE + it ;
                    if (ii < NACHUNKS-1 || i < N)
                    {
                        if (INSIDE (i+rowTile < fm) && INSIDE (jt+j1 < fn))
                        {
                            glA (i+rowTile, jt+j1) = shA [i + t*TILESIZE][jt];
                        }
                    }
                }
            }
        }
    #endif
}

//------------------------------------------------------------------------------
// undefine macros
//------------------------------------------------------------------------------

#undef EPSILON
#undef FACTORIZE
#undef M
#undef N
#undef glA

#undef WORKER
#undef ATHREADS
#undef MCHUNK

#undef BITTYROWS
#undef MYBITTYROW
#undef MYBITTYCOL
#undef shA
#undef shT
#undef shZ
#undef shRdiag
#undef shV1
#undef RSIGMA
#undef TAU
#undef INSIDE
#undef INSIDE
#undef nv
#undef it
#undef jt
#undef j1
#undef ACHUNKSIZE
#undef SAFELOAD
#undef NACHUNKS

#undef EDGE_CASE
#undef WHOLE_FRONT
#undef ROW_PANELSIZE
#endif
