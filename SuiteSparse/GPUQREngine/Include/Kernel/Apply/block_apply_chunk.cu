// =============================================================================
// === GPUQREngine/Include/Kernel/Apply/block_apply_chunk.cu ===================
// =============================================================================

//------------------------------------------------------------------------------
// block_apply_chunk macro
//------------------------------------------------------------------------------

// A = A - V*T'*V'*A, for a single chunk of N columns of A, starting at column
// j1 and ending at j1+N-1.
//
// This function uses fixed thread geometry and loop unrolling, which requires
// the geometry to be known at compile time for best efficiency.  It is then
// #include'd by the block_apply_x function (block_apply.cu).  The following
// terms are #define'd by each specific version:
//
//      ROW_PANELSIZE    # of row tiles in V and A
//      COL_PANELSIZE    # of column tiles in C and A
//      CBITTYROWS       # of rows in the C bitty block
//      CBITTYCOLS       # of cols in the C bitty block
//      ABITTYROWS       # of rows in the A bitty block
//      ABITTYCOLS       # of cols in the A bitty block
//
// The C bitty must cannot be larger than the A bitty block, since additional
// registers are used to buffer the A matrix while the C bitty block is being
// computed.  These buffer registers are not used while computing with the A
// bitty block, so for some variants of this kernel, they can be overlapped
// with the A bitty block.
//
// The ROW_PANELSIZE, COL_PANELSIZE, ROW_EDGE_CASE, and COL_EDGE_CASE are
// #define'd by the parent file(s) that include this file.  The *_EDGE_CASE
// macros are then #undefined here.  The bitty block dimensions are defined
// below.  This file is #include'd into block_apply.cu.  It is not a standalone
// function.

{

    //--------------------------------------------------------------------------
    // bitty block sizes
    //--------------------------------------------------------------------------

    #if (ROW_PANELSIZE == 3)

        #if (COL_PANELSIZE == 2)

            //------------------------------------------------------------------
            // 3-by-2 block apply
            //------------------------------------------------------------------

            // V is 3-by-1, C is 1-by-2, A is 3-by-2 (in # tiles)
            // 256 threads, each does a 4-by-2 block of C = T'*V'*A
            #define CBITTYROWS      4
            #define CBITTYCOLS      2
            // 384 threads, each does a 4-by-4 block of A = A-V*C
            #define ABITTYROWS      4
            #define ABITTYCOLS      4

        #else

            //------------------------------------------------------------------
            // 3-by-1 block apply
            //------------------------------------------------------------------

            // V is 3-by-1, C is 1-by-1, A is 3-by-1 (in # tiles)
            // 256 threads, each does a 2-by-2 block of C = T'*V'*A
            #define CBITTYROWS      2
            #define CBITTYCOLS      2
            // 384 threads, each does a 2-by-4 block of A = A-V*C
            #define ABITTYROWS      2
            #define ABITTYCOLS      4

        #endif

    #elif (ROW_PANELSIZE == 2)

        #if (COL_PANELSIZE == 2)

            //------------------------------------------------------------------
            // block_apply_2_by_2
            //------------------------------------------------------------------

            // V is 2-by-1, C is 1-by-2, A is 2-by-2 (in # tiles)
            // 256 threads, each does a 4-by-2 block of C = T'*V'*A
            #define CBITTYROWS      4
            #define CBITTYCOLS      2
            // 256 threads, each does a 4-by-4 block of A = A-V*C
            #define ABITTYROWS      4
            #define ABITTYCOLS      4

        #else

            //------------------------------------------------------------------
            // block_apply_2_by_1
            //------------------------------------------------------------------

            // V is 2-by-1, C is 1-by-1, A is 2-by-1 (in # tiles)
            // 256 threads, each does a 2-by-2 block of C = T'*V'*A
            #define CBITTYROWS      2
            #define CBITTYCOLS      2
            // 256 threads, each does a 2-by-4 block of A = A-V*C
            #define ABITTYROWS      2
            #define ABITTYCOLS      4


        #endif

    #else

        #if (COL_PANELSIZE == 2)

            //------------------------------------------------------------------
            // block_apply_1_by_2
            //------------------------------------------------------------------

            // V is 1-by-1, C is 1-by-2, A is 1-by-2 (in # tiles)
            // 256 threads, each does a 4-by-2 block of C = T'*V'*A
            #define CBITTYROWS      2
            #define CBITTYCOLS      4
            // 256 threads, each does a 4-by-2 block of A = A-V*C
            #define ABITTYROWS      2
            #define ABITTYCOLS      4

        #else

            //------------------------------------------------------------------
            // block_apply_1_by_1
            //------------------------------------------------------------------

            // V is 1-by-1, C is 1-by-1, A is 1-by-1 (in # tiles)
            // 256 threads, each does a 2-by-2 block of C = T'*V'*A
            #define CBITTYROWS      2
            #define CBITTYCOLS      2
            // 256 threads, each does a 2-by-2 block of A = A-V*C
            #define ABITTYROWS      2
            #define ABITTYCOLS      2

        #endif

    #endif

    //--------------------------------------------------------------------------
    // matrix sizes and thread geometry
    //--------------------------------------------------------------------------

    // For each outer iteration, C is M-by-N, V is (K+1)-by-M (with an extra
    // row for T), and A is K-by-N.
    #define K           (ROW_PANELSIZE * M)
    #define N           (COL_PANELSIZE * M)

    // threads to use for C=T'*(V'*A)
    #define CTHREADS    ((M * N) / (CBITTYROWS * CBITTYCOLS))

    // threads to use for A=A-V*C
    #define ATHREADS    ((K * N) / (ABITTYROWS * ABITTYCOLS))

    //--------------------------------------------------------------------------
    // bitty blocks for the computation
    //--------------------------------------------------------------------------

    // Each thread owns a bitty block of C for C=T'*V'*A.  The top left entry
    // owned by a thread is C(ic,jc).  Thread 0 does C(0,0), thread 1 does
    // C(1,0) ...
    #define ic          (threadIdx.x % (M/CBITTYROWS))
    #define jc          (threadIdx.x / (M/CBITTYROWS))
    #define MYCBITTYROW(ii) (ii * (M/CBITTYROWS) + ic)
    #define MYCBITTYCOL(jj) (jj * (N/CBITTYCOLS) + jc)

    // Each thread owns a bitty block of A for A=A-V*C, with top left entry
    // A(ia,ja).  Thread 0 does A(0,0), thread 1 does A(0,1), thread 2 does
    // A(0,2), ... so that global memory loads/stores are coallesced across a
    // warp.
    #define ia          (threadIdx.x / (N/ABITTYCOLS))
    #define ja          (threadIdx.x % (N/ABITTYCOLS))
    #define MYABITTYROW(ii) (ii * (K/ABITTYROWS) + ia)
    #define MYABITTYCOL(jj) (jj * (N/ABITTYCOLS) + ja)

    //--------------------------------------------------------------------------
    // loading the A matrix
    //--------------------------------------------------------------------------

    // Each thread loads a set of entries of A defined by iaload and jaload.
    // The first entry loaded by a thread is A(iaload,jaload), and then it
    // loads entries every ACHUNKSIZE rows after that (in the same column
    // jaload).
    #define iaload      (threadIdx.x / N)
    #define jaload      (threadIdx.x % N)
    #define ACHUNKSIZE  (NUMTHREADS / N)
    #define NACHUNKS    CEIL (HALFTILE*N, NUMTHREADS)

    int fjload = j1 + jaload ;

    //--------------------------------------------------------------------------
    // register allocation
    //--------------------------------------------------------------------------

    // C bitty block is no larger than the A bitty block, in both dimensions.
    double rbit [ABITTYROWS][ABITTYCOLS] ;
    double rrow [ABITTYROWS] ;
    double rcol [ABITTYCOLS] ;

    #if (CBITTYCOLS == ABITTYCOLS)
        // the A bitty block is too small to hold the A buffer
        double abuffer [NACHUNKS] ;
        #define rbitA(i) abuffer [i]
    #else
        // use the last column of the A bitty block for the A buffer
        #define rbitA(i) (rbit [i][ABITTYCOLS-1])
    #endif

    //--------------------------------------------------------------------------
    // edge case
    //--------------------------------------------------------------------------

    #ifdef ROW_EDGE_CASE
        // check if a row is inside the front.
        #define INSIDE_ROW(test) (test)
    #else
        // the row is guaranteed to reside inside the frontal matrix.
        #define INSIDE_ROW(test) (1)
    #endif

    #ifdef COL_EDGE_CASE
        // check if a column is inside the front.
        #define INSIDE_COL(test) (test)
    #else
        // the column is guaranteed to reside inside the frontal matrix.
        #define INSIDE_COL(test) (1)
    #endif

    bool aloader = INSIDE_COL (fjload < fn) ;

    //--------------------------------------------------------------------------
    // C = V'*A, where V is now in shared, and A is loaded from global
    //--------------------------------------------------------------------------

    // prefetch the first halftile of A from global to register
    #pragma unroll
    for (int ii = 0 ; ii < NACHUNKS ; ii++)
    {
        rbitA (ii) = 0 ;
    }
    #pragma unroll
    for (int ii = 0 ; ii < NACHUNKS ; ii++)
    {
        int i = ii * ACHUNKSIZE + iaload ;
        if (ii < NACHUNKS-1 || i < HALFTILE)
        {
            int fi = IFRONT (0, i) ;
            if (aloader && INSIDE_ROW (fi < fm))
            {
                rbitA (ii) = glF [fi * fn + fjload] ;
            }
        }
    }

    // The X=V*C computation in the prior iteration reads shC, but the same
    // space is used to load A from the frontal matrix in this iteration.
    __syncthreads ( ) ;

    // clear the C bitty block
    #pragma unroll
    for (int ii = 0 ; ii < CBITTYROWS ; ii++)
    {
        #pragma unroll
        for (int jj = 0 ; jj < CBITTYCOLS ; jj++)
        {
            rbit [ii][jj] = 0 ;
        }
    }

    // C=V'*A for the first tile of V, which is lower triangular
    #define FIRST_TILE
    #include "cevta_tile.cu"
    
    // Subsequent tiles of V are square.  Result is in C bitty block.
    for (int t = 1 ; t < ROW_PANELSIZE ; t++)
    {
        #include "cevta_tile.cu"
    }

    //--------------------------------------------------------------------------
    // write result of C=V'*A into shared, and clear the C bitty block
    //--------------------------------------------------------------------------

    if (CTHREADS == NUMTHREADS || threadIdx.x < CTHREADS)
    {
        #pragma unroll
        for (int ii = 0 ; ii < CBITTYROWS ; ii++)
        {
            int i = MYCBITTYROW (ii) ;
            #pragma unroll
            for (int jj = 0 ; jj < CBITTYCOLS ; jj++)
            {
                int j = MYCBITTYCOL (jj) ;
                shC [i][j] = rbit [ii][jj] ;
                rbit [ii][jj] = 0 ;
            }
        }
    }

    // make sure all of shC is available to all threads
    __syncthreads ( ) ;

    //--------------------------------------------------------------------------
    // C = triu(T)'*C, leaving the result in the C bitty block
    //--------------------------------------------------------------------------

    if (CTHREADS == NUMTHREADS || threadIdx.x < CTHREADS)
    {
        #pragma unroll
        for (int i = 0 ; i < M ; i++)
        {
            #pragma unroll
            for (int ii = 0 ; ii < CBITTYROWS ; ii++)
            {
                int j = MYCBITTYROW (ii) ;
                if (i <= j)
                {
                    rrow [ii] = ST (i,j) ;
                }
            }
            #pragma unroll
            for (int jj = 0 ; jj < CBITTYCOLS ; jj++)
            {
                int j = MYCBITTYCOL (jj) ;
                rcol [jj] = shC [i][j] ;
            }
            #pragma unroll
            for (int ii = 0 ; ii < CBITTYROWS ; ii++)
            {
                int j = MYCBITTYROW (ii) ;
                if (i <= j)
                {
                    #pragma unroll
                    for (int jj = 0 ; jj < CBITTYCOLS ; jj++)
                    {
                        rbit [ii][jj] += rrow [ii] * rcol [jj] ;
                    }
                }                
            }
        }
    }

    // We need syncthreads here because of the write-after-read hazard.  Each
    // thread reads the old C, above, and then C is modified below with the new
    // C, where newC = triu(T)'*oldC.
    __syncthreads ( ) ;

    //--------------------------------------------------------------------------
    // write the result of C = T'*C to shared memory
    //--------------------------------------------------------------------------

    if (CTHREADS == NUMTHREADS || threadIdx.x < CTHREADS)
    {
        #pragma unroll
        for (int ii = 0 ; ii < CBITTYROWS ; ii++)
        {
            int i = MYCBITTYROW (ii) ;
            #pragma unroll
            for (int jj = 0 ; jj < CBITTYCOLS ; jj++)
            {
                int j = MYCBITTYCOL (jj) ;
                shC [i][j] = rbit [ii][jj] ;
            }
        }
    }

    // All threads come here.  We need a syncthreads because
    // shC has been written above and must be read below in A=A-V*C.
    __syncthreads ( ) ;

    //--------------------------------------------------------------------------
    // A = A - V*C
    //--------------------------------------------------------------------------

    if (ATHREADS == NUMTHREADS || threadIdx.x < ATHREADS)
    {

        //----------------------------------------------------------------------
        // clear the A bitty block
        //----------------------------------------------------------------------

        #pragma unroll
        for (int ii = 0 ; ii < ABITTYROWS ; ii++)
        {
            #pragma unroll
            for (int jj = 0 ; jj < ABITTYCOLS ; jj++)
            {
                rbit [ii][jj] = 0 ;
            }
        }

        //----------------------------------------------------------------------
        // X = tril(V)*C, store result into register (rbit)
        //----------------------------------------------------------------------

        #pragma unroll
        for (int p = 0 ; p < M ; p++)
        {
            #pragma unroll
            for (int ii = 0 ; ii < ABITTYROWS ; ii++)
            {
                int i = MYABITTYROW (ii) ;
                if (i >= p)
                {
                    rrow [ii] = shV [1+i][p] ;
                }
            }
            #pragma unroll
            for (int jj = 0 ; jj < ABITTYCOLS ; jj++)
            {
                int j = MYABITTYCOL (jj) ;
                rcol [jj] = shC [p][j] ;
            }
            #pragma unroll
            for (int ii = 0 ; ii < ABITTYROWS ; ii++)
            {
                int i = MYABITTYROW (ii) ;
                if (i >= p)
                {
                    #pragma unroll
                    for (int jj = 0 ; jj < ABITTYCOLS ; jj++)
                    {
                        rbit [ii][jj] += rrow [ii] * rcol [jj] ;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // A = A - X, which finalizes the computation A = A - V*(T'*(V'*A))
        //----------------------------------------------------------------------
        
        #if (COL_PANELSIZE == 2)
        
            #pragma unroll
            for (int ii = 0 ; ii < ABITTYROWS ; ii++)
            {
                int i = MYABITTYROW (ii) ;
                int fi = IFRONT (i / M, i % M) ;
                #pragma unroll
                for (int jj = 0 ; jj < ABITTYCOLS ; jj++)
                {
                    int fj = j1 + MYABITTYCOL (jj) ;
                    if (INSIDE_ROW (fi < fm) && INSIDE_COL (fj < fn))
                    {
                        glF [fi * fn + fj] -= rbit [ii][jj] ;
                    }
                }
            }

        #else
        
            #pragma unroll
            for (int ii = 0 ; ii < ABITTYROWS ; ii++)
            {
                int i = MYABITTYROW (ii) ;
                int fi = IFRONT (i / M, i % M) ;
                #pragma unroll
                for (int jj = 0 ; jj < ABITTYCOLS ; jj++)
                {
                    int fj = j1 + MYABITTYCOL (jj) ;
                    if (INSIDE_ROW (fi < fm) && INSIDE_COL (fj < fn))
                    {
                        shV[i][MYABITTYCOL(jj)] = glF[fi*fn+fj] - rbit[ii][jj];
                    }
                    else
                    {
                        shV[i][MYABITTYCOL(jj)] = 0.0;
                    }
                }
            }
    
        #endif
    }

    //--------------------------------------------------------------------------
    // sync
    //--------------------------------------------------------------------------

    // The X=V*C computation in this iteration reads shC, but the same space is
    // used to load A from the frontal matrix in C=V'*A in the next iteration.
    // This final sync also ensures that all threads finish the block_apply
    // at the same time.  Thus, no syncthreads is needed at the start of a
    // subsequent function (the pipelined apply+factorize, for example).

    __syncthreads ( ) ;
}

//------------------------------------------------------------------------------
// undef's
//------------------------------------------------------------------------------

// The following #define's appear above.  Note that FIRST_TILE is not #undef'd
// since that is done by cevta_tile.cu.
#undef CBITTYROWS
#undef CBITTYCOLS
#undef ABITTYROWS
#undef ABITTYCOLS

#undef K
#undef N

#undef CTHREADS
#undef ATHREADS

#undef ic
#undef jc
#undef MYCBITTYROW
#undef MYCBITTYCOL

#undef ia
#undef ja
#undef MYABITTYROW
#undef MYABITTYCOL

#undef iaload
#undef jaload
#undef ACHUNKSIZE
#undef NACHUNKS

#undef rbitA
#undef INSIDE_ROW
#undef INSIDE_COL

// Defined in the parent file that includes this one.  Note that ROW_PANELSIZE
// is not #undef'd, since that is done in the parent.
#undef ROW_EDGE_CASE
#undef COL_EDGE_CASE
