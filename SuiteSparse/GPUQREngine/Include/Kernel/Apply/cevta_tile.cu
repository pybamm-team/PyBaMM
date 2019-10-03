// =============================================================================
// === GPUQREngine/Include/Kernel/Apply/cevta_tile.cu ==========================
// =============================================================================

//------------------------------------------------------------------------------
// cevta_tile macro
//------------------------------------------------------------------------------

// C=V'*A for a single tile, computed in two chunks of HALFTILE each.  Also
// loads in the A matrix into the SHA buffer.  The first tile is lower
// triangular (selected by defining FIRST_TILE in the caller).  Subsequent
// tiles are square (and FIRST_TILE is not defined).  This file is #include'd
// into block_apply_chunk.cu.  It is not a stand-alone function.

#ifdef FIRST_TILE
#define this_tile 0
#define TRIL(test) (test)
#else
#define this_tile t
#define TRIL(test) (1)
#endif

{
    #pragma unroll
    for (int p = 0 ; p < 2 ; p++)
    {

        //----------------------------------------------------------------------
        // move the prefetched A from register into shared for this halftile
        //----------------------------------------------------------------------

        // Write next halftile of 16 rows of A from register to shared, for
        // _this_ iteration of p.  The abuffer was read from global to register
        // in the prior iteration of p (the prefetch below), or in the prefetch
        // in block_apply_chunk.cu.  If column fjload is outside the front,
        // rbitA is all zero, which safely clears SHA.

        #pragma unroll
        for (int ii = 0 ; ii < NACHUNKS ; ii++)
        {
            int i = ii * ACHUNKSIZE + iaload ;
            if (ii < NACHUNKS-1 || i < HALFTILE)
            {
                SHA (p*HALFTILE+i, jaload) = rbitA (ii) ;
            }
        }

        // V and A for this iteration of p are now loaded into shared
        __syncthreads ( ) ;

        //----------------------------------------------------------------------
        // prefetch the next halftile of A from global into register
        //----------------------------------------------------------------------

        // Load the next halftile of A from global memory to register,
        // not for this iteration of p, but the next.

        if (p == 0)
        {
            // Read the next halftile of A for this row tile.
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
                    int fi = IFRONT (this_tile, HALFTILE+i) ;
                    if (aloader && INSIDE_ROW (fi < fm))
                    {
                        rbitA (ii) = glF [fi * fn + fjload] ;
                    }
                }
            }
        }
        else if (this_tile+1 < ROW_PANELSIZE)       // p is 1 for this case
        {
            // Read the first halftile of V and A for the next row tile,
            // but not if we are computing with the very last tile.
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
                    int fi = IFRONT (this_tile+1, i) ;
                    if (aloader && INSIDE_ROW (fi < fm))
                    {
                        rbitA (ii) = glF [fi * fn + fjload] ;
                    }
                }
            }
        }

        //----------------------------------------------------------------------
        // C = tril (V)'*A for this halffile
        //----------------------------------------------------------------------

        // For square tiles (all but the first tile of V), TRIL is always
        // true, and this code is simpler and faster as a result.

        if (CTHREADS == NUMTHREADS || threadIdx.x < CTHREADS)
        {
            // C=tril(V)'*A, compute with a halftile of A and V
            #pragma unroll
            for (int pp = 0 ; pp < HALFTILE ; pp++)
            {
                int i = p * HALFTILE + pp ;
                #pragma unroll
                for (int ii = 0 ; ii < CBITTYROWS ; ii++)
                {
                    int j = MYCBITTYROW (ii) ;
                    if (TRIL (i >= j))
                    {
                        rrow [ii] = SHV (this_tile, i, j) ;
                    }
                }
                #pragma unroll
                for (int jj = 0 ; jj < CBITTYCOLS ; jj++)
                {
                    int j = MYCBITTYCOL (jj) ;
                    rcol [jj] = SHA (i, j) ;
                }
                #pragma unroll
                for (int ii = 0 ; ii < CBITTYROWS ; ii++)
                {
                    if (TRIL (i >= MYCBITTYROW (ii)))
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
    }
}

#undef FIRST_TILE
#undef this_tile
#undef TRIL
