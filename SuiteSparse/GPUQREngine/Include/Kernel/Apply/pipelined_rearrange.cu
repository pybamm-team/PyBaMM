// =============================================================================
// === GPUQREngine/Include/Kernel/Apply/pipelined_rearrange.cu =================
// =============================================================================

//------------------------------------------------------------------------------
// pipelined_rearrange
//------------------------------------------------------------------------------

/*
    PSEUDO #define MACROS (copied from vt_factorize.cu)
        N
            The # of columns to operate on (should always be TILESIZE).
        INSIDE
            Substitute in a condition depending on compilation options.
            For this code, we always assume we need to check edge cases.
        NACHUNKS
            A chunking scheme used in the factorization kernel. We use
            the same layout and thread dimension for our tile load/stores.
        glA
            Shorthand for the index computation into the global A.
        shA
            Shorthand for accessing the shared memory tiles of A in the union.
        it
            Row indices of a tile owned by a thread.
        jt
            Col indices of a tile owned by a thread.
        ACHUNKSIZE
            The amount of A do load in a chunk
*/

#define N               (TILESIZE)

#define INSIDE(COND)    (COND)

// when all threads work on a tile.
// (N*N / NUMTHREADS) does not have to be an integer.  With a tile
// size of N=32, and NUMTHREADS=384, it isn't.  So compute the ceiling,
// and handle the clean up by testing i < N below.
#define NACHUNKS        CEIL (N*N, NUMTHREADS)

#define glA(i,j)        (myTask.F[((i)*fn + (j))])
#define shA             shMemory.factorize.A

// ACHUNKSIZE must be an integer
#define it              (threadIdx.x / N)
#define jt              (threadIdx.x % N)
#define ACHUNKSIZE      (NUMTHREADS / N)
    
/*
    NEW #define MACROS
        SAFELOAD
            Loads a tile from global memory. Checks edge cases.
        SH_TRANSFER
            Moves a tile within shared memory
        SAFESTORE
            Stores a tile back to global memory. Checks edge cases.
*/

#define SAFELOAD(SLOT, ROWTILE)                                             \
{                                                                           \
    int rowTile = (ROWTILE);                                                \
    if (INSIDE (rowTile != EMPTY))                                          \
    {                                                                       \
        /* load the tile of A from global memory */                         \
        for (int ii = 0 ; ii < NACHUNKS ; ii++)                             \
        {                                                                   \
            int i = ii * ACHUNKSIZE + it ;                                  \
            if (ii < NACHUNKS-1 || i < N)                                   \
            {                                                               \
                shA [i + (SLOT)*TILESIZE][jt] =                             \
                    (INSIDE (i+rowTile < fm) && INSIDE (jt+j1 < fn)) ?      \
                    glA (i+rowTile, jt+j1) : 0 ;                            \
            }                                                               \
        }                                                                   \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        /* clear the tile of A */                                           \
        for (int ii = 0 ; ii < NACHUNKS ; ii++)                             \
        {                                                                   \
            int i = ii * ACHUNKSIZE + it ;                                  \
            if (ii < NACHUNKS-1 || i < N)                                   \
            {                                                               \
                shA [i + SLOT*TILESIZE][jt] = 0 ;                           \
            }                                                               \
        }                                                                   \
    }                                                                       \
}                                                                           \


#define SH_TRANSFER(TO, FROM)                                               \
{                                                                           \
    for (int th=threadIdx.x; th<TILESIZE*TILESIZE; th+=blockDim.x)          \
    {                                                                       \
        int ito = (TILESIZE*(TO))   + (th / TILESIZE);                      \
        int ifr = (TILESIZE*(FROM)) + (th / TILESIZE);                      \
        int  j  =                     (th % TILESIZE);                      \
        shA[ito][j] = shA[ifr][j];                                          \
    }                                                                       \
}                                                                           \


#define SAFESTORE(SLOT, ROWTILE)                                            \
{                                                                           \
    int rowTile = ROWTILE;                                                  \
    if (INSIDE (rowTile != EMPTY))                                          \
    {                                                                       \
        for (int ii = 0 ; ii < NACHUNKS ; ii++)                             \
        {                                                                   \
            int i = ii * ACHUNKSIZE + it ;                                  \
            if (ii < NACHUNKS-1 || i < N)                                   \
            {                                                               \
                if (INSIDE (i+rowTile < fm) && INSIDE (jt+j1 < fn))         \
                {                                                           \
                    glA (i+rowTile, jt+j1) = shA [i + (SLOT)*TILESIZE][jt]; \
                }                                                           \
            }                                                               \
        }                                                                   \
    }                                                                       \
}                                                                           \


/* ALL THREADS PARTICIPATE */
{
        
    int delta     = myTask.extra[8];
    int secondMin = myTask.extra[9];
    int fc        = IsApplyFactorize;
    int j1        = myTask.extra[4] + TILESIZE;
    
    /*** DO MEMORY SHUFFLES ***/

    SAFESTORE(0, myTask.extra[0]);

    /* 0 <-- secondMin */
    if(delta != EMPTY && secondMin == delta)
    {
        SAFELOAD(0, myTask.extra[secondMin]);
    }
    else
    {
        SH_TRANSFER(0, secondMin);
    }

    /* secondMin <-- fc */
    if(fc != secondMin)
    {
        if(delta != EMPTY && fc >= delta)
        {
            SAFELOAD(secondMin, myTask.extra[fc]);
        }
        else
        {
            SH_TRANSFER(secondMin, fc);
        }
    }

    /* Hard-load D from global in the 2-3 case where [1] is secondMin. */
    if(fc == 3 && delta == 2 && secondMin == 1)
    {
        SAFELOAD(2, myTask.extra[2]);
    }

    /* Rearrange tiles so the tile store at the end doesn't explode.
       This is non-essential until the very end, so we can easilly justify
       piggybacking this integer shuffle to the next natural __syncthreads
       that we encounter. */
    __syncthreads();
    if(threadIdx.x == 0)
    {
        myTask.extra[4] = j1;
        myTask.AuxAddress[0] = myTask.AuxAddress[1];
        myTask.AuxAddress[1] = NULL;
        
        myTask.extra[0] = myTask.extra[secondMin];
        if(fc != secondMin)
        {
            myTask.extra[secondMin] = myTask.extra[fc];
        }
    }  
    __syncthreads();
}

#undef N
#undef INSIDE
#undef NACHUNKS
#undef glA
#undef shA
#undef it
#undef jt
#undef ACHUNKSIZE

#undef SAFELOAD
#undef SH_TRANSFER
#undef SAFESTORE
