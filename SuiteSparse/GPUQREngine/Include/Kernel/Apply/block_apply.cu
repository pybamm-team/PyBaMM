// =============================================================================
// === GPUQREngine/Include/Kernel/Apply/block_apply.cu =========================
// =============================================================================

//------------------------------------------------------------------------------
// block_apply.cu: generic device function to apply a block Householder
//------------------------------------------------------------------------------
//
// Computes C = V'*A ; C = T'*C ; A = A-V*C ; where:
// T is upper triangular of size M-by-M (a single tile, where M = TILESIZE)
// V is lower triangular of size K-by-M,
// C is rectangular of size M-by-N, and
// A is rectangular of size K-by-n, operated on blocks of N = 2*M or M columns
// at a time.
//
// V and T are stored in the same matrix in shared memory, of size K+1-by-M
// (since both have a non-trivial diagonal).  The first M+1 rows of this
// combined [V T] matrix are held outside the frontal matrix.  The remainder
// of V is held in the frontal matrix.
//
// The C matrix resides in shared memory, and exists only within this function.
//
// The A matrix resides in the frontal matrix, in global memory.  It is
// temporarily loaded into shared memory for C=V'*A, one halftile at a time. 
//
// All data starts and ends in global memory.  Shared memory is used to hold T,
// C and V during the computation.  All of A and all but the first 32 rows of V
// are held in the frontal matrix.  The V1 matrix (the first 32 rows of V) and
// all of T are held in a 33-by-32 matrix (VT), where V is lower triangular and
// T is upper triangular.
//
// All threads are used to load data from global memory.  Only CTHREADS threads
// are used to compute C=T'*V'*A.  ATHREADS threads compute A=A-V*C.
//
// The A matrix and V2 are held in the frontal matrix F, of size fm-by-fn. 
// All matrices are in row-major form.  Suppose F is 8-by-9 tiles:
//
//        0 1 2 3 4 5 6 7 8
//      0 . . . . . . . . .
//      1 . . . . . . . . .
//      2 . o . a a a a a .
//      3 . . . . . . . . .
//      4 . v . a a a a a .
//      5 . . . . . . . . .
//      6 . v . a a a a a .
//      7 . . . . . . . . .
//
// The "o", above, is not stored in the front, but in a 33-by-32 VT array in
// global memory outside the front.  It contains all of T, and the triangular
// part of V (V1).  The "v" holds the rectangular part of V (V2), and it is in
// the front.  The "a" entries are tiles that must be updated via A =
// A-V*T'*V'*A.
//
// In this example, the row tiles in the bundle are [2 4 6].  The rowTile vector
// holds [2 4 6]*32.  The column tiles are 1 (for V) and 3:7 for A, so jTile
// contains [1 3 7]*32.
//
// This code skips the load/store of entries in tiles that extend past the edge
// of the front.  However, jTile and rowTile must always be multiples of the
// tilesize (32), since the edge cases are only triggered by the edge of the
// frontal matrix.
//
// The task is defined by the following parameters:
//      double *F       pointer to frontal matrix in GPU global memory
//      double *VT[0]   pointer to [V1 T] matrix in GPU global memory
//      int fm          # of rows in the front (not # of tiles);
//                      only needed for edge cases.
//      int fn          # of columns in the front.  F is fm-by-fn,
//                      and takes space for fm*fn entries (each 8 bytes)
//                      in global memory.  F is stored in row major form.
//      rowTiles [3]    assuming the panel size is 3
//      jTile [3]       size is 3, regardless of panel size
//      Type            defines which variant of this kernel to call
//
// Future: the pipeline task, which will will do:
//      1) apply to a col tile
//          C = V'*A
//          C = T'*C
//          A2 = A-V*C       overwrite memory space V with the updated A2.
//      2) load growth tiles
//      3) factorize the target, A2, then save [V1 T] in its own space,
//         and save V1 and R in the front

//------------------------------------------------------------------------------

#define M TILESIZE

__device__ void BLOCK_APPLY ( )
{

    //--------------------------------------------------------------------------
    // grab my task from the queue
    //--------------------------------------------------------------------------

    double *glF = myTask.F ;
    int fn = myTask.fn ;
    int fm = myTask.fm ;

    //--------------------------------------------------------------------------
    // load V and T, using all threads
    //--------------------------------------------------------------------------

    {
        double (*glVT)[M] = (double (*)[M]) myTask.AuxAddress[0] ;
        // load the first row of T from the VT block, using one warp
        if (threadIdx.x < M)
        {
            SHV (0, -1, jv) = GLVT (-1, jv) ;
        }

        // load the first tile of V and T from the VT block
        for (int ii = 0 ; ii < NVCHUNKS ; ii++)
        {
            int i = ii * VCHUNKSIZE + iv ;
            if (ii < NVCHUNKS-1 || i < M)
            {
                SHV (0, i, jv) = GLVT (i, jv) ;
            }
        }
    }

    // load subsequent tiles of V from the frontal matrix
    {
        int j0 = myTask.extra [4] ;
        for (int t = 1 ; t < ROW_PANELSIZE ; t++)
        {
            // For the edge case, only check if the row is inside the front.
            // The column is always in the front because V always has M columns.
            for (int ii = 0 ; ii < NVCHUNKS ; ii++)
            {
                int i = ii * VCHUNKSIZE + iv ;
                if (ii < NVCHUNKS-1 || i < M)
                {
                    int fi = IFRONT (t,i) ;
                    SHV (t, i, jv) = (fi < fm) ? glF [fi * fn + (j0+jv)] : 0.0 ;
                }
            }
        }
    }

    //--------------------------------------------------------------------------
    // do the block apply:  A = A - V*T'*V'*A
    //--------------------------------------------------------------------------

    int j1 = myTask.extra [5] ;

    #if (COL_PANELSIZE == 1)

        // Apply the update to a single column tile.  jTile [2] is ignored.
        // Handle both row and column edge cases.
        #define ROW_EDGE_CASE
        #define COL_EDGE_CASE
        #include "block_apply_chunk.cu"

        // Do the rearrange for pipelined factorization.

        #include "pipelined_rearrange.cu"        


    #else

        // When COL_PANELSIZE is 2, this function iterates across all column
        // tiles in chunks of 2 column tiles at a time.

        #ifdef FANCY

            // This first operates on pairs of column tiles that fully fit into
            // the front, with no handling of the column edge case.  This is
            // followed by a cleanup phase that handles the last columns, with
            // the column edge case enabled.  This is slightly faster than the
            // simple code below.  (Fermi: 188 vs 185 Gflops, Kepler: 263 vs
            // 259).  However, the code doubles in length with this version.

            int jend = myTask.extra [6] - 2*M ;
            for ( ; j1 <= jend ; j1 += 2*M)
            {
                // Apply the update to columns j1 through j1+2*M-1.  Check for
                // rows outside the front.  All columns reside in the front.
                #define ROW_EDGE_CASE
                #include "block_apply_chunk.cu"
            }

            if (j1 < myTask.extra [6])
            {
                // Apply the update to columns j1 through the end of the front.
                // Check for both rows and columns outside the front
                #define ROW_EDGE_CASE
                #define COL_EDGE_CASE
                #include "block_apply_chunk.cu"
            }

        #else

            // Simple version:  always use the edge-handling code.

            int jend = myTask.extra [6] ;
            for ( ; j1 < jend ; j1 += 2*M)
            {
                // Apply the update to columns j1 through j1+2*M-1.
                // Check for both rows and columns outside the front
                #define ROW_EDGE_CASE
                #define COL_EDGE_CASE
                #include "block_apply_chunk.cu"
            }

        #endif

    #endif

}

//------------------------------------------------------------------------------
// #undefines of terms defined in the parent function
//------------------------------------------------------------------------------

#undef BLOCK_APPLY
#undef ROW_PANELSIZE
#undef COL_PANELSIZE
#undef M
