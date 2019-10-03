
#ifndef PARAMS_APPLY_HPP_
#define PARAMS_APPLY_HPP_

//------------------------------------------------------------------------------
// definitions for all variants of block_apply
//------------------------------------------------------------------------------

// maximum number of row and column tiles in a panel
#define MAX_ROW_TILES 3
#define MAX_COL_TILES 2

#define shV shMemory.apply.V
#define shC shMemory.apply.C

// each tile is 32-by-32, which is always M, for all variants
// (M is defined inside block_apply.cu, and then #undef'd there also)
// #define M TILESIZE

// V1 is held in the lower triangular part of the glVT array, including the
// diagonal.  Thg glVT array is of size (M+1)-by-M.  The upper triangular part
// holds T, (also including a diagonal)
#define GLVT(i,j)   (glVT [1+(i)][j])

// The shared array V is K-by-M (1 to 3 tiles of size M-by-M), with an
// extra column for padding.  It is indexed first by the t-th row tile, and
// then (i,j) within that t-th tile.  V is lower triangular, and shares its
// space with the upper triangular T matrix.
#define SHV(t,i,j)  (shV [1+ (t)*TILESIZE + (i)][j])

// Macros for accessing entries in a frontal matrix.  The A matrix and most of
// the V matrix reside in the frontal matrix as a set of tiles.  The row index
// of GLF(t,i,j) is defined by row tile t and by a row index i within that tile
// (i is in the range 0 to the tilesize-1).  Column j refers to the global
// column index in F.  fi = IFRONT(t,i) translates the tile t and row i inside
// that tile to an index fi which is in the range 0 to fm-1, which is an index
// into the front in global memory.
#define IFRONT(t,i) ((i) + myTask.extra [t])
#define GLF(t,i,j)  glF [IFRONT(t,i) * fn + (j)]

// C is used to buffer A, when computing C=V'*A
#define SHA(i,j)    (shC [i][j])

// T is upper triangular of size M-by-M, and shares its space with V
#define ST(i,j)     (shV [i][j])

// Each thread loads V(iv,jv) from global, and then iv+chunksize,
// iv+2*chunksize, etc.  With M = 32 and 384 threads, the chunksize is 12,
// and the number of chunks is 3.
#define iv          (threadIdx.x / TILESIZE)
#define jv          (threadIdx.x % TILESIZE)
#define VCHUNKSIZE  (NUMTHREADS / TILESIZE)
#define NVCHUNKS    CEIL (TILESIZE*TILESIZE, NUMTHREADS)

// device functions block_apply, one for each variant
__device__ void block_apply_3 ( ) ;
__device__ void block_apply_2 ( ) ;
__device__ void block_apply_1 ( ) ;
__device__ void block_apply_3_by_1 ( ) ;
__device__ void block_apply_2_by_1_( ) ;
__device__ void block_apply_1_by_1 ( ) ;

#endif
