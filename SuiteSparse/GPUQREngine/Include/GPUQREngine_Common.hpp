// =============================================================================
// === GPUQREngine/Include/GPUQREngine_Common.hpp ==============================
// =============================================================================
//
// This include file contains
//    - Thread geometry and related manifest constants
//    - Common macros and definitions
//
// =============================================================================

#ifndef GPUQRENGINE_COMMON_HPP
#define GPUQRENGINE_COMMON_HPP

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include "SuiteSparseGPU_Runtime.hpp"


/*** GPU Parameters **********************************************************/

//
// PANELSIZE  refers to the number of tiles that we can accomodate in
//            an apply or factorize tasks
// TILESIZE   refers to the square dimension of dense front data composed
//            as blocks into panels
// HALFTILE   refers to half the TILESIZE square dimension. This is used in
//            some kernels to optimize
// PADDING    refers to the amount of padding required to prevent shared memory
//            bank conflicts. In reality, our "tiles" are TILESIZE+PADDING in
//            square dimension size.
// NUMTHREADS refers to the number of threads with which we launch each kernel
// PACKASSEMBLY_SHMEM_MAPINTS
//            refers to the number of RiMap and RjMap integers that we can
//            keep in shared memory for pack assembly tasks. This value is
//            used in the decomposition of contribution blocks and the
//            subsequent construction of PackAssembly tasks.
#define PANELSIZE           (3)
#define TILESIZE            (32)
#define HALFTILE            (TILESIZE / 2)
#define PADDING             (1)

#define NUMTHREADS                     384
#define PACKASSEMBLY_SHMEM_MAPINTS     2024


/*** Common Macros ***********************************************************/

// ceiling of a/b for two integers a and b
#ifndef CEIL
#define CEIL(a,b)   (((a) + (b) - 1) / (b))
#endif

#ifndef MIN
#define MIN(x,y)    ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x,y)    ((x) > (y) ? (x) : (y))
#endif

#define EMPTY       (-1)
#define Int         SuiteSparse_long

// To render the buckets for visualization in graphvis, uncomment this next
// line, or compile the code with -DGPUQRENGINE_RENDER
// #define GPUQRENGINE_RENDER

// deprecated:
// and also enable one or both of these, as true:
// #define RENDER_DENSE_FACTORIZATION       false
// #define RENDER_SPARSE_FACTORIZATION      false

// To enable pipelining (combined apply-factorize tasks), uncomment this
// next 3 lines, or compile with -DGPUQRENGINE_PIPELINING
//  #ifndef GPUQRENGINE_PIPELINING
//  #define GPUQRENGINE_PIPELINING
//  #endif
// This is an experimental feature that is not fully tested.
// It should not be used in production use.

#endif
