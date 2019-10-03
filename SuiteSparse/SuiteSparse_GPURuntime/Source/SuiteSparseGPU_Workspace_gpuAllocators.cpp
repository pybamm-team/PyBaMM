// =============================================================================
// === SuiteSparse_GPURuntime/Source/SuiteSparseGPU_Workspace_gpuAllocators.cpp
// =============================================================================

#include "SuiteSparseGPU_Runtime.hpp"
#include <stdio.h>

// -----------------------------------------------------------------------------
// gpu_malloc: malloc memory on the GPU
// -----------------------------------------------------------------------------

void *Workspace::gpu_malloc(size_t nitems, size_t size_of_item)
{

    if (nitems == 0)
    {
        // make sure something gets allocated, to avoid spurious failures
        nitems = 1 ;
    }

    void *returner = NULL;

    size_t requestSize = nitems*size_of_item;

    // check for integer overflow
    if (requestSize != ((double) nitems) * size_of_item)
    {
        return (NULL) ;     // size_t overflow
    }

    cudaError_t result = cudaMalloc(&returner, requestSize);

    if(result != cudaSuccess)
    {
        return (NULL) ;     // failed to malloc on the GPU
    }

    return returner;
}

// -----------------------------------------------------------------------------
// gpu_calloc: calloc memory on the GPU
// -----------------------------------------------------------------------------

void *Workspace::gpu_calloc(size_t nitems, size_t size_of_item)
{

    if (nitems == 0)
    {
        // make sure something gets allocated, to avoid spurious failures
        nitems = 1 ;
    }

    void *returner = gpu_malloc(nitems, size_of_item);
    if(returner)
    {
        cudaError_t result = cudaMemset(returner, 0, nitems*size_of_item);

        if(result != cudaSuccess)
        {
            returner = gpu_free(returner);      // memset failed on GPU
        }
    }

    return returner;
}

// -----------------------------------------------------------------------------
// gpu_free: free memory on the GPU
// -----------------------------------------------------------------------------

void *Workspace::gpu_free(void *address)
{
    if(!address) return NULL;

    cudaError_t result = cudaFree(address);

    return NULL;        // not an error, but for convenience for the caller
}


// -----------------------------------------------------------------------------
// gpu_memset:  set entire GPU memory block to a single value
// -----------------------------------------------------------------------------

// unused.  Left commented out for possible future use.

// bool Workspace::gpu_memset(size_t value)
// {
//     cudaError_t result = cudaMemset(gpuReference, value, totalSize);
//     return (result == cudaSuccess) ;
// }

