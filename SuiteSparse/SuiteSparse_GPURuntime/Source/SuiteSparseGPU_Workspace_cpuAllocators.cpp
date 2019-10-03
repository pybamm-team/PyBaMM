// =============================================================================
// === SuiteSparse_GPURuntime/Source/SuiteSparseGPU_Workspace_cpuAllocators.cpp
// =============================================================================

#include "SuiteSparseGPU_Runtime.hpp"
#include <string.h>

// -----------------------------------------------------------------------------
// cpu_malloc: malloc memory on the CPU, optionally pagelocked
// -----------------------------------------------------------------------------

void *Workspace::cpu_malloc(size_t nitems, size_t size_of_item, bool pageLocked)
{

    if (nitems == 0)
    {
        /* make sure something gets allocated, to avoid spurious failures */
        nitems = 1 ;
    }

    /* If we're not pageLocked, just use the standard malloc. */
    void *returner;
    if(!pageLocked)
    {
        returner = SuiteSparse_malloc(nitems, size_of_item);
    }
    else
    {
        size_t requestSize = nitems * size_of_item;

        // check for integer overflow
        if (requestSize != ((double) nitems) * size_of_item)
        {
            return (NULL) ;     // size_t overflow
        }

        cudaError_t result = cudaMallocHost(&returner, requestSize);
        if (!result == cudaSuccess)
        {
            return (NULL) ;     // failed to malloc pagelocked memory
        }
    }

    return returner;
}

// -----------------------------------------------------------------------------
// cpu_calloc: calloc memory on the CPU, optionally pagelocked
// -----------------------------------------------------------------------------

void *Workspace::cpu_calloc(size_t nitems, size_t size_of_item, bool pageLocked)
{

    if (nitems == 0)
    {
        /* make sure something gets allocated, to avoid spurious failures */
        nitems = 1 ;
    }

    /* If we're not pageLocked, just use the standard calloc. */
    void *returner;
    if(!pageLocked)
    {
        returner = SuiteSparse_calloc(nitems, size_of_item);
    }
    else
    {
        // malloc pagelocked memory on the CPU
        returner = cpu_malloc (nitems, size_of_item, true) ;

        #if 0
        size_t requestSize = nitems * size_of_item;

        // check for integer overflow
        if (requestSize != ((double) nitems) * size_of_item)
        {
            return (NULL) ;     // size_t overflow
        }

        cudaError_t result = cudaMallocHost(&returner, requestSize);
        if (!result == cudaSuccess)
        { 
            return (NULL) ;     // failed to malloc pagelocked memory
        }
        #endif

        // set the memory to all zeros
        if (returner != NULL)
        {
            // size_t overflow already checked by cpu_malloc
            memset (returner, 0, nitems * size_of_item) ;
        }
    }

    return returner;
}

// -----------------------------------------------------------------------------
// cpu_free: free memory on the CPU, optionally pagelocked
// -----------------------------------------------------------------------------

void *Workspace::cpu_free(void *address, bool pageLocked)
{
    if(!address) return NULL;

    if(!pageLocked)
    {
        SuiteSparse_free(address);
    }
    else
    {
        cudaError_t result = cudaFreeHost(address);
    }

    return NULL;        // not an error, but for convenience for the caller
}

// -----------------------------------------------------------------------------
// cpu_memset:  set entire CPU memory block to a single value
// -----------------------------------------------------------------------------

// unused.  Left commented out for possible future use.

// bool Workspace::cpu_memset(size_t value)
// {
//     memset(cpuReference, value, totalSize);
//     return true;
// }

