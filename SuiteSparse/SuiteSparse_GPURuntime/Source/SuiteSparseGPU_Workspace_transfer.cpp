// =============================================================================
// SuiteSparse_GPURuntime/Source/SuiteSparseGPU_Workspace_transfer.cpp =========
// =============================================================================

#include "SuiteSparseGPU_Runtime.hpp"

// -----------------------------------------------------------------------------
// transfer: synchronous/asynchronous transfer of memory to/from the CPU/GPU
// -----------------------------------------------------------------------------

bool Workspace::transfer(cudaMemcpyKind direction, bool synchronous,
    cudaStream_t stream)
{
    /* Check inputs */
//  if(!cpuReference || (!gpuReference && !lazyAllocate)) return false;
    if(!cpuReference ||  !gpuReference                  ) return false;

//  // Handle lazy allocate (for possible future use)
//  if(direction == cudaMemcpyHostToDevice && lazyAllocate && !gpuReference)
//  {
//      gpuReference = Workspace::gpu_malloc(nitems, size_of_item);
//      if(!gpuReference) return false;
//  }

    // Set the src and dst depending on the direction.
    void *src = NULL, *dst = NULL;
    if(direction == cudaMemcpyHostToDevice)
    {
        src = cpuReference;
        dst = gpuReference;
    }
    else if(direction == cudaMemcpyDeviceToHost)
    {
        src = gpuReference;
        dst = cpuReference;
    }
    else
    {
        // Unhandled cudaMemcpyKind value in Workspace::transfer
        return false;
    }

    // Do the transfer and if synchronous wait until completed.
    cudaError_t result;
    if(synchronous)
    {
        result = cudaMemcpy(dst, src, totalSize, direction);
    }
    else
    {
        result = cudaMemcpyAsync(dst, src, totalSize, direction, stream);
    }

    if(result != cudaSuccess)
    {
        return false;       // memcpy failed
    }

    return true;            // success
}
