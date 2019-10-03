// =============================================================================
// === GPUQREngine/Include/GPUQREngine_internal.hpp ============================
// =============================================================================
//
// The GPUQREngine_internal is an internal global include file.
//
// =============================================================================

#ifndef GPUQRENGINE_INTERNAL_HPP
#define GPUQRENGINE_INTERNAL_HPP

#include "GPUQREngine_Common.hpp"
#include "GPUQREngine_TaskDescriptor.hpp"
#include "GPUQREngine_Front.hpp"
#include "GPUQREngine_Stats.hpp"
#include "GPUQREngine.hpp"

void GPUQREngine_UberKernel
(
    cudaStream_t kernelStream,      // The stream on which to launch the kernel
    TaskDescriptor *gpuWorkQueue,   // The list of work items for the GPU
    int numTasks                    // The # of items in the work list
);

QREngineResultCode GPUQREngine_Internal
(
    size_t gpuMemorySize,           // The total size of the GPU memory
    Front *fronts,                  // The list of fronts to factorize
    Int numFronts,                  // The number of fronts in the list
    Int *Parent = NULL,             // Map from front to its Parent
    Int *Childp = NULL,             // Child[Childp[f]] to Child[Childp[f+1]]
    Int *Child  = NULL,             // has all the children of front f.
    QREngineStats *stats = NULL     // An optional in-out parameter to capture
                                    // statistics
);

#endif
