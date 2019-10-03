// =============================================================================
// === GPUQREngine/Source/GPUQREngine_Internal.cpp =============================
// =============================================================================
//
// GPUQREngine_Internal is the call-down from the dense and sparse polymorphic
// wrappers. This code is responsible for maintaining the Scheduler and
// coordinating the factorization in a main loop.
//
// =============================================================================

#include "GPUQREngine_Internal.hpp"
#include "GPUQREngine_Scheduler.hpp"
#include "GPUQREngine_Stats.hpp"


QREngineResultCode GPUQREngine_Internal
(
    size_t gpuMemorySize,   // The total available GPU memory size in bytes
    Front *fronts,          // The list of fronts to factorize
    Int numFronts,          // The number of fronts to factorize
    Int *Parent,            // The front-to-parent mapping
    Int *Childp,            // Front-to-child column pointers
    Int *Child,             // Child permutation
                            // (Child[Childp[f]] to Child[Childp[f+1]] are all
                            // the front identifiers for front "f"'s children.
    QREngineStats *stats    // An optional parameter. If present, statistics
                            // are collected and passed back to the caller
                            // via this struct
)
{
    bool ok = true;

    /* Create the scheduler. */
    Scheduler *scheduler = (Scheduler*) SuiteSparse_calloc(1,sizeof(Scheduler));
    if (scheduler == NULL)
    {
        return QRENGINE_OUTOFMEMORY;
    }

    new (scheduler) Scheduler(fronts, numFronts, gpuMemorySize);
    ok = scheduler->memory_ok && scheduler->cuda_ok;

    /* If we encountered problems initializing the scheduler: */
    if(!ok)
    {
        bool memory_ok = scheduler->memory_ok ;
        bool cuda_ok = scheduler->cuda_ok ;
        if(scheduler)
        {
            scheduler->~Scheduler();
            scheduler = (Scheduler*) SuiteSparse_free (scheduler) ;
        }
        if(!memory_ok) return QRENGINE_OUTOFMEMORY;
        if(!cuda_ok) return QRENGINE_GPUERROR;
    }

    bool completed = false;
    while(!completed)
    {
//      #ifdef GPUQRENGINE_RENDER
//      scheduler->render();
//      #endif
        scheduler->fillWorkQueue();
        scheduler->transferData();

        // Launch the kernel and break out of the loop if we encountered
        // a cuda error.
        scheduler->launchKernel();
        if(!scheduler->cuda_ok) break;

        completed = scheduler->postProcess();
        scheduler->toggleQueue();
    }

    /* Report metrics back to the caller. */
    if(stats)
    {
        stats->kernelTime = scheduler->kernelTime;
        stats->numLaunches = scheduler->numKernelLaunches;
        stats->flopsActual = scheduler->gpuFlops;
    }

    /* Explicitly invoke the destructor */
    scheduler->~Scheduler();
    scheduler = (Scheduler*) SuiteSparse_free(scheduler);

    return QRENGINE_SUCCESS;
}
