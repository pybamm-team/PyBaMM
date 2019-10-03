// =============================================================================
// === GPUQREngine/Include/GPUQREngine_Scheduler.hpp ===========================
// =============================================================================
//
// The Scheduler is a principal class in the GPUQREngine.
//
// This class manages the input set of Fronts, creates BucketLists when
// necessary for factorization, and contains all logic required to coordinate
// the factorization and assembly tasks with the GPU.
//
// =============================================================================

#ifndef GPUQRENGINE_SCHEDULER_HPP
#define GPUQRENGINE_SCHEDULER_HPP

#include "GPUQREngine_Common.hpp"
#include "GPUQREngine_FrontState.hpp"
#include "GPUQREngine_TaskDescriptor.hpp"
#include "GPUQREngine_BucketList.hpp"
#include "GPUQREngine_LLBundle.hpp"
#include "GPUQREngine_Front.hpp"

#define SSGPU_MINAPPLYGRANULARITY 16

size_t ssgpu_maxQueueSize       // return size of scheduler queue
(
    size_t gpuMemorySize        // size of GPU memory, in bytes
) ;

class Scheduler
{
private:
    /* Scheduler.cpp */
    bool initialize(size_t gpuMemorySize);

    /* Scheduler_Front.cpp */
    bool pullFrontData(Int f);

    /* Scheduler_FillWorkQueue.cpp */
    void fillTasks
    (
        Int f,                          // INPUT: Current front
        TaskDescriptor *queue,          // INPUT: CPU Task entries
        Int *queueIndex                 // IN/OUT: The index of the current entry
    );

public:
    bool memory_ok;                     // Flag for the creating function to
                                        // determine whether we had enough
                                        // memory to initialize the Scheduler.
    bool cuda_ok;                       // Flag for the creating function to
                                        // determine whether we could
                                        // successfully invoke the cuda
                                        // initialization calls.

    Front *frontList;
    Int numFronts;
    Int numFrontsCompleted;

    int activeSet;

    BucketList *bucketLists;

    Int *afPerm;                        // Permutation of "active" fronts
    Int *afPinv;                        // Inverse permutation of "active" fronts
    Int numActiveFronts;

    Int maxQueueSize;
    Workspace *workQueues[2];
    Int numTasks[2];
    Int minApplyGranularity;            // The minimum number of tiles for which
                                        // we will group apply tasks

    bool *FrontDataPulled;              // A set of flags indicating whether R has
                                        // been pulled off the GPU.
    cudaEvent_t *eventFrontDataReady;   // A list of cudaEvents that are used to
                                        // coordinate when the R factor is ready
                                        // to be pulled from the GPU.
    cudaEvent_t *eventFrontDataPulled;  // A list of cudaEvents that are used to
                                        // coordinate when the R factor is finally
                                        // finished transfering off the GPU.

    // Use multiple CUDA streams to coordinate kernel launches and asynchronous
    // memory transfers between the host and the device:
    //   kernelStreams : Launch kernels on alternating streams
    //   H2D           : Asynchronous memory transfer stream (Host-to-Device)
    //   D2H           : Asynchronous memory transfer stream (Device-to-Host)
    cudaStream_t kernelStreams[2];
    cudaStream_t memoryStreamH2D;
    cudaStream_t memoryStreamD2H;

    /* Scheduler.cpp */
    void *operator new(long unsigned int, Scheduler* p){ return p; }
    Scheduler(Front *fronts, Int numFronts, size_t gpuMemorySize);
    ~Scheduler();

    /* Scheduler_Front.cpp */
    void activateFront
    (
        Int f                   // The index of the front to operate on
    );

    bool finishFront
    (
        Int f                   // The index of the front to operate on
    );

    void initializeBucketList
    (
        Int f                   // The index of the front to operate on
    )
    {
        // NOTE: tested by SPQR/Tcov, but not flagged as such in cov results
        BucketList *dlbl = (&bucketLists[f]);
        if(dlbl->useFlag) dlbl->Initialize();
    }

    /* Scheduler_TransferData.cpp */
    void transferData
    (
        void
    );

    /* Scheduler_FillWorkQueue.cpp */
    void fillWorkQueue
    (
        void
    );

    /* Scheduler_LaunchKernel.cpp */
    void launchKernel
    (
        void
    );

    /* Scheduler_PostProcess.cpp */
    bool postProcess
    (
        void
    );

    void toggleQueue
    (
        void
    )
    {
        activeSet ^= 1;
    }

    /* Stats */
    float kernelTime;
    Int numKernelLaunches;
    Int gpuFlops;

#ifdef GPUQRENGINE_RENDER
    /* Debug stuff */
    const char *TaskNames[21];
    const char *StateNames[9];
    int renderCount;
    void render();
#endif

#if 1
    void debugDumpFront(Front *front);
#endif
};

#endif
