// =============================================================================
// === GPUQREngine/Source/Scheduler.cpp ========================================
// =============================================================================
//
// This file contains code to construct, initialize, and destroy the Scheduler.
//
// To support debugging and code coverage tests, we use placement new in order
// to trap and exercise out-of-memory conditions within the operating system
// memory manager.
//
// =============================================================================
// The pattern in use in this file is the memory allocation self-contained
// within the constructor with concrete initialization codes appearing in the
// initializer.This practice is common in OO languages, such as Java in which
// the constructor is responsible for memory management AND initialization.
// =============================================================================

#include "GPUQREngine_Scheduler.hpp"

// -----------------------------------------------------------------------------
// Macro destructor
// -----------------------------------------------------------------------------

#define FREE_EVERYTHING \
    afPerm = (Int *) SuiteSparse_free(afPerm); \
    afPinv = (Int *) SuiteSparse_free(afPinv); \
    if(bucketLists) \
    { \
        for(Int f=0; f<numFronts; f++) \
        { \
            BucketList *dlbl = (&bucketLists[f]); \
            dlbl->~BucketList(); \
        } \
    } \
    bucketLists = (BucketList*) SuiteSparse_free(bucketLists); \
    FrontDataPulled = (bool *) SuiteSparse_free(FrontDataPulled); \
    eventFrontDataReady =(cudaEvent_t*) SuiteSparse_free(eventFrontDataReady); \
    eventFrontDataPulled=(cudaEvent_t*) SuiteSparse_free(eventFrontDataPulled);\
    for(int q=0; q<2; q++) workQueues[q] = Workspace::destroy(workQueues[q]); \
    if (kernelStreams[0] != NULL) cudaStreamDestroy(kernelStreams[0]); \
    if (kernelStreams[1] != NULL) cudaStreamDestroy(kernelStreams[1]); \
    if (memoryStreamH2D  != NULL) cudaStreamDestroy(memoryStreamH2D); \
    if (memoryStreamD2H  != NULL) cudaStreamDestroy(memoryStreamD2H); \
    kernelStreams[0] = NULL ; \
    kernelStreams[1] = NULL ; \
    memoryStreamH2D = NULL ; \
    memoryStreamD2H = NULL ;

// -----------------------------------------------------------------------------
// Scheduler constructor
// -----------------------------------------------------------------------------

Scheduler::Scheduler
(
    Front *fronts,
    Int numFronts,
    size_t gpuMemorySize
)
{
    frontList = fronts;
    this->numFronts = numFronts;
    numFrontsCompleted = 0;

    kernelStreams[0] = NULL ;
    kernelStreams[1] = NULL ;
    memoryStreamH2D = NULL ;
    memoryStreamD2H = NULL ;

    workQueues[0] = NULL ;
    workQueues[1] = NULL ;

    /* Allocate scheduler memory, checking for out of memory frequently.
     *  Remark: SuiteSparse_calloc is important to use here for "mongo" memory
     *          initializations because the objects have been designed
     *          using flags that trigger only if set to true.
     *          In other words, using malloc here could accidentally
     *          trigger unexpected behavior.
     */
    memory_ok = true;
    cuda_ok = true;

    afPerm = (Int*) SuiteSparse_calloc(numFronts, sizeof(Int));
    afPinv = (Int*) SuiteSparse_calloc(numFronts, sizeof(Int));
    bucketLists = (BucketList*)
        SuiteSparse_calloc(numFronts, sizeof(BucketList));
    FrontDataPulled = (bool*) SuiteSparse_calloc(numFronts, sizeof(bool));
    eventFrontDataReady =
        (cudaEvent_t*) SuiteSparse_calloc(numFronts, sizeof(cudaEvent_t));
    eventFrontDataPulled =
        (cudaEvent_t*) SuiteSparse_calloc(numFronts, sizeof(cudaEvent_t));

    if(!afPerm || !afPinv || !bucketLists || !FrontDataPulled
       || !eventFrontDataReady || !eventFrontDataPulled)
    {
        FREE_EVERYTHING ;
        memory_ok = false;
        return;
    }

    /* Scheduler memory has all been allocated by this point,
       or we have exited the constructor. */

    /* Initialize the scheduler and allocate memory for doubly-linked
       bucket lists.
       If this fails, we have either cuda_ok = false or memory_ok = false. */
    if(!initialize(gpuMemorySize))
    {
        FREE_EVERYTHING;
        // If cuda_ok is still true then we ran out of memory.
        // Else we had enough memory but failed the cuda calls.
        if(cuda_ok) memory_ok = false;
        return;
    }

    /* Stats fields */
    kernelTime = 0.0;
    numKernelLaunches = 0;
    gpuFlops = 0;

    #ifdef GPUQRENGINE_RENDER
    /* Debug fields */
    TaskNames[0] = "TASKTYPE_Nothing";
    TaskNames[1] = "TASKTYPE_GenericFactorize";
    TaskNames[2] = "TASKTYPE_FactorizeVT_3x1";
    TaskNames[3] = "TASKTYPE_FactorizeVT_2x1";
    TaskNames[4] = "TASKTYPE_FactorizeVT_1x1";
    TaskNames[5] = "TASKTYPE_FactorizeVT_3x1e";
    TaskNames[6] = "TASKTYPE_FactorizeVT_2x1e";
    TaskNames[7] = "TASKTYPE_FactorizeVT_1x1e";
    TaskNames[8] = "TASKTYPE_FactorizeVT_3x1w";
    TaskNames[9] = "TASKTYPE_GenericApply";
    TaskNames[10] = "TASKTYPE_Apply3";
    TaskNames[11] = "TASKTYPE_Apply2";
    TaskNames[12] = "TASKTYPE_Apply1";
    TaskNames[13] = "TASKTYPE_GenericApplyFactorize";
    TaskNames[14] = "TASKTYPE_Apply3_Factorize3";
    TaskNames[15] = "TASKTYPE_Apply3_Factorize2";
    TaskNames[16] = "TASKTYPE_Apply2_Factorize3";
    TaskNames[17] = "TASKTYPE_Apply2_Factorize2";
    TaskNames[18] = "TASKTYPE_Apply2_Factorize1";
    TaskNames[19] = "TASKTYPE_SAssembly";
    TaskNames[20] = "TASKTYPE_PackAssembly";

    StateNames[0] = "ALLOCATE_WAIT";
    StateNames[1] = "ASSEMBLE_S";
    StateNames[2] = "CHILD_WAIT";
    StateNames[3] = "FACTORIZE";
    StateNames[4] = "FACTORIZE_COMPLETE";
    StateNames[5] = "PARENT_WAIT";
    StateNames[6] = "PUSH_ASSEMBLE";
    StateNames[7] = "CLEANUP";
    StateNames[8] = "DONE";

    renderCount = 0;
    #endif
}

// -----------------------------------------------------------------------------
// Scheduler destructor
// -----------------------------------------------------------------------------

Scheduler::~Scheduler()
{
    FREE_EVERYTHING ;
}

// -----------------------------------------------------------------------------
// Scheduler::initialize
// -----------------------------------------------------------------------------
// Returns true if OK, false if out of memory or cuda initialization failed.
// -----------------------------------------------------------------------------
bool Scheduler::initialize
(
    size_t gpuMemorySize
)
{
    activeSet = 0;
    numActiveFronts = 0;
    minApplyGranularity = SSGPU_MINAPPLYGRANULARITY ;

    for(int pf=0; pf<numFronts; pf++)
    {
        /* Extract the front details from the frontListing. */
        Front *front = &(frontList[pf]);
        SparseMeta *meta = &(front->sparseMeta);
        Int f = front->fids;
        bool isDense = front->isDense();
        bool pushOnly = front->isPushOnly();

        /* Configure active front inverse permutation. */
        afPinv[f] = EMPTY;

        /* Configure the bucket list for each front. */
        BucketList *dlbl = (&bucketLists[f]);
        dlbl->useFlag = false;
        if(front->isTooBigForSmallQR())
        {
            /* Only allocate and initialize a bucketlist scheduler
               if we're doing more than a push assembly. */
            if(!pushOnly)
            {
                new (dlbl) BucketList(front, minApplyGranularity);
                if(!dlbl->memory_ok) return false;
                dlbl->gpuF = (&frontList[f])->gpuF;
            }
        }

        eventFrontDataReady[f] = NULL;
        eventFrontDataPulled[f] = NULL;
        FrontDataPulled[f] = false;

        /* If the front is dense, activate it immediately. */
        if(isDense)
        {
            /* Activate the front for factorization. */
            activateFront(f);
            initializeBucketList(f);
        }
        /* Else the front is sparse: */
        else
        {
            /* If the front has no children: */
            if(meta->nc == 0)
            {
                /* Activate the front. */
                activateFront(f);

                /* Initialize the bucket list for front f.
                   If we're only doing a push, bucketLists[f] will be NULL,
                   and this function does nothing. */
                initializeBucketList(f);
            }
        }
    }

    // determine the size of the work queue
    maxQueueSize = (Int) ssgpu_maxQueueSize (gpuMemorySize) ;

    for(int q=0; q<2; q++)
    {
        // malloc on both CPU (pagelocked) and GPU
        workQueues[q] = Workspace::allocate (maxQueueSize,  // CPU and GPU
            sizeof(TaskDescriptor), false, true, true, true) ;
        if(!workQueues[q]) return false;
        numTasks[q] = 0;
    }

    cuda_ok = cuda_ok && (cudaSuccess == cudaStreamCreate(&kernelStreams[0]));
    cuda_ok = cuda_ok && (cudaSuccess == cudaStreamCreate(&kernelStreams[1]));
    cuda_ok = cuda_ok && (cudaSuccess == cudaStreamCreate(&memoryStreamH2D));
    cuda_ok = cuda_ok && (cudaSuccess == cudaStreamCreate(&memoryStreamD2H));

    return cuda_ok;
}
