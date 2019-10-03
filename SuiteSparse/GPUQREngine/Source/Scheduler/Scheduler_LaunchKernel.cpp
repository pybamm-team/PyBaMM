// =============================================================================
// === GPUQREngine/Source/Scheduler_LaunchKernel.cpp ===========================
// =============================================================================
//
// This file wraps logic to launch the GPU kernel on alternating kernel streams
// coordinated by the Scheduler. The work lists also alternate to facilitate
// overlapping communication with computation. We use the CUDA events and
// streams model througout the Scheduler to coordinate asynchronous launches.
//
// =============================================================================

#include "GPUQREngine_Scheduler.hpp"
#include "GPUQREngine_Internal.hpp"
#include "GPUQREngine_Timing.hpp"


void Scheduler::launchKernel
(
    void
)
{
    /* Get the gpu pointer for the work queue. */
    Workspace *wsWorkQueue = workQueues[activeSet];
    TaskDescriptor *gpuWorkQueue = GPU_REFERENCE(wsWorkQueue, TaskDescriptor*);

    /* Get the kernel streams */
    cudaStream_t thisKernel = kernelStreams[activeSet];
    cudaStream_t lastKernel = kernelStreams[activeSet^1];

    /* Wait for the last kernel and all H2D memory transfers to finish. */
    cudaStreamSynchronize(lastKernel);
    cudaStreamSynchronize(memoryStreamH2D);

    /* Launch the kernel if we have valid tasks to do. */
    if(numTasks[activeSet] > 0)
    {
        /* Keep track of the number of launches. */
        numKernelLaunches++;

#ifdef TIMING
        float totalTime = 0.0;
        TIMER_INIT();
        TIMER_START();
#endif

        /* Launch the kernel. */
        GPUQREngine_UberKernel(thisKernel, gpuWorkQueue, numTasks[activeSet]);

#ifdef FORCE_SYNCHRONIZE
        cudaDeviceSynchronize();
        cuda_ok = cuda_ok && (cudaGetLastError() == cudaSuccess);
#endif

#ifdef TIMING
        TIMER_STOP(totalTime);
        TIMER_FINISH();
        kernelTime += totalTime;
#endif
    }

    /* Clear the number of tasks. */
    numTasks[activeSet] = 0;
}
