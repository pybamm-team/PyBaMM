// =============================================================================
// === GPUQREngine/Source/Scheduler_TransferData.cpp ===========================
// =============================================================================
//
// This file wraps logic surrounding the asynchronous transfer of data between
// the host and device. We transfer the workspace "surgically" meaning that
// the memory actually involved in the transfer is some subset of the full
// memory space allocated to the base pointer address. This results in less
// physical memory being transfered in any single transfer operation.
//
// We qsort the tasks by approximate amount of time required to perform the
// flops. We weigh certain tasks more than others in an effort to keep a more
// uniform distribution of task load on the GPU for each item in the work queue.
//   compareTaskTime is the comparator.
//
// =============================================================================

#include "GPUQREngine_Scheduler.hpp"
#include "GPUQREngine_TaskDescriptor.hpp"


int compareTaskTime (const void * a, const void * b)
{
    TaskDescriptor *ta = (TaskDescriptor*) a;
    TaskDescriptor *tb = (TaskDescriptor*) b;

    Int aFlops = getWeightedFlops(ta);
    Int bFlops = getWeightedFlops(tb);

    return bFlops - aFlops;
}

void Scheduler::transferData
(
    void
)
{
    /* Get the workspace */
    Workspace *wsWorkQueue = workQueues[activeSet];
    TaskDescriptor *queue = (TaskDescriptor*) wsWorkQueue->cpu();

    /* Accumulate the flops in this kernel launch. */
    for(Int t=0; t<numTasks[activeSet]; t++)
    {
        TaskDescriptor *task = &(queue[t]);
        Int flops = getFlops(task);
        gpuFlops += flops;
    }

    /* Sort the tasks in decreasing order by task time. */
    qsort (wsWorkQueue->cpu(), numTasks[activeSet], sizeof(TaskDescriptor),
        compareTaskTime);

    /* Surgically transfer the queue data from the scheduler onto the GPU: */
    Workspace wsQueueSurgical(numTasks[activeSet], sizeof(TaskDescriptor));
    wsQueueSurgical.assign(wsWorkQueue->cpu(), wsWorkQueue->gpu());
    wsQueueSurgical.transfer(cudaMemcpyHostToDevice, false, memoryStreamH2D);
    wsQueueSurgical.assign(NULL, NULL);
}
