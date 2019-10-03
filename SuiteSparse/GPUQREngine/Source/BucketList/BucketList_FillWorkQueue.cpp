// =============================================================================
// === GPUQREngine/Source/BucketList_FillWorkQueue.cpp =========================
// =============================================================================
//
// FillWorkQueue is responsible for filling the work queue with items and
// resolving generic TaskType entries on the bundles into concrete tasks
// to be performed by the GPU.
//
// This function should not be called for a particular front if we are at
// risk of exceeding the work queue.  The caller is responsible for this.
// The maximum number of tasks that can be placed in the queue is equal
// to (# row tiles) * (# col tiles) / applyGranularity, for any one front.
//
// =============================================================================

#include "GPUQREngine_BucketList.hpp"


// FillWorkQueue is responsible for filling the work queue with items and
// resolving generic TaskType entries on the bundles into concrete tasks
// to be performed by the GPU.
Int BucketList::FillWorkQueue
(
    TaskDescriptor *queue,  // The list of work items for the GPU
    Int *queueIndex         // The current index into the queue
)
{
    // Copy-in the current index
    Int qindex = *queueIndex;

    // Create and typecast object members to local variables.
    int fm          = (int) front->fm;
    int fn          = (int) front->fn;
    int numColTiles = (int) this->numColTiles;

    // For all bundles the bucket list is currently tracking:
    for (int i = 0; i < numBundles; i++)
    {
        LLBundle& bundle = Bundles[i];
        TaskType type = bundle.CurrentTask;
        int nativeBucket = (int) bundle.NativeBucket;

        // Configure for block task construction.
        int start = nativeBucket;

        // If the task type is a factorize:
        switch(type)
        {
            case TASKTYPE_GenericFactorize:
            {
                // General task configuration.
                TaskDescriptor task;
                bundle.gpuPack(&task);
                task.F = gpuF;
                task.fn = fn;
                task.fm = fm;

                // Set launch characteristics.
                int vtOwner = nativeBucket;
                task.extra[4] = TILESIZE * vtOwner;

                // Resolve the generic type to a specific type.

                // See if we need to consider edge cases.
                int lastColumn = (TILESIZE * vtOwner) + 31;
                bool isInternal = IsInternal(bundle, lastColumn);

                switch(bundle.Count)
                {
                    case 3:
                        task.Type = (isInternal ? TASKTYPE_FactorizeVT_3x1
                                                : TASKTYPE_FactorizeVT_3x1e);
                        break;
                    case 2:
                        task.Type = (isInternal ? TASKTYPE_FactorizeVT_2x1
                                                : TASKTYPE_FactorizeVT_2x1e);
                        break;
                    case 1:
                        task.Type = (isInternal ? TASKTYPE_FactorizeVT_1x1
                                                : TASKTYPE_FactorizeVT_1x1e);
                        break;
                }

                // Add the task to the queue.
                queue[qindex++] = task;

                break;
            }

            #ifdef GPUQRENGINE_PIPELINING
            case TASKTYPE_GenericApplyFactorize:
            {
                // General task configuration.
                TaskDescriptor task;
                bundle.gpuPack(&task);
                task.F = gpuF;
                task.fn = fn;
                task.fm = fm;

                // Set launch characteristics.
                int vtOwner = nativeBucket - 1;
                int from    = nativeBucket;
                int to      = MIN(nativeBucket + 1, numColTiles);
                task.extra[4] = TILESIZE * vtOwner;
                task.extra[5] = TILESIZE * from;
                task.extra[6] = TILESIZE * to;

                // Resolve the generic type to a specific type.
                int factorizeCount = bundle.Count;
                switch(bundle.ApplyCount)
                {
                    case 3:
                        switch(factorizeCount)
                        {
                            case 3:
                                task.Type = TASKTYPE_Apply3_Factorize3;
                                break;
                            case 2:
                                task.Type = TASKTYPE_Apply3_Factorize2;
                                break;
                            // case 1: never happens
                        }
                        break;
                    case 2:
                        switch(factorizeCount)
                        {
                            case 3:
                                task.Type = TASKTYPE_Apply2_Factorize3;
                                break;
                            case 2:
                                task.Type = TASKTYPE_Apply2_Factorize2;
                                break;
                            case 1:
                                task.Type = TASKTYPE_Apply2_Factorize1;
                                break;
                        }
                        break;
                    // case 1: never happens.  We never have an apply-factorize
                    // with one tile. A one-tile apply is considered a phantom
                    // bundle. We avoid the bogus rearrange.
                }

                // Add the task to the queue.
                queue[qindex++] = task;

                // Configure parameters to build the rest of the applies.
                start++;
                type = TASKTYPE_GenericApply;

                // INTENTIONALLY FALL THROUGH TO BUILD THE APPLIES
            }
            #endif

            case TASKTYPE_GenericApply:
            {
                for( ; start < numBuckets; start += ApplyGranularity)
                {
                    // General task configuration.
                    TaskDescriptor task;
                    bundle.gpuPack(&task);
                    task.F = gpuF;
                    task.fn = fn;
                    task.fm = fm;

                    // Set launch characteristics.
                    int vtOwner = nativeBucket - 1;
                    int from    = start;
                    int to      = MIN(start + ApplyGranularity, numColTiles);
                    task.extra[4] = TILESIZE * vtOwner;
                    task.extra[5] = TILESIZE * from;
                    task.extra[6] = TILESIZE * to;

                    // Resolve the generic type to a specific type.
                    switch(bundle.ApplyCount)
                    {
                        case 3: task.Type = TASKTYPE_Apply3; break;
                        case 2: task.Type = TASKTYPE_Apply2; break;
                        case 1: task.Type = TASKTYPE_Apply1; break;
                    }

                    // Add the task to the queue.
                    queue[qindex++] = task;
                }

                break;
            }
            default: break; // DEAD: no default case is ever used.
        }
    }

    // Compute the number of tasks we just built.
    Int numTasks = qindex - *queueIndex;

    // Copy-out the current index
    *queueIndex = qindex;

    return numTasks;
}
