// =============================================================================
// === GPUQREngine/Source/Scheduler_FillWorkQueue.cpp ==========================
// =============================================================================
//
// This file contains logic to run through the factorization state machine.
// At each stage, different operations are performed:
//
//   - ALLOCATE_WAIT
//     If a front is in ALLOCATE_WAIT then we are waiting for the memory to be
//     available on the gpu. This is a holding state and the front has no work
//     to do at this stage. A front is advanced out of this state by a child
//     front's PARENT_WAIT postprocessor code.
//
//   - ASSEMBLE_S
//     Fronts in ASSEMBLE_S build S Assembly tasks with a granularity specified
//     by the number of data movements we can tolerate on the GPU (for now this
//     value is hardcoded to 4).
//
//     If a front has no values, or if it has created all of its S Assembly
//     tasks then it is advanced to CHILD_WAIT
//
//   - CHILD_WAIT
//     A front in CHILD_WAIT must wait until all of its children's contribution
//     block data are pushed into it before it may proceed to FACTORIZE.
//     Fronts are advanced out of CHILD_WAIT by the postprocessor.
//
//   - FACTORIZE
//     Fronts in FACTORIZE either use the BucketList sub scheduler to manage
//     their factorization or they are small enough to simply use the SmallQR
//     GPU kernel to complete their factorization.
//     Once a front has added all of its factorize tasks and the BucketList
//     scheduler determines that the front has been factorized, the front is
//     advanced to FACTORIZE_COMPLETE, signalling that the set of tasks is the
//     last set of tasks to perform before the front is considered factorized.
//
//   - FACTORIZE_COMPLETE
//     Fronts in FACTORIZE_COMPLETE wait for a cuda event remarking that front
//     data (R factor) is ready to be transfered off of the GPU.
//     The postprocessor advances fronts out of FACTORIZE_COMPLETE into
//     PARENT_WAIT (if not staged) or CLEANUP (if staged).
//
//   - PARENT_WAIT
//     Fronts in PARENT_WAIT are waiting for their parent to be allocated in
//     postprocessing. Postprocessing advances these fronts to PUSH_ASSEMBLE
//     once the parent is ready for push assembly on the GPU. Alternatively
//     if the front has no parent, the postprocessor will advance the front to
//     CLEANUP.
//
//   - PUSH_ASSEMBLE
//     Fronts in PUSH_ASSEMBLE build PackAssemble tasks that move data from
//     the child's memory space into the parent's memory space on the GPU.
//     The granularity of these memory moves is dictated by the available
//     amount of shared memory. Once a front has added all of its PUSH_ASSEMBLE
//     tasks to the work queue, it is advanced to CLEANUP.
//
//   - CLEANUP
//     Fronts in CLEANUP are managed by the postprocessor. In general, fronts
//     in CLEANUP are waiting for their corresponding R factors to be transfered
//     off of the GPU.
//
//   - DONE
//     Fronts in DONE have no more work nor additional state transitions.
//     When all fronts are in the DONE state then the QREngine's work is done.
//
// =============================================================================

#include "GPUQREngine_Scheduler.hpp"
#include "GPUQREngine_GraphVizHelper.hpp"


// -----------------------------------------------------------------------------
// prototypes for local functions
// -----------------------------------------------------------------------------

TaskDescriptor buildSAssemblyTask
(
    Front *front,
    int pstart,
    int pend
);

TaskDescriptor buildPackAssemblyTask
(
    Front *front,
    int cistart,
    int ciend,
    int cjstart,
    int cjend
);

TaskDescriptor buildSmallQRTask
(
    Front *front
);

// -----------------------------------------------------------------------------
// Scheduler::fillWorkQueue
// -----------------------------------------------------------------------------

void Scheduler::fillWorkQueue
(
    void
)
{
    /* Get the queue that we want to fill. */
    TaskDescriptor *queue = (TaskDescriptor*) workQueues[activeSet]->cpu();

    Int taskIndex = 0;
    bool queueFull = (taskIndex == maxQueueSize);
    for(Int p=0; p<numActiveFronts && !queueFull; p++)
    {
        /* Get the front from the active fronts permutation. */
        Int f = afPerm[p];

        /* Fill the queue with tasks from f. */
        fillTasks(f, queue, &taskIndex);

        /* See if the queue is full. */
        queueFull = (taskIndex == maxQueueSize);
    }

    /* Save the number of tasks & bundles in this run. */
    numTasks[activeSet] = taskIndex;

#if 0
#define MORE_DETAIL
    /* Debug prints */
    if(numTasks[activeSet] > 0)
    {
//    printf("numTasks[%d] = %ld\n", activeSet, taskIndex);
#ifdef MORE_DETAIL
        for(Int t=0; t<numTasks[activeSet]; t++)
        {
            bool valid = false;
            for(Int p=0; p<numActiveFronts && !valid; p++)
            {
                Int f = afPerm[p];
                Front *front = (&frontList[f]);
                valid = (front->gpuF == queue[t].F && front->printMe);
            }
            if(!valid) continue;

            printf("Task %ld: \"%s\" JT0 %d JT1 %d JT2 %d JT3 %d\n",
              t, TaskNames[queue[t].Type],
              queue[t].extra[4], queue[t].extra[5], queue[t].extra[6],
              queue[t].extra[7]
            );
            printf("      : dim %d %d\n", queue[t].fm, queue[t].fn);
            printf("      : rowTiles %d %d %d %d\n",
              queue[t].extra[0], queue[t].extra[1], queue[t].extra[2],
              queue[t].extra[3]
            );
            printf("      : VT[0] %ld\n", queue[t].AuxAddress[0]);
            printf("      : VT[1] %ld\n", queue[t].AuxAddress[1]);
            printf("      : AA[2] %ld\n", queue[t].AuxAddress[2]);
            printf("      : AA[3] %ld\n", queue[t].AuxAddress[3]);
        }
#undef MORE_DETAIL
    }
#endif
    else
    {
    //    printf("  R factors are in flight from GPU.\n");
    }
#endif
}

// -----------------------------------------------------------------------------
// Scheduler::fillTasks
// -----------------------------------------------------------------------------

void Scheduler::fillTasks
(
    Int f,                      // INPUT: Current front
    TaskDescriptor *queue,      // INPUT: CPU Task entries
    Int *queueIndex             // IN/OUT: The index of the current entry
)
{
    Front *front = (&frontList[f]);
    SparseMeta *sparseMeta = &(front->sparseMeta);
    bool isDense = front->isDense();

    /* Maintain state transitions through the FSM. */
    FrontState state = front->state;
    FrontState nextState = state;

    /* Copy-in the currentIndex. */
    Int qindex = *queueIndex;

    /* Begin the finite state machine switch statement: */
    switch(state)
    {
        /* There's nothing to do if you're waiting to be allocated. */
        case ALLOCATE_WAIT:
            break;

        /* ASSEMBLE_S assumes that Scount > 0.
         * For fronts with no rows of S, the allocateFront member function
         * bypasses this state by setting the state to CHILD_WAIT. */
        case ASSEMBLE_S:
        {
            /* If we got here, we MUST be sparse. */
            // assert(!isDense);

            /* If we don't have any S entries, advance to the next state. */
            if(sparseMeta->Scount == 0)
            {
                nextState = CHILD_WAIT;
                break;
            }

            /* Unpack S assembly members. */
            int lastSIndex = sparseMeta->lastSIndex;

            /* Determine the iteration bounds for this task. */
            int movesPerThread = 4;
            int threadCount = NUMTHREADS;
            int Scount = sparseMeta->Scount;

            int pstart = lastSIndex;
            int pend = MIN(pstart + movesPerThread*threadCount, Scount);
            while(pstart != pend)
            {
                /* Build the S Assembly task. */
                queue[qindex++] = buildSAssemblyTask(front, pstart, pend);

                /* Save-through the lastSIndex then update pend */
                pstart = sparseMeta->lastSIndex = pend;
                pend = MIN(pstart + movesPerThread*threadCount, Scount);

                // If we just build the last task we can do in this run, break.
                if(qindex == maxQueueSize) break;
            }

            // If we finished building all of the S Assembly tasks, move to
            // child wait.
            if(pstart == pend) nextState = CHILD_WAIT;

            break;
        }

        // We cannot begin the factorization until all the children are pushed
        // into the current front.  Postprocessing handles this.
        case CHILD_WAIT:
            // assert(!isDense);
            break;

        case FACTORIZE:
        {
            /* If we have to schedule the fronts via the scheduler: */
            BucketList *Buckets = (&bucketLists[f]);
            if(Buckets->useFlag)
            {
                /* Only invoke the bucket scheduler if we have enough space in
                   the queue for the tasks it may spawn. The number of tasks
                   that might be spawned by a FillWorkQueue is bounded by the
                   number of rowTiles * (colTiles / ApplyGranularity) */
                Int numRowTiles = Buckets->numRowTiles;
                Int numColTiles = Buckets->numColTiles;
                Int applyGranularity = Buckets->ApplyGranularity;
                Int maxNumTasks = numRowTiles * numColTiles / applyGranularity;
                if(maxNumTasks < (maxQueueSize - qindex))
                {
                    /* Advance, Grow, and Create fresh bundles. */
                    Buckets->AdvanceBundles();
                    #ifdef GPUQRENGINE_PIPELINING
                    Buckets->GrowBundles();
                    #endif
                    Buckets->CreateBundles();
                    Buckets->FillWorkQueue(queue, &qindex);
//                  #ifdef GPUQRENGINE_RENDER
//                  // for development, debuging, to visualize the buckets
//                  GPUQREngine_RenderBuckets(Buckets);
//                  #endif
                    Buckets->PostProcess();
                }

                // If this was the last batch of factorize tasks,
                // remark that we're done.
                if(Buckets->IsDone()) nextState = FACTORIZE_COMPLETE;
            }
            /* Else this is a front that can be handled by SmallQR: */
            else
            {
                queue[qindex++] = buildSmallQRTask(front);
                nextState = FACTORIZE_COMPLETE;
            }

            break;
        }

        // Postprocessing records the eventFrontDataReady event and advances to
        // PARENT_WAIT.
        case FACTORIZE_COMPLETE:
            break;

        /* Wait for the parent to be allocated.
         * Postprocessing will advance this to PUSH_ASSEMBLE. */
        case PARENT_WAIT: break;

        case PUSH_ASSEMBLE:
        {
            bool iterDone = false;
            while(!iterDone)
            {
                /* Compute the iteration bounds for the task. */
                Int cm = sparseMeta->cm;
                Int cn = sparseMeta->cn;
                int cistart = sparseMeta->lastCiStart;
                int cjstart = sparseMeta->lastCjStart;
                int ciend = MIN(cistart+PACKASSEMBLY_SHMEM_MAPINTS, cm);
                int cjend = MIN(cjstart+PACKASSEMBLY_SHMEM_MAPINTS, cn);

                /* Build the pack assembly task */
                sparseMeta->gpuP = (&frontList[front->pids])->gpuF;
                queue[qindex++] = buildPackAssemblyTask(front, cistart,
                    ciend, cjstart, cjend);

                /* Encode the iteration pattern of (left-right, top-bottom). */
                bool endOfRow = (cjend == cn);
                bool moreRows = (ciend != cm);
                iterDone = (endOfRow && !moreRows);
                if(!endOfRow)
                {
                    // This is a rare occurence.  It is triggered by only a
                    // handful of matrices.  In particular, SPQR/Tcov triggers
                    // it with the SPQR/Matrix/Groebner_id2003_aug.mtx, when
                    // using METIS with the SPQR/Demo/qrdemo_gpu program.
                    cjstart = cjend;
                    // cistart = cistart; // no change to row
                }
                else if(endOfRow && moreRows)
                {
                    // This is also rare.  It is trigged by the Franz6 matrix,
                    // augmented by identity, when using METIS with the
                    // SPQR/Demo/qrdemo_gpu program.
                    cjstart = 0;
                    cistart = ciend;
                }
                else
                {
                    // assert(iterDone);
                    // assert(cjend == cn);
                    // assert(ciend == cm);
                }

                /* Save factorization state. */
                sparseMeta->lastCiStart = cistart;
                sparseMeta->lastCjStart = cjstart;

                // If we just built the last task we can do in this run, break.
                if(qindex == maxQueueSize) break;
            }

            // If we've built all of the packAssembly tasks, advance to CLEANUP.
            if(iterDone) nextState = CLEANUP;
            break;
        }

        /* Post-processing advances CLEANUP to DONE. */
        case CLEANUP: break;

        /* At this point, the data for the front can be freed.
         * When all fronts are freed, the factorization is complete. */
        case DONE:
            break;
    }

// #ifdef GPUQRENGINE_RENDER
#if 0
    if (f == 40) // (front->printMe)
    {
        printf("[FillWorkQueue] %g : %d -> %d\n", (double) f,
            state, nextState) ;
            // StateNames[state], StateNames[nextState]);
        debugDumpFront(front);
    }
#endif

    /* Save the factorization state. */
    front->state = nextState;

    /* Copy-out the indexes. */
    *queueIndex = qindex;
}

// -----------------------------------------------------------------------------
// buildSAssemblyTask
// -----------------------------------------------------------------------------

TaskDescriptor buildSAssemblyTask
(
    Front *front,
    int pstart,
    int pend
)
{
    SparseMeta *meta = &(front->sparseMeta);

    TaskDescriptor returner;
    returner.Type = TASKTYPE_SAssembly;
    returner.F = front->gpuF;
    returner.AuxAddress[0] = (double*) meta->gpuS;
    returner.fm = front->fm;
    returner.fn = front->fn;
    returner.extra[0] = meta->Scount;
    returner.extra[1] = pstart;
    returner.extra[2] = pend;
    return returner;
}

// -----------------------------------------------------------------------------
// buildPackAssemblyTask
// -----------------------------------------------------------------------------

TaskDescriptor buildPackAssemblyTask
(
    Front *front,
    int cistart,
    int ciend,
    int cjstart,
    int cjend
)
{
    SparseMeta *meta = &(front->sparseMeta);

    TaskDescriptor returner;
    returner.Type = TASKTYPE_PackAssembly;
    returner.fm = front->fm;
    returner.fn = front->fn;
    returner.F = front->gpuF;
    returner.AuxAddress[0] = meta->gpuC;
    returner.AuxAddress[1] = meta->gpuP;
    returner.AuxAddress[2] = (double*) meta->gpuRjmap;
    returner.AuxAddress[3] = (double*) meta->gpuRimap;
    returner.extra[0] = meta->pn;
    returner.extra[1] = meta->cm;
    returner.extra[2] = meta->cn;
    returner.extra[3] = (ciend-cistart) * (cjend-cjstart);
        // cTileSize, at most 1024x1024
    returner.extra[4] = cistart;
    returner.extra[5] = ciend;
    returner.extra[6] = cjstart;
    returner.extra[7] = cjend;
    return returner;
}

// -----------------------------------------------------------------------------
// buildSmallQRTask
// -----------------------------------------------------------------------------

TaskDescriptor buildSmallQRTask
(
    Front *front
)
{
    TaskDescriptor returner;
    returner.Type = TASKTYPE_FactorizeVT_3x1w;
    returner.F = front->gpuF;
    returner.fm = (int) front->fm;
    returner.fn = (int) front->fn;
    return returner;
}
