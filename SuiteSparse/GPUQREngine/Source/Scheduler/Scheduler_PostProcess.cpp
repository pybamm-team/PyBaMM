// =============================================================================
// === GPUQREngine/Source/Scheduler_PostProcess.cpp ============================
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
//     value is hardcoded to 4). A front remains in ASSEMBLE_S during
//     postprocessing only if the FillWorkQueue code was unable to add all of
//     the relevant tasks to the GPU work list.
//
//   - CHILD_WAIT
//     The postprocessor checks the number of pending children. When all the
//     children have pushed their data into the current front,
//     the postprocessor advances the front into the FACTORIZE state.
//
//   - FACTORIZE
//     Fronts in FACTORIZE are continuing their factorization as usual.
//     It is very possible that factorization has proceeded to the point where
//     the R factor is ready before the front has finished being factorized.
//     When this happens, the postprocessor kicks off an early async R transfer.
//
//   - FACTORIZE_COMPLETE
//     Fronts in FACTORIZE_COMPLETE initiate an async R transfer. The transfer
//     is idempotent, such that there is only one pending transfer request in
//     flight at any time. This allows for an early R pull in FACTORIZE as well
//     as large R factors to take multiple kernel launches before the transfer
//     completes. The asynchronous transfer uses the cuda events and streams
//     model.
//
//   - PARENT_WAIT
//     A front in PARENT_WAIT attempts to continue transfering the R factor
//     from the GPU to the CPU. Additionally, this code activates the parent
//     front, moving the parent from ALLOCATE_WAIT to its initial state.
//
//   - PUSH_ASSEMBLE
//     Fronts in PUSH_ASSEMBLE build PackAssemble tasks that move data from
//     the child's memory space into the parent's memory space on the GPU.
//     A front only stays in PUSH_ASSEMBLE if the FillWorkQueue code couldn't
//     add all of the required PackAssembly tasks to the GPU queue.
//
//   - CLEANUP
//     Fronts in CLEANUP wait for their corresponding R factors to be
//     transfered off of the GPU.
//
//   - DONE
//     Fronts in DONE have no more work nor additional state transitions.
//     When all fronts are in the DONE state then the QREngine's work is done.
//
// =============================================================================

#include "GPUQREngine_Scheduler.hpp"


bool Scheduler::postProcess
(
    void
)
{
    /* Post-process all active fronts. */
    for(Int p=0; p<numActiveFronts; p++)
    {
        /* Get the front from the "active fronts" permutation. */
        Int f = afPerm[p];

        Front *front = (&frontList[f]);
        SparseMeta *meta = &(front->sparseMeta);
        bool isDense = front->isDense();
        bool isSparse = front->isSparse();
        FrontState state = front->state;
        FrontState nextState = state;

        /* The post-processing we do depends on the state: */
        switch(state)
        {
            /* There's nothing to do if you're waiting to be allocated. */
            case ALLOCATE_WAIT:
                break;

            /* The only time we stay in ASSEMBLE_S is if we can't get to
             * adding the task to the work queue in a particular pass.
             * This happens when we have a ton of other work to do. */
            case ASSEMBLE_S: break;

            /* If we're in CHILD_WAIT, see if all of the children are ready. */
            case CHILD_WAIT:
            {
                // assert(isSparse);

                /* If all the children are ready then we can proceed. */
                int nc = meta->nc;
                if(nc == 0)
                {
                    initializeBucketList(f);
                    nextState = FACTORIZE;
                }
                break;
            }

            /* If we're in the middle of a factorization: */
            case FACTORIZE:

//              // IsRReadyEarly experimental feature : pulls R from the GPU
//              // R is computed but the contribution block is not.  This
//              // method is under development and not yet available for
//              // production use.
//              if(isSparse && (&bucketLists[f])->IsRReadyEarly()) {
//                  /* If we haven't created the event yet, create it. */
//                  if(eventFrontDataReady[f] == NULL) {
//                      // Piggyback the synchronization on the next kernel
//                      // launch.
//                      cudaEventCreate(&eventFrontDataReady[f]);
//                      cudaEventRecord(eventFrontDataReady[f],
//                      kernelStreams[activeSet^1]); }
//                  /* We must have created the event on the last kernel
//                     launch so try to pull R off the GPU. */ else {
//                     pullFrontData(f); } }

                break;

            // At this point, the R factor is ready to be pulled from the GPU.
            case FACTORIZE_COMPLETE:
            {
                /* If we haven't created the event yet, create it. */
                if(eventFrontDataReady[f] == NULL)
                {
                    // Piggyback the synchronization on the next kernel launch.
                    cudaEventCreate(&eventFrontDataReady[f]);
                    cudaEventRecord(eventFrontDataReady[f],
                        kernelStreams[activeSet^1]);
                }
                /* We must have created the event already during factorize,
                   so instead try to pull R off the GPU. */
                else
                {
                    pullFrontData(f);
                }

                /* If the front is dense or staged, then we can't assemble
                   into the parent, so just cleanup. */
                if(isDense || meta->isStaged)
                {
                    nextState = CLEANUP;
                }
                /* Else we're sparse and not staged so it means we have memory
                   to assemble into the parent. */
                else
                {
                    nextState = PARENT_WAIT;
                }
                break;
            }

            /* If we're waiting on the parent to be allocated: */
            case PARENT_WAIT:
            {
                // assert(isSparse);

                /* Make sure we're trying to pull the R factor off the GPU. */
                pullFrontData(f);

                // If we have a parent, allocate it and proceed to PUSH_ASSEMBLE
                Int pids = front->pids;
                if(pids != EMPTY)
                {
                    activateFront(pids);
                    nextState = PUSH_ASSEMBLE;
                }
                /* Else the parent is the dummy, so cleanup and move to done. */
                else
                {
                    nextState = CLEANUP;
                }

                break;
            }

            /* The only time we stay in PUSH_ASSEMBLE is if we can't get to
             * adding the task to the work queue in a particular pass.
             * This happens when we have a ton of other work to do. */
            case PUSH_ASSEMBLE:
                // assert(isSparse);
                break;

            /* If we're in CLEANUP then we need to free the front. */
            case CLEANUP:
            {
                /* If we were able to get the R factor and free the front. */
                if(pullFrontData(f) && finishFront(f))
                {
                    /* Update the parent's child count. */
                    Int pid = front->pids;
                    if(pid != EMPTY) (&frontList[pid])->sparseMeta.nc--;

                    /* Move to DONE. */
                    nextState = DONE;

                    /* Keep track of the # completed. */
                    numFrontsCompleted++;

                    /* Revisit the same position again since a front was
                     * swapped to the current location. */
                    p--;
                }
                break;
            }

            /* This is the done state with nothing to do. */
            case DONE:
                break;
        }

#if 0
        if(front->printMe)
        {
            printf("[PostProcessing] %g : %d -> %d\n", (double) (front->fidg),
                state, nextState);
                // StateNames[state], StateNames[nextState]);
            debugDumpFront(front);
        }
#endif

        /* Save the next state back to the frontDescriptor. */
        front->state = nextState;
    }

    // printf("%2.2f completed.\n", 100 * (double) numCompleted / (double)
    // numFronts);

    /* Return whether all the fronts are DONE. */
    return (numFronts == numFrontsCompleted);
}
