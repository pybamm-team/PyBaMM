// =============================================================================
// === GPUQREngine/Source/Scheduler_Front.cpp ==================================
// =============================================================================
//
// This file contains code to manage fronts within the scheduler.
//
// The following functions are implemented:
//
//  - activateFront
//    This function adds the front to the permutation of active fronts,
//    configures the inverse permutation for O(1) lookups, and sets the
//    initial factorization state of the front.
//
//  - pullFrontData
//    This function coordinates the asynchronous pull of the R factor off of
//    the GPU as soon as it is available. This function uses the cuda events
//    and streams model
//
//  - finishFront
//    This function is the inverse of activateFront. It removes the front from
//    the list of active fronts. The call is idempotent and coordinates with
//    the cuda events and streams responsible for pulling the R factor in order
//    to not accidentally free a front whose R factor is still in transit.
//
// =============================================================================

#include "GPUQREngine_Scheduler.hpp"


// -----------------------------------------------------------------------------
// Scheduler::activateFront
// -----------------------------------------------------------------------------

void Scheduler::activateFront
(
    Int f                                          // The front id to manipulate
)
{
    /* If the front has already been activated, exit early. */
    if(afPinv[f] != EMPTY) return;

    Front *front = (&frontList[f]);

    /* Add this front to the list of active fronts. */
    afPerm[numActiveFronts] = f;
    afPinv[f] = numActiveFronts;
    numActiveFronts++;

    /* If the front is dense then there are no rows of S to assemble. */
    if(front->isDense())
    {
        front->state = FACTORIZE ;
    }
    /* Else the front is sparse: */
    else
    {
        /* If we're only doing a push assembly, jump to parent wait. */
        if(front->sparseMeta.pushOnly)
        {
            front->state = PARENT_WAIT;
        }
        /* Else we are doing a full factorization of this front; assemble S. */
        else
        {
            front->state = ASSEMBLE_S;
        }
    }
}

// -----------------------------------------------------------------------------
// Scheduler::pullFrontData
// -----------------------------------------------------------------------------

bool Scheduler::pullFrontData
(
    Int f                                          // The front id to manipulate
)
{
    /* Grab the front descriptor. */
    Front *front = (&frontList[f]);

    /* If we're only doing a push assembly then there's nothing to pull. */
    if(front->isPushOnly()) return true;

    /* If we already pulled the R factor, return early. */
    if(FrontDataPulled[f]) return true;

    /* If the R factor isn't actually ready yet, return false.
     * This can happen if the kernel responsible for finishing the factorization
     * is running while we're trying to execute this subroutine. */
    // assert(eventFrontDataReady[f] != NULL);
    if(cudaEventQuery(eventFrontDataReady[f]) != cudaSuccess){ return false; }
    cudaEventDestroy(eventFrontDataReady[f]);

    /* Use an event to signal when the R factor is off the GPU. */
    cudaEventCreate(&eventFrontDataPulled[f]);

    /* Determine how many values to pull back from the GPU: */

    /* We always pull R. */
    Int numValuesToPull = front->getNumRValues();

    /* If we're doing a sparse factorization and this front is staged,
       we also need to pull the contribution block rows. */
    if(front->isStaged())
    {
        SparseMeta *meta = &(front->sparseMeta);
        numValuesToPull += meta->cm * front->fn;
    }

    /* Surgically transfer the data across the D2H stream. */
    Workspace wsR = Workspace(numValuesToPull, sizeof(double));
    wsR.assign(front->cpuR, front->gpuF);
    wsR.transfer(cudaMemcpyDeviceToHost, false, memoryStreamD2H);
    wsR.assign(NULL, NULL);

    /* Record the event to signal when R is off the GPU. */
    cudaEventRecord(eventFrontDataPulled[f]);

    /* Save and return that we've initiated the R factor pull. */
    return (FrontDataPulled[f] = true);
}

// -----------------------------------------------------------------------------
// Scheduler::finishFront
// -----------------------------------------------------------------------------

bool Scheduler::finishFront
(
    Int f                                          // The front id to manipulate
)
{
    /* If we've already freed the front, return early. */
    if(afPinv[f] == EMPTY) return true;

    Front *front = (&frontList[f]);

    /* If we're doing more than a push, we need to get the data off the GPU. */
    if(!front->isPushOnly())
    {
        /* Non-blocking guard to make sure front data is off the GPU. */
        if(cudaEventQuery(eventFrontDataPulled[f]) != cudaSuccess)
        {
            return false;
        }
        cudaEventDestroy(eventFrontDataPulled[f]);
    }

    /* Remove the front from the active fronts. */
    numActiveFronts--;
    if(numActiveFronts > 0)
    {
        /* Replace the active front slot with the last front in the list. */
        Int replacer = afPerm[numActiveFronts];
        Int position = afPinv[f];
        afPerm[position] = replacer;
        afPinv[replacer] = position;
    }
    afPinv[f] = EMPTY;

    /* If we got through this method, we have successfully freed the front. */
    return true;
}

// -----------------------------------------------------------------------------
// debugDumpFront
// -----------------------------------------------------------------------------

#if 1
void Scheduler::debugDumpFront(Front *front)
{
    Workspace *wsFront =
        Workspace::allocate (front->getNumFrontValues(),     // CPU, DEBUG ONLY
        sizeof(double), false, true, false, false);
    double *F = CPU_REFERENCE(wsFront, double*);
    Int fm = front->fm;
    Int fn = front->fn;
    wsFront->assign(wsFront->cpu(), front->gpuF);
    wsFront->transfer(cudaMemcpyDeviceToHost);
    printf("--- %g ---\n", (double) (front->fidg));

//  for(Int i=0; i<fm; i++)
//  {
//      for(Int j=0; j<fn; j++)
//      {
//          printf("%16.8e ", F[i*fn+j]);
//      }
//      printf("\n");
//  }

    for (Int j = 0 ; j < fn ; j++)
    {
        printf ("   --- column %ld of %ld\n", j, fn) ;
        for (Int i = 0 ; i < fm ; i++)
        {
            if (i == j) printf ("      [ diag:     ") ;
            else        printf ("      row %4ld    ", i) ;
            printf (" %10.4g", F [fn*i+j]) ;
            if (i == j) printf (" ]\n") ;
            else        printf ("\n") ;
        }
        printf ("\n") ;
    }

    printf("----------\n", front->fidg);
    wsFront->assign(wsFront->cpu(), NULL);
    wsFront = Workspace::destroy(wsFront);
}
#endif
