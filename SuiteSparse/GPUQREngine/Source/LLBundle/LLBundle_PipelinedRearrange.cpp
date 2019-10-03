// =============================================================================
// === GPUQREngine/Source/LLBundle_PipelinedRearrange.cpp ======================
// =============================================================================
//
// PipelinedRearrange reconfigures an LLBundle instance by swapping the
// SecondMin row tile to the top position, overwriting the Shadow. Any remaining
// delta tiles are merged into the bundle, and delta metadata is removed,
// allowing additional deltas to participate in future sweeps through the
// data structure.
//
// =============================================================================

#include "GPUQREngine_BucketList.hpp"


void LLBundle::PipelinedRearrange
(
    void
)
{
    Int *prev = Buckets->prev;
    Int *next = Buckets->next;

    /*** Move the min to the top. ***/
    if (First != SecondMin)
    {
        /* Remove second min */
        int smnext = next[SecondMin];
        int smprev = prev[SecondMin];
        if (smnext != EMPTY) prev[smnext] = smprev;
        if (smprev != EMPTY) next[smprev] = smnext;

        /* Add second min to the front of the list. */
        prev[First] = SecondMin;
        next[SecondMin] = First;
        prev[SecondMin] = EMPTY;
        First = SecondMin;

        /* If second min was the first of delta, update Delta. */
        if (SecondMin == Delta)
        {
            // NOTE: only used when GPUQRENGINE_PIPELINING #define'd
            Delta = smnext;     // PIPELINE
        }
        /* Else If the second min was somewhere in the original list. */
        else if(SecondMin != Last)
        {
            // NOTE: only used when GPUQRENGINE_PIPELINING #define'd

            /* If we have a delta, fill the gap using Delta. */
            if (Delta != EMPTY)
            {
                // Take a snapshot of Delta's state so we can update it later.
                int dnext = next[Delta];

                /* Insert the delta entry */
                if (smprev != EMPTY) next[smprev] = Delta;
                if (smnext != EMPTY) prev[smnext] = Delta;
                prev[Delta] = smprev;
                next[Delta] = smnext;

                /* Update Delta */
                Delta = dnext;
                if (Delta != EMPTY) prev[Delta] = EMPTY;
            }
            // Else if second min wasn't second from the last then we need to
            // actually swap in the last because it isn't automatically ordered
            // correctly.
            else if(smnext != Last)
            {
                /* Take a snapshot of Last's state so we can update it later. */
                int lprev = prev[Last];

                /* Insert the last entry */
                prev[Last] = smprev;
                next[Last] = (Last != smnext ? smnext : EMPTY);
                if (smprev != EMPTY) next[smprev] = Last;
                if (smnext != EMPTY) prev[smnext] = Last;

                /* Update Last */
                Last = lprev;
                next[Last] = EMPTY;
            }
        }
        /* Else SecondMin was the last entry, so update last. */
        else
        {
            Last = smprev;
            if(Last != EMPTY) next[Last] = EMPTY;
        }
    }
    SecondMin = EMPTY;

    // If we still have a delta component, glue it to the end of the list.

    if (Delta != EMPTY)
    {
        // NOTE: only used when GPUQRENGINE_PIPELINING #define'd
        next[Last] = Delta;     // PIPELINE
        prev[Delta] = Last;
        Delta = EMPTY;

        /* Move Last all the way to the last entry in Delta. */
        while (next[Last] != EMPTY) Last = next[Last];
    }
}
