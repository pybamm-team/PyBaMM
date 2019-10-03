// =============================================================================
// === GPUQREngine/Source/LLBundle_Advance.cpp =================================
// =============================================================================
//
// The LLBundle Advance member function advances the bundle, leaving behind the
// min tile (which becomes the bundle's Shadow). On advance, the bundle is
// scheduled for at least an Apply (it may be upgraded to ApplyFactorize if
// we're using pipelining).
//
// Advance returns a flag indicating whether or not the bundle evaporated as
// it advanced. This may happen if the bundle advances past the # of pivotal
// columns in the front (tall & skinny case) or if the bundle only had one
// front in it to begin with.
//
// =============================================================================

#include "GPUQREngine_BucketList.hpp"


bool LLBundle::Advance
(
    void
)
{
    Int *prev = Buckets->prev;
    Int *next = Buckets->next;

    bool stillAround;

    /* If the task was an apply, evaporate the bundle and
     * put everyone back into the native bucket. */
    if(CurrentTask == TASKTYPE_GenericApply)
    {
        bool triu = false;
        stillAround = false;
        int tile = First;
        First = EMPTY;
        while(tile != EMPTY)
        {
            int nextTile = next[tile];
            next[tile] = EMPTY;
            prev[tile] = EMPTY;
            Buckets->Insert(tile, NativeBucket, triu);
            tile = nextTile;
            Count--;
        }
    }
    /* Else if the task was a factorize or an apply-factorize: */
    else
    {
        bool triu = true;

        /* The bundle evaporated if we don't have a first tile.. */
        stillAround = (First != EMPTY);
        if(stillAround)
        {
            /* Put the leading tile back into its native bucket. */
            int tile = First;
            Shadow = tile;
            First = next[tile];
            Buckets->Insert(tile, NativeBucket, triu);
            Count--;

            /* By definition, the First doesn't have a previous element. */
            if(First != EMPTY) prev[First] = EMPTY;

            /* Advance the HouseholderBundle to the next colBucket. */
            NativeBucket++;

            /* See if we dropped off the ends of the earth. */
            stillAround = (NativeBucket < Buckets->numBuckets);

            /* Schedule the apply. */
            CurrentTask = TASKTYPE_GenericApply;
            ApplyCount = Count + 1; // The apply count considers the shadow.
        }
    }

    return stillAround;
}
