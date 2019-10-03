// =============================================================================
// === GPUQREngine/Source/LLBundle_UpdateSecondMinIndex.cpp ====================
// =============================================================================
//
// This file contains two functions that perform a scan through an LLBundle
// instance to discover the SecondMin rowtile and the Max, respectively.
//
// =============================================================================


#include "GPUQREngine_BucketList.hpp"

// -----------------------------------------------------------------------------
// LLBundle::UpdateSecondMinIndex
// -----------------------------------------------------------------------------

void LLBundle::UpdateSecondMinIndex
(
    void
)
{
    /* If we don't have a First, return early. */
    if(First == EMPTY) return;

    Int *next = Buckets->next;

    /* Scan to find the next second min index. */
    int inspect = SecondMin = next[First];
    while (inspect != EMPTY)
    {
        SecondMin = MIN(SecondMin, inspect);
        inspect = next[inspect];
    }
}

// -----------------------------------------------------------------------------
// LLBundle::UpdateMax
// -----------------------------------------------------------------------------

void LLBundle::UpdateMax
(
    void
)
{
    Int *next = Buckets->next;

    /* Scan to find the max. */
    Max = Shadow;
    for(Int tile=First; tile!=EMPTY; tile=next[tile]) Max = MAX(Max, tile);
}
