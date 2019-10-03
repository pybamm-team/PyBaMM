// =============================================================================
// === GPUQREngine/Source/LLBundle_AddTiles.cpp ================================
// =============================================================================
//
// This file contains two codes that add a rowtile to either the main bundle
// or to the bundle's delta (if pipelined factorization is enabled).
//
// =============================================================================

#include "GPUQREngine_BucketList.hpp"


#ifdef GPUQRENGINE_PIPELINING

void LLBundle::AddTileToDelta
(
    Int rowTile
)
{
    Count++;

    Int *prev = Buckets->prev;
    Int *next = Buckets->next;

    /* Add the first delta entry. */
    if (Delta == EMPTY)
    {
        Delta = rowTile;
        next[Delta] = EMPTY;
        prev[Delta] = EMPTY;
        SecondMin = MIN(SecondMin, Delta);
        return;
    }

    /* Add additional entries */
    int min = MIN(Delta, rowTile);
    int max = MAX(Delta, rowTile);

    /* If the Delta is still the best: */
    if (Delta == min)
    {
        int fnext = next[Delta];
        if (fnext != EMPTY) prev[fnext] = max;
        next[Delta] = max;
        prev[max] = Delta;
        next[max] = fnext;
    }
    /* Else the added guy is smaller than the first delta entry. */
    else
    {
        next[min] = Delta;
        prev[Delta] = min;
        prev[min] = EMPTY;
        Delta = min;
    }

    /* Keep track of SecondMin. */
    SecondMin = (SecondMin == EMPTY ? Delta : MIN(SecondMin, Delta));
}

#endif

void LLBundle::AddTileToSlots
(
    Int rowTile
)
{
    Count++;

    Int *prev = Buckets->prev;
    Int *next = Buckets->next;

    /* Keep track of the Max. */
    Max = MAX(Max, rowTile);

    /* Add the first entry. */
    if (First == EMPTY)
    {
        First = rowTile;
        Last = rowTile;
        next[First] = EMPTY;
        prev[First] = EMPTY;
        SecondMin = EMPTY;
        return;
    }

    /* Add additional entries */
    int min = MIN(First, rowTile);
    int max = MAX(First, rowTile);

    /* If the first is still the best: */
    if (First == min)
    {
        int fnext = next[First];
        if (fnext != EMPTY) prev[fnext] = max;
        next[First] = max;
        prev[max] = First;
        next[max] = fnext;
    }
    /* Else the added guy is smaller than First. */
    else
    {
        next[min] = First;
        prev[First] = min;
        prev[min] = EMPTY;
        First = min;
    }

    /* Keep track of SecondMin. */
    SecondMin = (SecondMin == EMPTY ? max : MIN(SecondMin, max));

    /* Update last, if needed. */
    if (next[Last] != EMPTY) Last = next[Last];
}
