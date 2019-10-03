// =============================================================================
// === GPUQREngine/Source/LLBundle_GPUPack.cpp =================================
// =============================================================================
//
// The GPUPack code converts the logical CPU representation of the Bundle and
// its state into a packet of metadata information that the GPU can act on.
//
// =============================================================================

#include "GPUQREngine_BucketList.hpp"


void LLBundle::gpuPack
(
    TaskDescriptor* cpuTask
)
{
    Int *next = Buckets->next;
    int i = 0;

    int delta = EMPTY;
    int secondMin = EMPTY;

    /* Pack the standard rowTiles... */
    int rowTile = (int) (Shadow != EMPTY ? Shadow : First);
    while(rowTile != EMPTY)
    {
        cpuTask->extra[i] = TILESIZE * rowTile;
        if(rowTile == SecondMin) secondMin = i;
        i++;

        rowTile = (int) (rowTile == Shadow ? First : next[rowTile]);
    }
    /* Pack the delta rowTiles... */
    rowTile = (int) Delta;
    delta = i;
    while(rowTile != EMPTY)
    {
        // NOTE: only used when GPUQRENGINE_PIPELINING #define'd
        cpuTask->extra[i] = TILESIZE * rowTile;
        if(rowTile == SecondMin) secondMin = i;
        i++;

        rowTile = (int) next[rowTile];
    }
    /* Clear the remaining rowtiles. */
    for( ; i<PANELSIZE+1; i++) cpuTask->extra[i] = EMPTY;

    /* Transfer ApplyFactorize members */
    cpuTask->extra[8] = delta;
    cpuTask->extra[9] = secondMin;

    /* Transfer VT assignments. */
    cpuTask->AuxAddress[0] = VT[0];
    cpuTask->AuxAddress[1] = VT[1];
}
