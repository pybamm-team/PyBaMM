// =============================================================================
// === GPUQREngine/Source/Scheduler_Render.cpp =================================
// =============================================================================
// === This code is used only for internal development and debugging ===========
// =============================================================================
//
// This file contains logic to render the current state of the scheduler.
// It colors each front in the frontal elimination tree according to its
// factorization state:
//
//    ALLOCATE_WAIT:      white
//    ASSEMBLE_S:         lightblue
//    CHILD_WAIT:         orange
//    FACTORIZE:          red
//    FACTORIZE_COMPLETE: maroon
//    PARENT_WAIT:        yellow
//    PUSH_ASSEMBLE:      blue
//    CLEANUP:            green
//    DONE:               green
//
// =============================================================================

#include "GPUQREngine_Internal.hpp"

#ifdef GPUQRENGINE_RENDER

#include "GPUQREngine_Scheduler.hpp"
#include <stdio.h>
#include <string.h>


void Scheduler::render
(
    void
)
{
    // if(!RENDER_SPARSE_FACTORIZATION) return;

    char filename[64];
    sprintf(filename, "state_%d.dot", renderCount++);
    FILE *output = fopen(filename, "w");

    fprintf(output, "digraph G {\n");
    fprintf(output, "rankdir=BT;");
    // fprintf(output, "edge [arrowhead=none];\n");
    // fprintf(output, "node [shape=point,style=filled];\n");
    fprintf(output, "node [shape=record,style=filled];\n");

    Int fnMax = 0;
    Int fmMax = 0;
    for(Int pf=0; pf<numFronts; pf++)
    {
        Front *front = (&frontList[pf]);
        fmMax = MAX(front->fm, fmMax);
        fnMax = MAX(front->fn, fnMax);
    }
    for(Int pf=0; pf<numFronts; pf++)
    {
        Front *front = (&frontList[pf]);
        Int fg = front->fidg;

        char fillcolor[16];
        switch(front->state)
        {
            case ALLOCATE_WAIT:      strcpy(fillcolor,"\"white\""); break;
            case ASSEMBLE_S:         strcpy(fillcolor,"\"lightblue\""); break;
            case CHILD_WAIT:         strcpy(fillcolor,"\"orange\""); break;
            case FACTORIZE:          strcpy(fillcolor,"\"red\""); break;
            case FACTORIZE_COMPLETE: strcpy(fillcolor,"\"maroon\""); break;
            case PARENT_WAIT:        strcpy(fillcolor,"\"yellow\""); break;
            case PUSH_ASSEMBLE:      strcpy(fillcolor,"\"blue\""); break;
            case CLEANUP:            strcpy(fillcolor,"\"green\""); break;
            case DONE:               strcpy(fillcolor,"\"green\""); break;
        }

        double height = MAX(0.10, 2.0 * ((double) front->fm / (double) fmMax));
        double width = MAX(0.10, 2.0 * ((double) front->fn / (double) fnMax));
        fprintf(output, "%ld [fillcolor=%s,width=%f,height=%f];\n",
            fg, fillcolor, width, height);
    }

    for(Int pf=0; pf<numFronts; pf++)
    {
        Front *front = (&frontList[pf]);
        Int fg = front->fidg;
        Int pg = front->pidg;
        if(pg != EMPTY)
        {
            fprintf(output, "%ld->%ld;\n", fg, pg);
        }
    }
    fprintf(output, "};\n");

    fclose(output);
}
#endif
