/* ========================================================================== */
/* === Source/Mongoose_Refinement.cpp ======================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_Refinement.hpp"
#include "Mongoose_BoundaryHeap.hpp"
#include "Mongoose_ImproveFM.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Logger.hpp"

namespace Mongoose
{

EdgeCutProblem *refine(EdgeCutProblem *graph, const EdgeCut_Options *options)
{
    Logger::tic(RefinementTiming);

    EdgeCutProblem *P             = graph->parent;
    Int cn               = graph->n;
    bool *cPartition     = graph->partition;
    double *fGains       = P->vertexGains;
    Int *fExternalDegree = P->externalDegree;

    /* Transfer cut costs and partition details upwards. */
    P->heuCost   = graph->heuCost;
    P->cutCost   = graph->cutCost;
    P->W0        = graph->W0;
    P->W1        = graph->W1;
    P->imbalance = graph->imbalance;

    /* For each vertex in the coarse graph. */
    for (Int k = 0; k < cn; k++)
    {
        /* Load up the inverse matching */
        Int v[3] = { -1, -1, -1 };
        v[0]     = P->invmatchmap[k];
        v[1]     = P->getMatch(v[0]);
        if (v[0] == v[1])
        {
            v[1] = -1;
        }
        else
        {
            v[2] = P->getMatch(v[1]);
            if (v[0] == v[2])
            {
                v[2] = -1;
            }
        }
        /* Transfer the partition choices to the fine level. */
        bool cp = cPartition[k];
        for (Int i = 0; i < 3 && v[i] != -1; i++)
        {
            Int vertex           = v[i];
            P->partition[vertex] = cp;
        }
    }
    /* See if we can relax the boundary constraint and recompute gains for
     * vertices on the boundary.
     * NOTE: For this, we only need to go through the set of vertices that
     * were on the boundary in the coarse representation. */
    for (Int h = 0; h < 2; h++)
    {
        /* Get the appropriate heap's data. */
        Int *heap = graph->bhHeap[h];
        Int size  = graph->bhSize[h];

        /* Go through all the boundary vertices. */
        for (Int hpos = 0; hpos < size; hpos++)
        {
            /* Get the coarse vertex from the heap. */
            Int k = heap[hpos];

            /* Load up the inverse matching */
            Int v[3] = { -1, -1, -1 };
            v[0]     = P->invmatchmap[k];
            v[1]     = P->getMatch(v[0]);
            if (v[0] == v[1])
            {
                v[1] = -1;
            }
            else
            {
                v[2] = P->getMatch(v[1]);
                if (v[0] == v[2])
                {
                    v[2] = -1;
                }
            }

            /* Relax the boundary constraint. */
            for (Int i = 0; i < 3 && v[i] != -1; i++)
            {
                Int vertex = v[i];

                double gain;
                Int externalDegree;
                calculateGain(P, options, vertex, &gain, &externalDegree);

                /* Only add relevant vertices to the boundary heap. */
                if (externalDegree > 0)
                {
                    fExternalDegree[vertex] = externalDegree;
                    fGains[vertex]          = gain;
                    bhInsert(P, vertex);
                }
            }
        }
    }

    /* Now that we're done with the coarse graph, we can release it. */
    graph->~EdgeCutProblem();

    Logger::toc(RefinementTiming);

    return P;
}

} // end namespace Mongoose
