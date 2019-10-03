/* ========================================================================== */
/* === Source/Mongoose_ImproveFM.cpp ======================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_ImproveFM.hpp"
#include "Mongoose_BoundaryHeap.hpp"
#include "Mongoose_Debug.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Logger.hpp"

namespace Mongoose
{

//-----------------------------------------------------------------------------
// Wrapper for Fidducia-Mattheyes cut improvement.
//-----------------------------------------------------------------------------
void improveCutUsingFM(EdgeCutProblem *graph, const EdgeCut_Options *options)
{
    Logger::tic(FMTiming);

    if (!options->use_FM)
        return;

    double heuCost = INFINITY;
    for (Int i = 0;
         i < options->FM_max_num_refinements && graph->heuCost < heuCost; i++)
    {
        heuCost = graph->heuCost;
        fmRefine_worker(graph, options);
    }

    Logger::toc(FMTiming);
}

//-----------------------------------------------------------------------------
// Make a number of partition moves while considering the impact on problem
// balance.
//-----------------------------------------------------------------------------
void fmRefine_worker(EdgeCutProblem *graph, const EdgeCut_Options *options)
{
    double *Gw          = graph->w;
    double W            = graph->W;
    Int **bhHeap        = graph->bhHeap;
    Int *bhSize         = graph->bhSize;
    Int *externalDegree = graph->externalDegree;
    double *gains       = graph->vertexGains;
    bool *partition     = graph->partition;

    /* Keep a stack of moved vertices. */
    Int *stack = graph->matchmap;
    Int head = 0, tail = 0;

    /* create & initialize a working cost and a best cost. */
    struct CutCost workingCost, bestCost;
    workingCost.heuCost = bestCost.heuCost = graph->heuCost;
    workingCost.cutCost = bestCost.cutCost = graph->cutCost;
    workingCost.W[0] = bestCost.W[0] = graph->W0;
    workingCost.W[1] = bestCost.W[1] = graph->W1;
    workingCost.imbalance = bestCost.imbalance = graph->imbalance;

    /* Tolerance and the linear penalty to assess. */
    double tol = options->soft_split_tolerance;
    double H   = graph->H;

    Int fmSearchDepth   = options->FM_search_depth;
    Int fmConsiderCount = options->FM_consider_count;
    Int i               = 0;
    bool productive     = true;
    for (; i < fmSearchDepth && productive; i++)
    {
        productive = false;

        /* Look for the best vertex to swap: */
        struct SwapCandidate bestCandidate;
        for (Int h = 0; h < 2; h++)
        {
            Int *heap = bhHeap[h];
            Int size  = bhSize[h];
            for (Int c = 0; c < fmConsiderCount && c < size; c++)
            {
                /* Read the vertex, and if it's marked, try the next one. */
                Int v = heap[c];
                if (graph->isMarked(v))
                {
                    continue;
                }

                /* Read the gain for the vertex. */
                double gain = gains[v];

                /* The balance penalty is the penalty to assess for the move. */
                double vertexWeight = (Gw) ? Gw[v] : 1;
                double imbalance    = workingCost.imbalance
                                   + (h ? -1.0 : 1.0) * (vertexWeight / W);
                double absImbalance = fabs(imbalance);
                double imbalanceDelta
                    = absImbalance - fabs(workingCost.imbalance);

                /* If the move hurts the balance past tol, add a penalty. */
                double balPenalty = 0.0;
                if (imbalanceDelta > 0 && absImbalance > tol)
                {
                    balPenalty = absImbalance * H;
                }

                /* Heuristic cost is the cut cost reduced by the gain for making
                 * this move. The gain for the move is amplified by any impact
                 * to the balance penalty. */
                double heuCost = workingCost.cutCost - (gain - balPenalty);

                /* If our heuristic value is better than the running one: */
                if (heuCost < bestCandidate.heuCost)
                {
                    bestCandidate.vertex       = v;
                    bestCandidate.partition    = static_cast<bool>(h);
                    bestCandidate.vertexWeight = vertexWeight;
                    bestCandidate.gain         = gain;
                    bestCandidate.bhPosition   = c;
                    bestCandidate.imbalance    = imbalance;
                    bestCandidate.heuCost      = heuCost;
                }
            }
        }

        /* If we were able to find the best unmoved boundary vertex: */
        if (bestCandidate.heuCost < INFINITY)
        {
            productive = true;
            graph->mark(bestCandidate.vertex);

            /* Move the vertex from the boundary into the move set. */
            bhRemove(graph, options, bestCandidate.vertex, bestCandidate.gain,
                     bestCandidate.partition, bestCandidate.bhPosition);
            stack[tail++] = bestCandidate.vertex;

            /* Swap & update the vertex and its neighbors afterwards. */
            fmSwap(graph, options, bestCandidate.vertex, bestCandidate.gain,
                   bestCandidate.partition);

            /* Update the cut cost. */
            workingCost.cutCost -= 2.0 * bestCandidate.gain;
            workingCost.W[bestCandidate.partition]
                -= bestCandidate.vertexWeight;
            workingCost.W[!bestCandidate.partition]
                += bestCandidate.vertexWeight;
            workingCost.imbalance = bestCandidate.imbalance;
            double absImbalance   = fabs(bestCandidate.imbalance);
            workingCost.heuCost
                = workingCost.cutCost
                  + (absImbalance > tol ? absImbalance * H : 0.0);

            /* Commit the cut if it's better. */
            if (workingCost.heuCost < bestCost.heuCost)
            {
                bestCost = workingCost;
                head     = tail;
                i        = 0;
            }
        }
    }

    /* We've exhausted our search space, so undo all suboptimal moves. */
    for (Int u = tail - 1; u >= head; u--)
    {
        Int vertex           = stack[u];
        Int bhVertexPosition = graph->BH_getIndex(vertex);

        /* Unmark this vertex. */
        graph->unmark(vertex);

        /* It is possible, although rare, that a vertex may have gone
         * from not in the boundary to an undo state that places it in
         * the boundary. It is also possible that a previous swap added
         * this vertex to the boundary already. */
        if (bhVertexPosition != -1)
        {
            bhRemove(graph, options, vertex, gains[vertex], partition[vertex],
                     bhVertexPosition);
        }

        /* Swap the partition and compute the impact on neighbors. */
        fmSwap(graph, options, vertex, gains[vertex], partition[vertex]);
        if (externalDegree[vertex] > 0)
            bhInsert(graph, vertex);
    }

    // clear the marks from all the vertices
    graph->clearMarkArray();

    /* Re-add any vertices that were moved that are still on the boundary. */
    for (Int k = 0; k < head; k++)
    {
        Int vertex = stack[k];
        if (externalDegree[vertex] > 0 && !graph->BH_inBoundary(vertex))
        {
            bhInsert(graph, vertex);
        }
    }

    // clear the marks from all the vertices
    graph->clearMarkArray();

    /* Save the best cost back into the graph. */
    graph->heuCost   = bestCost.heuCost;
    graph->cutCost   = bestCost.cutCost;
    graph->W0        = bestCost.W[0];
    graph->W1        = bestCost.W[1];
    graph->imbalance = bestCost.imbalance;
}

//-----------------------------------------------------------------------------
// This function swaps the partition of a vertex
//-----------------------------------------------------------------------------
void fmSwap(EdgeCutProblem *graph, const EdgeCut_Options *options, Int vertex, double gain,
            bool oldPartition)
{
    Int *Gp             = graph->p;
    Int *Gi             = graph->i;
    double *Gx          = graph->x;
    bool *partition     = graph->partition;
    double *gains       = graph->vertexGains;
    Int *externalDegree = graph->externalDegree;
    Int **bhHeap        = graph->bhHeap;
    Int *bhSize         = graph->bhSize;

    /* Swap partitions */
    bool newPartition = !oldPartition;
    partition[vertex] = newPartition;
    gains[vertex]     = -gain;

    /* Update neighbors. */
    Int exD = 0;
    for (Int p = Gp[vertex]; p < Gp[vertex + 1]; p++)
    {
        Int neighbor           = Gi[p];
        bool neighborPartition = partition[neighbor];
        bool sameSide          = (newPartition == neighborPartition);

        /* Update the bestCandidate vertex's external degree. */
        if (!sameSide)
            exD++;

        /* Update the neighbor's gain. */
        double edgeWeight   = (Gx) ? Gx[p] : 1;
        double neighborGain = gains[neighbor];
        neighborGain += 2 * (sameSide ? -edgeWeight : edgeWeight);
        gains[neighbor] = neighborGain;

        /* Update the neighbor's external degree. */
        Int neighborExD = externalDegree[neighbor];
        neighborExD += (sameSide ? -1 : 1);
        externalDegree[neighbor] = neighborExD;
        Int position             = graph->BH_getIndex(neighbor);

        /* If the neighbor was in a heap: */
        if (position != -1)
        {
            /* If it had its externalDegree reduced to 0, remove it from the
             * heap. */
            if (neighborExD == 0)
            {
                bhRemove(graph, options, neighbor, neighborGain,
                         neighborPartition, position);
            }
            /* If the neighbor is in the heap, we touched its gain
             * so make sure the heap property is satisfied. */
            else
            {
                Int v = neighbor;
                heapifyUp(graph, bhHeap[neighborPartition], gains, v, position,
                          neighborGain);
                v = bhHeap[neighborPartition][position];
                heapifyDown(graph, bhHeap[neighborPartition],
                            bhSize[neighborPartition], gains, v, position,
                            gains[v]);
            }
        }
        /* Else the neighbor wasn't in the heap so add it. */
        else
        {
            if (!graph->isMarked(neighbor))
            {
                ASSERT(!graph->BH_inBoundary(neighbor));
                bhInsert(graph, neighbor);
            }
        }
    }

    externalDegree[vertex] = exD;
}

//-----------------------------------------------------------------------------
// This function computes the gain of a vertex
//-----------------------------------------------------------------------------
void calculateGain(EdgeCutProblem *graph, const EdgeCut_Options *options, Int vertex,
                   double *out_gain, Int *out_externalDegree)
{
    (void)options; // Unused variable

    Int *Gp         = graph->p;
    Int *Gi         = graph->i;
    double *Gx      = graph->x;
    bool *partition = graph->partition;

    bool vp = partition[vertex];

    double gain        = 0.0;
    Int externalDegree = 0;
    for (Int p = Gp[vertex]; p < Gp[vertex + 1]; p++)
    {
        double ew     = (Gx ? Gx[p] : 1.0);
        bool sameSide = (partition[Gi[p]] == vp);
        gain += (sameSide ? -ew : ew);

        if (!sameSide)
            externalDegree++;
    }

    /* Save outputs */
    *out_gain           = gain;
    *out_externalDegree = externalDegree;
}

} // end namespace Mongoose
