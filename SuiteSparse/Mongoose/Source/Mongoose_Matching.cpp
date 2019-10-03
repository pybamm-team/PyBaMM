/* ========================================================================== */
/* === Source/Mongoose_Matching.cpp ========================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/**
 * For computing vertex matchings
 *
 * During coarsening, a matching of vertices is computed to determine
 * which vertices are combined together into supervertices. This can be done
 * using a number of different strategies, including Heavy Edge Matching and
 * Community/Brotherly (similar to 2-hop) Matching.
 */

#include "Mongoose_Matching.hpp"
#include "Mongoose_Debug.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Logger.hpp"

namespace Mongoose
{

//-----------------------------------------------------------------------------
// top-level matching code that serves as a multiple-dispatch system.
//-----------------------------------------------------------------------------
void match(EdgeCutProblem *graph, const EdgeCut_Options *options)
{
    Logger::tic(MatchingTiming);
    switch (options->matching_strategy)
    {
    case Random:
        matching_Random(graph, options);
        break;

    case HEM:
        matching_HEM(graph, options);
        break;

    case HEMSR:
        matching_HEM(graph, options);
        matching_SR(graph, options);
        break;

    case HEMSRdeg:
        matching_HEM(graph, options);
        matching_SRdeg(graph, options);
        break;
    }
    matching_Cleanup(graph, options);
    Logger::toc(MatchingTiming);
}

//-----------------------------------------------------------------------------
// Cleans up a matching by matching remaining unmatched vertices to themselves
//-----------------------------------------------------------------------------
void matching_Cleanup(EdgeCutProblem *graph, const EdgeCut_Options *options)
{
    (void)options; // Unused variable

    Int n   = graph->n;
    Int *Gp = graph->p;

    /* Match unmatched vertices to themselves. */
    for (Int k = 0; k < n; k++)
    {
        if (!graph->isMatched(k))
        {
            Int degree = Gp[k + 1] - Gp[k];
            if (degree == 0)
            {
                // Singleton!
                if (graph->singleton == -1)
                {
                    graph->singleton = k;
                }
                else
                {
                    graph->createMatch(k, graph->singleton, MatchType_Standard);
                    graph->singleton = -1;
                }
            }
            else
            {
                // Not a singleton
                if (options->do_community_matching)
                {
                    int i;
                    for (i = 0; i < graph->n; i++)
                    {
                        if (graph->matchtype[i] != MatchType_Community)
                            break;
                    }
                    graph->createCommunityMatch(i, k, MatchType_Community);
                }
                else
                {
                    graph->createMatch(k, k, MatchType_Orphan);
                }
            }
        }
    }

    if (graph->singleton != -1)
    {
        // Leftover singleton
        Int k = graph->singleton;
        if (options->do_community_matching)
        {
            int i;
            for (i = 0; i < graph->n; i++)
            {
                if (graph->matchtype[i] != MatchType_Community)
                    break;
            }
            graph->createCommunityMatch(i, k, MatchType_Community);
        }
        else
        {
            graph->createMatch(k, k, MatchType_Orphan);
        }
    }

#ifndef NDEBUG
    /* Every vertex must be matched in no more than a 3-way matching. */
    for (Int k = 0; k < n; k++)
    {
        if (options->do_community_matching)
        {
            if (!graph->isMatched(k))
                PR(("%ld is unmatched\n", k));
            ASSERT(graph->isMatched(k));
        }

        /* Load matching. */
        Int v[3] = { -1, -1, -1 };
        v[0]     = k;
        v[1]     = graph->getMatch(v[0]);

        if (v[1] == v[0])
        {
            v[1] = -1;
        }

        if (v[1] != -1)
        {
            v[2] = graph->getMatch(v[1]);
            if (v[2] == v[0])
                v[2] = -1;
        }

        if (options->do_community_matching)
        {
            if (v[2] != -1)
            {
                ASSERT(graph->getMatch(v[2]) == v[0]);
            }
            else
            {
                ASSERT(graph->getMatch(v[1]) == v[0]);
            }
        }
        else
        {
            if (v[1] != -1)
            {
                ASSERT(graph->getMatch(v[1]) == v[0]);
            }
            else
            {
                ASSERT(graph->getMatch(v[0]) == v[0]);
            }
        }
    }
#endif
}

//-----------------------------------------------------------------------------
// This is a random matching strategy
//-----------------------------------------------------------------------------
void matching_Random(EdgeCutProblem *graph, const EdgeCut_Options *options)
{
    (void)options; // Unused variable

    Int n   = graph->n;
    Int *Gp = graph->p;
    Int *Gi = graph->i;

    for (Int k = 0; k < n; k++)
    {
        /* Consider only unmatched vertices */
        if (graph->isMatched(k))
            continue;

        bool unmatched = true;
        for (Int p = Gp[k]; p < Gp[k + 1] && unmatched; p++)
        {
            Int neighbor = Gi[p];

            /* Consider only unmatched neighbors */
            if (graph->isMatched(neighbor))
                continue;

            unmatched = false;

            graph->createMatch(k, neighbor, MatchType_Standard);
        }
    }

#ifndef NDEBUG
    /* If we want to do expensive checks, make sure that every vertex is either:
     *     1) matched
     *     2) has no unmatched neighbors
     */
    for (Int k = 0; k < n; k++)
    {
        /* Check condition 1 */
        if (graph->matching[k])
            continue;
        /* Check condition 2 */
        for (Int p = Gp[k]; p < Gp[k + 1]; p++)
        {
            ASSERT(graph->matching[Gi[p]]);
        }
    }
#endif
}

//-----------------------------------------------------------------------------
// This is the implementation of stall-reducing matching
//-----------------------------------------------------------------------------
void matching_SR(EdgeCutProblem *graph, const EdgeCut_Options *options)
{
    Int n      = graph->n;
    Int *Gp    = graph->p;
    Int *Gi    = graph->i;
    double *Gx = graph->x;

#ifndef NDEBUG
    /* In order for us to use Passive-Aggressive matching,
     * all unmatched vertices must have matched neighbors. */
    for (Int k = 0; k < n; k++)
    {
        if (graph->isMatched(k))
            continue;
        for (Int p = Gp[k]; p < Gp[k + 1]; p++)
        {
            ASSERT(graph->isMatched(Gi[p]));
        }
    }
#endif

    for (Int k = 0; k < n; k++)
    {
        /* Consider only unmatched vertices */
        if (graph->isMatched(k))
            continue;

        Int heaviestNeighbor  = -1;
        double heaviestWeight = -1.0;

        for (Int p = Gp[k]; p < Gp[k + 1]; p++)
        {
            Int neighbor = Gi[p];

            /* Keep track of the heaviest. */
            double x = (Gx) ? Gx[p] : 1;
            if (x > heaviestWeight)
            {
                heaviestWeight   = x;
                heaviestNeighbor = neighbor;
            }
        }

        /* If we found a heaviest neighbor then begin resolving matches. */
        if (heaviestNeighbor != -1)
        {
            Int v = -1;
            for (Int p = Gp[heaviestNeighbor]; p < Gp[heaviestNeighbor + 1];
                 p++)
            {
                Int neighbor = Gi[p];
                if (graph->isMatched(neighbor))
                    continue;

                if (v == -1)
                {
                    v = neighbor;
                }
                else
                {
                    graph->createMatch(v, neighbor, MatchType_Brotherly);
                    v = -1;
                }
            }

            /* If we had a vertex left over: */
            if (v != -1)
            {
                if (options->do_community_matching)
                {
                    graph->createCommunityMatch(heaviestNeighbor, v,
                                                MatchType_Community);
                }
                else
                {
                    graph->createMatch(v, v, MatchType_Orphan);
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
// This uses the stall-reducing matching where we only try SR matching
// with vertices with degree above a user-defined threshold.
//-----------------------------------------------------------------------------
void matching_SRdeg(EdgeCutProblem *graph, const EdgeCut_Options *options)
{
    Int n   = graph->n;
    Int *Gp = graph->p;
    Int *Gi = graph->i;

    /* The brotherly threshold is the minimum degree a "high degree" vertex.
     * It is the options->degreeThreshold times the average degree. */
    double bt
        = options->high_degree_threshold * ((double)graph->nz / (double)graph->n);

#ifndef NDEBUG
    /* In order for us to use Passive-Aggressive matching,
     * all unmatched vertices must have matched neighbors. */
    for (Int k = 0; k < n; k++)
    {
        if (graph->isMatched(k))
            continue;
        for (Int p = Gp[k]; p < Gp[k + 1]; p++)
        {
            ASSERT(graph->isMatched(Gi[p]));
        }
    }
#endif

    for (Int k = 0; k < n; k++)
    {
        /* Consider only matched vertices */
        if (!graph->isMatched(k))
            continue;

        Int degree = Gp[k + 1] - Gp[k];
        if (degree >= (Int)bt)
        {
            Int v = -1;
            for (Int p = Gp[k]; p < Gp[k + 1]; p++)
            {
                Int neighbor = Gi[p];
                if (graph->isMatched(neighbor))
                    continue;

                if (v == -1)
                {
                    v = neighbor;
                }
                else
                {
                    graph->createMatch(v, neighbor, MatchType_Brotherly);
                    v = -1;
                }
            }

            /* If we had a vertex left over: */
            if (v != -1)
            {
                if (options->do_community_matching)
                {
                    graph->createCommunityMatch(k, v, MatchType_Community);
                }
                else
                {
                    graph->createMatch(v, v, MatchType_Orphan);
                }
            }
        }
    }

    ASSERT(graph->cn < n);
}

//-----------------------------------------------------------------------------
// This is a vanilla implementation of heavy edge matching
//-----------------------------------------------------------------------------
void matching_HEM(EdgeCutProblem *graph, const EdgeCut_Options *options)
{
    (void)options; // Unused variable

    Int n      = graph->n;
    Int *Gp    = graph->p;
    Int *Gi    = graph->i;
    double *Gx = graph->x;

    for (Int k = 0; k < n; k++)
    {
        /* Consider only unmatched vertices */
        if (graph->isMatched(k))
            continue;

        Int heaviestNeighbor  = -1;
        double heaviestWeight = -1.0;
        for (Int p = Gp[k]; p < Gp[k + 1]; p++)
        {
            Int neighbor = Gi[p];

            /* Consider only unmatched neighbors */
            if (graph->isMatched(neighbor))
                continue;

            /* Keep track of the heaviest. */
            double x = (Gx) ? Gx[p] : 1;
            if (x > heaviestWeight)
            {
                heaviestWeight   = x;
                heaviestNeighbor = neighbor;
            }
        }

        /* Match to the heaviest. */
        if (heaviestNeighbor != -1)
        {
            graph->createMatch(k, heaviestNeighbor, MatchType_Standard);
        }
    }

#ifndef NDEBUG
    /* If we want to do expensive checks, make sure that every vertex is either:
     *     1) matched
     *     2) has no unmatched neighbors
     */
    for (Int k = 0; k < n; k++)
    {
        /* Check condition 1 */
        if (graph->matching[k])
            continue;

        /* Check condition 2 */
        for (Int p = Gp[k]; p < Gp[k + 1]; p++)
        {
            ASSERT(graph->matching[Gi[p]]);
        }
    }
#endif
}

} // end namespace Mongoose
