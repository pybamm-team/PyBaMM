/* ========================================================================== */
/* === Include/Mongoose_EdgeCutProblem.hpp ================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/**
 * Graph data structure.
 *
 * Stores graph adjacency and weight information. Also used as a container for
 * storing information about matching, coarsening, and partitioning.
 */

// #pragma once
#ifndef MONGOOSE_EDGECUTPROBLEM_HPP
#define MONGOOSE_EDGECUTPROBLEM_HPP

#include "Mongoose_CSparse.hpp"
#include "Mongoose_Graph.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_EdgeCutOptions.hpp"

namespace Mongoose
{

class EdgeCutProblem
{
public:
    /** Graph Data ***********************************************************/
    Int n;     /** # vertices                      */
    Int nz;    /** # edges                         */
    Int *p;    /** Column pointers                 */
    Int *i;    /** Row indices                     */
    double *x; /** Edge weight                     */
    double *w; /** Node weight                     */
    double X;  /** Sum of edge weights             */
    double W;  /** Sum of vertex weights           */

    double H; /** Heuristic max penalty to assess */
    double worstCaseRatio;

    /** Partition Data *******************************************************/
    bool *partition;     /** T/F denoting partition side     */
    double *vertexGains; /** Gains for each vertex           */
    Int *externalDegree; /** # edges lying across the cut    */
    Int *bhIndex;        /** Index+1 of a vertex in the heap */
    Int *bhHeap[2];      /** Heap data structure organized by
                            boundaryGains descending         */
    Int bhSize[2];       /** Size of the boundary heap       */

    /** Cut Cost Metrics *****************************************************/
    double heuCost;   /** cutCost + balance penalty         */
    double cutCost;   /** Sum of edge weights in cut set    */
    Int cutSize;      /** Number of edges in cut set        */
    double W0;        /** Sum of partition 0 vertex weights */
    double W1;        /** Sum of partition 1 vertex weights */
    double imbalance; /** Degree to which the partitioning
                          is imbalanced, and this is
                          computed as (0.5 - W0/W).         */

    /** Matching Data ********************************************************/
    EdgeCutProblem *parent;    /** Link to the parent graph        */
    Int clevel;       /** Coarsening level for this graph */
    Int cn;           /** # vertices in coarse graph      */
    Int *matching;    /** Linked List of matched vertices */
    Int *matchmap;    /** Map from fine to coarse vertices */
    Int *invmatchmap; /** Map from coarse to fine vertices */
    Int *matchtype;   /** Vertex's match classification
                           0: Orphan
                           1: Standard (random, hem, shem)
                           2: Brotherly
                           3: Community                   */
    Int singleton;

    /* Constructor & Destructor */
    static EdgeCutProblem *create(const Int _n, const Int _nz, Int *_p = NULL,
                                  Int *_i = NULL, double *_x = NULL, double *_w = NULL);
    static EdgeCutProblem *create(const Graph *_graph);
    static EdgeCutProblem *create(EdgeCutProblem *_parent);
    ~EdgeCutProblem();
    void initialize(const EdgeCut_Options *options);

    /** Matching Functions ****************************************************/
    inline bool isMatched(Int vertex)
    {
        return (matching[vertex] > 0);
    }

    inline Int getMatch(Int vertex)
    {
        return (matching[vertex] - 1);
    }

    inline void createMatch(Int vertexA, Int vertexB, MatchType matchType)
    {
        matching[vertexA]  = (vertexB) + 1;
        matching[vertexB]  = (vertexA) + 1;
        invmatchmap[cn]    = vertexA;
        matchtype[vertexA] = matchType;
        matchtype[vertexB] = matchType;
        matchmap[vertexA]  = cn;
        matchmap[vertexB]  = cn;
        cn++;
    }

    inline void createCommunityMatch(Int vertexA, Int vertexB,
                                     MatchType matchType)
    {
        Int vm[4] = { -1, -1, -1, -1 };
        vm[0]     = vertexA;
        vm[1]     = getMatch(vm[0]);
        vm[2]     = getMatch(vm[1]);
        vm[3]     = getMatch(vm[2]);

        bool is3Way = (vm[0] == vm[3]);
        if (is3Way)
        {
            matching[vm[1]] = vertexA + 1;
            createMatch(vm[2], vertexB, matchType);
        }
        else
        {
            matching[vertexB]  = matching[vertexA];
            matching[vertexA]  = vertexB + 1;
            matchmap[vertexB]  = matchmap[vertexA];
            matchtype[vertexB] = matchType;
        }
    }

    /** Boundary Heap Functions ***********************************************/
    inline Int BH_getParent(Int a)
    {
        return ((a - 1) / 2);
    }

    inline Int BH_getLeftChild(Int a)
    {
        return (2 * a + 1);
    }

    inline Int BH_getRightChild(Int a)
    {
        return (2 * a + 2);
    }

    inline bool BH_inBoundary(Int v)
    {
        return (bhIndex[v] > 0);
    }

    inline void BH_putIndex(Int v, Int pos)
    {
        bhIndex[v] = (pos + 1);
    }

    inline Int BH_getIndex(Int v)
    {
        return (bhIndex[v] - 1);
    }

    /** Mark Array Functions **************************************************/
    inline void mark(Int index)
    {
        markArray[index] = markValue;
    }

    inline void unmark(Int index)
    {
        markArray[index] = 0;
    }

    inline bool isMarked(Int index)
    {
        return markArray[index] == markValue;
    }

    inline Int getMarkValue()
    {
        return markValue;
    }

    void clearMarkArray();
    void clearMarkArray(Int incrementBy);

private:
    EdgeCutProblem();

    /** Memory Management Flags ***********************************************/
    bool shallow_p;
    bool shallow_i;
    bool shallow_x;
    bool shallow_w;

    /** Mark Data *************************************************************/
    Int *markArray; /** O(n) mark array                 */
    Int markValue;  /** Mark array can be cleared in O(k)
                        by incrementing markValue.
                        Implicitly, a mark value less than
                        markValue is unmarked.          */
    void resetMarkArray();
    bool initialized; // Used to mark if the graph has been initialized
                      // previously.
};

} // end namespace Mongoose

#endif
