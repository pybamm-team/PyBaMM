/* ========================================================================== */
/* === Source/Mongoose_EdgeCutProblem.cpp =================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_EdgeCutProblem.hpp"

#include <algorithm>
#include <new>

namespace Mongoose
{

/* Constructor & Destructor */
EdgeCutProblem::EdgeCutProblem()
{
    n = nz = 0;
    p      = NULL;
    i      = NULL;
    x      = NULL;
    w      = NULL;
    X      = 0.0;
    W      = 0.0;
    H      = 0.0;

    partition      = NULL;
    vertexGains    = NULL;
    externalDegree = NULL;
    bhIndex        = NULL;
    bhHeap[0] = bhHeap[1] = NULL;
    bhSize[0] = bhSize[1] = 0;

    heuCost   = 0.0;
    cutCost   = 0.0;
    W0        = 0.0;
    W1        = 0.0;
    imbalance = 0.0;

    parent      = NULL;
    clevel      = 0;
    cn          = 0;
    matching    = NULL;
    matchmap    = NULL;
    invmatchmap = NULL;
    matchtype   = NULL;

    markArray = NULL;
    markValue = 1;
}

EdgeCutProblem *EdgeCutProblem::create(const Int _n, const Int _nz, Int *_p,
                                       Int *_i, double *_x, double *_w)
{
    void *memoryLocation = SuiteSparse_malloc(1, sizeof(EdgeCutProblem));
    if (!memoryLocation)
        return NULL;

    // Placement new
    EdgeCutProblem *graph = new (memoryLocation) EdgeCutProblem();

    graph->shallow_p = (_p != NULL);
    graph->shallow_i = (_i != NULL);
    graph->shallow_x = (_x != NULL);
    graph->shallow_w = (_w != NULL);

    size_t n = static_cast<size_t>(_n);
    graph->n = _n;

    size_t nz = static_cast<size_t>(_nz);
    graph->nz = _nz;

    graph->p = (graph->shallow_p)
               ? _p
               : (Int *)SuiteSparse_calloc(n + 1, sizeof(Int));
    graph->i
        = (graph->shallow_i) ? _i : (Int *)SuiteSparse_malloc(nz, sizeof(Int));
    graph->x = _x;
    graph->w = _w;
    graph->X = 0.0;
    graph->W = 0.0;
    graph->H = 0.0;
    if (!graph->p || !graph->i)
    {
        graph->~EdgeCutProblem();
        return NULL;
    }

    graph->partition      = (bool *)SuiteSparse_malloc(n, sizeof(bool));
    graph->vertexGains    = (double *)SuiteSparse_malloc(n, sizeof(double));
    graph->externalDegree = (Int *)SuiteSparse_calloc(n, sizeof(Int));
    graph->bhIndex        = (Int *)SuiteSparse_calloc(n, sizeof(Int));
    graph->bhHeap[0]      = (Int *)SuiteSparse_malloc(n, sizeof(Int));
    graph->bhHeap[1]      = (Int *)SuiteSparse_malloc(n, sizeof(Int));
    graph->bhSize[0] = graph->bhSize[1] = 0;
    if (!graph->partition || !graph->vertexGains || !graph->externalDegree
        || !graph->bhIndex || !graph->bhHeap[0] || !graph->bhHeap[1])
    {
        graph->~EdgeCutProblem();
        return NULL;
    }

    graph->heuCost   = 0.0;
    graph->cutCost   = 0.0;
    graph->W0        = 0.0;
    graph->W1        = 0.0;
    graph->imbalance = 0.0;

    graph->parent      = NULL;
    graph->clevel      = 0;
    graph->cn          = 0;
    graph->matching    = (Int *)SuiteSparse_calloc(n, sizeof(Int));
    graph->matchmap    = (Int *)SuiteSparse_malloc(n, sizeof(Int));
    graph->invmatchmap = (Int *)SuiteSparse_malloc(n, sizeof(Int));
    graph->matchtype   = (Int *)SuiteSparse_malloc(n, sizeof(Int));
    graph->markArray   = (Int *)SuiteSparse_calloc(n, sizeof(Int));
    graph->markValue   = 1;
    graph->singleton   = -1;
    if (!graph->matching || !graph->matchmap || !graph->invmatchmap
        || !graph->markArray || !graph->matchtype)
    {
        graph->~EdgeCutProblem();
        return NULL;
    }

    graph->initialized = false;

    return graph;
}

EdgeCutProblem *EdgeCutProblem::create(const Graph *_graph)
{
    EdgeCutProblem *graph = create(_graph->n, _graph->nz, _graph->p, _graph->i,
                                   _graph->x, _graph->w);

    return graph;
}

EdgeCutProblem *EdgeCutProblem::create(EdgeCutProblem *_parent)
{
    EdgeCutProblem *graph = create(_parent->cn, _parent->nz);

    if (!graph)
        return NULL;

    graph->x = (double *)SuiteSparse_malloc(_parent->nz, sizeof(double));
    graph->w = (double *)SuiteSparse_malloc(_parent->cn, sizeof(double));

    if (!graph->x || !graph->w)
    {
        graph->~EdgeCutProblem();
        return NULL;
    }

    graph->W      = _parent->W;
    graph->parent = _parent;
    graph->clevel = graph->parent->clevel + 1;

    return graph;
}

EdgeCutProblem::~EdgeCutProblem()
{
    p = (shallow_p) ? NULL : (Int *)SuiteSparse_free(p);
    i = (shallow_i) ? NULL : (Int *)SuiteSparse_free(i);
    x = (shallow_x) ? NULL : (double *)SuiteSparse_free(x);
    w = (shallow_w) ? NULL : (double *)SuiteSparse_free(w);

    partition      = (bool *)SuiteSparse_free(partition);
    vertexGains    = (double *)SuiteSparse_free(vertexGains);
    externalDegree = (Int *)SuiteSparse_free(externalDegree);
    bhIndex        = (Int *)SuiteSparse_free(bhIndex);
    bhHeap[0]      = (Int *)SuiteSparse_free(bhHeap[0]);
    bhHeap[1]      = (Int *)SuiteSparse_free(bhHeap[1]);
    matching       = (Int *)SuiteSparse_free(matching);
    matchmap       = (Int *)SuiteSparse_free(matchmap);
    invmatchmap    = (Int *)SuiteSparse_free(invmatchmap);
    matchtype      = (Int *)SuiteSparse_free(matchtype);

    markArray = (Int *)SuiteSparse_free(markArray);

    SuiteSparse_free(this);
}

/* Initialize a top level graph with a a set of options. */
void EdgeCutProblem::initialize(const EdgeCut_Options *options)
{
    (void)options; // Unused variable

    if (initialized)
    {
        // Graph has been previously initialized. We need to clear some extra
        // data structures to be able to reuse it.

        X = 0.0;
        W = 0.0;
        H = 0.0;

        bhSize[0] = bhSize[1] = 0;

        heuCost   = 0.0;
        cutCost   = 0.0;
        W0        = 0.0;
        W1        = 0.0;
        imbalance = 0.0;

        clevel = 0;
        cn     = 0;
        for (Int k = 0; k < n; k++)
        {
            externalDegree[k] = 0;
            bhIndex[k]        = 0;
            matching[k]       = 0;
        }
        singleton = -1;

        clearMarkArray();
    }

    Int *Gp    = p;
    double *Gx = x;
    double *Gw = w;

    /* Compute worst-case gains, and compute X. */
    double *gains = vertexGains;
    double min    = fabs((Gx) ? Gx[0] : 1);
    double max    = fabs((Gx) ? Gx[0] : 1);
    for (Int k = 0; k < n; k++)
    {
        W += (Gw) ? Gw[k] : 1;
        double sumEdgeWeights = 0.0;

        for (Int j = Gp[k]; j < Gp[k + 1]; j++)
        {
            double Gxj = (Gx) ? Gx[j] : 1;
            sumEdgeWeights += Gxj;

            if (fabs(Gxj) < min)
            {
                min = fabs(Gxj);
            }
            if (fabs(Gxj) > max)
            {
                max = fabs(Gxj);
            }
        }

        gains[k] = -sumEdgeWeights;
        X += sumEdgeWeights;
    }
    H = 2.0 * X;

    // May need to correct tolerance for very ill-conditioned problems
    worstCaseRatio = max / (1E-9 + min);

    initialized = true;
}

void EdgeCutProblem::clearMarkArray()
{
    markValue += 1;
    if (markValue < 0)
    {
        resetMarkArray();
    }
}

void EdgeCutProblem::clearMarkArray(Int incrementBy)
{
    markValue += incrementBy;
    if (markValue < 0)
    {
        resetMarkArray();
    }
}

void EdgeCutProblem::resetMarkArray()
{
    markValue = 1;
    for (Int k = 0; k < n; k++)
    {
        markArray[k] = 0;
    }
}

} // end namespace Mongoose
