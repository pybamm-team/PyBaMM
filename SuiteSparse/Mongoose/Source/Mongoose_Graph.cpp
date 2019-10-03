/* ========================================================================== */
/* === Source/Mongoose_Graph.cpp ============================================ */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_Graph.hpp"

#include <algorithm>
#include <new>

namespace Mongoose
{

/* Constructor & Destructor */
Graph::Graph()
{
    n = nz = 0;
    p      = NULL;
    i      = NULL;
    x      = NULL;
    w      = NULL;
}

Graph *Graph::create(const Int _n, const Int _nz, Int *_p, Int *_i, double *_x,
                     double *_w)
{
    void *memoryLocation = SuiteSparse_malloc(1, sizeof(Graph));
    if (!memoryLocation)
        return NULL;

    // Placement new
    Graph *graph = new (memoryLocation) Graph();

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

    if (!graph->p || !graph->i)
    {
        graph->~Graph();
        return NULL;
    }

    return graph;
}

// Creates graph using a shallow copy of the matrix
// Note that this does not free the matrix arrays when done
Graph *Graph::create(cs *matrix)
{
    Graph *graph = create(std::max(matrix->n, matrix->m), matrix->p[matrix->n],
                          matrix->p, matrix->i, matrix->x);
    if (!graph)
    {
        return NULL;
    }

    return graph;
}

Graph *Graph::create(cs *matrix, bool free_when_done)
{
    void *memoryLocation = SuiteSparse_malloc(1, sizeof(Graph));
    if (!memoryLocation)
        return NULL;

    // Placement new
    Graph *graph = new (memoryLocation) Graph();

    if (!graph)
    {
        return NULL;
    }

    graph->n = std::max(matrix->n, matrix->m);
    graph->nz = matrix->p[matrix->n];
    graph->p = matrix->p;
    graph->i = matrix->i;
    graph->x = matrix->x;

    graph->shallow_p = !free_when_done;
    graph->shallow_i = !free_when_done;
    graph->shallow_x = !free_when_done;

    return graph;
}

Graph::~Graph()
{
    p = (shallow_p) ? NULL : (Int *)SuiteSparse_free(p);
    i = (shallow_i) ? NULL : (Int *)SuiteSparse_free(i);
    x = (shallow_x) ? NULL : (double *)SuiteSparse_free(x);
    w = (shallow_w) ? NULL : (double *)SuiteSparse_free(w);

    SuiteSparse_free(this);
}

} // end namespace Mongoose
