/* ========================================================================== */
/* === Include/Mongoose_Graph.hpp =========================================== */
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
 * Stores graph adjacency and weight information.
 */

// #pragma once
#ifndef MONGOOSE_GRAPH_HPP
#define MONGOOSE_GRAPH_HPP

#include "Mongoose_CSparse.hpp"
#include "Mongoose_Internal.hpp"

namespace Mongoose
{

class Graph
{
public:
    /** Graph Data ***********************************************************/
    Int n;     /** # vertices                      */
    Int nz;    /** # edges                         */
    Int *p;    /** Column pointers                 */
    Int *i;    /** Row indices                     */
    double *x; /** Edge weight                     */
    double *w; /** Node weight                     */

    /* Constructors & Destructor */
    static Graph *create(const Int _n, const Int _nz, Int *_p = NULL,
                         Int *_i = NULL, double *_x = NULL, double *_w = NULL);
    static Graph *create(cs *matrix);
    static Graph *create(cs *matrix, bool free_when_done);
    ~Graph();

private:
    Graph();

    /** Memory Management Flags ***********************************************/
    bool shallow_p;
    bool shallow_i;
    bool shallow_x;
    bool shallow_w;
};

} // end namespace Mongoose

#endif
