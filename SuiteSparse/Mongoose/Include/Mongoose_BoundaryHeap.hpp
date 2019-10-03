/* ========================================================================== */
/* === Include/Mongoose_BoundaryHeap.hpp ==================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_BOUNDARYHEAP_HPP
#define MONGOOSE_BOUNDARYHEAP_HPP

#include "Mongoose_EdgeCutOptions.hpp"
#include "Mongoose_EdgeCutProblem.hpp"
#include "Mongoose_Internal.hpp"

namespace Mongoose
{

void bhLoad(EdgeCutProblem *, const EdgeCut_Options *);
void bhClear(EdgeCutProblem *);
void bhInsert(EdgeCutProblem *, Int vertex);

void bhRemove(EdgeCutProblem *, const EdgeCut_Options *, Int vertex, double gain, bool partition,
              Int bhPosition);

void heapifyUp(EdgeCutProblem *, Int *bhHeap, double *gains, Int vertex, Int position,
               double gain);

void heapifyDown(EdgeCutProblem *, Int *bhHeap, Int size, double *gains, Int vertex,
                 Int position, double gain);

} // end namespace Mongoose

#endif
