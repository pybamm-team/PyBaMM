/* ========================================================================== */
/* === Include/Mongoose_Coarsening.hpp ====================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/**
 * Coarsening of a graph given a previously determined matching
 *
 * In order to operate on extremely large graphs, a pre-processing is
 * done to reduce the size of the graph while maintaining its overall structure.
 * Given a matching of vertices with other vertices (e.g. heavy edge matching,
 * random, etc.), coarsening constructs the new, coarsened graph.
 */

// #pragma once
#ifndef MONGOOSE_COARSENING_HPP
#define MONGOOSE_COARSENING_HPP

#include "Mongoose_EdgeCutOptions.hpp"
#include "Mongoose_EdgeCutProblem.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Matching.hpp"

namespace Mongoose
{

EdgeCutProblem *coarsen(EdgeCutProblem *, const EdgeCut_Options *);

} // end namespace Mongoose

#endif
