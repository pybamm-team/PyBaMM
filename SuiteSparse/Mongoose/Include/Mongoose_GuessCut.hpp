/* ========================================================================== */
/* === Include/Mongoose_GuessCut.hpp ======================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_GUESSCUT_HPP
#define MONGOOSE_GUESSCUT_HPP

#include "Mongoose_BoundaryHeap.hpp"
#include "Mongoose_CutCost.hpp"
#include "Mongoose_Graph.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_EdgeCutOptions.hpp"

namespace Mongoose
{

bool guessCut(EdgeCutProblem *, const EdgeCut_Options *);

} // end namespace Mongoose

#endif
