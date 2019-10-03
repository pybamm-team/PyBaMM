/* ========================================================================== */
/* === Include/Mongoose_Matching.hpp ======================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_MATCHING_HPP
#define MONGOOSE_MATCHING_HPP

#include "Mongoose_EdgeCutOptions.hpp"
#include "Mongoose_EdgeCutProblem.hpp"
#include "Mongoose_Internal.hpp"

namespace Mongoose
{

void match(EdgeCutProblem *, const EdgeCut_Options *);

void matching_Random(EdgeCutProblem *, const EdgeCut_Options *);
void matching_HEM(EdgeCutProblem *, const EdgeCut_Options *);
void matching_SR(EdgeCutProblem *, const EdgeCut_Options *);
void matching_SRdeg(EdgeCutProblem *, const EdgeCut_Options *);
void matching_Cleanup(EdgeCutProblem *, const EdgeCut_Options *);

} // end namespace Mongoose

#endif
