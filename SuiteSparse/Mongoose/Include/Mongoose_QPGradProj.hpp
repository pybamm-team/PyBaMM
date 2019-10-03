/* ========================================================================== */
/* === Include/Mongoose_QPGradProj.hpp ====================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_QPGRADPROJ_HPP
#define MONGOOSE_QPGRADPROJ_HPP

#include "Mongoose_EdgeCutOptions.hpp"
#include "Mongoose_EdgeCutProblem.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_QPDelta.hpp"

namespace Mongoose
{

double QPGradProj(EdgeCutProblem *, const EdgeCut_Options *, QPDelta *);

} // end namespace Mongoose

#endif
