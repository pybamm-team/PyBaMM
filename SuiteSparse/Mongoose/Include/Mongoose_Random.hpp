/* ========================================================================== */
/* === Include/Mongoose_Random.hpp ========================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_RANDOM_HPP
#define MONGOOSE_RANDOM_HPP

#include "Mongoose_Internal.hpp"

namespace Mongoose
{

Int random();
void setRandomSeed(Int seed);

} // end namespace Mongoose

#endif
