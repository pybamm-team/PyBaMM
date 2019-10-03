/* ========================================================================== */
/* === Include/Mongoose_Internal.hpp ======================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_INTERNAL_HPP
#define MONGOOSE_INTERNAL_HPP

#define FREESET_DEBUG 0

#if __cplusplus > 199711L
#define CPP11_OR_LATER true
#else
#define CPP11_OR_LATER false
#endif

/* Dependencies */
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>

/* Memory Management */
#include "SuiteSparse_config.h"

namespace Mongoose
{

/* Type definitions */
typedef SuiteSparse_long Int;
#ifndef MAX_INT
#define MAX_INT SuiteSparse_long_max
#endif

/* Enumerations */
enum MatchingStrategy
{
    Random   = 0,
    HEM      = 1,
    HEMSR    = 2,
    HEMSRdeg = 3
};

enum InitialEdgeCutType
{
    InitialEdgeCut_QP           = 0,
    InitialEdgeCut_Random       = 1,
    InitialEdgeCut_NaturalOrder = 2
};

enum MatchType
{
    MatchType_Orphan    = 0,
    MatchType_Standard  = 1,
    MatchType_Brotherly = 2,
    MatchType_Community = 3
};

} // end namespace Mongoose

#endif
