/* ========================================================================== */
/* === Include/Mongoose_QPNapsack.hpp ======================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_QPNAPSACK_HPP
#define MONGOOSE_QPNAPSACK_HPP

#include "Mongoose_Internal.hpp"

namespace Mongoose
{

double QPNapsack    /* return the final lambda */
    (double *x,     /* holds y on input, and the solution x on output */
     Int n,         /* size of x, constraint lo <= a'x <= hi */
     double lo,     /* partition lower bound */
     double hi,     /* partition upper bound */
     double *a,     /* vector of nodal weights */
     double Lambda, /* initial guess for lambda */
     const Int *FreeSet_status,
     /* FreeSet_status[i] = +1,-1, or 0 on input, for 3 cases:
        x_i = 1, 0, or 0< x_i< 1 */
     double *w,  /* work array of size n */
     Int *heap1, /* work array of size n+1 */
     Int *heap2, /* work array of size n+1 */
     double tol);

} // end namespace Mongoose

#endif
