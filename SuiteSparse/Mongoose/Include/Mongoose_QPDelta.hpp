/* ========================================================================== */
/* === Include/Mongoose_QPDelta.hpp ========================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_QPDELTA_HPP
#define MONGOOSE_QPDELTA_HPP

#include "Mongoose_Internal.hpp"

namespace Mongoose
{

class QPDelta
{
private:
    static const Int WXSIZE = 3;
    static const Int WISIZE = 2;

public:
    double *x; /* current estimate of solution                    */

    // FreeSet:
    Int nFreeSet;        /* number of i such that 0 < x_i < 1               */
    Int *FreeSet_status; /* ix_i = +1,-1, or 0 if x_i = 1,0, or 0 < x_i < 1 */
    Int *FreeSet_list;   /* list for free indices                    */
    //---

    double *gradient; /* gradient at current x                           */
    double *D;        /* max value along the column.                     */

    double lo; // lo <= a'*x <= hi must always hold
    double hi;

    // workspace
    Int *wi[WISIZE];
    double *wx[WXSIZE];

    Int its;
    double err;
    Int ib;   // ib =  0 means lo < b < hi
              // ib = +1 means b == hi
              // ib = -1 means b == lo
    double b; // b = a'*x

    double lambda;

    static QPDelta *Create(Int numVars);
    ~QPDelta();

#ifndef NDEBUG
    double check_cost;
#endif
};

} // end namespace Mongoose

#endif
