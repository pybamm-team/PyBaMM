/* ========================================================================== */
/* === Include/Mongoose_Debug.hpp =========================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_DEBUG_HPP
#define MONGOOSE_DEBUG_HPP

#include "Mongoose_CSparse.hpp"
#include "Mongoose_EdgeCutOptions.hpp"
#include "Mongoose_EdgeCutProblem.hpp"
#include "Mongoose_QPDelta.hpp"

#include <cassert>
#include <cstdio>

// Mongoose Logic Macros
#undef IMPLIES
#define IMPLIES(p, q) (!(p) || ((p) && (q)))
#undef IFF
#define IFF(p, q) (IMPLIES(p, q) && IMPLIES(q, p))

// turn off debugging
#ifndef NDEBUG
#define NDEBUG
#endif

// turn off debug printing
#ifndef NPRINT
#define NPRINT
#endif

// Uncomment this line to enable debugging.  Mongoose will be very slow.
// #undef NDEBUG

// Uncomment this line to enable debug printing.  Mongoose will be very slow
// and produce massive amounts of debug output.
// #undef NPRINT

// ASSERT macro, example usage: ASSERT (x > 0) ;
// where x is required to be positive.  An error occurs if x <= 0.
#undef ASSERT
#ifndef NDEBUG
#define ASSERT(expression) (assert(expression))
#define ASSERT_TEXT(expression, text) (assert(expression &&text))
#else
#define ASSERT(expression)
#define ASSERT_TEXT(expression, text)
#endif

// PR macro, example usage: PR (("stuff here %g %g\n", x, y)) ;
#undef PR
#undef FFLUSH
#ifndef NPRINT
#define PR(s) printf s
#define FFLUSH                                                                 \
    {                                                                          \
        fflush(stdout);                                                        \
        fflush(stderr);                                                        \
    }
#else
#define PR(s)
#define FFLUSH
#endif

// DEBUG macro, example usage: DEBUG (statement) ;
#undef DEBUG
#ifndef NDEBUG
#define DEBUG(s) s
#else
#define DEBUG(s)
#endif

#ifndef NDEBUG
#include <algorithm>
#endif

namespace Mongoose
{

void print(cs *G);

void print(EdgeCutProblem *G);

void QPcheckCom(EdgeCutProblem *G, const EdgeCut_Options *O, QPDelta *QP, bool check_b,
                Int nFreeSet, double b);

void FreeSet_dump(const char *where, Int n, Int *FreeSet_list, Int nFreeSet,
                  Int *FreeSet_status, Int verbose, double *x);

} // end namespace Mongoose

#endif
