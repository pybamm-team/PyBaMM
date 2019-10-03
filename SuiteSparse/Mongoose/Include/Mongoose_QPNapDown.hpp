/* ========================================================================== */
/* === Include/Mongoose_QPNapDown.hpp ======================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_QPNAPDOWN_HPP
#define MONGOOSE_QPNAPDOWN_HPP

#include "Mongoose_Internal.hpp"

namespace Mongoose
{

double QPNapDown       /* return lambda */
    (const double *x,  /* holds y on input, not modified */
     Int n,            /* size of x */
     double lambda,    /* initial guess for the shift */
     const double *a,  /* input constraint vector */
     double b,         /* input constraint scalar */
     double *breakpts, /* break points */
     Int *bound_heap,  /* work array */
     Int *free_heap    /* work array */
    );

} // end namespace Mongoose

#endif
