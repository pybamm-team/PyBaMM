/* ========================================================================== */
/* === Include/Mongoose_QPMaxHeap.hpp ======================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

// #pragma once
#ifndef MONGOOSE_QPMAXHEAP_HPP
#define MONGOOSE_QPMAXHEAP_HPP

#include "Mongoose_Internal.hpp"

namespace Mongoose
{

void QPMaxHeap_build(Int *heap, /* on input, an unsorted set of elements */
                     Int size,  /* size of the heap */
                     double *x);

Int QPMaxHeap_delete /* return new size of heap */
    (Int *heap,      /* containing indices into x, 1..n on input */
     Int size,       /* size of the heap */
     const double *x /* not modified */
    );

void QPMaxHeapify(Int p,          /* start at vertex p in the heap */
                  Int *heap,      /* size n, containing indices into x */
                  Int size,       /* heap [ ... nheap] is in use */
                  const double *x /* not modified */
);

Int QPMaxHeap_add(Int leaf,        /* the new leaf */
                  Int *heap,       /* size n, containing indices into x */
                  const double *x, /* not modified */
                  Int size /* number of elements in heap not counting new one */
);

void QPmaxheap_check(Int *heap,   /* vector of size n+1 */
                     double *x,   /* vector of size n */
                     Int size,    /* # items in heap */
                     Int n, Int p /* start checking at heap [p] */
);

} // end namespace Mongoose

#endif
