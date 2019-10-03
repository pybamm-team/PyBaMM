/* ========================================================================== */
/* === Source/Mongoose_QPMaxHeap.cpp ======================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/* ========================================================================== */
/* === maxheap ============================================================== */
/* ========================================================================== */

#include "Mongoose_QPMaxHeap.hpp"
#include "Mongoose_Internal.hpp"

namespace Mongoose
{

/* ========================================================================== */
/* === QPMaxHeap_build ====================================================== */
/* ========================================================================== */

/* build a max heap in heap [1..nheap] */

void QPMaxHeap_build(Int *heap, /* on input, an unsorted set of elements */
                     Int size,  /* number of elements to build into the heap */
                     double *x)
{
    for (Int p = size / 2; p >= 1; p--)
        QPMaxHeapify(p, heap, size, x);
}

/* ========================================================================== */
/* === QPMaxHeap_delete ===================================================== */
/* ========================================================================== */

/* delete the top element in a max heap */

Int QPMaxHeap_delete /* return new size of heap */
    (Int *heap,      /* containing indices into x, 1..n on input */
     Int size,       /* number of items in heap */
     const double *x /* not modified */
    )
{
    if (size <= 1)
        return 0;

    /* Replace top element with last element. */
    heap[1] = heap[size];
    size--;
    QPMaxHeapify(1, heap, size, x);
    return size;
}

/* ========================================================================== */
/* === QPMaxHeap_add ======================================================== */
/* ========================================================================== */

/* add a new leaf to a max heap */

Int QPMaxHeap_add(Int leaf,        /* the new leaf */
                  Int *heap,       /* size n, containing indices into x */
                  const double *x, /* not modified */
                  Int size /* number of elements in heap not counting new one */
)
{
    Int l, lnew, lold;
    double xold;

    size++;
    lold       = size;
    heap[lold] = leaf;
    xold       = x[leaf];
    while (lold > 1)
    {
        lnew        = lold / 2;
        l           = heap[lnew];
        double xnew = x[l];

        /* swap new and old */
        if (xnew < xold)
        {
            heap[lnew] = leaf;
            heap[lold] = l;
        }
        else
        {
            return size;
        }

        lold = lnew;
    }

    return size;
}

/* ========================================================================== */
/* === QPMaxHeapify ========================================================= */
/* ========================================================================== */

/* heapify starting at vertex p.  On input, the heap at vertex p satisfies   */
/* the heap property, except for heap [p] itself.  On output, the whole heap */
/* satisfies the heap property. */

void QPMaxHeapify(Int p,          /* start at vertex p in the heap */
                  Int *heap,      /* size n, containing indices into x */
                  Int size,       /* heap [ ... nheap] is in use */
                  const double *x /* not modified */
)
{
    Int left, right, e, hleft, hright;
    double xe, xleft, xright;

    e  = heap[p];
    xe = x[e];

    while (true)
    {
        left  = p * 2;
        right = left + 1;

        if (right <= size)
        {
            hleft  = heap[left];
            hright = heap[right];
            xleft  = x[hleft];
            xright = x[hright];

            if (xleft > xright)
            {
                if (xe < xleft)
                {
                    heap[p] = hleft;
                    p       = left;
                }
                else
                {
                    heap[p] = e;
                    return;
                }
            }
            else
            {
                if (xe < xright)
                {
                    heap[p] = hright;
                    p       = right;
                }
                else
                {
                    heap[p] = e;
                    return;
                }
            }
        }
        else
        {
            if (left <= size)
            {
                hleft = heap[left];
                xleft = x[hleft];
                if (xe < xleft)
                {
                    heap[p] = hleft;
                    p       = left;
                }
            }
            heap[p] = e;
            return;
        }
    }
}

} // end namespace Mongoose
