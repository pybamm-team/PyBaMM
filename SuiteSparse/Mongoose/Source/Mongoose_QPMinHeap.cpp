/* ========================================================================== */
/* === Source/Mongoose_QPMinHeap.cpp ======================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/* ========================================================================== */
/* === minheap ============================================================== */
/* ========================================================================== */

#include "Mongoose_QPMinHeap.hpp"
#include "Mongoose_Internal.hpp"

namespace Mongoose
{

/* ========================================================================== */
/* === QPMinHeap_build ====================================================== */
/* ========================================================================== */

/* build a min heap in heap [1..nheap] */

void QPMinHeap_build(Int *heap, /* on input, an unsorted set of elements */
                     Int size,  /* number of elements to build into the heap */
                     double *x)
{
    Int p;

    for (p = size / 2; p >= 1; p--)
    {
        QPMinHeapify(p, heap, size, x);
    }
}

/* ========================================================================== */
/* === QPMinHeap_delete ===================================================== */
/* ========================================================================== */

/* delete the top element in a min heap */

Int QPMinHeap_delete /* return new size of heap */
    (Int *heap,      /* containing indices into x, 1..n on input */
     Int size,       /* number of items in heap */
     const double *x /* not modified */
    )
{
    if (size <= 1)
    {
        return (0);
    }

    /* move element from the end of the heap to the top */
    heap[1] = heap[size];
    size--;
    QPMinHeapify(1, heap, size, x);
    return (size);
}

/* ========================================================================== */
/* === QPMinHeap_add ======================================================== */
/* ========================================================================== */

/* add a new leaf to a min heap */

Int QPMinHeap_add(
    Int leaf,        /* the new leaf */
    Int *heap,       /* size n, containing indices into x */
    const double *x, /* not modified */
    Int nheap        /* number of elements in heap not counting new one */
)
{
    Int l, lnew, lold;
    double xold;

    nheap++;
    lold       = nheap;
    heap[lold] = leaf;
    xold       = x[leaf];
    while (lold > 1)
    {
        lnew        = lold / 2;
        l           = heap[lnew];
        double xnew = x[l];

        /* swap new and old */
        if (xnew > xold)
        {
            heap[lnew] = leaf;
            heap[lold] = l;
        }
        else
        {
            return (nheap);
        }

        lold = lnew;
    }
    return (nheap);
}

/* ========================================================================== */
/* === QPMinHeapify ========================================================= */
/* ========================================================================== */

/* heapify starting at vertex p.  On input, the heap at vertex p satisfies    */
/* the heap property, except for heap [p] itself.  On output, the whole heap  */
/* satisfies the heap property. */

void QPMinHeapify(Int p,          /* start at vertex p in the heap */
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
            if (xleft < xright)
            {
                if (xe > xleft)
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
                if (xe > xright)
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
                if (xe > xleft)
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
