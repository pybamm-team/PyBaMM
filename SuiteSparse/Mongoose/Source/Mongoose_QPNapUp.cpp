/* ========================================================================== */
/* === Source/Mongoose_QPNapUp.cpp ========================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/* ========================================================================== */
/* === QPNapUp ============================================================== */
/* ========================================================================== */
/* Find x that minimizes ||x-y|| while satisfying the constraints
   0 <= x <= 1, a'x = b.
   The algorithm is described in the napsack comments.
   It is assumed that the starting guess lambda for the dual multiplier is <=
   the correct multiplier. Hence, lambda will be increased.  The slope of
   the dual function, neglecting b, starts out larger than b. We stop
   when we reach b. We assume that a >= 0, so that as lambda increases,
   x_i (lambda) decreases. Hence, the only bound variables that can become
   free are those with x_i (lambda) >= 1 */

#include "Mongoose_QPNapUp.hpp"
#include "Mongoose_Debug.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Logger.hpp"
#include "Mongoose_QPMinHeap.hpp"

namespace Mongoose
{

double QPNapUp         /* return lambda */
    (const double *x,  /* holds y on input, not modified */
     const Int n,      /* size of x */
     double lambda,    /* initial guess for the shift */
     const double *a,  /* input constraint vector */
     double b,         /* input constraint scalar */
     double *breakpts, /* break points */
     Int *bound_heap,  /* work array */
     Int *free_heap    /* work array */
    )
{
    Int i, k, e, maxsteps, n_bound, n_free;
    double ai, asum, a2sum, minbound, minfree, t;

    minbound = INFINITY;
    minfree  = INFINITY;

    /* -------------------------------------------------------------- */
    /* construct the heaps */
    /* -------------------------------------------------------------- */

    n_bound = 0;
    n_free  = 0;
    asum    = 0.;
    a2sum   = 0.;

    for (i = 0; i < n; i++)
    {
        ai        = (a) ? a[i] : 1;
        double xi = x[i] - ai * lambda;
        if (xi > 1.)
        {
            n_bound++;
            bound_heap[n_bound] = i;
            asum += ai;
            t           = (x[i] - 1.) / ai;
            minbound    = std::min(minbound, t);
            breakpts[i] = t;
        }
        else if (xi > 0.)
        {
            n_free++;
            free_heap[n_free] = i;
            asum += x[i] * ai;
            a2sum += ai * ai;
            t           = x[i] / ai;
            minfree     = std::min(minfree, t);
            breakpts[i] = t;
        }
    }

    maxsteps = 2 * n + 1;
    for (k = 1; k <= maxsteps; k++)
    {
        /*------------------------------------------------------------------- */
        /* check to see if zero slope achieved without changing the free set  */
        /* remember that the slope must always be adjusted by b               */
        /*------------------------------------------------------------------- */
        double new_break = std::min(minfree, minbound);
        double s         = asum - new_break * a2sum;
        if ((s <= b) || (new_break == INFINITY)) /* done */
        {
            if (a2sum != 0.)
            {
                lambda = (asum - b) / a2sum;
            }
            return (lambda);
        }
        lambda = new_break;

        if (k == 1)
        {
            QPMinHeap_build(free_heap, n_free, breakpts);
            QPMinHeap_build(bound_heap, n_bound, breakpts);
        }

        /* -------------------------------------------------------------- */
        /* update the heaps */
        /* -------------------------------------------------------------- */

        if (n_free > 0)
        {
            while (breakpts[e = free_heap[1]] <= lambda)
            {
                ai = (a) ? a[e] : 1;
                a2sum -= ai * ai;
                asum -= ai * x[e];
                n_free = QPMinHeap_delete(free_heap, n_free, breakpts);
                if (n_free == 0)
                {
                    a2sum = 0.;
                    break;
                }
            }
        }

        if (n_bound > 0)
        {
            while (breakpts[e = bound_heap[1]] <= lambda)
            {
                n_bound = QPMinHeap_delete(bound_heap, n_bound, breakpts);
                ai      = (a) ? a[e] : 1;
                a2sum += ai * ai;
                asum += ai * (x[e] - 1.);
                t           = x[e] / ai;
                breakpts[e] = t;
                n_free      = QPMinHeap_add(e, free_heap, breakpts, n_free);
                if (n_bound == 0)
                    break;
            }
        }

        /*------------------------------------------------------------------- */
        /* get the biggest entry in each heap */
        /*------------------------------------------------------------------- */

        minfree  = (n_free > 0 ? breakpts[free_heap[1]] : INFINITY);
        minbound = (n_bound > 0 ? breakpts[bound_heap[1]] : INFINITY);
    }

    /* this should not happen */
    ASSERT(false);
    lambda = 0.;
    return lambda;
}

} // end namespace Mongoose
