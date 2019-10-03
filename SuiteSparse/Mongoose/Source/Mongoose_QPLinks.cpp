/* ========================================================================== */
/* === Source/Mongoose_QPLinks.cpp ========================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_QPLinks.hpp"
#include "Mongoose_Debug.hpp"
#include "Mongoose_Internal.hpp"

namespace Mongoose
{

bool QPLinks(EdgeCutProblem *graph, const EdgeCut_Options *options, QPDelta *QP)
{
    (void)options; // Unused variable

    /* Inputs */
    double *x = QP->x;

    /* Unpack structures. */
    Int n      = graph->n;
    Int *Ep    = graph->p;
    Int *Ei    = graph->i;
    double *Ex = graph->x;
    double *a  = graph->w;

    /* working array */
    double *D           = QP->D;
    Int *FreeSet_status = QP->FreeSet_status;
    Int *FreeSet_list   = QP->FreeSet_list;
    double *grad        = QP->gradient; /* gradient at current x */

    // FreeSet is empty
    Int nFreeSet = 0;

    double s = 0.; // a'x

    for (Int k = 0; k < n; k++)
    {
        grad[k] = (0.5 - x[k]) * D[k];
    }

    for (Int k = 0; k < n; k++)
    {
        double xk = x[k];
        if (xk < 0. || xk > 1.)
        {
            // Error!
            return false;
        }

        s += ((a) ? a[k] : 1) * xk;
        double r = 0.5 - xk;
        for (Int p = Ep[k]; p < Ep[k + 1]; p++)
        {
            grad[Ei[p]] += r * ((Ex) ? Ex[p] : 1);
        }
        if (xk >= 1.)
        {
            FreeSet_status[k] = 1;
        }
        else if (xk <= 0.)
        {
            FreeSet_status[k] = -1;
        }
        else
        {
            // add k to the FreeSet
            FreeSet_status[k]        = 0;
            FreeSet_list[nFreeSet++] = k;
            //---
        }
    }

    QP->nFreeSet = nFreeSet;
    QP->b        = s; // a'x

    DEBUG(FreeSet_dump("QPLinks:done", n, FreeSet_list, nFreeSet,
                       FreeSet_status, 1, x));

    // Adjust bounds to force feasibility
    if (s > QP->hi)
        QP->hi = s;
    if (s < QP->lo)
        QP->lo = s;

    // Note that b can be less than lo or greater than hi.
    // b starts between: lo < b < hi
    QP->ib = (s <= QP->lo ? -1 : s < QP->hi ? 0 : 1);

#ifndef NDEBUG
    // for debugging only
    QP->check_cost = INFINITY;

    // ib is shorthand for these tests:
    Int ib = QP->ib;
    PR(("QPlinks: target "
        "%g GW %g ib %ld lo %g b %g hi %g b-lo %g hi-b %g\n",
        options->target_split, graph->W, ib, QP->lo, QP->b, QP->hi,
        (QP->b) - (QP->lo), (QP->hi) - (QP->b)));
    fflush(stdout);
    fflush(stderr);

    double minVal = INFINITY;
    double maxVal = 1;
    for (Int k = 0; k < n; k++)
    {
        if (minVal > x[k])
        {
            minVal = x[k];
        }
        if (maxVal < x[k])
        {
            maxVal = x[k];
        }
    }
    minVal     = std::max(minVal, 1E-8);
    maxVal     = std::max(maxVal, 1E-8);
    double eps = maxVal * minVal * n;
    ASSERT(IMPLIES((ib == -1), (fabs(QP->b - QP->lo) < eps))); // b = lo
    ASSERT(IMPLIES((ib == 0), ((QP->lo < (QP->b + eps))
                               && (QP->b < (QP->hi + eps)))));    // lo < b <hi
    ASSERT(IMPLIES((ib == +1), (fabs(QP->b - QP->hi) < eps)));    // b = hi
    ASSERT((QP->lo <= (QP->b + eps) && QP->b <= (QP->hi + eps))); // x feasible
#endif

    return true;
}

} // end namespace Mongoose
