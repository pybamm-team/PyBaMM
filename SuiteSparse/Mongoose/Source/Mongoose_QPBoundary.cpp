/* ========================================================================== */
/* === Source/Mongoose_QPBoundary.cpp ======================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/*
   Move all components of x to boundary of the feasible region

    0 <= x <= 1, a'x = b, lo <= b <= hi

   while decreasing the cost function. The algorithm has the following parts

   1. For each i in the free set, see if x_i can be feasibly pushed to either
      boundary while decreasing the cost.

   2. For each i in the bound set, see if x_i can be feasibly flipped to
      opposite boundary while decreasing the cost.

   3. For each i in the free list with a_{ij} = 0 and with j free,
      move either x_i or x_j to the boundary while decreasing
      the cost. The adjustments has the form x_i = s/a_i and x_j = -s/a_j
      where s is a scalar factor. These adjustments must decrease cost.

   4. For the remaining i in the free list, take pair x_i and x_j and
      apply adjustments of the same form as in #2 above to push at least one
      component to boundary. The quadratic terms can only decrease the
      cost function. We choose the sign of s such that g_i x_i + g_j x_j <= 0.
      Hence, this adjustment cannot increase the cost.
 */

/* ========================================================================== */

#include "Mongoose_QPBoundary.hpp"
#include "Mongoose_Debug.hpp"
#include "Mongoose_Internal.hpp"

#define EMPTY (-1)

namespace Mongoose
{

void QPBoundary(EdgeCutProblem *graph, const EdgeCut_Options *options, QPDelta *QP)
{
    (void)options; // Unused variable
    /* ---------------------------------------------------------------------- */
    /* Step 0. read in the needed arrays                                      */
    /* ---------------------------------------------------------------------- */

    /* input and output */

    //--- FreeSet
    Int nFreeSet        = QP->nFreeSet;
    Int *FreeSet_list   = QP->FreeSet_list; /* list for free indices */
    Int *FreeSet_status = QP->FreeSet_status;
    /* FreeSet_status [i] = +1, -1, or 0
       if x_i = 1, 0, or 0 < x_i < 1 */
    //---

    PR(("Mongoose_QPBoundary nFreeSet %ld\n", nFreeSet));

    if (nFreeSet == 0)
    {
        // quick return if FreeSet is empty
        return;
    }

    double *x    = QP->x;        /* current estimate of solution */
    double *grad = QP->gradient; /* gradient at current x */
    Int ib       = QP->ib;       /* ib = +1, -1, or 0 ,
         if b = hi, lo, or lo < b < hi, respectively.  Note there are cases
         where roundoff occurs, and ib can be zero even though b == lo or
         b == hi.  The value of be can even be < lo or > hi, but only by a tiny
         amount of roundoff error.  This is OK. */

    double b = QP->b; /* current value for a'x */

    /* problem specification for the graph G */
    Int n      = graph->n; /* problem dimension */
    double *Ex = graph->x; /* numerical values for edge weights */
    Int *Ei    = graph->i; /* adjacent vertices for each vertex */
    Int *Ep    = graph->p; /* points into Ex or Ei */
    double *a  = graph->w; /* a'x = b, lo <= b <= hi */

    double lo = QP->lo;
    double hi = QP->hi;

    /* work array */
    double *D = QP->D; /* diagonal of quadratic */

    PR(("\n----- QPBoundary start: [\n"));
    DEBUG(QPcheckCom(graph, options, QP, 1, QP->nFreeSet, QP->b)); // check b

    /* ---------------------------------------------------------------------- */
    /* Step 1. if lo < b < hi, then for each free k,                          */
    /*         see if x_k can be pushed to 0 or 1                             */
    /* ---------------------------------------------------------------------- */

    DEBUG(FreeSet_dump("QPBoundary start", n, FreeSet_list, nFreeSet,
                       FreeSet_status, 0, x));

    PR(("Boundary 1 start: ib %ld lo %g b %g hi %g b-lo %g hi-b %g\n", ib, lo,
        b, hi, b - lo, hi - b));

    Int kfree2 = 0;
    for (Int kfree = 0; kfree < nFreeSet; kfree++)
    {
        // Once b becomes bounded, the remainder of the FreeSet is unchanged,
        // and no further changes are made to x.  However, this loop must still
        // continue, so as to compact the FreeSet from deletions made by earlier
        // iterations.

        // get the next k from the FreeSet
        Int k = FreeSet_list[kfree];

        PR(("Step 1: k %ld  x[k] %g  ib %ld b %g\n", k, x[k], ib, b));

        // only modify x[k] if ib == 0 (which means lo < b < hi)
        if (ib == 0)
        {
            double delta_xk;
            double ak = (a) ? a[k] : 1;
            if (grad[k] > 0.0)
            {
                // decrease x [k]
                delta_xk = (b - lo) / ak; // note that delta_xk > 0
                if (delta_xk < x[k])
                {
                    // x [k] decreases by delta_xk but does not hit zero
                    // b hits the lower bound, lo
                    ib = -1;
                    b  = lo;
                    x[k] -= delta_xk;
                    //--- keep k in the FreeSet
                    FreeSet_list[kfree2++] = k;
                }
                else
                {
                    // x [k] hits lower bound of zero
                    // b does not hit lo; still between lower and upper bound
                    delta_xk          = x[k];
                    x[k]              = 0.;
                    FreeSet_status[k] = -1;
                    b -= delta_xk * ak;
                    //--- remove k from the FreeSet by not incrementing kfree2
                }
            }
            else
            {
                // increase x [k]
                delta_xk = (b - hi) / ak; // note that delta_xk < 0
                if (delta_xk < x[k] - 1.)
                {
                    // x [k] hits upper bound of one
                    // b does not reach hi; still between lower and upper bound
                    delta_xk          = x[k] - 1.;
                    x[k]              = 1.;
                    FreeSet_status[k] = +1;
                    b -= delta_xk * ak;
                    //--- remove k from the FreeSet by not incrementing kfree2
                }
                else
                {
                    // x [k] increases by -delta_xk but does not hit one
                    // b hits the upper bound, hi.
                    ib = +1;
                    b  = hi;
                    x[k] -= delta_xk;
                    //--- keep k in the FreeSet
                    FreeSet_list[kfree2++] = k;
                }
            }
            // x [k] has dropped by delta_xk, so update the gradient
            for (Int p = Ep[k]; p < Ep[k + 1]; p++)
            {
                grad[Ei[p]] += delta_xk * ((Ex) ? Ex[p] : 1);
            }
            grad[k] += delta_xk * D[k];
        }
        else
        {
            // b is at lo or hi and thus x [k] is not changed.
            // Once this happens, the remainder of this loop does this next
            // step only, and no further changes are made to x and the FreeSet.
            //--- keep k in the FreeSet
            FreeSet_list[kfree2++] = k;
        }
    }

    // update the size of the FreeSet, after pruning
    nFreeSet = kfree2;

    /* ---------------------------------------------------------------------- */
    /* Step 2. Examine flips of x_k from 0 to 1 or from 1 to 0 */
    /* ---------------------------------------------------------------------- */

    PR(("Boundary step 2:\n"));

    for (Int k = 0; k < n; k++)
    {
        Int FreeSet_status_k = FreeSet_status[k];
        if (FreeSet_status_k == 0)
        {
            // k is in FreeSet so it cannot be simply flipped 0->1 or 1->0
            continue;
        }

        // k not in FreeSet, so no changes here to FreeSet

        double ak = (a) ? a[k] : 1;
        if (FreeSet_status_k > 0) /* try changing x_k from 1 to 0 */
        {
            if (b - ak >= lo)
            {
                if (0.5 * D[k] + grad[k] >= 0) /* flip lowers cost */
                {
                    b -= ak;
                    ib                = (b <= lo ? -1 : 0);
                    x[k]              = 0.0;
                    FreeSet_status[k] = -1;
                }
            }
        }
        else /* try changing x_k from 0 to 1 */
        {
            if (b + ak <= hi)
            {
                if (grad[k] - 0.5 * D[k] <= 0) /* flip lowers cost */
                {
                    b += ak;
                    ib                = (b >= hi ? 1 : 0);
                    x[k]              = 1.0;
                    FreeSet_status[k] = +1;
                }
            }
        }

        if (FreeSet_status_k != FreeSet_status[k])
        {
            if (FreeSet_status_k == 1) /* x [k] was 1, now it is 0 */
            {
                for (Int p = Ep[k]; p < Ep[k + 1]; p++)
                {
                    grad[Ei[p]] += (Ex) ? Ex[p] : 1;
                }
                grad[k] += D[k];
            }
            else /* x [k] was 0, now it is 1 */
            {
                for (Int p = Ep[k]; p < Ep[k + 1]; p++)
                {
                    grad[Ei[p]] -= (Ex) ? Ex[p] : 1;
                }
                grad[k] -= D[k];
            }
        }
        // DEBUG (QPcheckCom (graph, options, QP, 1, nFreeSet, b)) ;         //
        // check b
    }

    /* ---------------------------------------------------------------------- */
    // quick return if FreeSet is now empty
    /* ---------------------------------------------------------------------- */

    if (nFreeSet == 0)
    {
        PR(("Boundary quick: ib %ld lo %g b %g hi %g b-lo %g hi-b %g\n", ib, lo,
            b, hi, b - lo, hi - b));
        QP->nFreeSet = nFreeSet;
        QP->b        = b;
        QP->ib       = ib;
        PR(("------- QPBoundary end ]\n"));
        return;
    }

    /* ---------------------------------------------------------------------- */
    /* Step 3. Search for a_{ij} = 0 in the free index set */
    /* ---------------------------------------------------------------------- */

    // look for where both i and j are in the FreeSet,
    // but i and j are not adjacent in the graph G.

    DEBUG(FreeSet_dump("step 3", n, FreeSet_list, nFreeSet, FreeSet_status, 0,
                       x));

    // for each j in FreeSet, except for the last one
    for (Int jfree = 0; jfree < nFreeSet - 1; jfree++)
    {

        // get j from the FreeSet
        Int j = FreeSet_list[jfree];
        if (j == EMPTY)
        {
            // j has already been deleted, skip it
            continue;
        }

        /* -------------------------------------------------------------- */
        /* find i and j both free and where a_{ij} = 0 */
        /* -------------------------------------------------------------- */

        // mark all vertices i adjacent to j in the FreeSet
        for (Int p = Ep[j]; p < Ep[j + 1]; p++)
        {
            Int i = Ei[p];
            ASSERT(i != j); // graph has no self edges
            graph->mark(i);
        }
        graph->mark(j);

        // for each i that follows after j in the FreeSet
        for (Int ifree = jfree + 1; ifree < nFreeSet; ifree++)
        {

            // get i from the FreeSet
            Int i = FreeSet_list[ifree];
            if (i == EMPTY)
            {
                // i has already been deleted it; skip it
                continue;
            }

            if (!graph->isMarked(i))
            {
                // vertex i is not adjacent to j in the graph G
                double aj = (a) ? a[j] : 1;
                double ai = (a) ? a[i] : 1;
                double xi = x[i];
                double xj = x[j];

                /* cost change if x_j increases dx_j = s/a_j, dx_i = s/a_i */
                double s;
                Int bind1, bind2;
                if (aj * (1. - xj) < ai * xi) // x_j hits upper bound
                {
                    s     = aj * (1. - xj);
                    bind1 = 1;
                }
                else /* x_i hits lower bound */
                {
                    s     = ai * xi;
                    bind1 = 0;
                }
                double dxj = s / aj;
                double dxi = -s / ai;
                double c1  = (grad[j] - .5 * D[j] * dxj) * dxj
                            + (grad[i] - .5 * D[i] * dxi) * dxi;

                /* cost change if x_j decreases dx_j = s/a_j, dx_i = s/a_i */
                if (aj * xj < ai * (1. - xi)) // x_j hits lower bound
                {
                    s     = -aj * xj;
                    bind2 = -1;
                }
                else /* x_i hits upper bound */
                {
                    s     = -ai * (1. - xi);
                    bind2 = 0;
                }
                dxj       = s / aj;
                dxi       = -s / ai;
                double c2 = (grad[j] - 0.5 * D[j] * dxj) * dxj
                            + (grad[i] - 0.5 * D[i] * dxi) * dxi;

                Int new_FreeSet_status;
                if (c1 < c2) /* increase x_j */
                {
                    if (bind1 == 1)
                    {
                        // j is bound (not i) and x_j becomes 1
                        dxj  = 1. - xj;
                        dxi  = -aj * dxj / ai;
                        x[j] = 1.;
                        x[i] += dxi;
                        new_FreeSet_status = +1; /* j is bound at 1 */
                    }
                    else // bind1 is zero
                    {
                        // i is bound (not j) and x_i becomes 0
                        dxi  = -xi;
                        dxj  = -ai * dxi / aj;
                        x[i] = 0.;
                        x[j] += dxj;
                        new_FreeSet_status = -1; /* i is bound at 0 */
                    }
                }
                else
                {
                    if (bind2 == -1)
                    {
                        // j is bound (not i) and x_j becomes 0
                        bind1 = 1;
                        x[j]  = 0.;
                        x[i] += dxi;
                        new_FreeSet_status = -1; /* j is bound at 0 */
                    }
                    else /* x_i = 1 */
                    {
                        // i is bound (not j) and x_i becomes 1
                        bind1 = 0;
                        x[i]  = 1;
                        x[j] += dxj;
                        new_FreeSet_status = +1; /* i is bound at 1 */
                    }
                }

                for (Int p = Ep[j]; p < Ep[j + 1]; p++)
                {
                    grad[Ei[p]] -= ((Ex) ? Ex[p] : 1) * dxj;
                }
                for (Int p = Ep[i]; p < Ep[i + 1]; p++)
                {
                    grad[Ei[p]] -= ((Ex) ? Ex[p] : 1) * dxi;
                }
                grad[j] -= D[j] * dxj;
                grad[i] -= D[i] * dxi;

                // Remove either i or j from the FreeSet.  Note that it
                // is possible for both x[i] and x[j] to reach their bounds
                // at the same time.  Only one is removed from the FreeSet;
                // the other will be removed later.

                if (bind1)
                {
                    // remove j from the FreeSet by setting its place to EMPTY
                    PR(("(b1):remove j = %ld from the FreeSet\n", j));
                    ASSERT(j == FreeSet_list[jfree]);
                    ASSERT(FreeSet_status[j] == 0);
                    FreeSet_list[jfree] = EMPTY;
                    FreeSet_status[j]   = new_FreeSet_status;
                    ASSERT(FreeSet_status[j] != 0);
                    //---
                    // no longer consider j, so skip all of remainder of i loop
                    break;
                }
                else
                {
                    // remove i from the FreeSet by setting its place to EMPTY
                    PR(("(b2):remove i = %ld from the FreeSet\n", i));
                    ASSERT(i == FreeSet_list[ifree]);
                    ASSERT(FreeSet_status[i] == 0);
                    FreeSet_list[ifree] = EMPTY;
                    FreeSet_status[i]   = new_FreeSet_status;
                    ASSERT(FreeSet_status[i] != 0);
                    //---
                    // keep j, and consider it with the next i
                    continue;
                }
            }
        }

        // clear the marks from all the vertices
        graph->clearMarkArray();
    }

    // remove deleted vertices from the FreeSet
    kfree2 = 0;
    for (Int kfree = 0; kfree < nFreeSet; kfree++)
    {
        Int k = FreeSet_list[kfree];
        if (k != EMPTY)
        {
            // keep k in the FreeSet
            FreeSet_list[kfree2++] = k;
            ASSERT(0 <= k && k < n);
            ASSERT(FreeSet_status[k] == 0);
        }
    }
    nFreeSet = kfree2;

    DEBUG(FreeSet_dump("step 3 done", n, FreeSet_list, nFreeSet, FreeSet_status,
                       1, x));

    DEBUG(QPcheckCom(graph, options, QP, 1, nFreeSet, b)); // check b

#ifndef NDEBUG
    // the vertices in the FreeSet now form a single clique.  Check this.
    // this test is for debug mode only
    ASSERT(nFreeSet >= 1); // we can have 1 or more vertices still in FreeSet
    for (Int kfree = 0; kfree < nFreeSet; kfree++)
    {
        // j must be adjacent to all other vertices in the FreeSet
        Int j               = FreeSet_list[kfree];
        Int nfree_neighbors = 0;
        for (Int p = Ep[j]; p < Ep[j + 1]; p++)
        {
            Int i = Ei[p];
            ASSERT(i != j);
            if (FreeSet_status[i] == 0)
                nfree_neighbors++;
        }
        ASSERT(nfree_neighbors == nFreeSet - 1);
    }
#endif

    /* ---------------------------------------------------------------------- */
    /* Step 4. dxj = s/aj, dxi = -s/ai, choose s with g_j dxj + g_i dxi <= 0 */
    /* ---------------------------------------------------------------------- */

    DEBUG(FreeSet_dump("step 4 starts", n, FreeSet_list, nFreeSet,
                       FreeSet_status, 0, x));

    // consider pairs of vertices in the FreeSet, until only one is left
    while (nFreeSet > 1)
    {
        /* free variables: 0 < x_j < 1 */
        /* choose s so that first derivative terms decrease */

        // i and j are the last two vertex in the FreeSet_list, as in:
        // FreeSet_list = [ .... i j ]
        // at the end of this iteration, one will be deleted, thus becoming
        // FreeSet_list = [ .... j ]
        // or
        // FreeSet_list = [ .... i ]
        Int j = FreeSet_list[nFreeSet - 1];
        ASSERT(FreeSet_status[j] == 0);
        Int i = FreeSet_list[nFreeSet - 2];
        ASSERT(FreeSet_status[i] == 0);

        double ai = (a) ? a[i] : 1;
        double aj = (a) ? a[j] : 1;
        double xi = x[i];
        double xj = x[j];

        Int new_FreeSet_status;
        Int bind1;
        double dxj, dxi, s = grad[j] / aj - grad[i] / ai;

        if (s < 0.) /* increase x_j */
        {
            if (aj * (1. - xj) < ai * xi) /* x_j hits upper bound */
            {
                dxj  = 1. - xj;
                dxi  = -aj * dxj / ai;
                x[j] = 1.;
                x[i] += dxi;
                new_FreeSet_status = +1;
                bind1              = 1; /* x_j is bound at 1 */
            }
            else /* x_i hits lower bound */
            {
                dxi  = -xi;
                dxj  = -ai * dxi / aj;
                x[i] = 0.;
                x[j] += dxj;
                new_FreeSet_status = -1;
                bind1              = 0; /* x_i is bound at 0 */
            }
        }
        else /* decrease x_j */
        {
            if (aj * xj < ai * (1. - xi)) /* x_j hits lower bound */
            {
                dxj  = -xj;
                dxi  = -aj * dxj / ai;
                x[j] = 0;
                x[i] += dxi;
                new_FreeSet_status = -1;
                bind1              = 1; /* x_j is bound */
            }
            else /* x_i hits upper bound */
            {
                dxi  = 1 - xi;
                dxj  = -ai * dxi / aj;
                x[i] = 1;
                x[j] += dxj;
                new_FreeSet_status = +1;
                bind1              = 0; /* x_i is bound */
            }
        }

        for (Int k = Ep[j]; k < Ep[j + 1]; k++)
        {
            grad[Ei[k]] -= ((Ex) ? Ex[k] : 1) * dxj;
        }
        for (Int k = Ep[i]; k < Ep[i + 1]; k++)
        {
            grad[Ei[k]] -= ((Ex) ? Ex[k] : 1) * dxi;
        }
        grad[j] -= D[j] * dxj;
        grad[i] -= D[i] * dxi;

        // ---------------------------------------------------------------------
        // the following 2 cases define the next j in the iteration:
        // ---------------------------------------------------------------------

        // Remove either i or j from the FreeSet.  Note that it is possible for
        // both x[i] and x[j] to reach their bounds at the same time.  Only one
        // is removed from the FreeSet; the other will be removed later.

        if (bind1)
        {
            // j is bound.
            // remove j from the FreeSet, and keep i.  The FreeSet_list was
            // FreeSet_list = [ .... i j ] becomes FreeSet_list = [ .... i ]
            PR(("(b3):remove j = %ld from the FreeSet\n", j));
            ASSERT(FreeSet_status[j] == 0);
            FreeSet_status[j] = new_FreeSet_status;
            ASSERT(FreeSet_status[j] != 0);
        }
        else
        {
            // i is bound.
            // remove i from the FreeSet, and keep j.  The FreeSet_list was
            // FreeSet_list = [ .... i j ] becomes FreeSet_list = [ .... j ]
            PR(("(b4):remove i = %ld from the FreeSet\n", i));
            ASSERT(FreeSet_status[i] == 0);
            FreeSet_status[i] = new_FreeSet_status;
            ASSERT(FreeSet_status[i] != 0);
            // shift j down by one in the list, thus discarding j.
            ASSERT(FreeSet_list[nFreeSet - 2] == i);
            FreeSet_list[nFreeSet - 2] = j;
        }

        // one fewer vertex in the FreeSet (i or j removed)
        nFreeSet--;

        DEBUG(FreeSet_dump("step 4", n, FreeSet_list, nFreeSet, FreeSet_status,
                           0, x));
        DEBUG(QPcheckCom(graph, options, QP, 1, nFreeSet, b)); // check b
    }

    DEBUG(FreeSet_dump("wrapup", n, FreeSet_list, nFreeSet, FreeSet_status, 0,
                       x));

    /* ---------------------------------------------------------------------- */
    /* step 5: at most one free variable remaining */
    /* ---------------------------------------------------------------------- */

    ASSERT(nFreeSet == 0 || nFreeSet == 1);

    PR(("Step 5: ib %ld lo %g b %g hi %g b-lo %g hi-b %g\n", ib, lo, b, hi,
        b - lo, hi - b));

    if (nFreeSet == 1) /* j is free, optimize over x [j] */
    {
        // j is the first and only item in the FreeSet
        Int j = FreeSet_list[0];
        PR(("ONE AND ONLY!! j = %ld x[j] %g\n", j, x[j]));

        Int bind1  = 0;
        double aj  = (a) ? a[j] : 1;
        double dxj = (hi - b) / aj;
        PR(("dxj %g  x[j] %g  (1-x[j]): %g\n", dxj, x[j], 1 - x[j]));
        if (dxj < 1. - x[j])
        {
            bind1 = 1;
        }
        else
        {
            dxj = 1. - x[j];
        }

        Int bind2  = 0;
        double dxi = (lo - b) / aj;
        PR(("dxi %g  x[j] %g  (-x[j]): %g\n", dxi, x[j], -x[j]));
        if (dxi > -x[j])
        {
            bind2 = 1;
        }
        else
        {
            dxi = -x[j];
        }

        double c1 = (grad[j] - 0.5 * D[j] * dxj) * dxj;
        double c2 = (grad[j] - 0.5 * D[j] * dxi) * dxi;
        if (c1 <= c2) /* x [j] += dxj */
        {
            if (bind1)
            {
                PR(("bind1: xj changes from %g", x[j]));
                x[j] += dxj;
                PR((" to %g, b now at hi\n", x[j]));
                ib = +1;
                b  = hi;
            }
            else
            {
                x[j] = 1.;
                b += dxj * aj;
                /// remove j from the FreeSet, which is now empty
                PR(("(b5):remove j = %ld from FreeSet, now empty\n", j));
                ASSERT(FreeSet_status[j] == 0);
                FreeSet_status[j] = 1;
                ASSERT(FreeSet_status[j] != 0);
                nFreeSet--;
                ASSERT(nFreeSet == 0);
            }
        }
        else /* x [j] += dxi */
        {
            dxj = dxi;
            if (bind2)
            {
                PR(("bind2: xj changes from %g", x[j]));
                x[j] += dxj;
                PR((" to %g, b now at lo\n", x[j]));
                ib = -1;
                b  = lo;
            }
            else
            {
                x[j] = 0.;
                b += dxj * aj;
                /// remove j from the FreeSet, which is now empty
                PR(("(b6):remove j = %ld from FreeSet, now empty\n", j));
                ASSERT(FreeSet_status[j] == 0);
                FreeSet_status[j] = -1;
                ASSERT(FreeSet_status[j] != 0);
                nFreeSet--;
                ASSERT(nFreeSet == 0);
            }
        }

        if (dxj != 0.)
        {
            for (Int p = Ep[j]; p < Ep[j + 1]; p++)
            {
                grad[Ei[p]] -= ((Ex) ? Ex[p] : 1) * dxj;
            }
            grad[j] -= D[j] * dxj;
        }
    }

    /* ---------------------------------------------------------------------- */
    // wrapup
    /* ---------------------------------------------------------------------- */

    PR(("QBboundary, done:\n"));
    DEBUG(FreeSet_dump("QPBoundary: done ", n, FreeSet_list, nFreeSet,
                       FreeSet_status, 0, x));
    ASSERT(nFreeSet == 0 || nFreeSet == 1);
    PR(("Boundary done: ib %ld lo %g b %g hi %g b-lo %g hi-b %g\n", ib, lo, b,
        hi, b - lo, hi - b));

    QP->nFreeSet = nFreeSet;
    QP->b        = b;
    QP->ib       = ib;

    // clear the marks from all the vertices
    graph->clearMarkArray();

    DEBUG(QPcheckCom(graph, options, QP, 1, nFreeSet, b)); // check b
    PR(("----- QPBoundary end ]\n"));
}

} // end namespace Mongoose
