/* ========================================================================== */
/* === Source/Mongoose_QPGradProj.cpp ======================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/**
 * Gradient projection algorithm
 *
 * Apply gradient projection algorithm to the quadratic program which
 * arises in graph partitioning:
 *
 * min (1-x)'(D+A)x subject to lo <= b <= hi, a'x = b, 0 <= x <= 1
 *
 * The gradient at the current point is provided as input, and the
 * gradient is updated in each iteration.
 */

#include "Mongoose_QPGradProj.hpp"
#include "Mongoose_Debug.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Logger.hpp"
#include "Mongoose_QPNapsack.hpp"

#define EMPTY (-1)

namespace Mongoose
{

// save the current state of the solution, just before returning from QPGradProj
inline void saveContext(EdgeCutProblem *graph, QPDelta *QP, Int it, double err,
                        Int nFreeSet, Int ib, double lo, double hi)
{
    QP->its      = it;
    QP->err      = err;
    QP->nFreeSet = nFreeSet;
    double b     = 0.0;
    if (ib != 0)
    {
        b = (ib > 0 ? hi : lo);
    }
    else
    {
        for (Int k = 0; k < graph->n; k++)
            b += ((graph->w) ? graph->w[k] : 1) * QP->x[k];
    }
    QP->ib = ib;
    QP->b  = b;
}

double QPGradProj(EdgeCutProblem *graph, const EdgeCut_Options *options, QPDelta *qpDelta)
{

    PR(("\n------- QPGradProj start: [\n"));
    DEBUG(QPcheckCom(graph, options, qpDelta, 0, qpDelta->nFreeSet,
                     -999999)); // do not check b

    /* ---------------------------------------------------------------------- */
    /* Unpack the relevant structures                                         */
    /* ---------------------------------------------------------------------- */

    double tol  = options->gradproj_tolerance;
    double *wx1 = qpDelta->wx[0]; /* work array for napsack and here as y */
    double *wx2 = qpDelta->wx[1]; /* work array for napsack and here as Dgrad */
    double *wx3 = qpDelta->wx[2]; /* work array used here for d=y-x */
    Int *wi1    = qpDelta->wi[0]; /* work array for napsack
                                and here as changeList */
    Int *wi2 = qpDelta->wi[1];    /* work array only for napsack */

    /* Output and Input */
    double *x = qpDelta->x; /* current estimate of solution            */
    Int *FreeSet_status = qpDelta->FreeSet_status;
    /* FreeSet_status [i] = +1,-1, or 0 if x_i = 1,0, or 0 < x_i < 1 */

    Int nFreeSet = qpDelta->nFreeSet; /* number of i such that 0 < x_i < 1 */
    Int *FreeSet_list = qpDelta->FreeSet_list; /* list of free indices */

    double *grad = qpDelta->gradient; /* gradient at current x */

    /* Unpack the problem's parameters. */
    Int n      = graph->n; /* problem dimension */
    Int *Ep    = graph->p; /* points into Ex or Ei */
    Int *Ei    = graph->i; /* adjacent vertices for each vertex */
    double *Ex = graph->x; /* edge weights */
    double *Ew = graph->w; /* vertex weights; a'x = b, lo <= b <= hi */

    double lo = qpDelta->lo;
    double hi = qpDelta->hi;

    double *D = qpDelta->D; /* diagonal of quadratic */

    /* gradient projection parameters */
    Int limit = options->gradproj_iteration_limit; /* max number of iterations */

    /* work arrays */
    double *y     = wx1;
    double *wx    = wx2;
    double *d     = wx3;
    double *Dgrad = wx; /* gradient change       ; used in napsack as wx  */

    /* components of x change; used in napsack as wi1 */
    Int *changeList     = wi1;
    Int *changeLocation = wi2;

    /* compute error, take step along projected gradient */
    Int ib = 0; /* initialize ib so that lo < b < hi */
    // double lambda = 0.;
    double lambda = qpDelta->lambda;
    Int it        = 0;
    double err    = INFINITY;

    DEBUG(FreeSet_dump("QPGradProj: start", n, FreeSet_list, nFreeSet,
                       FreeSet_status, 0, x));

    while (err > tol)
    {

        PR(("top of QPgrad while loop\n"));
        DEBUG(FreeSet_dump("QPGradProj:0", n, FreeSet_list, nFreeSet,
                           FreeSet_status, 0, x));
        DEBUG(
            QPcheckCom(graph, options, qpDelta, 0, qpDelta->nFreeSet, -999999));

#ifndef NDEBUG
        // check grad
        {
            // for debugging, just use malloc
            double s       = 0.;
            double *mygrad = (double *)malloc((n + 1) * sizeof(double));
            for (Int k = 0; k < n; k++)
            {
                mygrad[k] = (0.5 - x[k]) * D[k];
            }
            for (Int k = 0; k < n; k++)
            {
                double xk = x[k];
                s += ((Ew) ? Ew[k] : 1) * xk;
                double r = 0.5 - xk;
                for (Int p = Ep[k]; p < Ep[k + 1]; p++)
                {
                    mygrad[Ei[p]] += r * ((Ex) ? Ex[p] : 1);
                }
            }
            double maxerr = 0.;
            for (Int k = 0; k < n; k++)
            {
                double err = fabs(grad[k] - mygrad[k]);
                maxerr     = std::max(maxerr, err);
            }
            // PR (("check grad %g\n", maxerr)) ;
            double adj_tol = std::max(
                log10(options->gradproj_tolerance * graph->worstCaseRatio),
                options->gradproj_tolerance);
            ASSERT(maxerr < adj_tol);
            free(mygrad);
        }
#endif

        /* Moving in the gradient direction. */
        for (Int k = 0; k < n; k++)
            y[k] = x[k] - grad[k];

        /* Run the napsack. */
        lambda = QPNapsack(y, n, lo, hi, Ew, lambda, FreeSet_status, wx, wi1,
                           wi2, tol);

        /* Compute the maximum error. */
        err = -INFINITY;
        for (Int k = 0; k < n; k++)
            err = std::max(err, fabs(y[k] - x[k]));

        /* If we converged or got exhausted, save context and exit. */
        if ((err <= tol) || (it >= limit))
        {
            PR(("QPGradProj exhausted:"));
            saveContext(graph, qpDelta, it, err, nFreeSet, ib, lo, hi);
            DEBUG(QPcheckCom(graph, options, qpDelta, 1, qpDelta->nFreeSet,
                             qpDelta->b));
            DEBUG(FreeSet_dump("QPGradProj exhausted", n, FreeSet_list,
                               nFreeSet, FreeSet_status, 0, x));
            PR(("------- QPGradProj end ]\n"));
            return err;
        }

        it++;

        /* compute stepsize st = g_F'g_F/-g_F'(A+D)g_F */
        for (Int k = 0; k < n; k++)
            Dgrad[k] = 0.;

        DEBUG(FreeSet_dump("QPGradProj:1", n, FreeSet_list, nFreeSet,
                           FreeSet_status, 0, x));

        // for each i in the FreeSet:
        for (Int ifree = 0; ifree < nFreeSet; ifree++)
        {
            /* compute -(A+D)g_F */
            Int i    = FreeSet_list[ifree];
            double s = grad[i];
            for (Int p = Ep[i]; p < Ep[i + 1]; p++)
            {
                Dgrad[Ei[p]] -= s * ((Ex) ? Ex[p] : 1);
            }
            Dgrad[i] -= s * D[i];
        }

        double st_num = 0.;
        double st_den = 0.;

        DEBUG(FreeSet_dump("QPGradProj:2", n, FreeSet_list, nFreeSet,
                           FreeSet_status, 0, x));

        for (Int jfree = 0; jfree < nFreeSet; jfree++)
        {
            Int j = FreeSet_list[jfree];
            st_num += grad[j] * grad[j];
            st_den += grad[j] * Dgrad[j];
        }

        /* st = g_F'g_F/-g_F'(A+D)g_F unless the denominator <= 0 */
        if (st_den > 0.)
        {
            // PR (("change y\n")) ;
            double st = std::max(st_num / st_den, 0.001);
            for (Int j = 0; j < n; j++)
                y[j] = x[j] - st * grad[j];
            lambda = QPNapsack(y, n, lo, hi, Ew, lambda, FreeSet_status, wx,
                               wi1, wi2, tol);
        }

        /* otherwise st = 1 and y is as computed above */
        Int nc   = 0; /* number of changes (number of j for which y_j != x_j) */
        double s = 0.;
        for (Int j = 0; j < n; j++)
            Dgrad[j] = 0.;

        // consider vertices j in the FreeSet_list
        for (Int jfree = 0; jfree < nFreeSet; jfree++)
        {
            Int j = FreeSet_list[jfree];
            ASSERT(FreeSet_status[j] == 0);
            double t = y[j] - x[j];
            if (t != 0.)
            {
                // PR (("changeList: we shall consider j %ld t %g\n", j, t)) ;
                d[j] = t;
                s += t * grad[j]; /* derivative in the direction y - x */
                // add j to the changeList and keep track of its position
                // in the FreeSet_list, in case we need to remove it from
                // the FreeSet.
                changeList[nc]    = j;
                changeLocation[j] = jfree;
                nc++;
                for (Int p = Ep[j]; p < Ep[j + 1]; p++)
                {
                    Dgrad[Ei[p]] -= ((Ex) ? Ex[p] : 1) * t;
                }
                Dgrad[j] -= D[j] * t;
            }
        }

        // now consider vertices j not in the FreeSet_list
        for (Int j = 0; j < n; j++)
        {
            if (FreeSet_status[j] == 0)
            {
                // j is in the FreeSet, so skip it (already done above)
                continue;
            }
            double t = y[j] - x[j];
            if (t != 0.)
            {
                // PR (("changeList: we shall consider j %ld t %g\n", j, t)) ;
                d[j] = t;
                s += t * grad[j]; /* derivative in the direction y - x */
                changeList[nc]    = j;
                changeLocation[j] = EMPTY; // j not in FreeSet
                nc++;
                for (Int p = Ep[j]; p < Ep[j + 1]; p++)
                {
                    Dgrad[Ei[p]] -= ((Ex) ? Ex[p] : 1) * t;
                }
                Dgrad[j] -= D[j] * t;
            }
        }

        // PR (("directional derivative s = %g\n", s)) ;

        /* If directional derivative has wrong sign, save context and exit. */
        if (s >= 0.)
        {
            PR(("QPGradProj directional derivative has wrong sign\n"));
            saveContext(graph, qpDelta, it, err, nFreeSet, ib, lo, hi);
            DEBUG(FreeSet_dump("QPGradProj wrong sign", n, FreeSet_list,
                               nFreeSet, FreeSet_status, 0, x));
            PR(("------- QPGradProj end ]\n"));
            return err;
        }

#ifndef NDEBUG
        // lo <= a'y <= hi should hold
        {
            double aty = 0., atx = 0.;
            for (Int j = 0; j < n; j++)
            {
                aty += ((Ew) ? Ew[j] : 1) * y[j];
                atx += ((Ew) ? Ew[j] : 1) * x[j];
            }
            bool good_aty = ((aty - lo) / (lo + tol) >= -tol)
                            && ((hi - aty) / (hi + tol) >= -tol);
            bool good_atx = ((atx - lo) / (lo + tol) >= -tol)
                            && ((hi - atx) / (hi + tol) >= -tol);
            if (!good_aty || !good_atx)
            {
                if (!good_aty)
                {
                    PR(("BAD ATY: lo %g a'y %g hi %g tol %g\n", lo, aty, hi,
                        tol));
                }
                if (!good_atx)
                {
                    PR(("BAD ATX: lo %g a'y %g hi %g tol %g\n", lo, atx, hi,
                        tol));
                }
                FFLUSH;
            }
            ASSERT(((aty - lo) / (lo + tol) >= -tol));
            ASSERT(((hi - aty) / (hi + tol) >= -tol));
            ASSERT(((atx - lo) / (lo + tol) >= -tol));
            ASSERT(((hi - atx) / (hi + tol) >= -tol));
        }
#endif

        double t = 0.;
        for (Int k = 0; k < nc; k++)
        {
            Int j = changeList[k];
            t += Dgrad[j] * d[j]; /* -dg'd */
        }

        // PR (("MIN ATTAINED AT Y? s %g t %g s+t %g\n", s, t, s+t)) ;

        if (s + t <= 0) /* min attained at y, slope at y <= 0 */
        {
            // PR (("min attained at y: s %g t %g s+t %g\n", s, t, s+t)) ;
            ib = (lambda > 0 ? 1 : lambda < 0 ? -1 : 0);
            for (Int k = 0; k < nc; k++)
            {
                Int j     = changeList[k];
                double yj = y[j];
                x[j]      = yj;

                Int bind; /* -1 = no change, 0 = free, +1 = bind */

                Int FreeSet_status_j = FreeSet_status[j];
                if (FreeSet_status_j > 0)
                {
                    if (yj == 0.)
                    {
                        // j changes from +1 to -1.  no change to FreeSet
                        FreeSet_status_j = -1;
                        bind             = -1;
                    }
                    else
                    {
                        // j changes from +1 to 0.  add j to FreeSet
                        FreeSet_status_j = 0;
                        bind             = 0;
                    }
                }
                else if (FreeSet_status_j < 0)
                {
                    if (yj == 1.0)
                    {
                        // j changes from -1 to 1.  no change to FreeSet
                        FreeSet_status_j = 1;
                        bind             = -1;
                    }
                    else
                    {
                        // j changes from -1 to 0.  add j to FreeSet
                        FreeSet_status_j = 0;
                        bind             = 0;
                    }
                }
                else /* x_j currently free, but it may become bound */
                {
                    if (yj == 1.0) /* x_j hits upper bound */
                    {
                        // j changes from 0 to 1,  remove from FreeSet
                        FreeSet_status_j = 1;
                        bind             = 1;
                    }
                    else if (yj == 0.) /* x_j hits lower bound */
                    {
                        // j changes from 0 to -1,  remove from FreeSet
                        FreeSet_status_j = -1;
                        bind             = 1;
                    }
                    else
                    {
                        // j remains 0.  no change to FreeSet
                        FreeSet_status_j = 0;
                        bind             = -1;
                    }
                }

                if (bind == 0)
                {
                    // add j to the FreeSet
                    ASSERT(FreeSet_status[j] != 0);
                    ASSERT(changeLocation[j] == EMPTY);
                    FreeSet_status[j]        = 0;
                    FreeSet_list[nFreeSet++] = j;
                    //---
                }
                else if (bind == 1)
                {
                    // remove j from the FreeSet
                    ASSERT(FreeSet_status[j] == 0);
                    FreeSet_status[j] = FreeSet_status_j;
                    ASSERT(FreeSet_status[j] != 0);
                    Int jfree = changeLocation[j];
                    ASSERT(0 <= jfree && jfree < nFreeSet);
                    ASSERT(FreeSet_list[jfree] == j);
                    FreeSet_list[jfree] = EMPTY;
                    //---
                }
                else // bind == -1, no change to the FreeSet
                {
                    FreeSet_status[j] = FreeSet_status_j;
                }
            }
            for (Int j = 0; j < n; j++)
            {
                grad[j] += Dgrad[j];
            }
        }
        else /* partial step towards y, st < 1 */
        {
            if ((ib > 0 && lambda <= 0) || (ib < 0 && lambda >= 0))
            {
                ib = 0;
            }

            double st = -s / t;
            // PR (("partial step towards y, st %g\n", st)) ;
            for (Int k = 0; k < nc; k++)
            {
                Int j = changeList[k];
                if (FreeSet_status[j] != 0) /* x_j became free */
                {
                    // add j to the FreeSet
                    ASSERT(FreeSet_status[j] != 0);
                    ASSERT(changeLocation[j] == EMPTY);
                    FreeSet_status[j]        = 0;
                    FreeSet_list[nFreeSet++] = j;
                    //---
                }

                /*  else x_j is free before and after step */
                x[j] += st * d[j];
            }

            for (Int k = 0; k < n; k++)
            {
                grad[k] += st * Dgrad[k];
            }
        }

        // prune any EMPTY entries from the FreeSet
        Int jfree2 = 0;
        for (Int jfree = 0; jfree < nFreeSet; jfree++)
        {
            Int j = FreeSet_list[jfree];
            if (j != EMPTY)
            {
                ASSERT(FreeSet_status[j] == 0);
                FreeSet_list[jfree2++] = j;
            }
        }
        nFreeSet = jfree2;

        DEBUG(FreeSet_dump("QPGradProj:6", n, FreeSet_list, nFreeSet,
                           FreeSet_status, 0, x));

        // do not check b
        PR(("QPGradProj continues:\n"));
        qpDelta->nFreeSet = nFreeSet;
        DEBUG(
            QPcheckCom(graph, options, qpDelta, 0, qpDelta->nFreeSet, -999999));
    }

    DEBUG(FreeSet_dump("QPGradProj end", n, FreeSet_list, nFreeSet,
                       FreeSet_status, 0, x));

    PR(("------- QPGradProj end ]\n"));
    return err;
}

} // end namespace Mongoose
