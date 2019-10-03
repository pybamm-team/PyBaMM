/* ========================================================================== */
/* === Source/Mongoose_QPNapsack.cpp ======================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/* ========================================================================== */
/* === QPNapsack ============================================================ */
/* ========================================================================== */
/*  Find x that minimizes ||x-y|| while satisfying 0 <= x <= 1,
    a'x = b, lo <= b <= hi.  It is assumed that the column vector a is strictly
    positive since, in our application, the vector a is the vertex weights,
    which are >= 1. If a is NULL, then it is assumed that a is identically 1.
    The approach is to solve the dual problem obtained by introducing
    a multiplier lambda for the constraint a'x = b.  The dual function is

    L (lambda) = min { ||x-y||^2 + lambda (a'x - b): 0 <= x <= 1, lo <= b <= hi}

    The dual function is concave. It is continuously differentiable
    except at lambda = 0.  If mu denotes the maximizer of the dual function,
    then the solution of the primal problem is

    x = proj (y - mu*a) ,

    where proj (z) is the projection of z onto the set { x : 0 <= x <= 1}.
    Thus we have

       proj (z)_i = 1   if z_i >= 1,
                    0   if z_i <= 0,
                    z_i otherwise  .

    Note that for any lambda, the minimizing x in the dual function is

       x (lambda) = proj (y - lambda*a).

    The slope of the dual function is

      L'(lambda) = a'proj (x(lambda)) - hi (if lambda > 0)
                   a'proj (x(lambda)) - lo (if lambda < 0)

    The minimizing b in the dual function is b = hi if lambda > 0 and b = lo
    if b <= 0.  When L' (lamdbda) = 0 with lambda != 0, either x'a = hi or
    x'a = lo.  The minimum is attained at lambda = 0 if and only if the
    slope of L is negative at lambda = 0+ and positive at lambda = 0-.
    This is equivalent to the inequalities

              lo <= a' proj (y) <= hi .

    The solution technique is to start with an initial guess lambda for
    mu and search for a zero of L'. We have the following cases:

    1. lambda >= 0, L'(lambda+) >= 0: mu >= lambda. If L' = 0, then done.
                                    Otherwise, increase lambda using napup until
                                    slope vanishes

    2. lambda <= 0, L'(lambda-) <= 0: mu <= lambda. If L' = 0, then done.
                                    Otherwise, decrease lambda using napdown
                                    until slope vanishes

    3. lambda >= 0, L'(lambda+)  < 0: If L' (0-) < 0, then mu < 0. Call napdown
                                    with lambda = 0 as starting guess.  If
                                    L' (0+) > 0, then 0 < mu < lambda. Call
                                    napdown with given starting guess lambda.
                                    Otherwise, if L' (0+) <= 0, then mu = 0.

    4. lambda <= 0, L'(lambda-)  > 0: If L' (0+) > 0, then mu > 0. Call napup
                                    with lambda = 0 as starting guess.  If
                                    L' (0-) < 0, then lambda < mu < 0.  Call
                                    napup with given starting guess lambda.
                                    Otherwise, if L' (0-) >= 0, then mu = 0.

    By the "free set" we mean those i for which 0 < x_i (lambda) < 1.  The
    total time taken by napsack is O (n + h log n), where n is the size of y,
    h is the number of times an element of x (lambda) moves off the boundary
    into the free set (entries between zero and one) plus the number of times
    elements move from the free set to the opposite boundary.  A heap is used
    to hold the entries in the boundary and in the free set.  If the slope
    vanishes at either the starting lambda or at lambda = 0, then no heap is
    constructed, and the time is just O (n).

    If we have a guess for which components of x will be free at the optimal
    solution, then we can obtain a good guess for the starting lambda by
    setting the slope of the dual function to zero and solving for lambda.  If
    FreeSet_status is not NULL, then the FreeSet_status array is used to
    compute a starting guess for lambda based on the estimated free indices.
    Note that FreeSet_status is an INPUT array, it is not modified by this
    routine.
   ========================================================================== */

#include "Mongoose_Debug.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Logger.hpp"
#include "Mongoose_QPNapDown.hpp"
#include "Mongoose_QPNapUp.hpp"

#include <cfloat>

namespace Mongoose
{

#ifndef NDEBUG
void checkatx(double *x, double *a, Int n, double lo, double hi, double tol)
{
    double atx = 0.;
    int ok     = 1;
    for (Int k = 0; k < n; k++)
    {
        if (x[k] < 0.)
        {
            ok = 0;
            PR(("x [%ld] = %g < 0!\n", k, x[k]));
        }
        if (x[k] > 1.)
        {
            ok = 0;
            PR(("x [%ld] = %g > 1!\n", k, x[k]));
        }
        if (a != NULL)
        {
            double ak = (a) ? a[k] : 1;
            PR(("a'x = %g * %g = %g\n", ak, x[k], ak * x[k]));
            atx += ak * x[k];
        }
        else
        {
            PR(("a'x = %g * %g = %g\n", 1.0, x[k], x[k]));
            atx += x[k];
        }
    }
    if (atx < lo - tol)
    {
        ok = 0;
    }
    if (atx > hi + tol)
    {
        ok = 0;
    }
    if (!ok)
    {
        PR(("tol = %g\n", tol));
        PR(("napsack error! lo %g a'x %g hi %g\n", lo, atx, hi));
        FFLUSH;
        ASSERT(0);
    }
}
#endif

double QPNapsack    /* return the final lambda */
    (double *x,     /* holds y on input, and the solution x on output */
     Int n,         /* size of x, constraint lo <= a'x <= hi */
     double lo,     /* partition lower bound */
     double hi,     /* partition upper bound */
     double *Gw,    /* vector of nodal weights */
     double Lambda, /* initial guess for lambda */
     const Int *FreeSet_status,
     /* FreeSet_status [i] = +1,-1, or 0 on input,
        for 3 cases: x_i =1,0, or 0< x_i< 1.  Not modified. */
     double *w,  /* work array of size n   */
     Int *heap1, /* work array of size n+1 */
     Int *heap2, /* work array of size n+1 */
     double tol  /* Gradient projection tolerance */
    )
{
    (void)tol; // unused variable except during debug
    double lambda = Lambda;
    PR(("QPNapsack start [\n"));

    /* ---------------------------------------------------------------------- */
    /* compute starting guess if FreeSet_status is provided and lambda != 0 */
    /* ---------------------------------------------------------------------- */

    if ((FreeSet_status != NULL) && (lambda != 0))
    {
        double asum  = (lambda > 0 ? -hi : -lo);
        double a2sum = 0.;

        for (Int k = 0; k < n; k++)
        {
            if (FreeSet_status[k] == 1)
            {
                asum += (Gw) ? Gw[k] : 1;
            }
            else if (FreeSet_status[k] == 0)
            {
                double ai = (Gw) ? Gw[k] : 1;
                asum += x[k] * ai;
                a2sum += ai * ai;
            }
        }

        if (a2sum != 0.)
            lambda = asum / a2sum;
    }

    /* ---------------------------------------------------------------------- */
    /* compute the initial slope */
    /* ---------------------------------------------------------------------- */

    double slope = 0;
    for (Int k = 0; k < n; k++)
    {
        double xi = x[k] - ((Gw) ? Gw[k] : 1) * lambda;
        if (xi >= 1.)
        {
            slope += ((Gw) ? Gw[k] : 1);
        }
        else if (xi > 0.)
        {
            slope += ((Gw) ? Gw[k] : 1) * xi;
        }
    }
    PR(("slope %g lo %g hi %g\n", slope, lo, hi));

    /* remember: must still adjust slope by "-hi" or "-lo" for its final value
     */

    if ((lambda >= 0.) && (slope >= hi)) /* case 1 */
    {
        if (slope > hi)
        {
            PR(("napsack case 1 up\n"));
            lambda = QPNapUp(x, n, lambda, Gw, hi, w, heap1, heap2);
            lambda = std::max(0., lambda);
        }
        else
        {
            PR(("napsack case 1 nothing\n"));
        }
    }
    else if ((lambda <= 0.) && (slope <= lo)) /* case 2 */
    {
        if (slope < lo)
        {
            PR(("napsack case 2 down\n"));
            lambda = QPNapDown(x, n, lambda, Gw, lo, w, heap1, heap2);
            lambda = std::min(lambda, 0.);
        }
        else
        {
            PR(("napsack case 2 nothing\n"));
        }
    }
    else /* case 3 or 4 */
    {
        if (lambda != 0.)
        {
            double slope0 = 0.;
            for (Int k = 0; k < n; k++)
            {
                double xi = x[k];
                if (xi >= 1.)
                {
                    slope0 += ((Gw) ? Gw[k] : 1);
                }
                else if (xi > 0.)
                {
                    slope0 += ((Gw) ? Gw[k] : 1) * xi;
                }
            }

            if ((lambda >= 0) && (slope < hi)) /* case 3 */
            {
                if (slope0 < lo)
                {
                    PR(("napsack case 3a down\n"));
                    lambda = 0.;
                    lambda = QPNapDown(x, n, lambda, Gw, lo, w, heap1, heap2);
                    if (lambda > 0.)
                    {
                        lambda = 0.;
                    }
                }
                else if (slope0 > hi)
                {
                    PR(("napsack case 3b down\n"));
                    lambda = QPNapDown(x, n, lambda, Gw, hi, w, heap1, heap2);
                    if (lambda < 0.)
                        lambda = 0.;
                }
                else
                {
                    PR(("napsack case 3c nothing\n"));
                    lambda = 0.;
                }
            }
            else /* ( (lambda <= 0) && (slope > lo) )  case 4 */
            {
                if (slope0 > hi)
                {
                    PR(("napsack case 4a up\n"));
                    lambda = 0.;
                    lambda = QPNapUp(x, n, lambda, Gw, hi, w, heap1, heap2);
                    lambda = std::max(lambda, 0.);
                }
                else if (slope0 < lo)
                {
                    PR(("napsack case 4b up\n"));
                    lambda = QPNapUp(x, n, lambda, Gw, lo, w, heap1, heap2);
                    lambda = std::min(0., lambda);
                }
                else
                {
                    PR(("napsack case 4c nothing\n"));
                    lambda = 0.;
                }
            }
        }
        else /* lambda == 0 */
        {
            if (slope < hi) /* case 3 */
            {
                if (slope < lo)
                {
                    PR(("napsack case 3d down\n"));
                    lambda = QPNapDown(x, n, lambda, Gw, lo, w, heap1, heap2);
                    lambda = std::min(0., lambda);
                }
                else
                {
                    PR(("napsack case 3e nothing\n"));
                }
            }
            else /* ( slope > lo )                    case 4 */
            {
                if (slope > hi)
                {
                    PR(("napsack case 4d up\n"));
                    lambda = QPNapUp(x, n, lambda, Gw, hi, w, heap1, heap2);
                    lambda = std::max(lambda, 0.);
                }
                else
                {
                    PR(("napsack case 4e nothing\n"));
                }
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* replace x by x (lambda) */
    /* ---------------------------------------------------------------------- */

    PR(("lambda %g\n", lambda));
    double atx    = 0;
    Int last_move = 0;
    for (Int k = 0; k < n; k++)
    {
        double xi = x[k] - ((Gw) ? Gw[k] : 1) * lambda;

        if (xi < 0)
        {
            x[k] = 0;
        }
        else if (xi > 1)
        {
            x[k] = 1;
        }
        else
        {
            x[k]      = xi;
            last_move = k;
        }

        double newatx = atx + ((Gw) ? Gw[k] : 1) * x[k];

        // Correction step if we go too far
        if (newatx > hi)
        {
            double diff = hi - atx - FLT_MIN;
            // Need diff = Gw[k] * x[k], so...
            x[k]   = diff / ((Gw) ? Gw[k] : 1);
            newatx = atx + ((Gw) ? Gw[k] : 1) * x[k];
        }
        atx = newatx;
    }

    // Correction step if we didn't go far enough
    for (Int kk = 0; kk < n && atx < lo; kk++)
    {
        Int k = last_move;
        atx -= ((Gw) ? Gw[k] : 1) * x[k];
        double diff = lo - atx;
        // Need diff = Gw[k] * x[k], so...
        x[k] = std::min(1., diff / ((Gw) ? Gw[k] : 1));
        atx += ((Gw) ? Gw[k] : 1) * x[k];
        last_move = (k + 1) % n;
    }

#ifndef NDEBUG
    // Define check tolerance by lambda values
    double atx_tol = log10(std::max(fabs(lambda), fabs(Lambda))
                           / (1e-9 + std::min(fabs(lambda), fabs(Lambda))));
    atx_tol        = std::max(atx_tol, tol);

    checkatx(x, Gw, n, lo, hi, atx_tol);
#endif

    PR(("QPNapsack done ]\n"));

    return lambda;
}

} // end namespace Mongoose
