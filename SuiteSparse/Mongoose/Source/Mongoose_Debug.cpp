/* ========================================================================== */
/* === Source/Mongoose_Debug.cpp ============================================ */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_Debug.hpp"

namespace Mongoose
{

#ifndef NDEBUG

// print a CSparse matrix
void print(cs *G)
{
    ASSERT(G);
    Int *Gp    = G->p;
    Int *Gi    = G->i;
    double *Gx = G->x;
    for (Int j = 0; j < G->n; j++)
    {
        for (int p = Gp[j]; p < Gp[j + 1]; p++)
        {
            PR(("G(%ld,%ld) = %g\n", Gi[p], j, (Gx) ? Gx[p] : 1));
        }
    }
}

// print a Mongoose graph
void print(Graph *G)
{
    ASSERT(G);
    Int *Gp    = G->p;
    Int *Gi    = G->i;
    double *Gx = G->x;
    for (Int j = 0; j < G->n; j++)
    {
        for (int p = Gp[j]; p < Gp[j + 1]; p++)
        {
            PR(("G(%ld,%ld) = %g\n", Gi[p], j, (Gx) ? Gx[p] : 1));
        }
    }
}

//------------------------------------------------------------------------------
// QPcheckCom
//------------------------------------------------------------------------------

// Check that the QPcom data structure is consistent and cost decreases

#define ERROR                                                                  \
    {                                                                          \
        FFLUSH;                                                                \
        ASSERT(0);                                                             \
    }

void QPcheckCom(EdgeCutProblem *G, const EdgeCut_Options *O, QPDelta *QP, bool check_b,
                Int nFreeSet, // use this instead of QP->nFreeSet
                double b      // use this instead of QP->b
)
{
    Int i, j, k, l;
    double s, t;

    ASSERT(G);
    ASSERT(O);
    ASSERT(QP);

    //--- FreeSet
    //  Int nFreeSet = QP->nFreeSet ;  /* number of i such that 0 < x_i < 1 */
    Int *FreeSet_list   = QP->FreeSet_list; /* list for free indices */
    Int *FreeSet_status = QP->FreeSet_status;
    /* FreeSet_status [i] = +1, -1, or 0 if x_i = 1, 0, or 0 < x_i < 1*/
    //---

    double *x = QP->x; /* current estimate of solution */

    /* problem specification */
    Int n      = G->n; /* problem dimension */
    double *Ex = G->x; /* numerical values for edge weights */
    Int *Ei    = G->i; /* adjacent vertices for each vertex */
    Int *Ep    = G->p; /* points into Ex or Ei */
    double *a  = G->w; /* a'x = b, lo <= b <= hi */

    double lo    = QP->lo;
    double hi    = QP->hi;
    double *D    = QP->D;        /* diagonal of quadratic */
    double *grad = QP->gradient; /* gradient at current x */
    double tol   = std::max(log10(O->gradproj_tolerance * G->worstCaseRatio),
                          O->gradproj_tolerance);

    // get workspace
    Int *w0       = (Int *)calloc(n + 1, sizeof(Int));          // [
    double *gtemp = (double *)malloc((n + 1) * sizeof(double)); // [

    ASSERT(w0);
    ASSERT(gtemp);

    /* check that lo <= hi */
    if (lo > hi)
    {
        PR(("lo %e > hi %e\n", lo, hi));
        ERROR;
    }

    /* check feasibility */
    if (a == NULL)
    {
        // a is implicitly all 1's
        s = (double)n;
        t = 0.;
    }
    else
    {
        s = 0.;
        t = 0.;
        for (j = 0; j < n; j++)
        {
            if (a[j] > 0)
                s += a[j];
            else
                t += a[j];
        }
    }

    if (s < lo)
    {
        PR(("lo %e > largest possible value %e\n", lo, s));
        ERROR;
    }
    if (t > hi)
    {
        PR(("hi %e < smallest possible value %e\n", hi, t));
        ERROR;
    }

    Int *ix = FreeSet_status;

    /* check that nFreeSet = number of zeroes in ix, and ix agrees with x */
    i = 0;
    for (j = 0; j < n; j++)
    {
        if (ix[j] > 1)
        {
            PR(("ix [%ld] = %ld (> 1)\n", j, ix[j]));
            ERROR;
        }
        else if (ix[j] < -1)
        {
            PR(("ix [%ld] = %ld (< -1)\n", j, ix[j]));
            ERROR;
        }
        else if (ix[j] == 0)
            i++;
        k = 0;
        if (ix[j] == 1)
        {
            if (x[j] != 1.)
                k = 1;
        }
        else if (ix[j] == -1)
        {
            if (x[j] != 0.)
                k = 1;
        }
        if (k)
        {
            PR(("ix [%ld] = %ld while x = %e\n", j, ix[j], x[j]));
            ERROR;
        }
        if ((x[j] > 1 + tol) || (x[j] < -tol))
        {
            PR(("x [%ld] = %e outside range [0, 1]", j, x[j]));
            ERROR;
        }
    }

    PR(("i %ld nFreeSet %ld\n", i, nFreeSet));

    if (i != nFreeSet)
    {
        PR(("free indices in ix is %ld, nFreeSet = %ld\n", i, nFreeSet));
        ERROR;
    }

    /* check that FreeSet is valid */
    for (Int ifree = 0; ifree < nFreeSet; ifree++)
    {
        i = FreeSet_list[ifree];
        if ((i < 0) || (i > n))
        {
            PR(("FreeSet_list [%ld] = %ld, out of range [0, %ld]\n", ifree, i,
                n));
            ERROR;
        }
        if (w0[i] != 0)
        {
            PR(("FreeSet_list [%ld] = %ld, repeats\n", ifree, i));
            ERROR;
        }
        if (ix[i] != 0)
        {
            PR(("FreeSet_list [%ld] = %ld, is not free\n", ifree, i));
            ERROR;
        }
        w0[i] = 1;
    }

    for (Int ifree = 0; ifree < nFreeSet; ifree++)
    {
        i     = FreeSet_list[ifree];
        w0[i] = 0;
    }

    /* check that b is correct */

    s = 0.;
    if (a == NULL)
        for (j = 0; j < n; j++)
            s += x[j];
    else
        for (j = 0; j < n; j++)
            s += x[j] * a[j];
    // PR (("CHECK BOUNDS: lo %g s=a'x %g hi %g\n", lo, s, hi)) ;
    if (check_b)
    {
        if (fabs(b - s) > tol)
        {
            PR(("QP->b = %e while a'x = %e\n", b, s));
            ERROR;
        }
    }
    if (s < lo - tol)
    {
        PR(("a'x TOO LO: a'x = %e < lo = %e\n", s, lo));
        ERROR;
    }
    if (s > hi + tol)
    {
        PR(("a'x TOO HI: a'x = %e > hi = %e\n", s, hi));
        ERROR;
    }

    /* check that grad is correct */

    for (j = 0; j < n; j++)
        gtemp[j] = (.5 - x[j]) * D[j];
    double newcost = 0.;
    if (Ex == NULL)
    {
        for (j = 0; j < n; j++)
        {
            s = .5 - x[j];
            t = 0.;
            for (k = Ep[j]; k < Ep[j + 1]; k++)
            {
                gtemp[Ei[k]] += s;
                t += x[Ei[k]];
            }
            newcost += (t + x[j] * D[j]) * (1. - x[j]);
        }
    }
    else
    {
        for (j = 0; j < n; j++)
        {
            s = .5 - x[j];
            t = 0.;
            for (k = Ep[j]; k < Ep[j + 1]; k++)
            {
                /* Ex is not NULL */
                gtemp[Ei[k]] += s * Ex[k];
                t += Ex[k] * x[Ei[k]];
            }
            newcost += (t + x[j] * D[j]) * (1. - x[j]);
        }
    }
    s = 0.;
    for (j = 0; j < n; j++)
        s = std::max(s, fabs(gtemp[j] - grad[j]));
    if (s > tol)
    {
        PR(("error (%e) in grad: current grad, true grad, x:\n", s));
        for (j = 0; j < n; j++)
        {
            double ack = fabs(gtemp[j] - grad[j]);
            PR(("j: %5ld grad: %15.6e gtemp: %15.6e err: %15.6e "
                "x: %15.6e",
                j, grad[j], gtemp[j], ack, x[j]));
            if (ack > tol)
                PR((" ACK!"));
            PR(("\n"));
        }
        PR(("tol = %g\n", tol));
        ERROR;
    }

    /* check that cost decreases */

    if (newcost / QP->check_cost > (1 + tol))
    {
        PR(("cost increases, old %30.15e new %30.15e\n", QP->check_cost,
            newcost));
        ERROR;
    }
    QP->check_cost = newcost;
    PR(("cost: %30.15e\n", newcost));

    free(gtemp); // ]
    free(w0);    // ]
}

//------------------------------------------------------------------------------
// FreeSet_dump
//------------------------------------------------------------------------------

void FreeSet_dump(const char *where, Int n, Int *FreeSet_list, Int nFreeSet,
                  Int *FreeSet_status, Int verbose, double *x)
{
    Int death = 0;

    if (verbose)
    {
        PR(("\ndump FreeSet (%s): nFreeSet %ld n %ld jfirst %ld\n", where,
            nFreeSet, n, (nFreeSet == 0) ? -1 : FreeSet_list[0]));
    }

    for (Int kfree = 0; kfree < nFreeSet; kfree++)
    {
        Int k = FreeSet_list[kfree];
        if (verbose)
        {
            PR(("    k %ld \n", k));
            ASSERT(k >= 0 && k < n);
            PR((" FreeSet_status %ld\n", FreeSet_status[k]));
            ASSERT(FreeSet_status[k] == 0);
            if (x != NULL)
            {
                PR((" x: %g\n", x[k]));
            }
            PR(("\n"));
        }
    }

    Int nFree2 = 0;
    Int nHi    = 0;
    Int nLo    = 0;
    for (Int j = 0; j < n; j++)
    {
        FFLUSH;
        if (FreeSet_status[j] == 0)
        {
            // note that in rare cases, x[j] can be 0 or 1 yet
            // still be in the FreeSet.
            if (x != NULL)
                ASSERT(-1E-8 <= x[j] && x[j] <= 1. + 1E-8);
            nFree2++;
        }
        else if (FreeSet_status[j] == 1)
        {
            if (x != NULL)
                ASSERT(x[j] >= 1.);
            nHi++;
        }
        else if (FreeSet_status[j] == -1)
        {
            if (x != NULL)
                ASSERT(x[j] <= 0.);
            nLo++;
        }
        else
        {
            ASSERT(0);
        }
    }
    if (verbose)
    {
        PR(("    # that have FreeSet_status of zero: %ld\n", nFree2));
        PR(("    # that have FreeSet_status of one:  %ld\n", nHi));
        PR(("    # that have FreeSet_status of -1    %ld\n", nLo));
    }
    if (nFreeSet != nFree2)
    {
        PR(("ERROR nFree2 (%ld) nFreeSet %ld\n", nFree2, nFreeSet));
        ASSERT(0);
    }
    PR(("bye\n"));
}
#endif

} // end namespace Mongoose
