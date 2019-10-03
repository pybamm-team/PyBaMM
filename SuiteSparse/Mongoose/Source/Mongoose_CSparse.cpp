/* ========================================================================== */
/* === Source/Mongoose_CSparse.cpp ========================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

/**
 * Fundamental sparse matrix operations.
 *
 * A subset of the CSparse library is used for its sparse matrix data
 * structure and efficient fundamental matrix operations, such as adding,
 * transposing, and converting from triplet to CSC form.
 */
#include "Mongoose_CSparse.hpp"
#include "Mongoose_Debug.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Logger.hpp"

namespace Mongoose
{

double cs_cumsum(csi *p, csi *c, csi n);
csi cs_scatter(const cs *A, csi j, double beta, csi *w, double *x, csi mark,
               cs *C, csi nz);
cs *cs_done(cs *C, void *w, void *x, csi ok);

/**
 * C = A'
 *
 * @details Given a CSparse matrix @p A, cs_transpose returns a new matrix that
 * is the transpose of @p A. In other words, C(j,i) = A(i,j).
 * The second parameter, @p values, allows for the copying of the A->x array.
 * If @p values = 0, the A->x array is ignored, and C->x is left NULL. If
 * @p values != 0, then the A->x array is copied (transposed) to C->x.
 *
 * @code
 * cs *C = cs_transpose(A, 0); // C is A' but C->x = NULL
 * cs *D = cs_transpose(A, 1); // D is A' and C->x is transpose of A->x
 * @endcode
 *
 * @param A Matrix to be transposed
 * @param values 1 to transpose the values A->x, 0 to ignore
 * @return A transposed version of A
 * @note Allocates memory for the returned matrix, but frees on error.
 */
cs *cs_transpose(const cs *A, csi values)
{
    csi p, q, j, *Cp, *Ci, n, m, *Ap, *Ai, *w;
    double *Cx, *Ax;
    cs *C;
    ASSERT(A != NULL);
    ASSERT(CS_CSC(A));
    m  = A->m;
    n  = A->n;
    Ap = A->p;
    Ai = A->i;
    Ax = A->x;
    C  = cs_spalloc(n, m, Ap[n], values && Ax, 0); /* allocate result */
    w  = (csi *)SuiteSparse_calloc(static_cast<size_t>(m),
                                  sizeof(csi)); /* get workspace */
    if (!C || !w)
        return (cs_done(C, w, NULL, 0)); /* out of memory */

    Cp = C->p;
    Ci = C->i;
    Cx = C->x;
    for (p = 0; p < Ap[n]; p++)
        w[Ai[p]]++;      /* row counts */
    cs_cumsum(Cp, w, m); /* row pointers */
    for (j = 0; j < n; j++)
    {
        for (p = Ap[j]; p < Ap[j + 1]; p++)
        {
            Ci[q = w[Ai[p]]++] = j; /* place A(i,j) as entry C(j,i) */
            if (Cx)
                Cx[q] = Ax[p];
        }
    }
    return (cs_done(C, w, NULL, 1)); /* success; release w and return C */
}

/**
 * C = alpha*A + beta*B
 *
 * @details Given two CSparse matrices @p A and @p B, and scaling constants
 * @p alpha and @p beta, cs_add returns a new matrix such that
 * C = alpha*A + beta*B. In other words, C(i,j) = alpha*A(i,j) + beta*B(i,j).
 *
 * @code
 * cs *C = cs_add(A, B, 1, 1);     // C = A + B
 * cs *D = cs_add(A, B, 0.5, 0.5); // C = 0.5*A + 0.5*B
 * cs *E = cs_add(A, B, 1, 0);     // C = A
 * @endcode
 *
 * @param A Matrix to be added to B.
 * @param B Matrix to be added to A
 * @param alpha Scalar (double) value to scale all elements of A matrix by
 * @param beta Scalar (double) value to scale all elements of B matrix by
 * @return A matrix such that C = alpha*A + beta*B
 * @note Allocates memory for the returned matrix, but frees on error.
 */
cs *cs_add(const cs *A, const cs *B, double alpha, double beta)
{
    csi p, j, nz = 0, anz, *Cp, *Ci, *Bp, m, n, bnz, *w, values;
    double *x, *Bx, *Cx;
    cs *C;
    ASSERT(A != NULL);
    ASSERT(CS_CSC(A));
    ASSERT(B != NULL);
    ASSERT(CS_CSC(B));
    ASSERT(A->m == B->m);
    ASSERT(A->n == B->n);
    m      = A->m;
    anz    = A->p[A->n];
    n      = B->n;
    Bp     = B->p;
    Bx     = B->x;
    bnz    = Bp[n];
    w      = (csi *)SuiteSparse_calloc(static_cast<size_t>(m),
                                  sizeof(csi)); /* get workspace */
    values = (A->x != NULL) && (Bx != NULL);
    x      = values ? (double *)SuiteSparse_malloc(static_cast<size_t>(m),
                                              sizeof(double))
               : NULL;                          /* get workspace */
    C = cs_spalloc(m, n, anz + bnz, values, 0); /* allocate result*/
    if (!C || !w || (values && !x))
        return (cs_done(C, w, x, 0));
    Cp = C->p;
    Ci = C->i;
    Cx = C->x;
    for (j = 0; j < n; j++)
    {
        Cp[j] = nz; /* column j of C starts here */
        nz    = cs_scatter(A, j, alpha, w, x, j + 1, C, nz); /* alpha*A(:,j)*/
        nz    = cs_scatter(B, j, beta, w, x, j + 1, C, nz);  /* beta*B(:,j) */
        if (values)
            for (p = Cp[j]; p < nz; p++)
                Cx[p] = x[Ci[p]];
    }
    Cp[n] = nz;                   /* finalize the last column of C */
    return (cs_done(C, w, x, 1)); /* success; release workspace, return C */
}

/**
 * C = compressed-column form of a triplet matrix T
 *
 * Given a CSparse matrix @p T in triplet form, cs_compress returns
 * the same matrix in compressed sparse column (CSC) format.
 *
 * A triplet form matrix is a simple listing of the nonzeros in the matrix and
 * their row, column coordinates. For example, the 2x2 identity matrix would be
 * specified with the following triplets in (row, column, value) format:
 * (1, 1, 1) and (2, 2, 1).
 *
 * @param T Matrix in triplet form to be compressed
 * @return A compressed sparse column (CSC) format matrix representing the
 * triplet form matrix T passed in.
 * @note Uses an O(n) workspace w, and frees on error.
 */
cs *cs_compress(const cs *T)
{
    csi m, n, nz, k, *Cp, *Ci, *w, *Ti, *Tj;
    double *Cx, *Tx;
    cs *C;
    ASSERT(CS_TRIPLET(T));
    m  = T->m;
    n  = T->n;
    Ti = T->i;
    Tj = T->p;
    Tx = T->x;
    nz = T->nz;
    C  = cs_spalloc(m, n, nz, Tx != NULL, 0); /* allocate result */
    w  = (csi *)SuiteSparse_calloc(static_cast<size_t>(n),
                                  sizeof(csi)); /* get workspace */
    if (!C || !w)
        return (cs_done(C, w, NULL, 0)); /* out of memory */
    Cp = C->p;
    Ci = C->i;
    Cx = C->x;
    for (k = 0; k < nz; k++)
        w[Tj[k]]++;      /* column counts */
    cs_cumsum(Cp, w, n); /* column pointers */
    for (k = 0; k < nz; k++)
    {
        csi p = w[Tj[k]]++;
        Ci[p] = Ti[k]; /* A(i,j) is the pth entry in C */
        if (Cx)
            Cx[p] = Tx[k];
    }
    return (cs_done(C, w, NULL, 1)); /* success; release w and return C */
}

/**
 * p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1]
 * into c
 *
 * @param p A vector of size @p n to be summed. On return, p is overwritten such
 * that p[i] contains the cumulative sum of p[0..i] that was passed in.
 * @param c A vector of size @p n used as a workspace. On return, c[i] contains
 * the cumulative sum of p[0..i].
 * @param n The dimension of @p p and @p c
 * @return The cumulative sum of c[0..n-1]
 */
double cs_cumsum(csi *p, csi *c, csi n)
{
    csi i, nz = 0;
    double nz2 = 0;
    ASSERT(p != NULL);
    ASSERT(c != NULL);
    for (i = 0; i < n; i++)
    {
        p[i] = nz;
        nz += c[i];
        nz2 += c[i]; /* also in double to avoid csi overflow */
        c[i] = p[i]; /* also copy p[0..n-1] back into c[0..n-1]*/
    }
    p[n] = nz;
    return (nz2); /* return sum (c [0..n-1]) */
}

//-----------------------------------------------------------------------------
// x = x + beta * A(:,j), where x is a dense vector and A(:,j) is sparse
//-----------------------------------------------------------------------------
csi cs_scatter(const cs *A, csi j, double beta, csi *w, double *x, csi mark,
               cs *C, csi nz)
{
    csi p, *Ap, *Ai, *Ci;
    double *Ax;
    ASSERT(CS_CSC(A));
    ASSERT(CS_CSC(C));
    ASSERT(w != NULL);
    Ap = A->p;
    Ai = A->i;
    Ax = A->x;
    Ci = C->i;
    for (p = Ap[j]; p < Ap[j + 1]; p++)
    {
        csi i = Ai[p]; /* A(i,j) is nonzero */
        if (w[i] < mark)
        {
            w[i]     = mark; /* i is new entry in column j */
            Ci[nz++] = i;    /* add i to pattern of C(:,j) */
            if (x)
                x[i] = beta * Ax[p]; /* x(i) = beta*A(i,j) */
        }
        else if (x)
            x[i] += beta * Ax[p]; /* i exists in C(:,j) already */
    }
    return (nz);
}

//-----------------------------------------------------------------------------
// utility functions
//-----------------------------------------------------------------------------
/* allocate a sparse matrix (triplet form or compressed-column form) */
cs *cs_spalloc(csi m, csi n, csi nzmax, csi values, csi triplet)
{
    cs *A
        = (cs *)SuiteSparse_calloc(1, sizeof(cs)); /* allocate the cs struct */
    if (!A)
        return (NULL); /* out of memory */
    A->m     = m;      /* define dimensions and nzmax */
    A->n     = n;
    A->nzmax = nzmax = std::max(nzmax, 1L);
    A->nz            = triplet ? 0 : -1; /* allocate triplet or comp.col */
    A->p             = (csi *)SuiteSparse_malloc(
        static_cast<size_t>(triplet ? nzmax : n + 1), sizeof(csi));
    A->i = (csi *)SuiteSparse_malloc(static_cast<size_t>(nzmax), sizeof(csi));
    A->x = values ? (double *)SuiteSparse_malloc(static_cast<size_t>(nzmax),
                                                 sizeof(double))
                  : NULL;
    return ((!A->p || !A->i || (values && !A->x)) ? cs_spfree(A) : A);
}

/* release a sparse matrix */
cs *cs_spfree(cs *A)
{
    if (!A)
        return (NULL); /* do nothing if A already NULL */
    SuiteSparse_free(A->p);
    SuiteSparse_free(A->i);
    SuiteSparse_free(A->x);
    return (
        (cs *)SuiteSparse_free(A)); /* release the cs struct and return NULL */
}

/* release workspace and return a sparse matrix result */
cs *cs_done(cs *C, void *w, void *x, csi ok)
{
    SuiteSparse_free(w); /* release workspace */
    SuiteSparse_free(x);
    return (ok ? C : cs_spfree(C)); /* return result if OK, else release it */
}

} // end namespace Mongoose
