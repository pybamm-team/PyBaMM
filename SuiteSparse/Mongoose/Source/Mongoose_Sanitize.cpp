/* ========================================================================== */
/* === Source/Mongoose_Sanitize.cpp ========================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * Mongoose Graph Partitioning Library  Copyright (C) 2017-2018,
 * Scott P. Kolodziej, Nuri S. Yeralan, Timothy A. Davis, William W. Hager
 * Mongoose is licensed under Version 3 of the GNU General Public License.
 * Mongoose is also available under other licenses; contact authors for details.
 * -------------------------------------------------------------------------- */

#include "Mongoose_Sanitize.hpp"
#include "Mongoose_Internal.hpp"

using namespace std;

namespace Mongoose
{

cs *sanitizeMatrix(cs *compressed_A, bool symmetricTriangular,
                   bool makeEdgeWeightsBinary)
{
    cs *cleanMatrix;
    if (symmetricTriangular)
    {
        cleanMatrix = mirrorTriangular(compressed_A);
    }
    else
    {
        cs *A_transpose = cs_transpose(compressed_A, 1);
        if (!A_transpose)
        {
            return NULL;
        }
        cleanMatrix = cs_add(compressed_A, A_transpose, 0.5, 0.5);
        cs_spfree(A_transpose);
    }

    if (!cleanMatrix)
    {
        return NULL;
    }

    removeDiagonal(cleanMatrix);

    cs *cleanMatrix_transpose = cs_transpose(cleanMatrix, 1);
    cs_spfree(cleanMatrix);

    if (!cleanMatrix_transpose)
    {
        return NULL;
    }
    cleanMatrix = cs_transpose(cleanMatrix_transpose, 1);
    cs_spfree(cleanMatrix_transpose);
    if (!cleanMatrix)
    {
        return NULL;
    }

    if (cleanMatrix->x)
    {
        for (Int p = 0; p < cleanMatrix->p[cleanMatrix->n]; p++)
        {
            if (makeEdgeWeightsBinary)
            {
                // Make edge weights binary
                if (cleanMatrix->x[p] != 0)
                {
                    cleanMatrix->x[p] = 1;
                }
            }
            else
            {
                // Force edge weights to be positive
                cleanMatrix->x[p] = fabs(cleanMatrix->x[p]);
            }
        }
    }

    return cleanMatrix;
}

void removeDiagonal(cs *A)
{
    Int n      = A->n;
    Int *Ap    = A->p;
    Int *Ai    = A->i;
    double *Ax = A->x;
    Int nz     = 0;
    Int old_Ap = Ap[0];

    for (Int j = 0; j < n; j++)
    {
        for (Int p = old_Ap; p < Ap[j + 1]; p++)
        {
            if (Ai[p] != j)
            {
                Ai[nz] = Ai[p];
                if (Ax)
                    Ax[nz] = Ax[p];
                nz++;
            }
        }
        old_Ap    = Ap[j + 1];
        Ap[j + 1] = nz;
    }
}

// Requires A to be a triangular matrix with no diagonal.
cs *mirrorTriangular(cs *A)
{
    if (!A)
        return NULL;
    Int A_n  = A->n;
    Int A_nz = A->p[A_n];
    Int B_nz = 2 * A_nz;

    bool values = (A->x != NULL);

    // allocate B in triplet form, with values Bx if A has values
    cs *B = cs_spalloc(A_n, A_n, B_nz, values, 1);
    if (!B)
        return NULL;

    Int *Ap    = A->p;
    Int *Ai    = A->i;
    double *Ax = A->x;
    Int *Bj    = B->p;
    Int *Bi    = B->i;
    double *Bx = B->x;
    Int nz     = 0;

    for (Int j = 0; j < A_n; j++)
    {
        for (Int p = Ap[j]; p < Ap[j + 1]; p++)
        {
            Bi[nz] = Ai[p];
            Bj[nz] = j;
            if (values)
                Bx[nz] = Ax[p];
            nz++;
            Bi[nz] = j;
            Bj[nz] = Ai[p];
            if (values)
                Bx[nz] = Ax[p];
            nz++;
        }
    }
    B->nz = nz;
    cs *C = cs_compress(B);
    cs_spfree(B);

    return C;
}

} // end namespace Mongoose
