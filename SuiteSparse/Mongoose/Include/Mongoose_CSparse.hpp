/* ========================================================================== */
/* === Include/Mongoose_CSparse.hpp ========================================= */
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
 * transposing, and converting from triplet to CSC form.  This version
 * uses the same integer (csi or SuiteSparse_long) as the Int in Mongoose.
 */

// #pragma once
#ifndef MONGOOSE_CSPARSE_HPP
#define MONGOOSE_CSPARSE_HPP

#include "Mongoose_Internal.hpp"
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif

/* same as Int in Mongoose */
typedef SuiteSparse_long csi;

/* CSparse Macros */
#ifndef CS_CSC
#define CS_CSC(A) ((A) && ((A)->nz == -1))
#endif
#ifndef CS_TRIPLET
#define CS_TRIPLET(A) ((A) && ((A)->nz >= 0))
#endif

namespace Mongoose
{

/* --- primary CSparse routines and data structures ------------------------- */
typedef struct cs_sparse /* matrix in compressed-column or triplet form */
{
    csi nzmax; /* maximum number of entries */
    csi m;     /* number of rows */
    csi n;     /* number of columns */
    csi *p;    /* column pointers (size n+1) or col indices (size nzmax) */
    csi *i;    /* row indices, size nzmax */
    double *x; /* numerical values, size nzmax */
    csi nz;    /* # of entries in triplet matrix, -1 for compressed-col */
} cs;

cs *cs_add(const cs *A, const cs *B, double alpha, double beta);
cs *cs_transpose(const cs *A, csi values);

cs *cs_compress(const cs *T);

cs *cs_spalloc(csi m, csi n, csi nzmax, csi values, csi triplet);
cs *cs_spfree(cs *A);

} // end namespace Mongoose

#endif
