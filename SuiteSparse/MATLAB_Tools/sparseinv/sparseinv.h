#ifndef _SPARSEINV_H_
#define _SPARSEINV_H_
#include <stddef.h>
#ifdef MATLAB_MEX_FILE
#include "mex.h"
#define Int mwSignedIndex
#else
#define Int ptrdiff_t
#endif

Int sparseinv       /* returns 1 if OK, 0 if failure */
(
    /* inputs, not modified on output: */
    Int n,          /* L, U, D, and Z are n-by-n */

    Int *Lp,        /* L is sparse, lower triangular, stored by column */
    Int *Li,        /* the row indices of L must be sorted */
    double *Lx,     /* diagonal of L, if present, is ignored */

    double *d,      /* diagonal of D, of size n */

    Int *Up,        /* U is sparse, upper triangular, stored by row */
    Int *Uj,        /* the column indices of U need not be sorted */
    double *Ux,     /* diagonal of U, if present, is ignored */

    Int *Zp,        /* Z is sparse, stored by column */
    Int *Zi,        /* the row indices of Z must be sorted */

    /* output, not defined on input: */ 
    double *Zx,

    /* workspace: */
    double *z,      /* size n, zero on input, restored as such on output */
    Int *Zdiagp,    /* size n */
    Int *Lmunch     /* size n */
) ;

#endif
