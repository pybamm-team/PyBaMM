#include "cs_mex.h"
/* cs_usolve: x=U\b.  U must be sparse and upper triangular.  b must be a
 * full or sparse vector.  x is full or sparse, depending on b.
 *
 * Time taken is O(flop count), which may be less than n if b is sparse,
 * depending on U and b.
 *
 * This function works with MATLAB 7.2, but is not perfectly compatible with
 * the requirements of a MATLAB mexFunction when b is sparse.  X is returned
 * as an unsorted sparse vector.  Also, this mexFunction temporarily modifies
 * its input, U, by modifying U->p (in the cs_dfs function) and then restoring
 * it.  This could be corrected by creating a copy of U->p
 * (see cs_dmperm_mex.c), but this would take O(n) time, destroying the
 * O(flop count) time complexity of this function.
 *
 * Note that b cannot be sparse complex.  This function does not support
 * sparse complex U and b because the sparse x=U\b only accesses part of the
 * matrix U.  Converting U from a MATLAB complex matrix to a CXSparse complex
 * matrix requires all of U to be accessed, defeating the purpose of this
 * function.
 *
 * U can be sparse complex, but in that case b must be full real or complex,
 * not sparse.
 */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT top, nz, p, *xi, n ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = cs_usolve(U,b)") ;
    }
    if (mxIsSparse (pargin [1]))
    {
        cs_dl Umatrix, Bmatrix, *U, *B, *X ;
        double *x ;
        if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1]))
        {
            mexErrMsgTxt ("sparse complex case not supported") ;
        }
        U = cs_dl_mex_get_sparse (&Umatrix, 1, 1, pargin [0]) ;/* get U */
        n = U->n ;
        B = cs_dl_mex_get_sparse (&Bmatrix, 0, 1, pargin [1]) ;/* get sparse b*/
        cs_mex_check (0, n, 1, 0, 1, 1, pargin [1]) ;
        xi = cs_dl_malloc (2*n, sizeof (CS_INT)) ;          /* get workspace */
        x  = cs_dl_malloc (n, sizeof (double)) ;
        top = cs_dl_spsolve (U, B, 0, xi, x, NULL, 0) ;     /* x = U\b */
        X = cs_dl_spalloc (n, 1, n-top, 1, 0) ;     /* create sparse x*/
        X->p [0] = 0 ;
        nz = 0 ;
        for (p = top ; p < n ; p++)
        {
            X->i [nz] = xi [p] ;
            X->x [nz++] = x [xi [p]] ;
        }
        X->p [1] = nz ;
        pargout [0] = cs_dl_mex_put_sparse (&X) ;
        cs_free (x) ;
        cs_free (xi) ;
    }
    else if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1]))
    {
#ifndef NCOMPLEX
        cs_cl Umatrix, *U ;
        cs_complex_t *x ;
        U = cs_cl_mex_get_sparse (&Umatrix, 1, pargin [0]) ;    /* get U */
        n = U->n ;
        x = cs_cl_mex_get_double (n, pargin [1]) ;              /* x = b */
        cs_cl_usolve (U, x) ;                                   /* x = U\x */
        cs_free (U->x) ;
        pargout [0] = cs_cl_mex_put_double (n, x) ;             /* return x */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Umatrix, *U ;
        double *x, *b ;
        U = cs_dl_mex_get_sparse (&Umatrix, 1, 1, pargin [0]) ; /* get U */
        n = U->n ;
        b = cs_dl_mex_get_double (n, pargin [1]) ;              /* get b */
        x = cs_dl_mex_put_double (n, b, &(pargout [0])) ;       /* x = b */
        cs_dl_usolve (U, x) ;                                   /* x = U\x */
    }
}
