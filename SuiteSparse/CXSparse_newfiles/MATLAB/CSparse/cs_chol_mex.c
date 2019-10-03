#include "cs_mex.h"
/* cs_chol: sparse Cholesky factorization */
void mexFunction (int nargout, mxArray *pargout [ ], int nargin,
    const mxArray *pargin [ ])
{
    CS_INT order, n, drop, *p ;
    if (nargout > 2 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: [L,p] = cs_chol(A,drop)") ;
    }
    drop = (nargin == 1) ? 1 : mxGetScalar (pargin [1]) ;
    order = (nargout > 1) ? 1 : 0 ;                 /* determine ordering */
    if (mxIsComplex (pargin [0]))
    {
#ifndef NCOMPLEX
        cs_cl Amatrix, *A ;
        cs_cls *S ;
        cs_cln *N ;
        A = cs_cl_mex_get_sparse (&Amatrix, 1, pargin [0]) ;    /* get A */
        n = A->n ;
        S = cs_cl_schol (order, A) ;                /* symbolic Cholesky */
        N = cs_cl_chol (A, S) ;                     /* numeric Cholesky */
        if (!N) mexErrMsgTxt ("cs_chol failed: not positive definite\n") ;
        cs_free (A->x) ;
        if (drop) cs_cl_dropzeros (N->L) ;          /* drop zeros if requested*/
        pargout [0] = cs_cl_mex_put_sparse (&(N->L)) ;      /* return L */
        if (nargout > 1)
        {
            p = cs_cl_pinv (S->pinv, n) ;           /* p=pinv' */
            pargout [1] = cs_dl_mex_put_int (p, n, 1, 1) ; /* return p */
        }
        cs_cl_nfree (N) ;
        cs_cl_sfree (S) ;
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Amatrix, *A ;
        cs_dls *S ;
        cs_dln *N ;
        A = cs_dl_mex_get_sparse (&Amatrix, 1, 1, pargin [0]) ; /* get A */
        n = A->n ;
        S = cs_dl_schol (order, A) ;                /* symbolic Cholesky */
        N = cs_dl_chol (A, S) ;                     /* numeric Cholesky */
        if (!N) mexErrMsgTxt ("cs_chol failed: not positive definite\n") ;
        if (drop) cs_dl_dropzeros (N->L) ;          /* drop zeros if requested*/
        pargout [0] = cs_dl_mex_put_sparse (&(N->L)) ; /* return L */
        if (nargout > 1)
        {
            p = cs_dl_pinv (S->pinv, n) ;                   /* p=pinv' */
            pargout [1] = cs_dl_mex_put_int (p, n, 1, 1) ; /* return p */
        }
        cs_dl_nfree (N) ;
        cs_dl_sfree (S) ;
    }
}
