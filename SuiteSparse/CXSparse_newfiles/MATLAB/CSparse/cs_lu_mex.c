#include "cs_mex.h"
/* cs_lu: sparse LU factorization, with optional fill-reducing ordering */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT n, order, *p ;
    double tol ;
    if (nargout > 4 || nargin > 3 || nargin < 1)
    {
        mexErrMsgTxt ("Usage: [L,U,p,q] = cs_lu (A,tol)") ;
    }
    if (nargin == 2)                        /* determine tol and ordering */
    {
        tol = mxGetScalar (pargin [1]) ;
        order = (nargout == 4) ? 1 : 0 ;    /* amd (A+A'), or natural */
    }
    else
    {
        tol = 1 ;
        order = (nargout == 4) ? 2 : 0 ;    /* amd(S'*S) w/dense rows or I */
    }
    if (mxIsComplex (pargin [0]))
    {
#ifndef NCOMPLEX
        cs_cls *S ;
        cs_cln *N ;
        cs_cl Amatrix, *A, *D ;
        A = cs_cl_mex_get_sparse (&Amatrix, 1, pargin [0]) ;    /* get A */
        n = A->n ;
        S = cs_cl_sqr (order, A, 0) ;       /* symbolic ordering, no QR bound */
        N = cs_cl_lu (A, S, tol) ;          /* numeric factorization */
        if (!N) mexErrMsgTxt ("cs_lu failed (singular, or out of memory)") ;
        cs_cl_free (A->x) ;                 /* complex copy no longer needed */
        cs_cl_dropzeros (N->L) ;            /* drop zeros from L and sort it */
        D = cs_cl_transpose (N->L, 1) ;
        cs_cl_spfree (N->L) ;
        N->L = cs_cl_transpose (D, 1) ;
        cs_cl_spfree (D) ;
        cs_cl_dropzeros (N->U) ;            /* drop zeros from U and sort it */
        D = cs_cl_transpose (N->U, 1) ;
        cs_cl_spfree (N->U) ;
        N->U = cs_cl_transpose (D, 1) ;
        cs_cl_spfree (D) ;
        p = cs_cl_pinv (N->pinv, n) ;                       /* p=pinv' */
        pargout [0] = cs_cl_mex_put_sparse (&(N->L)) ;      /* return L */
        pargout [1] = cs_cl_mex_put_sparse (&(N->U)) ;      /* return U */
        pargout [2] = cs_dl_mex_put_int (p, n, 1, 1) ;      /* return p */
        /* return Q */
        if (nargout == 4) pargout [3] = cs_dl_mex_put_int (S->q, n, 1, 0) ;
        cs_cl_nfree (N) ;
        cs_cl_sfree (S) ;
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dls *S ;
        cs_dln *N ;
        cs_dl Amatrix, *A, *D ;
        A = cs_dl_mex_get_sparse (&Amatrix, 1, 1, pargin [0]) ; /* get A */
        n = A->n ;
        S = cs_dl_sqr (order, A, 0) ;       /* symbolic ordering, no QR bound */
        N = cs_dl_lu (A, S, tol) ;          /* numeric factorization */
        if (!N) mexErrMsgTxt ("cs_lu failed (singular, or out of memory)") ;
        cs_dl_dropzeros (N->L) ;            /* drop zeros from L and sort it */
        D = cs_dl_transpose (N->L, 1) ;
        cs_dl_spfree (N->L) ;
        N->L = cs_dl_transpose (D, 1) ;
        cs_dl_spfree (D) ;
        cs_dl_dropzeros (N->U) ;            /* drop zeros from U and sort it */
        D = cs_dl_transpose (N->U, 1) ;
        cs_dl_spfree (N->U) ;
        N->U = cs_dl_transpose (D, 1) ;
        cs_dl_spfree (D) ;
        p = cs_dl_pinv (N->pinv, n) ;                       /* p=pinv' */
        pargout [0] = cs_dl_mex_put_sparse (&(N->L)) ;      /* return L */
        pargout [1] = cs_dl_mex_put_sparse (&(N->U)) ;      /* return U */
        pargout [2] = cs_dl_mex_put_int (p, n, 1, 1) ;      /* return p */
        /* return Q */
        if (nargout == 4) pargout [3] = cs_dl_mex_put_int (S->q, n, 1, 0) ;
        cs_dl_nfree (N) ;
        cs_dl_sfree (S) ;
    }
}
