#include "cs_mex.h"
/* cs_qr: sparse QR factorization */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT m, n, order, *p ;
    if (nargout > 5 || nargin != 1)
    {
        mexErrMsgTxt ("Usage: [V,beta,p,R,q] = cs_qr(A)") ;
    }
    order = (nargout == 5) ? 3 : 0 ;        /* determine ordering */
    m = mxGetM (pargin [0]) ;
    n = mxGetN (pargin [0]) ;
    if (m < n) mexErrMsgTxt ("A must have # rows >= # columns") ;
    if (mxIsComplex (pargin [0]))
    {
#ifndef NCOMPLEX
        cs_cls *S ;
        cs_cln *N ;
        cs_cl Amatrix, *A, *D ;
        A = cs_cl_mex_get_sparse (&Amatrix, 0, pargin [0]) ;    /* get A */
        S = cs_cl_sqr (order, A, 1) ;       /* symbolic QR ordering & analysis*/
        N = cs_cl_qr (A, S) ;               /* numeric QR factorization */
        cs_free (A->x) ;
        if (!N) mexErrMsgTxt ("qr failed") ;
        cs_cl_dropzeros (N->L) ;            /* drop zeros from V and sort */
        D = cs_cl_transpose (N->L, 1) ;
        cs_cl_spfree (N->L) ;
        N->L = cs_cl_transpose (D, 1) ;
        cs_cl_spfree (D) ;
        cs_cl_dropzeros (N->U) ;            /* drop zeros from R and sort */
        D = cs_cl_transpose (N->U, 1) ;
        cs_cl_spfree (N->U) ;
        N->U = cs_cl_transpose (D, 1) ;
        cs_cl_spfree (D) ;
        m = N->L->m ;                               /* m may be larger now */
        p = cs_cl_pinv (S->pinv, m) ;                   /* p = pinv' */
        pargout [0] = cs_cl_mex_put_sparse (&(N->L)) ;  /* return V */
        cs_dl_mex_put_double (n, N->B, &(pargout [1])) ;   /* return beta */
        pargout [2] = cs_dl_mex_put_int (p, m, 1, 1) ;  /* return p */
        pargout [3] = cs_cl_mex_put_sparse (&(N->U)) ;  /* return R */
        pargout [4] = cs_dl_mex_put_int (S->q, n, 1, 0) ;  /* return q */
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
        A = cs_dl_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ; /* get A */
        S = cs_dl_sqr (order, A, 1) ;       /* symbolic QR ordering & analysis*/
        N = cs_dl_qr (A, S) ;               /* numeric QR factorization */
        if (!N) mexErrMsgTxt ("qr failed") ;
        cs_dl_dropzeros (N->L) ;            /* drop zeros from V and sort */
        D = cs_dl_transpose (N->L, 1) ;
        cs_dl_spfree (N->L) ;
        N->L = cs_dl_transpose (D, 1) ;
        cs_dl_spfree (D) ;
        cs_dl_dropzeros (N->U) ;            /* drop zeros from R and sort */
        D = cs_dl_transpose (N->U, 1) ;
        cs_dl_spfree (N->U) ;
        N->U = cs_dl_transpose (D, 1) ;
        cs_dl_spfree (D) ;
        m = N->L->m ;                               /* m may be larger now */
        p = cs_dl_pinv (S->pinv, m) ;                   /* p = pinv' */
        pargout [0] = cs_dl_mex_put_sparse (&(N->L)) ;  /* return V */
        cs_dl_mex_put_double (n, N->B, &(pargout [1])) ;   /* return beta */
        pargout [2] = cs_dl_mex_put_int (p, m, 1, 1) ;  /* return p */
        pargout [3] = cs_dl_mex_put_sparse (&(N->U)) ;  /* return R */
        pargout [4] = cs_dl_mex_put_int (S->q, n, 1, 0) ;  /* return q */
        cs_dl_nfree (N) ;
        cs_dl_sfree (S) ;
    }
}
