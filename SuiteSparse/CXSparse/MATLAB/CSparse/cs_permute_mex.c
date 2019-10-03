#include "cs_mex.h"
/* cs_permute: permute a sparse matrix */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT ignore, *P, *Q, *Pinv, m, n ;
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: C = cs_permute(A,p,q)") ;
    }
    m = mxGetM (pargin [0]) ;
    n = mxGetN (pargin [0]) ;
    P = cs_dl_mex_get_int (m, pargin [1], &ignore, 1) ; /* get P */
    Q = cs_dl_mex_get_int (n, pargin [2], &ignore, 1) ; /* get Q */
    Pinv = cs_pinv (P, m) ;                 /* P = Pinv' */
    if (mxIsComplex (pargin [0]))
    {
#ifndef NCOMPLEX
        cs_cl Amatrix, *A, *C, *D ;
        A = cs_cl_mex_get_sparse (&Amatrix, 0, pargin [0]) ;    /* get A */
        C = cs_cl_permute (A, Pinv, Q, 1) ;         /* C = A(p,q) */
        cs_cl_free (A->x) ;
        D = cs_cl_transpose (C, 1) ;    /* sort C via double transpose */
        cs_cl_spfree (C) ;
        C = cs_cl_transpose (D, 1) ;
        cs_cl_spfree (D) ;
        pargout [0] = cs_cl_mex_put_sparse (&C) ;           /* return C */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Amatrix, *A, *C, *D ;
        A = cs_dl_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
        C = cs_dl_permute (A, Pinv, Q, 1) ;         /* C = A(p,q) */
        D = cs_dl_transpose (C, 1) ;    /* sort C via double transpose */
        cs_dl_spfree (C) ;
        C = cs_dl_transpose (D, 1) ;
        cs_dl_spfree (D) ;
        pargout [0] = cs_dl_mex_put_sparse (&C) ;           /* return C */
    }
    cs_free (Pinv) ;
    cs_free (P) ;
    cs_free (Q) ;
}
