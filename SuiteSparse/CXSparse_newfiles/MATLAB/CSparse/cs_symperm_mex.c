#include "cs_mex.h"
/* cs_symperm: symmetric permutation of a symmetric sparse matrix. */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT ignore, n, *P, *Pinv ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: C = cs_symperm(A,p)") ;
    }
    if (mxIsComplex (pargin [0]))
    {
#ifndef NCOMPLEX
        cs_cl Amatrix, *A, *C, *D ;
        A = cs_cl_mex_get_sparse (&Amatrix, 1, pargin [0]) ;
        n = A->n ;
        P = cs_dl_mex_get_int (n, pargin [1], &ignore, 1) ;     /* get P */
        Pinv = cs_cl_pinv (P, n) ;                              /* P=Pinv' */
        C = cs_cl_symperm (A, Pinv, 1) ;                        /* C = A(p,p) */
        D = cs_cl_transpose (C, 1) ;                            /* sort C */
        cs_cl_spfree (C) ;
        C = cs_cl_transpose (D, 1) ;
        cs_cl_spfree (D) ;
        pargout [0] = cs_cl_mex_put_sparse (&C) ;               /* return C */
        cs_free (P) ;
        cs_free (Pinv) ;
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Amatrix, *A, *C, *D ;
        A = cs_dl_mex_get_sparse (&Amatrix, 1, 1, pargin [0]) ;
        n = A->n ;
        P = cs_dl_mex_get_int (n, pargin [1], &ignore, 1) ;     /* get P */
        Pinv = cs_dl_pinv (P, n) ;                              /* P=Pinv' */
        C = cs_dl_symperm (A, Pinv, 1) ;                        /* C = A(p,p) */
        D = cs_dl_transpose (C, 1) ;                            /* sort C */
        cs_dl_spfree (C) ;
        C = cs_dl_transpose (D, 1) ;
        cs_dl_spfree (D) ;
        pargout [0] = cs_dl_mex_put_sparse (&C) ;               /* return C */
        cs_free (P) ;
        cs_free (Pinv) ;
    }
}
