#include "cs_mex.h"
/* cs_utsolve: solve a lower triangular system U'*x=b */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: x = cs_utsolve(U,b)") ;
    }
    if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1]))
    {
#ifndef NCOMPLEX
        cs_cl Umatrix, *U ;
        cs_complex_t *x ;
        U = cs_cl_mex_get_sparse (&Umatrix, 1, pargin [0]) ;    /* get U */
        x = cs_cl_mex_get_double (U->n, pargin [1]) ;           /* x = b */
        cs_cl_utsolve (U, x) ;                                  /* x = U'\x */
        cs_free (U->x) ;
        pargout [0] = cs_cl_mex_put_double (U->n, x) ;          /* return x */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Umatrix, *U ;
        double *x, *b ;
        U = cs_dl_mex_get_sparse (&Umatrix, 1, 1, pargin [0]) ; /* get U */
        b = cs_dl_mex_get_double (U->n, pargin [1]) ;           /* get b */
        x = cs_dl_mex_put_double (U->n, b, &(pargout [0])) ;    /* x = b */
        cs_dl_utsolve (U, x) ;                                  /* x = U'\x */
    }
}
