#include "cs_mex.h"
/* cs_ltsolve: solve an upper triangular system L'*x=b */
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
        mexErrMsgTxt ("Usage: x = cs_ltsolve(L,b)") ;
    }
    if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1]))
    {
#ifndef NCOMPLEX
        cs_cl Lmatrix, *L ;
        cs_complex_t *x ;
        L = cs_cl_mex_get_sparse (&Lmatrix, 1, pargin [0]) ;    /* get L */
        x = cs_cl_mex_get_double (L->n, pargin [1]) ;           /* x = b */
        cs_cl_ltsolve (L, x) ;                                  /* x = L'\x */
        cs_free (L->x) ;
        pargout [0] = cs_cl_mex_put_double (L->n, x) ;          /* return x */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Lmatrix, *L ;
        double *x, *b ;
        L = cs_dl_mex_get_sparse (&Lmatrix, 1, 1, pargin [0]) ; /* get L */
        b = cs_dl_mex_get_double (L->n, pargin [1]) ;           /* get b */
        x = cs_dl_mex_put_double (L->n, b, &(pargout [0])) ;    /* x = b */
        cs_dl_ltsolve (L, x) ;                                  /* x = L'\x */
    }
}
