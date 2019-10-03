#include "cs_mex.h"
/* cs_cholsol: solve A*x=b using a sparse Cholesky factorization */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT order ;
    if (nargout > 1 || nargin < 2 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: x = cs_cholsol(A,b,order)") ;
    }
    order = (nargin < 3) ? 1 : mxGetScalar (pargin [2]) ;
    order = CS_MAX (order, 0) ;
    order = CS_MIN (order, 3) ;
    if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1]))
    {
#ifndef NCOMPLEX
        cs_cl *A, Amatrix ;
        cs_complex_t *x ;
        A = cs_cl_mex_get_sparse (&Amatrix, 1, pargin [0]) ;    /* get A */
        x = cs_cl_mex_get_double (A->n, pargin [1]) ;           /* x = b */
        if (!cs_cl_cholsol (order, A, x))                       /* x = A\x */
        {
            mexErrMsgTxt ("A not positive definite") ;
        }
        cs_free (A->x) ;
        pargout [0] = cs_cl_mex_put_double (A->n, x) ;      /* return x */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl *A, Amatrix ;
        double *x, *b ;
        A = cs_dl_mex_get_sparse (&Amatrix, 1, 1, pargin [0]) ; /* get A */
        b = cs_dl_mex_get_double (A->n, pargin [1]) ;           /* get b */
        x = cs_dl_mex_put_double (A->n, b, &(pargout [0])) ;    /* x = b */
        if (!cs_dl_cholsol (order, A, x))                   /* x = A\x */
        {
            mexErrMsgTxt ("A not positive definite") ;
        }
    }
}
