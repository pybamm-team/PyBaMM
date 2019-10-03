#include "cs_mex.h"
/* cs_lusol: solve A*x=b using a sparse LU factorization */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double tol ;
    CS_INT order ;
    if (nargout > 1 || nargin < 2 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: x = cs_lusol(A,b,order,tol)") ;
    }
    order = (nargin < 3) ? 2 : mxGetScalar (pargin [2]) ;
    order = CS_MAX (order, 0) ;
    order = CS_MIN (order, 3) ;
    if (nargin == 2)
    {
        tol = 1 ;                           /* normal partial pivoting */
    }
    else if (nargin == 3)
    {
        tol = (order == 1) ? 0.001 : 1 ;    /* tol = 0.001 for amd(A+A') */
    }
    else
    {
        tol = mxGetScalar (pargin [3]) ;
    }
    if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1]))
    {
#ifndef NCOMPLEX
        cs_cl *A, Amatrix ;
        cs_complex_t *x ;
        A = cs_cl_mex_get_sparse (&Amatrix, 1, pargin [0]) ;    /* get A */
        x = cs_cl_mex_get_double (A->n, pargin [1]) ;           /* x = b */
        if (!cs_cl_lusol (order, A, x, tol))                    /* x = A\x */
        {
            mexErrMsgTxt ("failed (singular or out of memory)") ;
        }
        cs_cl_free (A->x) ;     /* complex copy no longer needed */
        pargout [0] = cs_cl_mex_put_double (A->n, x) ;          /* return x */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl *A, Amatrix ;
        double *x, *b ;
        A = cs_dl_mex_get_sparse (&Amatrix, 1, 1, pargin [0]) ;    /* get A */
        b = cs_dl_mex_get_double (A->n, pargin [1]) ;           /* get b */
        x = cs_dl_mex_put_double (A->n, b, &(pargout [0])) ;    /* x = b */
        if (!cs_dl_lusol (order, A, x, tol))                    /* x = A\x */
        {
            mexErrMsgTxt ("failed (singular or out of memory)") ;
        }
    }
}
