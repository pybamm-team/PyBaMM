#include "cs_mex.h"
/* z = cs_gaxpy (A,x,y) computes z = A*x+y */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: z = cs_gaxpy(A,x,y)") ;
    }
    if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1])
        || mxIsComplex (pargin [2]))
    {
#ifndef NCOMPLEX
        cs_cl Amatrix, *A ;
        cs_complex_t *x, *z ;
        A = cs_cl_mex_get_sparse (&Amatrix, 0, pargin [0]) ;/* get A */
        x = cs_cl_mex_get_double (A->n, pargin [1]) ;       /* get x */
        z = cs_cl_mex_get_double (A->m, pargin [2]) ;       /* z = y */
        cs_cl_gaxpy (A, x, z) ;                             /* z = z + A*x */
        cs_free (x) ;
        cs_free (A->x) ;
        pargout [0] = cs_cl_mex_put_double (A->m, z) ;      /* return z */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Amatrix, *A ;
        double *x, *y, *z ;
        A = cs_dl_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;/* get A */
        x = cs_dl_mex_get_double (A->n, pargin [1]) ;       /* get x */
        y = cs_dl_mex_get_double (A->m, pargin [2]) ;       /* get y */
        z = cs_dl_mex_put_double (A->m, y, &(pargout [0])) ;   /* z = y */
        cs_dl_gaxpy (A, x, z) ;                             /* z = z + A*x */
    }
}
