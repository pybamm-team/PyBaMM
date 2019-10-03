#include "cs_mex.h"
/* cs_qrsol: solve least squares or underdetermined problem */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT k, order ;
    if (nargout > 1 || nargin < 2 || nargin > 3)
    {
        mexErrMsgTxt ("Usage: x = cs_qrsol(A,b,order)") ;
    }
    order = (nargin < 3) ? 3 : mxGetScalar (pargin [2]) ;
    order = CS_MAX (order, 0) ;
    order = CS_MIN (order, 3) ;

    if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1]))
    {
#ifndef NCOMPLEX
        cs_cl *A, Amatrix ;
        cs_complex_t *x, *b ;
        A = cs_cl_mex_get_sparse (&Amatrix, 0, pargin [0]) ;    /* get A */
        b = cs_cl_mex_get_double (A->m, pargin [1]) ;           /* get b */
        x = cs_dl_calloc (CS_MAX (A->m, A->n), sizeof (cs_complex_t)) ;
        for (k = 0 ; k < A->m ; k++) x [k] = b [k] ;            /* x = b */
        cs_free (b) ;
        if (!cs_cl_qrsol (order, A, x))                         /* x = A\x */
        {
            mexErrMsgTxt ("QR solve failed") ;
        }
        pargout [0] = cs_cl_mex_put_double (A->n, x) ;          /* return x */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl *A, Amatrix ;
        double *x, *b ;
        A = cs_dl_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;     /* get A */
        b = cs_dl_mex_get_double (A->m, pargin [1]) ;               /* get b */
        x = cs_dl_calloc (CS_MAX (A->m, A->n), sizeof (double)) ;   /* x = b */
        for (k = 0 ; k < A->m ; k++) x [k] = b [k] ;
        if (!cs_dl_qrsol (order, A, x))                         /* x = A\x */
        {
            mexErrMsgTxt ("QR solve failed") ;
        }
        cs_dl_mex_put_double (A->n, x, &(pargout [0])) ;        /* return x */
        cs_free (x) ;
    }
}
