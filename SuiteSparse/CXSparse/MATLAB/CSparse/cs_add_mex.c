#include "cs_mex.h"
/* cs_add: sparse matrix addition */

#ifndef NCOMPLEX
static cs_complex_t get_complex (const mxArray *a)
{
    cs_complex_t s = mxGetScalar (a) ;
    if (mxIsComplex (a))
    {
        double *z = mxGetPi (a) ;
        s += I * z [0] ;
    }
    return (s) ;
}
#endif

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    if (nargout > 1 || nargin < 2 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: C = cs_add(A,B,alpha,beta)") ;
    }
    if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1])
        || (nargin > 2 && mxIsComplex (pargin [2]))
        || (nargin > 3 && mxIsComplex (pargin [3])))
    {
#ifndef NCOMPLEX
        cs_complex_t alpha, beta ;
        cs_cl Amatrix, Bmatrix, *A, *B, *C, *D ;
        A = cs_cl_mex_get_sparse (&Amatrix, 0, pargin [0]) ;    /* get A */
        B = cs_cl_mex_get_sparse (&Bmatrix, 0, pargin [1]) ;    /* get B */
        alpha = (nargin < 3) ? 1 : get_complex (pargin [2]) ;   /* get alpha */
        beta  = (nargin < 4) ? 1 : get_complex (pargin [3]) ;   /* get beta */
        C = cs_cl_add (A,B,alpha,beta) ;    /* C = alpha*A + beta *B */
        cs_cl_dropzeros (C) ;           /* drop zeros */
        D = cs_cl_transpose (C, 1) ;    /* sort result via double transpose */
        cs_cl_spfree (C) ;
        C = cs_cl_transpose (D, 1) ;
        cs_cl_spfree (D) ;
        pargout [0] = cs_cl_mex_put_sparse (&C) ;       /* return C */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        double alpha, beta ;
        cs_dl Amatrix, Bmatrix, *A, *B, *C, *D ;
        A = cs_dl_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
        B = cs_dl_mex_get_sparse (&Bmatrix, 0, 1, pargin [1]) ;    /* get B */
        alpha = (nargin < 3) ? 1 : mxGetScalar (pargin [2]) ;   /* get alpha */
        beta  = (nargin < 4) ? 1 : mxGetScalar (pargin [3]) ;   /* get beta */
        C = cs_dl_add (A,B,alpha,beta) ;        /* C = alpha*A + beta *B */
        cs_dl_dropzeros (C) ;           /* drop zeros */
        D = cs_dl_transpose (C, 1) ;    /* sort result via double transpose */
        cs_dl_spfree (C) ;
        C = cs_dl_transpose (D, 1) ;
        cs_dl_spfree (D) ;
        pargout [0] = cs_dl_mex_put_sparse (&C) ;       /* return C */
    }
}
