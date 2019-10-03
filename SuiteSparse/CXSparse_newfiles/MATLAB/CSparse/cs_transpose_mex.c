#include "cs_mex.h"
/* C = cs_transpose (A), computes C=A', where A must be sparse.
   C = cs_transpose (A,kind) computes C=A.' if kind <= 0, C=A' if kind > 0 */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT values ;
    if (nargout > 1 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: C = cs_transpose(A,kind)") ;
    }
    values = (nargin > 1) ? mxGetScalar (pargin [1]) : 1 ;
    values = (values <= 0) ? -1 : 1 ;
    if (mxIsComplex (pargin [0]))
    {
#ifndef NCOMPLEX
        cs_cl Amatrix, *A, *C ;
        A = cs_cl_mex_get_sparse (&Amatrix, 0, pargin [0]) ;    /* get A */
        C = cs_cl_transpose (A, values) ;                       /* C = A' */
        pargout [0] = cs_cl_mex_put_sparse (&C) ;               /* return C */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Amatrix, *A, *C ;
        A = cs_dl_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ; /* get A */
        C = cs_dl_transpose (A, values) ;                       /* C = A' */
        pargout [0] = cs_dl_mex_put_sparse (&C) ;               /* return C */
    }
}
