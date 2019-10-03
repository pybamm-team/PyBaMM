#include "cs_mex.h"
/* cs_multiply: sparse matrix multiply */
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
        mexErrMsgTxt ("Usage: C = cs_multiply(A,B)") ;
    }
    if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1]))
    {
#ifndef NCOMPLEX
        cs_cl A1matrix, B1matrix, *A, *B, *C, *D, *A1, *B1 ;
        A1 = cs_cl_mex_get_sparse (&A1matrix, 0, pargin [0]) ;
        A = cs_cl_transpose (A1, 1) ;
        cs_cl_free (A1->x) ;            /* complex copy no longer needed */
        B1 = cs_cl_mex_get_sparse (&B1matrix, 0, pargin [1]) ;
        B = cs_cl_transpose (B1, 1) ;
        cs_cl_free (B1->x) ;            /* complex copy no longer needed */
        D = cs_cl_multiply (B,A) ;              /* D = B'*A' */
        cs_cl_spfree (A) ;
        cs_cl_spfree (B) ;
        cs_cl_dropzeros (D) ;                   /* drop zeros from D */
        C = cs_cl_transpose (D, 1) ;            /* C = D', so C is sorted */
        cs_cl_spfree (D) ;
        pargout [0] = cs_cl_mex_put_sparse (&C) ;       /* return C */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Amatrix, Bmatrix, *A, *B, *C, *D ;
        A = cs_dl_transpose (cs_dl_mex_get_sparse (&Amatrix, 0,1, pargin[0]),1);
        B = cs_dl_transpose (cs_dl_mex_get_sparse (&Bmatrix, 0,1, pargin[1]),1);
        D = cs_dl_multiply (B,A) ;              /* D = B'*A' */
        cs_dl_spfree (A) ;
        cs_dl_spfree (B) ;
        cs_dl_dropzeros (D) ;                   /* drop zeros from D */
        C = cs_dl_transpose (D, 1) ;            /* C = D', so C is sorted */
        cs_dl_spfree (D) ;
        pargout [0] = cs_dl_mex_put_sparse (&C) ;       /* return C */
    }
}
