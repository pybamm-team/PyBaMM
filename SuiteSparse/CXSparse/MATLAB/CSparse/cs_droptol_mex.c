#include "cs_mex.h"
/* cs_droptol: remove small entries from A */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT j, k ;
    double tol ;
    if (nargout > 1 || nargin != 2)
    {
        mexErrMsgTxt ("Usage: C = cs_droptol(A,tol)") ;
    }
    tol = mxGetScalar (pargin [1]) ;                        /* get tol */

    if (mxIsComplex (pargin [0]))
    {
#ifndef NCOMPLEX
        cs_cl Amatrix, *C, *A ;
        A = cs_cl_mex_get_sparse (&Amatrix, 0, pargin [0]) ;    /* get A */
        C = cs_cl_spalloc (A->m, A->n, A->nzmax, 1, 0) ;        /* C = A */
        for (j = 0 ; j <= A->n ; j++) C->p [j] = A->p [j] ;
        for (k = 0 ; k < A->nzmax ; k++) C->i [k] = A->i [k] ;
        for (k = 0 ; k < A->nzmax ; k++) C->x [k] = A->x [k] ;
        cs_cl_droptol (C, tol) ;                            /* drop from C */
        pargout [0] = cs_cl_mex_put_sparse (&C) ;           /* return C */
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl Amatrix, *C, *A ;
        A = cs_dl_mex_get_sparse (&Amatrix, 0, 1, pargin [0]) ;    /* get A */
        C = cs_dl_spalloc (A->m, A->n, A->nzmax, 1, 0) ;        /* C = A */
        for (j = 0 ; j <= A->n ; j++) C->p [j] = A->p [j] ;
        for (k = 0 ; k < A->nzmax ; k++) C->i [k] = A->i [k] ;
        for (k = 0 ; k < A->nzmax ; k++) C->x [k] = A->x [k] ;
        cs_dl_droptol (C, tol) ;                            /* drop from C */
        pargout [0] = cs_dl_mex_put_sparse (&C) ;                   /* return C */
    }
}
