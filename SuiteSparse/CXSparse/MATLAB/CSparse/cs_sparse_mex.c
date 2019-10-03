#include "cs_mex.h"
/* cs_sparse: convert triplet form into compress-column form sparse matrix */
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
        mexErrMsgTxt ("Usage: A = cs_sparse(i,j,x)") ;
    }
    if (mxIsComplex (pargin [2]))
    {
#ifndef NCOMPLEX
        cs_cl *A, *C, *T, Tmatrix ;
        T = &Tmatrix ;                  /* get i,j,x and copy to triplet form */
        T->nz = mxGetM (pargin [0]) ;
        T->p = cs_dl_mex_get_int (T->nz, pargin [0], &(T->n), 1) ;
        T->i = cs_dl_mex_get_int (T->nz, pargin [1], &(T->m), 1) ;
        cs_mex_check (1, T->nz, 1, 0, 0, 1, pargin [2]) ;
        T->x = cs_cl_mex_get_double (T->nz, pargin [2]) ;
        T->nzmax = T->nz ;
        C = cs_cl_compress (T) ;                /* create sparse matrix C */
        cs_cl_dupl (C) ;                        /* remove duplicates from C */
        cs_cl_dropzeros (C) ;                   /* remove zeros from C */
        A = cs_cl_transpose (C, -1) ;           /* A=C.' */
        cs_cl_spfree (C) ;
        pargout [0] = cs_cl_mex_put_sparse (&A) ;       /* return A */
        cs_free (T->p) ;
        cs_free (T->i) ;
        cs_free (T->x) ;                        /* free copy of complex values*/
#else
        mexErrMsgTxt ("complex matrices not supported") ;
#endif
    }
    else
    {
        cs_dl *A, *C, *T, Tmatrix ;
        T = &Tmatrix ;                  /* get i,j,x and copy to triplet form */
        T->nz = mxGetM (pargin [0]) ;
        T->p = cs_dl_mex_get_int (T->nz, pargin [0], &(T->n), 1) ;
        T->i = cs_dl_mex_get_int (T->nz, pargin [1], &(T->m), 1) ;
        cs_mex_check (1, T->nz, 1, 0, 0, 1, pargin [2]) ;
        T->x = mxGetPr (pargin [2]) ;
        T->nzmax = T->nz ;
        C = cs_dl_compress (T) ;                /* create sparse matrix C */
        cs_dl_dupl (C) ;                        /* remove duplicates from C */
        cs_dl_dropzeros (C) ;                   /* remove zeros from C */
        A = cs_dl_transpose (C, 1) ;            /* A=C' */
        cs_dl_spfree (C) ;
        pargout [0] = cs_dl_mex_put_sparse (&A) ;       /* return A */
        cs_free (T->p) ;
        cs_free (T->i) ;
    }
}
