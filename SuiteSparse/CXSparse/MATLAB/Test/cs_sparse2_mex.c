#include "cs_mex.h"
/* A = cs_sparse2 (i,j,x), removing duplicates and numerically zero entries,
 * and returning A sorted (test cs_entry) */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT k, m, n, nz, *Ti, *Tj ;
    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: A = cs_sparse2(i,j,x)") ;
    }
    nz = mxGetM (pargin [0]) ;
    Ti = cs_dl_mex_get_int (nz, pargin [0], &m, 1) ;
    Tj = cs_dl_mex_get_int (nz, pargin [1], &n, 1) ;
    cs_mex_check (1, nz, 1, 0, 0, 1, pargin [2]) ;
    if (mxIsComplex (pargin [2]))
    {
#ifdef NCOMPLEX
        mexErrMsgTxt ("complex case not supported") ;
#else
        cs_complex_t *Tx ;
        cs_cl *A, *C, *T ;
        Tx = cs_cl_mex_get_double (nz, pargin [2]) ;
        T = cs_cl_spalloc (n, m, 1, 1, 1) ;
        for (k = 0 ; k < nz ; k++)
        {
            cs_cl_entry (T, Tj [k], Ti [k], Tx [k]) ;
        }
        C = cs_cl_compress (T) ;
        cs_cl_spfree (T) ;
        cs_cl_dupl (C) ;
        cs_cl_dropzeros (C) ;
        A = cs_cl_transpose (C, -1) ;
        cs_cl_spfree (C) ;
        pargout [0] = cs_cl_mex_put_sparse (&A) ;
        cs_free (Tx) ;
#endif
    }
    else
    {
        double *Tx ;
        cs_dl *A, *C, *T ;
        Tx = mxGetPr (pargin [2]) ;
        T = cs_dl_spalloc (n, m, 1, 1, 1) ;
        for (k = 0 ; k < nz ; k++)
        {
            cs_dl_entry (T, Tj [k], Ti [k], Tx [k]) ;
        }
        C = cs_dl_compress (T) ;
        cs_dl_spfree (T) ;
        cs_dl_dupl (C) ;
        cs_dl_dropzeros (C) ;
        A = cs_dl_transpose (C, 1) ;
        cs_dl_spfree (C) ;
        pargout [0] = cs_dl_mex_put_sparse (&A) ;
    }
    cs_free (Ti) ;
    cs_free (Tj) ;
}
