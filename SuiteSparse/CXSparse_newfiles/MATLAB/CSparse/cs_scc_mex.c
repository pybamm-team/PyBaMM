#include "cs_mex.h"
/* [p,r] = cs_scc (A) finds the strongly connected components of A */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs_dl Amatrix, *A ;
    cs_dld *D ;
    CS_INT n, j, *Ap2 ;
    if (nargout > 2 || nargin != 1)
    {
        mexErrMsgTxt ("Usage: [p,r] = cs_scc(A)") ;
    }
    A = cs_dl_mex_get_sparse (&Amatrix, 1, 0, pargin [0]) ;     /* get A */
    /* cs_scc modifies A->p and then restores it (in cs_dfs).  Avoid the issue
     * of a mexFunction modifying its input (even temporarily) by making a copy
     * of A->p.  This issue does not arise in cs_dmperm, because that function
     * applies cs_scc to a submatrix C, not to A directly. */
    n = A->n ;
    Ap2 = cs_dl_malloc (n+1, sizeof (CS_INT)) ;
    for (j = 0 ; j <= n ; j++) Ap2 [j] = A->p [j] ;
    A->p = Ap2 ;
    D = cs_dl_scc (A) ;                                 /* find conn. comp. */
    pargout [0] = cs_dl_mex_put_int (D->p, n, 1, 0) ;           /* return p */
    pargout [1] = cs_dl_mex_put_int (D->r, D->nb+1, 1, 0) ;     /* return r */
    cs_dl_dfree (D) ;
    cs_free (Ap2) ;     /* free the copy of A->p */
}
