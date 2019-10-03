// =============================================================================
// === spqr_singletons mexFunction =============================================
// =============================================================================

#include "spqr_mx.hpp"
#include "spqr.hpp"

// Finds the row and column singletons of a sparse matrix.  Note that this
// function uses "non-usercallable" functions from SuiteSparseQR.
// See spqr_singletons.m for details.
//
// [p q n1rows n1cols tol] = spqr_singletons (A)
// [p q n1rows n1cols] = spqr_singletons (A, tol)

#define TRUE 1
#define FALSE 0 

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    Long *P, *Q, *Rp, *Pinv ;
    double *Ax, dummy, tol ;
    Long m, n, anz, is_complex, n1rows, n1cols, i, k ;
    cholmod_sparse *A, Amatrix, *Y ;
    cholmod_common Common, *cc ;

    // -------------------------------------------------------------------------
    // start CHOLMOD and set parameters
    // -------------------------------------------------------------------------

    cc = &Common ;
    cholmod_l_start (cc) ;
    spqr_mx_config (SPUMONI, cc) ;

    // -------------------------------------------------------------------------
    // check inputs
    // -------------------------------------------------------------------------

    if (nargout > 5)
    {
        mexErrMsgIdAndTxt ("MATLAB:maxlhs", "Too many output arguments") ;
    }
    if (nargin < 1)
    {
        mexErrMsgIdAndTxt ("MATLAB:minrhs", "Not enough input arguments") ;
    }
    if (nargin > 2)
    {
        mexErrMsgIdAndTxt ("MATLAB:maxrhs", "Too many input arguments") ;
    }

    // -------------------------------------------------------------------------
    // get the input matrix A and convert to merged-complex if needed
    // -------------------------------------------------------------------------

    if (!mxIsSparse (pargin [0]))
    {
        mexErrMsgIdAndTxt ("QR:invalidInput", "A must be sparse") ;
    }

    A = spqr_mx_get_sparse (pargin [0], &Amatrix, &dummy) ;
    m = A->nrow ;
    n = A->ncol ;
    is_complex = mxIsComplex (pargin [0]) ;
    Ax = spqr_mx_merge_if_complex (pargin [0], is_complex, &anz, cc) ; 
    if (is_complex)
    {
        // A has been converted from real or zomplex to complex
        A->x = Ax ;
        A->z = NULL ;
        A->xtype = CHOLMOD_COMPLEX ;
    }

    // -------------------------------------------------------------------------
    // get the tolerance
    // -------------------------------------------------------------------------

    if (nargin < 2)
    {
        tol = is_complex ? spqr_tol <Complex> (A,cc) : spqr_tol <double> (A,cc);
    }
    else
    {
        tol = mxGetScalar (pargin [1]) ;
    }

    // -------------------------------------------------------------------------
    // find the singletons
    // -------------------------------------------------------------------------

    if (is_complex)
    {
        spqr_1colamd <Complex> (SPQR_ORDERING_NATURAL, tol, 0, A,
            &Q, &Rp, &Pinv, &Y, &n1cols, &n1rows, cc) ;
    }
    else
    {
        spqr_1colamd <double> (SPQR_ORDERING_NATURAL, tol, 0, A,
            &Q, &Rp, &Pinv, &Y, &n1cols, &n1rows, cc) ;
    }

    // -------------------------------------------------------------------------
    // free unused outputs from spqr_1colamd, and the merged-complex copy of A
    // -------------------------------------------------------------------------

    cholmod_l_free (n1rows+1, sizeof (Long), Rp, cc) ;
    cholmod_l_free_sparse (&Y, cc) ;
    if (is_complex)
    {
        // this was allocated by merge_if_complex
        cholmod_l_free (anz, sizeof (Complex), Ax, cc) ;
    }

    // -------------------------------------------------------------------------
    // find P from Pinv
    // -------------------------------------------------------------------------

    P = (Long *) cholmod_l_malloc (m, sizeof (Long), cc) ;
    for (i = 0 ; i < m ; i++)
    {
        k = Pinv ? Pinv [i] : i ;
        P [k] = i ;
    }
    cholmod_l_free (m, sizeof (Long), Pinv, cc) ;

    // -------------------------------------------------------------------------
    // return results
    // -------------------------------------------------------------------------

    pargout [0] = spqr_mx_put_permutation (P, m, TRUE, cc) ;
    cholmod_l_free (m, sizeof (Long), P, cc) ;
    if (nargout > 1) pargout [1] = spqr_mx_put_permutation (Q, n, TRUE, cc) ;
    cholmod_l_free (n, sizeof (Long), Q, cc) ;
    if (nargout > 2) pargout [2] = mxCreateDoubleScalar ((double) n1rows) ;
    if (nargout > 3) pargout [3] = mxCreateDoubleScalar ((double) n1cols) ;
    if (nargout > 4) pargout [4] = mxCreateDoubleScalar (tol) ;

    cholmod_l_finish (cc) ;
}
