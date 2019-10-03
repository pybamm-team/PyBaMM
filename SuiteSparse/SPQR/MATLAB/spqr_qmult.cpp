// =============================================================================
// === spqr_qmult mexFunction ==================================================
// =============================================================================

#include "spqr_mx.hpp"

//  Multiply Q times X, where Q is stored as a struct, in Householder form.
//
//  method = 0: Y = Q'*X    default
//  method = 1: Y = Q*X 
//  method = 2: Y = X*Q'
//  method = 3: Y = X*Q
//
//  Usage:
//
//  Y = spqr_qmult (Q,X,method) ;
//
//  where Q is the struct from [Q,R,E] = spqr (A,opts) with
//  opts.Q = 'Householder'
//
//  Q.H: Householder vectors (m-by-nh).  In each column, the nonzero entry with
//      the smallest row index must be equal to 1.0.
//  Q.Tau: Householder coefficients (1-by-nh).
//  Q.P: inverse row permutation.  P [i] = k if row i of X is row k of H and Y.

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    mxArray *Hmatlab, *Tau, *P ;
    Long *HPinv, *Yp, *Yi ;
    double *Hx, *Xx, *Tx, *Px, dummy ;
    Long m, n, k, nh, nb, p, i, method, mh, gotP, X_is_sparse, is_complex, hnz,
        tnz, xnz, inuse, count ;
    cholmod_sparse *Ysparse, *H, Hmatrix, *Xsparse, Xsmatrix ;
    cholmod_dense *Ydense, *Xdense, Xdmatrix, *HTau, HTau_matrix ;
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

    // nargin can be 2 or 3
    // nargout can be 0 or 1

    if (nargout > 1)
    {
        mexErrMsgIdAndTxt ("MATLAB:maxlhs", "Too many output arguments") ;
    }
    if (nargin < 2)
    {
        mexErrMsgIdAndTxt ("MATLAB:minrhs", "Not enough input arguments") ;
    }
    if (nargin > 3)
    {
        mexErrMsgIdAndTxt ("MATLAB:maxrhs", "Too many input arguments") ;
    }

    if (!mxIsStruct (pargin [0]))
    {
        mexErrMsgIdAndTxt ("QR:invalidInput", "invalid Q (must be a struct)") ;
    }

    // -------------------------------------------------------------------------
    // get H, Tau, and P from the Q struct
    // -------------------------------------------------------------------------

    i = mxGetFieldNumber (pargin [0], "H") ;
    if (i < 0)
    {
        mexErrMsgIdAndTxt ("QR:invalidInput", "invalid Q struct") ;
    }
    Hmatlab = mxGetFieldByNumber (pargin [0], 0, i) ;
    nh = mxGetN (Hmatlab) ;
    if (!mxIsSparse (Hmatlab))
    {
        mexErrMsgIdAndTxt ("QR:invalidInput", "H must be sparse") ;
    }
    i = mxGetFieldNumber (pargin [0], "Tau") ;
    if (i < 0)
    {
        mexErrMsgIdAndTxt ("QR:invalidInput", "invalid Q struct") ;
    }
    Tau = mxGetFieldByNumber (pargin [0], 0, i) ;
    if (nh != mxGetNumberOfElements (Tau))
    {
        mexErrMsgIdAndTxt ("QR:invalidInput",
            "H and Tau must have the same number of columns") ;
    }

    is_complex = mxIsComplex (Tau) || mxIsComplex (Hmatlab) ||
        mxIsComplex (pargin [1]) ;

    // -------------------------------------------------------------------------
    // get the Householder vectors
    // -------------------------------------------------------------------------

    H = spqr_mx_get_sparse (Hmatlab, &Hmatrix, &dummy) ;
    mh = H->nrow ;
    Hx = spqr_mx_merge_if_complex (Hmatlab, is_complex, &hnz, cc) ;
    if (is_complex)
    {
        // H has been converted from real or zomplex to complex
        H->x = Hx ;
        H->z = NULL ;
        H->xtype = CHOLMOD_COMPLEX ;
    }

    // -------------------------------------------------------------------------
    // get Tau
    // -------------------------------------------------------------------------

    HTau = spqr_mx_get_dense (Tau, &HTau_matrix, &dummy) ;
    Tx = spqr_mx_merge_if_complex (Tau, is_complex, &tnz, cc) ;
    if (is_complex)
    {
        // HTau has been converted from real or zomplex to complex
        HTau->x = Tx ;
        HTau->z = NULL ;
        HTau->xtype = CHOLMOD_COMPLEX ;
    }

    // -------------------------------------------------------------------------
    // get method
    // -------------------------------------------------------------------------

    if (nargin < 3)
    {
        method = 0 ;
    }
    else
    {
        method = (Long) mxGetScalar (pargin [2]) ;
        if (method < 0 || method > 3)
        {
            mexErrMsgIdAndTxt ("QR:invalidInput", "invalid method") ;
        }
    }

    // -------------------------------------------------------------------------
    // get X
    // -------------------------------------------------------------------------

    m = mxGetM (pargin [1]) ;
    n = mxGetN (pargin [1]) ;
    X_is_sparse = mxIsSparse (pargin [1]) ;
    Xsparse = NULL ;
    if (X_is_sparse)
    {
        Xsparse = spqr_mx_get_sparse (pargin [1], &Xsmatrix, &dummy) ;
    }
    else
    {
        Xdense = spqr_mx_get_dense (pargin [1], &Xdmatrix, &dummy) ;
    }
    Xx = spqr_mx_merge_if_complex (pargin [1], is_complex, &xnz, cc) ;
    if (is_complex)
    {
        // X has been converted from real or zomplex to complex
        if (X_is_sparse)
        {
            Xsparse->x = Xx ;
            Xsparse->xtype = CHOLMOD_COMPLEX ;
        }
        else
        {
            Xdense->x = Xx ;
            Xdense->xtype = CHOLMOD_COMPLEX ;
        }
    }

    if (method == 0 || method == 1)
    {
        if (mh != m)
        {
            mexErrMsgIdAndTxt ("QR:invalidInput",
                "H and X must have same number of rows") ;
        }
    }
    else
    {
        if (mh != n)
        {
            mexErrMsgIdAndTxt ("QR:invalidInput",
                "# of cols of X must equal # of rows of H") ;
        }
    }

    // -------------------------------------------------------------------------
    // get P
    // -------------------------------------------------------------------------

    i = mxGetFieldNumber (pargin [0], "P") ;
    gotP = (i >= 0) ;
    HPinv = NULL ;

    if (gotP)
    {
        // get P from the H struct
        P = mxGetFieldByNumber (pargin [0], 0, i) ;
        if (mxGetNumberOfElements (P) != mh)
        {
            mexErrMsgIdAndTxt ("QR:invalidInput",
                "P must be a vector of length equal to # rows of H") ;
        }
        HPinv = (Long *) cholmod_l_malloc (mh, sizeof (Long), cc) ;
        Px = mxGetPr (P) ;
        for (i = 0 ; i < mh ; i++)
        {
            HPinv [i] = (Long) (Px [i] - 1) ;
            if (HPinv [i] < 0 || HPinv [i] >= mh)
            {
                mexErrMsgIdAndTxt ("QR:invalidInput", "invalid permutation") ;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Y = Q'*X, Q*X, X*Q or X*Q'
    // -------------------------------------------------------------------------

    if (is_complex)
    {
        if (X_is_sparse)
        {
            Ysparse = SuiteSparseQR_qmult <Complex> (method, H,
                HTau, HPinv, Xsparse, cc) ;
            pargout [0] = spqr_mx_put_sparse (&Ysparse, cc) ;
        }
        else
        {
            Ydense = SuiteSparseQR_qmult <Complex> (method, H,
                HTau, HPinv, Xdense, cc) ;
            pargout [0] = spqr_mx_put_dense (&Ydense, cc) ;
        }
    }
    else
    {
        if (X_is_sparse)
        {
            Ysparse = SuiteSparseQR_qmult <double> (method, H,
                HTau, HPinv, Xsparse, cc) ;
            pargout [0] = spqr_mx_put_sparse (&Ysparse, cc) ;
        }
        else
        {
            Ydense = SuiteSparseQR_qmult <double> (method, H,
                HTau, HPinv, Xdense, cc) ;
            pargout [0] = spqr_mx_put_dense (&Ydense, cc) ;
        }
    }

    // -------------------------------------------------------------------------
    // free workspace
    // -------------------------------------------------------------------------

    cholmod_l_free (mh, sizeof (Long), HPinv, cc) ;

    if (is_complex)
    {
        // free the merged copies of the real parts of the H and Tau matrices
        cholmod_l_free (hnz, sizeof (Complex), Hx, cc) ;
        cholmod_l_free (tnz, sizeof (Complex), Tx, cc) ;
        cholmod_l_free (xnz, sizeof (Complex), Xx, cc) ;
    }
    cholmod_l_finish (cc) ;

#if 0
    // malloc count for testing only ...
    spqr_mx_get_usage (pargout [0], 1, &inuse, &count, cc) ;
    if (inuse != cc->memory_inuse || count != cc->malloc_count)
    {
        mexErrMsgIdAndTxt ("QR:internalError", "memory leak!") ;
    }
#endif
}
