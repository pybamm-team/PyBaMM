// =============================================================================
// === spqr mexFunction ========================================================
// =============================================================================

#include "spqr_mx.hpp"

// This function mimics the existing MATLAB QR syntax, with extensions.
// See spqr.m for details.

#define EMPTY (-1)
#define TRUE 1
#define FALSE 0 

// =============================================================================
// === is_zero =================================================================
// =============================================================================

// Return TRUE if the MATLAB matrix s is a scalar equal to zero

static int is_zero (const mxArray *s)
{
    return (mxIsNumeric (s)
            && (mxGetNumberOfElements (s) == 1)
            && (mxGetScalar (s) == 0)) ;
}


// =============================================================================
// === spqr mexFunction ========================================================
// =============================================================================

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    Long *Ap, *Ai, *E, *Bp, *Bi, *HPinv ;
    double *Ax, *Bx, dummy, tol ;
    Long m, n, anz, bnz, is_complex, econ, A_complex, B_complex ;
    spqr_mx_options opts ;
    cholmod_sparse *A, Amatrix, *R, *Q, *Csparse, Bsmatrix, *Bsparse, *H ;
    cholmod_dense *Cdense, Bdmatrix, *Bdense, *HTau ;
    cholmod_common Common, *cc ;

    double t0 = (nargout > 3) ? SuiteSparse_time ( ) : 0 ;

    // -------------------------------------------------------------------------
    // start CHOLMOD and set parameters
    // -------------------------------------------------------------------------

    cc = &Common ;
    cholmod_l_start (cc) ;
    spqr_mx_config (SPUMONI, cc) ;

    // -------------------------------------------------------------------------
    // check inputs
    // -------------------------------------------------------------------------

    // nargin can be 1, 2, or 3
    // nargout can be 0, 1, 2, 3, or 4

    if (nargout > 4)
    {
        mexErrMsgIdAndTxt ("MATLAB:maxlhs", "Too many output arguments") ;
    }
    if (nargin < 1)
    {
        mexErrMsgIdAndTxt ("MATLAB:minrhs", "Not enough input arguments") ;
    }
    if (nargin > 3)
    {
        mexErrMsgIdAndTxt ("MATLAB:maxrhs", "Too many input arguments") ;
    }

    // -------------------------------------------------------------------------
    // get the input matrix A
    // -------------------------------------------------------------------------

    if (!mxIsSparse (pargin [0]))
    {
        mexErrMsgIdAndTxt ("QR:invalidInput", "A must be sparse") ;
    }

    A = spqr_mx_get_sparse (pargin [0], &Amatrix, &dummy) ;
    Ap = (Long *) A->p ;
    Ai = (Long *) A->i ;
    m = A->nrow ;
    n = A->ncol ;
    A_complex = mxIsComplex (pargin [0]) ;

    // -------------------------------------------------------------------------
    // determine usage and parameters
    // -------------------------------------------------------------------------

    if (nargin == 1)
    {

        // ---------------------------------------------------------------------
        // [ ] = qr (A)
        // ---------------------------------------------------------------------

        spqr_mx_get_options (NULL, &opts, m, nargout, cc) ;
        // R = qr (A)
        // [Q,R] = qr (A)
        // [Q,R,E] = qr (A)

    }
    else if (nargin == 2)
    {

        // ---------------------------------------------------------------------
        // [ ] = qr (A,0), [ ] = qr (A,opts), or [ ] = qr (A,B)
        // ---------------------------------------------------------------------

        if (is_zero (pargin [1]))
        {

            // -----------------------------------------------------------------
            // [ ... ] = qr (A,0)
            // -----------------------------------------------------------------

            spqr_mx_get_options (NULL, &opts, m, nargout, cc) ;
            opts.econ = n ;
            opts.permvector = TRUE ;
            // R = qr (A,0)
            // [Q,R] = qr (A,0)
            // [Q,R,E] = qr (A,0)

        }
        else if (mxIsEmpty (pargin [1]) || mxIsStruct (pargin [1]))
        {

            // -----------------------------------------------------------------
            // [ ] = qr (A,opts)
            // -----------------------------------------------------------------

            spqr_mx_get_options (pargin [1], &opts, m, nargout, cc) ;
            // R = qr (A,opts)
            // [Q,R] = qr (A,opts)
            // [Q,R,E] = qr (A,opts)

        }
        else
        {

            // -----------------------------------------------------------------
            // [ ] = qr (A,B)
            // -----------------------------------------------------------------

            spqr_mx_get_options (NULL, &opts, m, nargout, cc) ;
            opts.haveB = TRUE ;
            opts.Qformat = SPQR_Q_DISCARD ;
            if (nargout <= 1)
            {
                mexErrMsgIdAndTxt ("MATLAB:minlhs",
                    "Not enough output arguments") ;
            }
            // [C,R] = qr (A,B)
            // [C,R,E] = qr (A,B)

        }

    }
    else // if (nargin == 3)
    {

        // ---------------------------------------------------------------------
        // [ ] = qr (A,B,opts)
        // ---------------------------------------------------------------------

        if (is_zero (pargin [2]))
        {

            // -----------------------------------------------------------------
            // [ ] = qr (A,B,0)
            // -----------------------------------------------------------------

            spqr_mx_get_options (NULL, &opts, m, nargout, cc) ;
            opts.econ = n ;
            opts.permvector = TRUE ;
            opts.haveB = TRUE ;
            opts.Qformat = SPQR_Q_DISCARD ;
            if (nargout <= 1)
            {
                mexErrMsgIdAndTxt ("MATLAB:minlhs",
                    "Not enough output arguments") ;
            }
            // [C,R] = qr (A,B,0)
            // [C,R,E] = qr (A,B,0)

        }
        else if (mxIsEmpty (pargin [2]) || mxIsStruct (pargin [2]))
        {

            // -----------------------------------------------------------------
            // [ ] = qr (A,B,opts)
            // -----------------------------------------------------------------

            spqr_mx_get_options (pargin [2], &opts, m, nargout, cc) ;
            opts.haveB = TRUE ;
            opts.Qformat = SPQR_Q_DISCARD ;
            if (nargout <= 1)
            {
                mexErrMsgIdAndTxt ("MATLAB:minlhs",
                    "Not enough output arguments") ;
            }
            // [C,R] = qr (A,B,opts)
            // [C,R,E] = qr (A,B,opts)

        }
        else
        {
            mexErrMsgIdAndTxt ("QR:invalidInput", "Invalid opts argument") ;
        }
    }

    int order = opts.ordering ;
    tol = opts.tol ;
    econ = opts.econ ;

    // -------------------------------------------------------------------------
    // get A and convert to merged-complex if needed
    // -------------------------------------------------------------------------

    if (opts.haveB)
    {
        B_complex = mxIsComplex (pargin [1]) ;
    }
    else
    {
        B_complex = FALSE ;
    }

    is_complex = (A_complex || B_complex) ;
    Ax = spqr_mx_merge_if_complex (pargin [0], is_complex, &anz, cc) ; 
    if (is_complex)
    {
        // A has been converted from real or zomplex to complex
        A->x = Ax ;
        A->z = NULL ;
        A->xtype = CHOLMOD_COMPLEX ;
    }

    // -------------------------------------------------------------------------
    // analyze, factorize, and get the results
    // -------------------------------------------------------------------------

    if (opts.haveB)
    {

        // ---------------------------------------------------------------------
        // get B, and convert to complex if necessary
        // ---------------------------------------------------------------------

        if (!mxIsNumeric (pargin [1]))
        {
            mexErrMsgIdAndTxt ("QR:invalidInput", "invalid non-numeric B") ;
        }
        if (mxGetM (pargin [1]) != m)
        {
            mexErrMsgIdAndTxt ("QR:invalidInput",
                "A and B must have the same number of rows") ;
        }

        // convert from real or zomplex to complex
        Bx = spqr_mx_merge_if_complex (pargin [1], is_complex, &bnz, cc) ;

        int B_is_sparse = mxIsSparse (pargin [1]) ;
        if (B_is_sparse)
        {
            Bsparse = spqr_mx_get_sparse (pargin [1], &Bsmatrix, &dummy) ;
            Bdense = NULL ;
            if (is_complex)
            {
                // Bsparse has been converted from real or zomplex to complex
                Bsparse->x = Bx ;
                Bsparse->z = NULL ;
                Bsparse->xtype = CHOLMOD_COMPLEX ;
            }
        }
        else
        {
            Bsparse = NULL ;
            Bdense = spqr_mx_get_dense (pargin [1], &Bdmatrix, &dummy) ;
            if (is_complex)
            {
                // Bdense has been converted from real or zomplex to complex
                Bdense->x = Bx ;
                Bdense->z = NULL ;
                Bdense->xtype = CHOLMOD_COMPLEX ;
            }
        }

        // ---------------------------------------------------------------------
        // [C,R,E] = qr (A,B,...) or [C,R] = qr (A,B,...)
        // ---------------------------------------------------------------------

        if (is_complex)
        {

            // -----------------------------------------------------------------
            // [C,R,E] = qr (A,B): complex case
            // -----------------------------------------------------------------

            if (B_is_sparse)
            {
                // B and C are both sparse and complex
                SuiteSparseQR <Complex> (order, tol, econ, A, Bsparse,
                    &Csparse, &R, &E, cc) ;
                pargout [0] = spqr_mx_put_sparse (&Csparse, cc) ;
            }
            else
            {
                // B and C are both dense and complex
                SuiteSparseQR <Complex> (order, tol, econ, A, Bdense,
                    &Cdense, &R, &E, cc) ;
                pargout [0] = spqr_mx_put_dense (&Cdense, cc) ;
            }

        }
        else
        {

            // -----------------------------------------------------------------
            // [C,R,E] = qr (A,B): real case
            // -----------------------------------------------------------------

            if (B_is_sparse)
            {
                // B and C are both sparse and real
                SuiteSparseQR <double> (order, tol, econ, A, Bsparse,
                    &Csparse, &R, &E, cc) ;
                pargout [0] = spqr_mx_put_sparse (&Csparse, cc) ;
            }
            else
            {
                // B and C are both dense and real
                SuiteSparseQR <double> (order, tol, econ, A, Bdense,
                    &Cdense, &R, &E, cc) ;
                pargout [0] = spqr_mx_put_dense (&Cdense, cc) ;
            }
        }

        pargout [1] = spqr_mx_put_sparse (&R, cc) ;

    }
    else if (nargout <= 1)
    {

        // ---------------------------------------------------------------------
        // R = qr (A) or R = qr (A,opts)
        // ---------------------------------------------------------------------

        if (is_complex)
        {
            SuiteSparseQR <Complex> (0, tol, econ, A, &R, NULL, cc) ;
        }
        else
        {
            SuiteSparseQR <double> (0, tol, econ, A, &R, NULL, cc) ;
        }
        pargout [0] = spqr_mx_put_sparse (&R, cc) ;

    }
    else
    {

        // ---------------------------------------------------------------------
        // [Q,R,E] = qr (A,...) or [Q,R] = qr (A,...)
        // ---------------------------------------------------------------------

        if (opts.Qformat == SPQR_Q_DISCARD)
        {

            // -----------------------------------------------------------------
            // Q is discarded, and Q = [ ] is returned as a placeholder
            // -----------------------------------------------------------------

            if (is_complex)
            {
                SuiteSparseQR <Complex> (order, tol, econ, A, &R, &E, cc);
            }
            else
            {
                SuiteSparseQR <double> (order, tol, econ, A, &R, &E, cc) ;
            }
            pargout [0] = mxCreateDoubleMatrix (0, 0, mxREAL) ;

        }
        else if (opts.Qformat == SPQR_Q_MATRIX)
        {

            // -----------------------------------------------------------------
            // Q is a sparse matrix
            // -----------------------------------------------------------------

            if (is_complex)
            {
                SuiteSparseQR <Complex> (order, tol, econ, A, &Q, &R, &E, cc) ;
            }
            else
            {
                SuiteSparseQR <double> (order, tol, econ, A, &Q, &R, &E, cc) ;
            }
            pargout [0] = spqr_mx_put_sparse (&Q, cc) ;

        }
        else
        {

            // -----------------------------------------------------------------
            // H is kept, and Q is a struct containing H, Tau, and P
            // -----------------------------------------------------------------

            mxArray *Tau, *P, *Hmatlab ;
            if (is_complex)
            {
                SuiteSparseQR <Complex> (order, tol, econ, A,
                    &R, &E, &H, &HPinv, &HTau, cc) ;
            }
            else
            {
                SuiteSparseQR <double> (order, tol, econ, A,
                    &R, &E, &H, &HPinv, &HTau, cc) ;
            }

            Tau = spqr_mx_put_dense (&HTau, cc) ;
            Hmatlab = spqr_mx_put_sparse (&H, cc) ;

            // Q.P contains the inverse row permutation
            P = mxCreateDoubleMatrix (1, m, mxREAL) ;
            double *Tx = mxGetPr (P) ;
            for (Long i = 0 ; i < m ; i++)
            {
                Tx [i] = HPinv [i] + 1 ;
            }

            // return Q
            const char *Qstruct [ ] = { "H", "Tau", "P" } ;
            pargout [0] = mxCreateStructMatrix (1, 1, 3, Qstruct) ;
            mxSetFieldByNumber (pargout [0], 0, 0, Hmatlab) ;
            mxSetFieldByNumber (pargout [0], 0, 1, Tau) ;
            mxSetFieldByNumber (pargout [0], 0, 2, P) ;

        }
        pargout [1] = spqr_mx_put_sparse (&R, cc) ;
    }

    // -------------------------------------------------------------------------
    // return E
    // -------------------------------------------------------------------------

    if (nargout > 2)
    {
        pargout [2] = spqr_mx_put_permutation (E, n, opts.permvector, cc) ;
    }

    // -------------------------------------------------------------------------
    // free copy of merged-complex, if needed
    // -------------------------------------------------------------------------

    if (is_complex)
    {
        // this was allocated by merge_if_complex
        cholmod_l_free (anz, sizeof (Complex), Ax, cc) ;
        if (opts.haveB)
        {
            cholmod_l_free (bnz, sizeof (Complex), Bx, cc) ;
        }
    }

    // -------------------------------------------------------------------------
    // info output
    // -------------------------------------------------------------------------

    if (nargout > 3)
    {
        double flops = cc->SPQR_flopcount ;
        double t = SuiteSparse_time ( ) - t0 ;
        pargout [3] = spqr_mx_info (cc, t, flops) ;
    }

    cholmod_l_finish (cc) ;
    if (opts.spumoni > 0) spqr_mx_spumoni (&opts, is_complex, cc) ;
}
