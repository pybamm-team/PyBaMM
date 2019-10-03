// =============================================================================
// === spqr_mx =================================================================
// =============================================================================

// Utility routines used by the SuiteSparseQR mexFunctions

#include "spqr_mx.hpp"

// =============================================================================
// === spqr_mx_config ==========================================================
// =============================================================================

// Define function pointers and other parameters for a mexFunction

int spqr_mx_config (Long spumoni, cholmod_common *cc)
{
    if (cc == NULL) return (FALSE) ;

    // cholmod_l_solve must return a real or zomplex X for MATLAB
    cc->prefer_zomplex = TRUE ;

    // printing and error handling
    if (spumoni == 0)
    {
	// do not print anything from within CHOLMOD
	cc->print = -1 ;
	SuiteSparse_config.printf_func = NULL ;
    }
    else
    {
	// spumoni = 1: print warning and error messages.  cholmod_l_print_*
	//	routines will print a one-line summary of each object printed.
	// spumoni = 2: also print a short summary of each object.
	cc->print = spumoni + 2 ;
    }

    // error handler
    cc->error_handler = spqr_mx_error ;
    spqr_spumoni = spumoni ;

    // Turn off METIS memory guard.  It is not needed, because mxMalloc will
    // safely terminate the mexFunction and free any workspace without killing
    // all of MATLAB.  This assumes spqr_make was used to compile SuiteSparseQR
    // for MATLAB.
    cc->metis_memory = 0.0 ;

    cc->SPQR_grain = 1 ;    // opts.grain
    cc->SPQR_small = 1e6 ;  // opts.small
    cc->SPQR_shrink = 1 ;   // controls SPQR shrink realloc
    cc->SPQR_nthreads = 0 ; // number of TBB threads (0 = default)

    return (TRUE) ;
}


// =============================================================================
// === spqr_mx_spumoni =========================================================
// =============================================================================

// For spumoni output.  Have fun!

void spqr_mx_spumoni
(
    spqr_mx_options *opts,
    int is_complex,             // TRUE if complex, FALSE if real
    cholmod_common *cc
)
{
    // -------------------------------------------------------------------------
    // author and copyright
    // -------------------------------------------------------------------------

    mexPrintf (
    "\nSuiteSparseQR, version %d.%d.%d (%s), a multithreaded\n"
    "multifrontal sparse QR factorization method, (c) Timothy A. Davis,\n"
    "http://www.suitesparse.com\n\n",
        SPQR_MAIN_VERSION, SPQR_SUB_VERSION, SPQR_SUBSUB_VERSION, SPQR_DATE) ;

    // -------------------------------------------------------------------------
    // installation
    // -------------------------------------------------------------------------

    if (opts->spumoni > 1)
    {
        mexPrintf (
            "Installation:\n"
            "    size of mwIndex:  %d bytes\n"
            "    size of int:      %d bytes\n"
            "    size of BLAS int: %d bytes\n",
            sizeof (mwIndex), sizeof (int), sizeof (BLAS_INT)) ;
#ifdef HAVE_TBB
        mexPrintf ("    compiled with Intel Threading Building Blocks (TBB)\n");
#endif
#ifndef NEXPERT
        mexPrintf ("    compiled with opts.solution='min2norm' option\n") ;
#endif
#ifndef NPARTITION
        mexPrintf (
            "    compiled with opts.ordering='metis' option\n") ;
#endif
    }

    // -------------------------------------------------------------------------
    // input options
    // -------------------------------------------------------------------------

    mexPrintf ("Input options:\n") ;

    if (opts->tol <= SPQR_DEFAULT_TOL)
    {
        mexPrintf ("    opts.tol <= %g (use default tolerance)\n",
            (double) SPQR_DEFAULT_TOL) ;
    }
    else if (opts->tol < 0)
    {
        mexPrintf ("    %g < opts.tol < 0 (use no tolerance)\n",
            (double) SPQR_DEFAULT_TOL) ;
    }
    else
    {
        mexPrintf ("    opts.tol = %g (column 2-norm tolerance)\n", opts->tol) ;
    }

    mexPrintf ("    opts.Q = '") ;
    switch (opts->Qformat)
    {
        case SPQR_Q_DISCARD:      mexPrintf ("discard'\n") ;     break ;
        case SPQR_Q_MATRIX:       mexPrintf ("matrix'\n") ;      break ;
        case SPQR_Q_HOUSEHOLDER:  mexPrintf ("Householder'\n") ; break ;
        default:                  mexPrintf ("undefined'\n") ;   break ;
    }

    mexPrintf ("    opts.econ = %ld"
        " (# of rows of R is max (min (m, opts.econ), r)\n"
        "        where m = size(A,1) and r = rank(A) estimate)\n",
        opts->econ) ;

    mexPrintf ("    opts.ordering = '") ;
    switch (opts->ordering)
    {
        case SPQR_ORDERING_FIXED:   mexPrintf ("fixed'\n") ;   break ;
        case SPQR_ORDERING_NATURAL: mexPrintf ("natural'\n") ; break ;
        case SPQR_ORDERING_COLAMD:  mexPrintf ("colamd'\n") ;  break ;
        case SPQR_ORDERING_GIVEN:   mexPrintf ("given'\n") ;   break ;
        case SPQR_ORDERING_CHOLMOD: mexPrintf ("best'\n") ;    break ;
        case SPQR_ORDERING_AMD:     mexPrintf ("amd'\n") ;     break ;
#ifndef NPARTITION
        case SPQR_ORDERING_METIS:   mexPrintf ("metis'\n") ;   break ;
#endif
        case SPQR_ORDERING_DEFAULT: mexPrintf ("default'\n") ; break ;
        default: mexPrintf ("undefined'\n") ; break ;
    }

    mexPrintf ("    opts.permutation = ") ;
    if (opts->permvector)
    {
        mexPrintf ("'vector'\n") ;
    }
    else
    {
        mexPrintf ("'matrix'\n") ;
    }

    mexPrintf ("    opts.solution = ") ;
    if (opts->min2norm)
    {
        mexPrintf ("'min2norm'\n") ;
    }
    else
    {
        mexPrintf ("'basic'\n") ;
    }

#ifdef HAVE_TBB
    mexPrintf (
    "    opts.grain = %g, opts.small = %g, opts.nthreads = %d\n"
    "        The analysis for TBB parallelism constructs a task graph by\n"
    "        merging nodes in the frontal elimination tree, which is itself\n"
    "        a coalescing of the column elimination tree with one node per\n"
    "        column in the matrix.  The goal of the merging heuristic is to\n"
    "        create a task graph whose leaf nodes have flop counts >=\n"
    "        max ((total flops)/opts.grain, opts.small).  If opts.grain <= 1,\n"
    "        then no TBB parallelism is exploited.  The current default is\n"
    "        opts.grain=1 because in the current version of TBB and OpenMP,\n"
    "        TBB parallelism conflicts with BLAS OpenMP-based parallelism.\n"
    "        This will be resolved in a future version.\n"
    "        opts.nthreads is the number of threads that TBB should use.\n"
    "        If zero, the TBB default is used, which is normally the total\n"
    "        number of cores your computer has.\n",
        cc->SPQR_grain, cc->SPQR_small, cc->SPQR_nthreads) ;
#endif

    // -------------------------------------------------------------------------
    // output statistics
    // -------------------------------------------------------------------------

    mexPrintf ("Output statistics:\n") ;
    mexPrintf ("    upper bound on nnz(R): %ld\n",        cc->SPQR_istat [0]) ;
    mexPrintf ("    upper bound on nnz(H): %ld\n",        cc->SPQR_istat [1]) ;
    mexPrintf ("    number of frontal matrices: %ld\n",   cc->SPQR_istat [2]) ;
    mexPrintf ("    # tasks in TBB task tree: %ld\n",     cc->SPQR_istat [3]) ;
    mexPrintf ("    rank(A) estimate: %ld\n",             cc->SPQR_istat [4]) ;
    mexPrintf ("    # of column singletons: %ld\n",       cc->SPQR_istat [5]) ;
    mexPrintf ("    # of singleton rows: %ld\n",          cc->SPQR_istat [6]) ;
    mexPrintf ("    upper bound on flop count: %g\n",
        cc->SPQR_flopcount_bound * (is_complex ? 4 : 1)) ;
    mexPrintf ("    column 2-norm tolerance used: %g\n",  cc->SPQR_tol_used) ;
    mexPrintf ("    actual memory usage: %g (bytes)\n",
        ((double) cc->memory_usage)) ;

    // Place additional spumoni info here ...
}


// =============================================================================
// === get_option ==============================================================
// =============================================================================

// get a single string or numeric option from the MATLAB options struct

static int get_option
(
    // inputs:
    const mxArray *mxopts,      // the MATLAB struct
    const char *field,          // the field to get from the MATLAB struct

    // outputs:
    double *x,                  // double value of the field, if present
    Long *x_present,            // true if x is present
    char **s,                   // char value of the field, if present;
                                // must be mxFree'd by caller when done

    // workspace and parameters
    cholmod_common *cc
)
{
    Long f ;
    mxArray *p ;

    if (cc == NULL) return (FALSE) ;

    // find the field number
    if (mxopts == NULL || mxIsEmpty (mxopts) || !mxIsStruct (mxopts))
    {
        // mxopts is not present, or [ ], or not a struct
        f = EMPTY ;
    }
    else
    {
        // f will be EMPTY (-1) if the field is not present
        f = mxGetFieldNumber (mxopts, field) ;
    }

    // get the field, or NULL if not present
    if (f == EMPTY)
    {
        p = NULL ;
    }
    else
    {
        p = mxGetFieldByNumber (mxopts, 0, f) ;
    }

    *x_present = FALSE ;
    if (s != NULL)
    {
        *s = NULL ;
    }

    if (p == NULL)
    {
        // option not present
        return (TRUE) ;
    }
    if (mxIsNumeric (p))
    {
        // option is numeric
        if (x == NULL)
        {
            mexPrintf ("opts.%s field must be a string\n", field) ;
            mexErrMsgIdAndTxt ("QR:invalidInput", "invalid option") ;
        }
        *x = mxGetScalar (p) ;
        *x_present = TRUE ;
    }
    else if (mxIsChar (p))
    {
        // option is a MATLAB string; convert it to a C-style string
        if (s == NULL)
        {
            mexPrintf ("opts.%s field must be a numeric value\n", field) ;
            mexErrMsgIdAndTxt ("QR:invalidInput", "invalid option") ;
        }
        *s = mxArrayToString (p) ;
    }
    return (TRUE) ;
}


// =============================================================================
// === spqr_mx_get_options =====================================================
// =============================================================================

// get all the options from the MATLAB struct.  Defaults for the C struct:
//
//      opts.econ = m ;
//      opts.tol = -2, which means that the default tol must be computed
//      opts.ordering = colamd if E present, fixed otherwise
//      opts.permvector = FALSE ;
//      opts.Qformat = SPQR_Q_MATRIX ;
//      opts.haveB = FALSE ;
//      opts.grain = 1, use 2*opts.nthreads for parallelism
//      opts.small = 1e6
//      opts.shrink = 1
//      opts.nthreads = 0
//      opts.spumoni = SPUMONI
//      opts.min2norm = FALSE

int spqr_mx_get_options
(
    const mxArray *mxopts,
    spqr_mx_options *opts,
    Long m,
    int nargout,
    cholmod_common *cc
)
{
    double x ;
    char *s ;
    Long x_present ;

    if (cc == NULL) return (FALSE) ;

    // -------------------------------------------------------------------------
    // spumoni: an integer that defaults to SPUMONI (see spqr_mx.hpp)
    // -------------------------------------------------------------------------

    get_option (mxopts, "spumoni", &x, &x_present, NULL, cc) ;
    opts->spumoni = x_present ? ((Long) x) : SPUMONI ;
    spqr_spumoni = opts->spumoni ;

    // -------------------------------------------------------------------------
    // econ: an integer that defaults to m
    // -------------------------------------------------------------------------

    get_option (mxopts, "econ", &x, &x_present, NULL, cc) ;
    opts->econ = x_present ? ((Long) x) : m ;

    // -------------------------------------------------------------------------
    // tol: a double with default computed later
    // -------------------------------------------------------------------------

    get_option (mxopts, "tol", &x, &x_present, NULL, cc) ;
    opts->tol = x_present ? x : SPQR_DEFAULT_TOL ;

    // -------------------------------------------------------------------------
    // cc->SPQR_grain: defaults to 1 (no TBB parallelism)
    // -------------------------------------------------------------------------

    get_option (mxopts, "grain", &x, &x_present, NULL, cc) ;
    cc->SPQR_grain = x_present ? x : 1 ;

    // -------------------------------------------------------------------------
    // cc->SPQR_small: defaults to 1e6 (min flop count in a TBB task)
    // -------------------------------------------------------------------------

    get_option (mxopts, "small", &x, &x_present, NULL, cc) ;
    cc->SPQR_small = x_present ? x : 1e6 ;

    // -------------------------------------------------------------------------
    // cc->SPQR_shrink: defaults to 1; for testing malloc/realloc only!
    // -------------------------------------------------------------------------

    // NOTE: this feature is not documented in "help spqr".  Nor should it be.
    get_option (mxopts, "shrink", &x, &x_present, NULL, cc) ;
    cc->SPQR_shrink = x_present ? ((int) x) : 1 ;

    // -------------------------------------------------------------------------
    // cc->SPQR_nthreads: defaults to 0; # of threads to use
    // -------------------------------------------------------------------------

    // nthreads = 0 means to use the TBB default
    get_option (mxopts, "nthreads", &x, &x_present, NULL, cc) ;
    cc->SPQR_nthreads = x_present ? ((int) x) : 0 ;
    cc->SPQR_nthreads = MAX (0, cc->SPQR_nthreads) ;

    // -------------------------------------------------------------------------
    // ordering: a string (fixed, natural, colamd, metis, or cholmod)
    // -------------------------------------------------------------------------

    if (nargout < 3)
    {
        // R = qr ( ), [C,R] = qr ( ), or [Q,R] = qr ( ) usages
        opts->ordering = SPQR_ORDERING_FIXED ;
    }
    else
    {
        // [C,R,E] = qr ( ) or [Q,R,E] = qr ( ) usages
        opts->ordering = SPQR_ORDERING_DEFAULT ;
        get_option (mxopts, "ordering", NULL, &x_present, &s, cc) ;
        if (s != NULL)
        {
            if (strcmp (s, "fixed") == 0)
            {
                opts->ordering = SPQR_ORDERING_FIXED ;
            }
            else if (strcmp (s, "natural") == 0)
            {
                opts->ordering = SPQR_ORDERING_NATURAL ;
            }
            else if (strcmp (s, "colamd") == 0)
            {
                opts->ordering = SPQR_ORDERING_COLAMD ;
            }
            else if (strcmp (s, "amd") == 0)
            {
                opts->ordering = SPQR_ORDERING_AMD ;
            }
#ifndef NPARTITION
            else if (strcmp (s, "metis") == 0)
            {
                opts->ordering = SPQR_ORDERING_METIS ;
            }
#endif
            else if (strcmp (s, "best") == 0)
            {
                opts->ordering = SPQR_ORDERING_BEST ;
            }
            else if (strcmp (s, "bestamd") == 0)
            {
                opts->ordering = SPQR_ORDERING_BESTAMD ;
            }
            else if (strcmp (s, "default") == 0)
            {
                opts->ordering = SPQR_ORDERING_DEFAULT ;
            }
            else
            {
                mexErrMsgIdAndTxt ("QR:invalidInput","invalid ordering option");
            }
            MXFREE (s) ;
        }
    }

    // -------------------------------------------------------------------------
    // Q: a string ("matrix", "discard", or "Householder")
    // -------------------------------------------------------------------------

    get_option (mxopts, "Q", NULL, &x_present, &s, cc) ;
    if (nargout <= 1)
    {
        opts->Qformat = SPQR_Q_DISCARD ;
    }
    else
    {
        // for [Q,R] = qr ( ) or [Q,R,E] = qr ( )
        // this will be overridden for [C,R] = qr ( ) case
        opts->Qformat = SPQR_Q_MATRIX ;
    }
    if (s != NULL)
    {
        if (strcmp (s, "matrix") == 0)
        {
            opts->Qformat = SPQR_Q_MATRIX ;
        }
        else if (strcmp (s, "discard") == 0)
        {
            opts->Qformat = SPQR_Q_DISCARD ;
        }
        else if (strcmp (s, "Householder") == 0)
        {
            opts->Qformat = SPQR_Q_HOUSEHOLDER ;
        }
        else
        {
            mexErrMsgIdAndTxt ("QR:invalidInput", "invalid Q option") ;
        }
        MXFREE (s) ;
    }

    // -------------------------------------------------------------------------
    // permutation: a string ("matrix" or "vector")
    // -------------------------------------------------------------------------

    get_option (mxopts, "permutation", NULL, &x_present, &s, cc) ;
    opts->permvector = FALSE ;
    if (s != NULL)
    {
        if (strcmp (s, "matrix") == 0)
        {
            opts->permvector = FALSE ;
        }
        else if (strcmp (s, "vector") == 0)
        {
            opts->permvector = TRUE ;
        }
        else
        {
            mexErrMsgIdAndTxt ("QR:invalidInput", "invalid permutation option");
        }
        MXFREE (s) ;
    }

    // -------------------------------------------------------------------------
    // solution: a string ("min2norm" or "basic"), default is "basic"
    // -------------------------------------------------------------------------

    get_option (mxopts, "solution", NULL, &x_present, &s, cc) ;
    opts->min2norm = FALSE ;
    if (s != NULL)
    {
        if (strcmp (s, "min2norm") == 0)
        {
#ifndef NEXPERT
            opts->min2norm = TRUE ;
#else
            mexErrMsgIdAndTxt ("QR:invalidInput",
                "invalid solution option; min2norm option not installed");
#endif
        }
        else if (strcmp (s, "basic") == 0)
        {
            opts->min2norm = FALSE ;
        }
        else
        {
            mexErrMsgIdAndTxt ("QR:invalidInput", "invalid solution option");
        }
        MXFREE (s) ;
    }

    // if haveB becomes TRUE, Qformat must be set to "discard"
    opts->haveB = FALSE ;
    return (TRUE) ;
}


// =============================================================================
// === put_values ==============================================================
// =============================================================================

static int put_values
(
    Long nz,
    mxArray *A,
    double *Ax,         // complex case: size 2*nz and freed on return,
                        // real case: size nz, not freed on return.
    Long is_complex,
    cholmod_common *cc
)
{
    Long imag_all_zero = TRUE ;

    if (is_complex)
    {
        // A is complex, stored in interleaved form; split it for MATLAB
        Long k ;
        double z, *Ax2, *Az2 ;
        MXFREE (mxGetPi (A)) ;
        // Ax2 and Az2 will never be NULL, even if nz == 0
        Ax2 = (double *) cholmod_l_malloc (nz, sizeof (double), cc) ;
        Az2 = (double *) cholmod_l_malloc (nz, sizeof (double), cc) ;
        for (k = 0 ; k < nz ; k++)
        {
            Ax2 [k] = Ax [2*k] ;
            z = Ax [2*k+1] ;
            if (z != 0)
            {
                imag_all_zero = FALSE ;
            }
            Az2 [k] = z ;
        }
        mxSetPr (A, Ax2) ;
        if (imag_all_zero)
        {
            // free the imaginary part, converting A to real
            cholmod_l_free (1, sizeof (double), Az2, cc) ;
            // the imaginary part can be NULL, for MATLAB
            Az2 = NULL ;
        }
        mxSetPi (A, Az2) ;
        // NOTE: the input Ax is freed
        cholmod_l_free (nz, sizeof (Complex), Ax, cc) ;
    }
    else
    {
        // A is real; just set Ax and return (do not free Ax) 
        if (Ax == NULL)
        {
            // don't give MATLAB a null pointer
            Ax = (double *) cholmod_l_malloc (1, sizeof (double), cc) ;
        }
        mxSetPr (A, Ax) ;
    }
    return (TRUE) ;
}


// =============================================================================
// === spqr_mx_put_sparse ======================================================
// =============================================================================

// Creates a MATLAB version of a CHOLMOD sparse matrix.  The CHOLMOD sparse
// matrix is destroyed.  Both real and complex matrices are supported.

mxArray *spqr_mx_put_sparse
(
    cholmod_sparse **Ahandle,	// CHOLMOD version of the matrix
    cholmod_common *cc
)
{
    mxArray *Amatlab ;
    cholmod_sparse *A ;
    Long nz, is_complex ;

    A = *Ahandle ;
    is_complex = (A->xtype != CHOLMOD_REAL) ;
    Amatlab = mxCreateSparse (0, 0, 0, is_complex ? mxCOMPLEX: mxREAL) ;
    mxSetM (Amatlab, A->nrow) ;
    mxSetN (Amatlab, A->ncol) ;
    mxSetNzmax (Amatlab, A->nzmax) ;
    MXFREE (mxGetJc (Amatlab)) ;
    MXFREE (mxGetIr (Amatlab)) ;
    MXFREE (mxGetPr (Amatlab)) ;
    mxSetJc (Amatlab, (mwIndex *) A->p) ;
    mxSetIr (Amatlab, (mwIndex *) A->i) ;

    nz = cholmod_l_nnz (A, cc) ;
    put_values (nz, Amatlab, (double *) A->x, is_complex, cc) ;

    A->p = NULL ;
    A->i = NULL ;
    A->x = NULL ;
    A->z = NULL ;
    cholmod_l_free_sparse (Ahandle, cc) ;
    return (Amatlab) ;
}


// =============================================================================
// === spqr_mx_put_dense2 ======================================================
// =============================================================================

// Create a MATLAB dense matrix (real or complex)

mxArray *spqr_mx_put_dense2
(
    Long m,
    Long n,
    double *Ax,         // size nz if real; size 2*nz if complex
    int is_complex,
    cholmod_common *cc
)
{
    mxArray *A ;

    if (cc == NULL) return (NULL) ;

    A = mxCreateDoubleMatrix (0, 0, is_complex ? mxCOMPLEX : mxREAL) ;
    mxSetM (A, m) ;
    mxSetN (A, n) ; 
    MXFREE (mxGetPr (A)) ;
    MXFREE (mxGetPi (A)) ;
    put_values (m*n, A, Ax, is_complex, cc) ;
    return (A) ;
}

// =============================================================================
// === spqr_mx_put_dense =======================================================
// =============================================================================

// Creates a MATLAB version of a CHOLMOD dense matrix.  The CHOLMOD dense
// matrix is destroyed.  Both real and complex matrices are supported.
// The leading dimension must be equal to the number of rows.

mxArray *spqr_mx_put_dense
(
    cholmod_dense **Ahandle,	// CHOLMOD version of the matrix
    cholmod_common *cc
)
{
    mxArray *Amatlab ;
    cholmod_dense *A ;

    A = *Ahandle ;

    if (A->d != A->nrow)
    {
        mexErrMsgIdAndTxt ("QR:internalError", "internal error!") ;
    }

    Amatlab = spqr_mx_put_dense2 (A->nrow, A->ncol, (double *) A->x,
        A->xtype != CHOLMOD_REAL, cc) ;

    A->x = NULL ;
    A->z = NULL ;
    cholmod_l_free_dense (Ahandle, cc) ;
    return (Amatlab) ;
}


// =============================================================================
// === spqr_mx_put_permutation =================================================
// =============================================================================

// Return a permutation to MATLAB, as a dense permutation vector or as a
// sparse matrix

mxArray *spqr_mx_put_permutation
(
    Long *P,        // size n permutation vector
    Long n, 
    int vector,     // if TRUE, return a vector; otherwise return a matrix

    // workspace and parameters
    cholmod_common *cc
)
{
    mxArray *Pmatlab ;
    double *Ex ;
    Long *Ep, *Ei, j, k ;

    if (cc == NULL) return (NULL) ;

    if (vector)
    {
        // return P as a permutation vector
        Pmatlab = mxCreateDoubleMatrix (1, n, mxREAL) ;
        Ex = mxGetPr (Pmatlab) ;
        for (k = 0 ; k < n ; k++)
        {
            Ex [k] = (double) ((P ? P [k] : k) + 1) ;
        }
    }
    else
    {
        // return E as a permutation matrix
        Pmatlab = mxCreateSparse (n, n, MAX (n,1), mxREAL) ;
        Ep = (Long *) mxGetJc (Pmatlab) ;
        Ei = (Long *) mxGetIr (Pmatlab) ;
        Ex = mxGetPr (Pmatlab) ;
        for (k = 0 ; k < n ; k++)
        {
            j = P ? P [k] : k ;
            Ep [k] = k ;
            Ei [k] = j ;
            Ex [k] = 1 ;
        }
        Ep [n] = n ;
    }
    return (Pmatlab) ;
}


// =============================================================================
// === spqr_mx_merge_if_complex ================================================
// =============================================================================

// Convert a real or MATLAB-style split-complex matrix into a merged-complex
// matrix usable by SuiteSparseQR and LAPACK.  Alternatively, just return
// a pointer to the real MATLAB matrix if make_complex is FALSE.  Note that
// the input matrix can actually be real, in which case an all-zero imaginary
// part is created.

double *spqr_mx_merge_if_complex
(
    // inputs, not modified
    const mxArray *A,
    int make_complex,       // if TRUE, return value is Complex
    // output
    Long *p_nz,             // number of entries in A

    // workspace and parameters
    cholmod_common *cc
)
{
    Long nz, m, n ;
    double *X, *Xx, *Xz ;

    if (cc == NULL) return (NULL) ;

    m = mxGetM (A) ;
    n = mxGetN (A) ;
    Xx = mxGetPr (A) ;
    Xz = mxGetPi (A) ;

    if (mxIsSparse (A))
    {
        Long *Ap = (Long *) mxGetJc (A) ;
        nz = Ap [n] ;
    }
    else
    {
        nz = m*n ;
    }
    if (make_complex)
    {
        // Note the typecast and sizeof (...) intentionally do not match
        X = (double *) cholmod_l_malloc (nz, sizeof (Complex), cc) ;
        for (Long k = 0 ; k < nz ; k++)
        {
            X [2*k  ] = Xx [k] ;
            X [2*k+1] = Xz ? (Xz [k]) : 0 ;
        }
    }
    else
    {
        X = Xx ;
    }
    *p_nz = nz ;
    return (X) ;
}


// =============================================================================
// === spqr_mx_get_sparse ======================================================
// =============================================================================

// Create a shallow CHOLMOD copy of a MATLAB sparse matrix.  No memory is
// allocated.  The resulting matrix A must not be modified.

cholmod_sparse *spqr_mx_get_sparse
(
    const mxArray *Amatlab, // MATLAB version of the matrix
    cholmod_sparse *A,	    // CHOLMOD version of the matrix
    double *dummy 	    // a pointer to a valid scalar double
)
{
    Long *Ap ;
    A->nrow = mxGetM (Amatlab) ;
    A->ncol = mxGetN (Amatlab) ;
    A->p = (Long *) mxGetJc (Amatlab) ;
    A->i = (Long *) mxGetIr (Amatlab) ;
    Ap = (Long *) A->p ;
    A->nzmax = Ap [A->ncol] ;
    A->packed = TRUE ;
    A->sorted = TRUE ;
    A->nz = NULL ;
    A->itype = CHOLMOD_LONG ;
    A->dtype = CHOLMOD_DOUBLE ;     // FUTURE: support single
    A->stype = 0 ;

    if (mxIsEmpty (Amatlab))
    {
	// this is not dereferenced, but the existence (non-NULL) of these
	// pointers is checked in CHOLMOD
	A->x = dummy ;
	A->z = dummy ;
	A->xtype = mxIsComplex (Amatlab) ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL ;
    }
    else if (mxIsDouble (Amatlab))
    {
	A->x = mxGetPr (Amatlab) ;
	A->z = mxGetPi (Amatlab) ;
	A->xtype = mxIsComplex (Amatlab) ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL ;
    }
    else
    {
	// only sparse complex/real double matrices are supported
        mexErrMsgIdAndTxt ("QR:invalidInput", "matrix type not supported") ;
    }

    return (A) ;
}


// =============================================================================
// === spqr_mx_get_dense =======================================================
// =============================================================================

// Create a shallow CHOLMOD copy of a MATLAB dense matrix.  No memory is
// allocated.  Only double (real and zomplex) matrices are supported.  The
// resulting matrix B must not be modified.

cholmod_dense *spqr_mx_get_dense
(
    const mxArray *Amatlab, // MATLAB version of the matrix
    cholmod_dense *A,	    // CHOLMOD version of the matrix
    double *dummy	    // a pointer to a valid scalar double
)
{
    A->nrow = mxGetM (Amatlab) ;
    A->ncol = mxGetN (Amatlab) ;
    A->d = A->nrow ;
    A->nzmax = A->nrow * A->ncol ;
    A->dtype = CHOLMOD_DOUBLE ;

    if (mxIsEmpty (Amatlab))
    {
	A->x = dummy ;
	A->z = dummy ;
    }
    else if (mxIsDouble (Amatlab))
    {
	A->x = mxGetPr (Amatlab) ;
	A->z = mxGetPi (Amatlab) ;
    }
    else
    {
	// only full double matrices supported
        mexErrMsgIdAndTxt ("QR:invalidInput", "matrix type not supported") ;
    }
    A->xtype = mxIsComplex (Amatlab) ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL ;

    return (A) ;
}

// =============================================================================
// === spqr_mx_get_usage =======================================================
// =============================================================================

// Determine the memory usage of an mxArray.  Used for testing only.

void spqr_mx_get_usage
(
    mxArray *A,         // mxArray to check
    int tight,          // if true, then nnz(A) must equal nzmax(A)
    Long *p_usage,      // bytes used
    Long *p_count,      // # of malloc'd blocks
    cholmod_common *cc
)
{
    Long nz, m, n, nzmax, is_complex, usage, count, *Ap ;
    m = mxGetM (A) ;
    n = mxGetN (A) ;
    is_complex = mxIsComplex (A) ;
    if (mxIsSparse (A))
    {
        nzmax = mxGetNzmax (A) ;
        Ap = (Long *) mxGetJc (A) ;
        nz = MAX (Ap [n], 1) ;
        if (tight && nz != nzmax)
        {
            // This should never occur.
            mexErrMsgIdAndTxt ("QR:internalError", "nnz (A) < nzmax (A)!") ;
        }
        usage = sizeof (Long) * (n+1 + nz) ;
        count = 2 ;
    }
    else
    {
        nz = MAX (m*n,1) ;
        nzmax = nz ;
        usage = 0 ;
        count = 0 ;
    }
    if (is_complex)
    {
        usage += nzmax * sizeof (double) * 2 ;
        count += 2 ;
    }
    else
    {
        usage += nzmax * sizeof (double) ;
        count += 1 ;
    }
    *p_usage = usage ;
    *p_count = count ;
}

// =============================================================================
// === spqr_mx_id ==============================================================
// =============================================================================

// This is used for debugging only, in the ASSERT macro:

char spqr_mx_debug_string [200] ;       // global variable; debugging only
char *spqr_mx_id (int line)
{
    sprintf (spqr_mx_debug_string, "QR:Line_%d", line) ;
    return (spqr_mx_debug_string) ;
}


// =============================================================================
// === spqr_mx_info ============================================================
// =============================================================================

mxArray *spqr_mx_info       // return a struct with info statistics
(
    cholmod_common *cc,
    double t,               // total time, < 0 if not computed
    double flops            // flop count, < 0 if not computed
)
{
    Long ninfo ;
    mxArray *s ;

    const char *info_struct [ ] =
    {
        "nnzR_upper_bound",             // 0: nnz(R) bound
        "nnzH_upper_bound",             // 1: nnz(H) bound
        "number_of_frontal_matrices",   // 2: nf
        "number_of_TBB_tasks",          // 3: ntasks
        "rank_A_estimate",              // 4: rank
        "number_of_column_singletons",  // 5: n1cols
        "number_of_singleton_rows",     // 6: n1rows
        "ordering",                     // 7: ordering used
        "memory_usage_in_bytes",        // 8: memory usage
        "flops_upper_bound",            // 9: upper bound on flop count
                                        //    (excluding backsolve)
        "tol",                          // 10: column norm tolerance used
        "number_of_TBB_threads",        // 11: # threads used
        "norm_E_fro",                   // 12: norm of dropped diag of R

        // compilation options
        "spqr_compiled_with_TBB",       // 13: compiled with TBB or not
        "spqr_compiled_with_METIS",     // 14: compiled with METIS or not

        // only if flops >= 0:
        "analyze_time",                 // 15: analyze time
        "factorize_time",               // 16: factorize time (and apply Q')
        "solve_time",                   // 17: R\C backsolve only
        "total_time",                   // 18: total x=A\b in seconds
        "flops"                         // 19: actual flops (incl backsolve)
    } ;

    ninfo = (flops < 0) ? 15 : 20 ;

    s = mxCreateStructMatrix (1, 1, ninfo, info_struct) ;

    for (Long k = 0 ; k <= 6 ; k++)
    {
        mxSetFieldByNumber (s, 0, k,
            mxCreateDoubleScalar ((double) cc->SPQR_istat [k])) ;
    }

    // get the ordering used.  Note that "default", "best", and "cholmod"
    // are not among the possible results, since they are meta-orderings
    // that select among AMD, COLAMD, and/or METIS.
    mxArray *ord ;
    switch (cc->SPQR_istat [7])
    {
        case SPQR_ORDERING_FIXED:
        case SPQR_ORDERING_GIVEN: 
            ord = mxCreateString ("fixed") ;
            break ;
        case SPQR_ORDERING_NATURAL:
            ord = mxCreateString ("natural") ;
            break ;
        case SPQR_ORDERING_COLAMD:
            ord = mxCreateString ("colamd") ;
            break ;
        case SPQR_ORDERING_AMD:
            ord = mxCreateString ("amd") ;
            break ;
#ifndef NPARTITION
        case SPQR_ORDERING_METIS:
            ord = mxCreateString ("metis") ;
            break ;
#endif
        default:
            ord = mxCreateString ("unknown") ;
            break ;
    }
    mxSetFieldByNumber (s, 0, 7, ord) ;

    mxSetFieldByNumber (s, 0, 8,
        mxCreateDoubleScalar ((double) cc->memory_usage)) ;
    mxSetFieldByNumber (s, 0, 9,
        mxCreateDoubleScalar (cc->SPQR_flopcount_bound)) ;
    mxSetFieldByNumber (s, 0, 10, mxCreateDoubleScalar (cc->SPQR_tol_used)) ;

    int nthreads = cc->SPQR_nthreads ;
    if (nthreads <= 0)
    {
        mxSetFieldByNumber (s, 0, 11, mxCreateString ("default"));
    }
    else
    {
        mxSetFieldByNumber (s, 0, 11, mxCreateDoubleScalar ((double) nthreads));
    }

    mxSetFieldByNumber (s, 0, 12, mxCreateDoubleScalar (cc->SPQR_norm_E_fro)) ;

#ifdef HAVE_TBB
    mxSetFieldByNumber (s, 0, 13, mxCreateString ("yes")) ;
#else
    mxSetFieldByNumber (s, 0, 13, mxCreateString ("no")) ;
#endif

#ifndef NPARTITION
    mxSetFieldByNumber (s, 0, 14, mxCreateString ("yes")) ;
#else
    mxSetFieldByNumber (s, 0, 14, mxCreateString ("no")) ;
#endif

    if (flops >= 0)
    {
        mxSetFieldByNumber (s, 0, 15,
            mxCreateDoubleScalar (cc->SPQR_analyze_time)) ;
        mxSetFieldByNumber (s, 0, 16,
            mxCreateDoubleScalar (cc->SPQR_factorize_time)) ;
        mxSetFieldByNumber (s, 0, 17,
            mxCreateDoubleScalar (cc->SPQR_solve_time)) ;
        mxSetFieldByNumber (s, 0, 18, mxCreateDoubleScalar (t)) ;
        mxSetFieldByNumber (s, 0, 19, mxCreateDoubleScalar (flops)) ;
    }
    return (s) ;
}
