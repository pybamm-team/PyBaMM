/* ========================================================================== */
/* === CHOLMOD/MATLAB/chol2 mexFunction ===================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* Numeric R'R factorization.  Note that LL' and LDL' are faster than R'R
 * and use less memory.  The R'R factorization methods use triu(A), just like
 * MATLAB's built-in chol.
 *
 * R = chol2 (A)		same as R = chol (A), just faster
 * [R,p] = chol2 (A)		save as [R,p] = chol(A), just faster
 * [R,p,q] = chol2 (A)		factorizes A(q,q) into R'*R
 *
 * A must be sparse.  It can be complex or real.
 *
 * R is returned with no explicit zero entries.  This means it might not be
 * chordal, and R cannot be passed back to CHOLMOD for an update/downdate or
 * for a fast simplicial solve.  spones (R) will be equal to the R returned
 * by symbfact2 if no numerically zero entries are dropped, or a subset
 * otherwise.
 */

#include "cholmod_matlab.h"

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    double dummy = 0, *px ;
    cholmod_sparse Amatrix, *A, *Lsparse, *R ;
    cholmod_factor *L ;
    cholmod_common Common, *cm ;
    Long n, minor ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */ 
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* convert to packed LL' when done */
    cm->final_asis = FALSE ;
    cm->final_super = FALSE ;
    cm->final_ll = TRUE ;
    cm->final_pack = TRUE ;
    cm->final_monotonic = TRUE ;

    /* no need to prune entries due to relaxed supernodal amalgamation, since
     * zeros are dropped with sputil_drop_zeros instead */
    cm->final_resymbol = FALSE ;

    cm->quick_return_if_not_posdef = (nargout < 2) ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin != 1 || nargout > 3)
    {
	mexErrMsgTxt ("usage: [R,p,q] = chol2 (A)") ;
    }

    n = mxGetN (pargin [0]) ;

    if (!mxIsSparse (pargin [0]) || n != mxGetM (pargin [0]))
    {
    	mexErrMsgTxt ("A must be square and sparse") ;
    }

    /* get input sparse matrix A.  Use triu(A) only */
    A = sputil_get_sparse (pargin [0], &Amatrix, &dummy, 1) ;

    /* use natural ordering if no q output parameter */
    if (nargout < 3)
    {
	cm->nmethods = 1 ;
	cm->method [0].ordering = CHOLMOD_NATURAL ;
	cm->postorder = FALSE ;
    }

    /* ---------------------------------------------------------------------- */
    /* analyze and factorize */
    /* ---------------------------------------------------------------------- */

    L = cholmod_l_analyze (A, cm) ;
    cholmod_l_factorize (A, L, cm) ;

    if (nargout < 2 && cm->status != CHOLMOD_OK)
    {
	mexErrMsgTxt ("matrix is not positive definite") ;
    }

    /* ---------------------------------------------------------------------- */
    /* convert L to a sparse matrix */
    /* ---------------------------------------------------------------------- */

    /* the conversion sets L->minor back to n, so get a copy of it first */
    minor = L->minor ;
    Lsparse = cholmod_l_factor_to_sparse (L, cm) ;
    if (Lsparse->xtype == CHOLMOD_COMPLEX)
    {
	/* convert Lsparse from complex to zomplex */
	cholmod_l_sparse_xtype (CHOLMOD_ZOMPLEX, Lsparse, cm) ;
    }

    if (minor < n)
    {
	/* remove columns minor to n-1 from Lsparse */
	sputil_trim (Lsparse, minor, cm) ;
    }

    /* drop zeros from Lsparse */
    sputil_drop_zeros (Lsparse) ;

    /* Lsparse is lower triangular; conjugate transpose to get R */
    R = cholmod_l_transpose (Lsparse, 2, cm) ;
    cholmod_l_free_sparse (&Lsparse, cm) ;

    /* ---------------------------------------------------------------------- */
    /* return results to MATLAB */
    /* ---------------------------------------------------------------------- */

    /* return R */
    pargout [0] = sputil_put_sparse (&R, cm) ;

    /* return minor (translate to MATLAB convention) */
    if (nargout > 1)
    {
	pargout [1] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
	px = mxGetPr (pargout [1]) ;
	px [0] = ((minor == n) ? 0 : (minor+1)) ;
    }

    /* return permutation */
    if (nargout > 2)
    {
	pargout [2] = sputil_put_int (L->Perm, n, 1) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace and the CHOLMOD L, except for what is copied to MATLAB */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free_factor (&L, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /*
    if (cm->malloc_count != (3 + mxIsComplex (pargout[0]))) mexErrMsgTxt ("!") ;
    */
}
