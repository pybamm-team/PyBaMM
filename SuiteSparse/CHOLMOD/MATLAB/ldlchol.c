/* ========================================================================== */
/* === CHOLMOD/MATLAB/ldlchol mexFunction =================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* Numeric LDL' factorization.  Note that LL' and LDL' are faster than R'R
 * and use less memory.  The LDL' factorization methods use tril(A).
 * The unit diagonal L can be obtained with tril(LD,-1)+speye(n) and the
 * diagonal D can be obtained with D = diag(diag(LD)) ;
 *
 * LD = ldlchol (A)		return the LDL' factorization of A
 * [LD,p] = ldlchol (A)		like [R,p] = chol(A), except LD is always square
 * [LD,p,q] = ldlchol (A)	factorizes A(q,q) into L*D*L'
 *
 * LD = ldlchol (A,beta)	return the LDL' factorization of A*A'+beta*I
 * [LD,p] = ldlchol (A,beta)	like [R,p] = chol(A*A'+beta+I)
 * [LD,p,q] = ldlchol (A,beta)	factorizes A(q,:)*A(q,:)'+beta*I into L*D*L'
 *
 * Explicit zeros may appear in the LD matrix.  The pattern of LD matches the
 * pattern of L as computed by symbfact2, even if some entries in LD are
 * explicitly zero.  This is to ensure that ldlupdate works properly.  You must
 * NOT modify LD in MATLAB itself and then use ldlupdate if LD contains
 * explicit zero entries; ldlupdate will fail catastrophically in this case.
 *
 * You MAY modify LD in MATLAB if you do not pass it back to ldlupdate.  Just be
 * aware that LD contains explicit zero entries, contrary to the standard
 * practice in MATLAB of removing those entries from all sparse matrices.
 *
 * Note that CHOLMOD uses a supernodal LL' factorization and then converts it
 * to LDL' for large matrices, and a simplicial LDL' factorization otherwise.
 * Two-by-two block pivoting is not performed, in any case.  Thus, ldlchol
 * will not be able to perform an LDL' factorization of an arbitrary indefinite
 * matrix.  The matrix
 *
 *  0 1
 *  1 0
 * 
 * will fail, for example.  You can tell CHOLMOD to always use its simplicial
 * LDL' factorization by adding the statement
 *
 *	cm->supernodal = CHOLMOD_SIMPLICIAL ;
 *
 * but ldlchol will be much slower for large matrices.  It still will not be
 * able to handle the above matrix, but it can then handle matrices such as
 *
 *  -2  1
 *   1 -2
 *
 * or any other symmetric matrix for which all leading minors are
 * well-conditioned.
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
    double dummy = 0, beta [2], *px ;
    cholmod_sparse Amatrix, *A, *Lsparse ;
    cholmod_factor *L ;
    cholmod_common Common, *cm ;
    Long n, minor ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */ 
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* convert to packed LDL' when done */
    cm->final_asis = FALSE ;
    cm->final_super = FALSE ;
    cm->final_ll = FALSE ;
    cm->final_pack = TRUE ;
    cm->final_monotonic = TRUE ;

    /* since numerically zero entries are NOT dropped from the symbolic
     * pattern, we DO need to drop entries that result from supernodal
     * amalgamation. */
    cm->final_resymbol = TRUE ;

    cm->quick_return_if_not_posdef = (nargout < 2) ;

    /* This will disable the supernodal LL', which will be slow. */
    /* cm->supernodal = CHOLMOD_SIMPLICIAL ; */

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 1 || nargin > 2 || nargout > 3)
    {
	mexErrMsgTxt ("usage: [L,p,q] = ldlchol (A,beta)") ;
    }

    n = mxGetM (pargin [0]) ;

    if (!mxIsSparse (pargin [0]))
    {
    	mexErrMsgTxt ("A must be sparse") ;
    }
    if (nargin == 1 && n != mxGetN (pargin [0]))
    {
    	mexErrMsgTxt ("A must be square") ;
    }

    /* get sparse matrix A, use tril(A)  */
    A = sputil_get_sparse (pargin [0], &Amatrix, &dummy, -1) ; 

    if (nargin == 1)
    {
	A->stype = -1 ;	    /* use lower part of A */
	beta [0] = 0 ;
	beta [1] = 0 ;
    }
    else
    {
	A->stype = 0 ;	    /* use all of A, factorizing A*A' */
	beta [0] = mxGetScalar (pargin [1]) ;
	beta [1] = 0 ;
    }

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
    cholmod_l_factorize_p (A, beta, NULL, 0, L, cm) ;

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

    /* ---------------------------------------------------------------------- */
    /* return results to MATLAB */
    /* ---------------------------------------------------------------------- */

    /* return L as a sparse matrix (it may contain numerically zero entries) */
    pargout [0] = sputil_put_sparse (&Lsparse, cm) ;

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
    if (cm->malloc_count != 3 + mxIsComplex (pargout[0])) mexErrMsgTxt ("!") ;
    */
}
