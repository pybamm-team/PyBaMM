/* ========================================================================== */
/* === CHOLMOD/MATLAB/cholmod mexFunction =================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* Supernodal sparse Cholesky backslash, x = A\b.  Factorizes PAP' in LL' then
 * solves a sparse linear system.  Uses the diagonal and upper triangular part
 * of A only.  A must be sparse.  b can be sparse or dense.
 *
 * Usage:
 *
 *	x = cholmod2 (A, b)
 *	[x stats] = cholmod2 (A, b, ordering)	% a scalar: 0,-1,-2, or -3
 *	[x stats] = cholmod2 (A, b, p)		% a permutation vector
 *
 * The 3rd argument select the ordering method to use.  If not present or -1,
 * the default ordering strategy is used (AMD, and then try METIS if AMD finds
 * an ordering with high fill-in, and use the best method tried).
 *
 * Other options for the ordering parameter:
 *
 *	0   natural (no etree postordering)
 *	-1  use CHOLMOD's default ordering strategy (AMD, then try METIS)
 *	-2  AMD, and then try NESDIS (not METIS) if AMD has high fill-in
 *	-3  use AMD only
 *	-4  use METIS only
 *	-5  use NESDIS only
 *	-6  natural, but with etree postordering
 *	p   user permutation (vector of size n, with a permutation of 1:n)
 *
 * stats(1)	estimate of the reciprocal of the condition number
 * stats(2)	ordering used:
 *		    0: natural, 1: given, 2:amd, 3:metis, 4:nesdis, 5:colamd,
 *		    6: natural but postordered.
 * stats(3)	nnz(L)
 * stats(4)	flop count in Cholesky factorization.  Excludes solution
 *		    of upper/lower triangular systems, which can be easily
 *		    computed from stats(3) (roughly 4*nnz(L)*size(b,2)).
 * stats(5)	memory usage in MB.
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
    double dummy = 0, rcond, *p ;
    cholmod_sparse Amatrix, Bspmatrix, *A, *Bs, *Xs ;
    cholmod_dense Bmatrix, *X, *B ;
    cholmod_factor *L ;
    cholmod_common Common, *cm ;
    Long n, B_is_sparse, ordering, k, *Perm ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */ 
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* There is no supernodal LDL'.  If cm->final_ll = FALSE (the default), then
     * this mexFunction will use a simplicial LDL' when flops/lnz < 40, and a
     * supernodal LL' otherwise.  This may give suprising results to the MATLAB
     * user, so always perform an LL' factorization by setting cm->final_ll
     * to TRUE. */

    cm->final_ll = TRUE ;
    cm->quick_return_if_not_posdef = TRUE ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargout > 2 || nargin < 2 || nargin > 3)
    {
	mexErrMsgTxt ("usage: [x,rcond] = cholmod2 (A,b,ordering)") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || (n != mxGetN (pargin [0])))
    {
    	mexErrMsgTxt ("A must be square and sparse") ;
    }
    if (n != mxGetM (pargin [1]))
    {
    	mexErrMsgTxt ("# of rows of A and B must match") ;
    }

    /* get sparse matrix A.  Use triu(A) only. */
    A = sputil_get_sparse (pargin [0], &Amatrix, &dummy, 1) ;

    /* get sparse or dense matrix B */
    B = NULL ;
    Bs = NULL ;
    B_is_sparse = mxIsSparse (pargin [1]) ;
    if (B_is_sparse)
    {
	/* get sparse matrix B (unsymmetric) */
	Bs = sputil_get_sparse (pargin [1], &Bspmatrix, &dummy, 0) ;
    }
    else
    {
	/* get dense matrix B */
	B = sputil_get_dense (pargin [1], &Bmatrix, &dummy) ;
    }

    /* get the ordering option */
    if (nargin < 3)
    {
	/* use default ordering */
	ordering = -1 ;
    }
    else
    {
	/* use a non-default option */
	ordering = mxGetScalar (pargin [2]) ;
    }

    p = NULL ;
    Perm = NULL ;

    if (ordering == 0)
    {
	/* natural ordering */
	cm->nmethods = 1 ;
	cm->method [0].ordering = CHOLMOD_NATURAL ;
	cm->postorder = FALSE ;
    }
    else if (ordering == -1)
    {
	/* default strategy ... nothing to change */
    }
    else if (ordering == -2)
    {
	/* default strategy, but with NESDIS in place of METIS */
	cm->default_nesdis = TRUE ;
    }
    else if (ordering == -3)
    {
	/* use AMD only */
	cm->nmethods = 1 ;
	cm->method [0].ordering = CHOLMOD_AMD ;
	cm->postorder = TRUE ;
    }
    else if (ordering == -4)
    {
	/* use METIS only */
	cm->nmethods = 1 ;
	cm->method [0].ordering = CHOLMOD_METIS ;
	cm->postorder = TRUE ;
    }
    else if (ordering == -5)
    {
	/* use NESDIS only */
	cm->nmethods = 1 ;
	cm->method [0].ordering = CHOLMOD_NESDIS ;
	cm->postorder = TRUE ;
    }
    else if (ordering == -6)
    {
	/* natural ordering, but with etree postordering */
	cm->nmethods = 1 ;
	cm->method [0].ordering = CHOLMOD_NATURAL ;
	cm->postorder = TRUE ;
    }
    else if (ordering == -7)
    {
	/* always try both AMD and METIS, and pick the best */
	cm->nmethods = 2 ;
	cm->method [0].ordering = CHOLMOD_AMD ;
	cm->method [1].ordering = CHOLMOD_METIS ;
	cm->postorder = TRUE ;
    }
    else if (ordering >= 1)
    {
	/* assume the 3rd argument is a user-provided permutation of 1:n */
	if (mxGetNumberOfElements (pargin [2]) != n)
	{
	    mexErrMsgTxt ("invalid input permutation") ;
	}
	/* copy from double to integer, and convert to 0-based */
	p = mxGetPr (pargin [2]) ;
	Perm = cholmod_l_malloc (n, sizeof (Long), cm) ;
	for (k = 0 ; k < n ; k++)
	{
	    Perm [k] = p [k] - 1 ;
	}
	/* check the permutation */
	if (!cholmod_l_check_perm (Perm, n, n, cm))
	{
	    mexErrMsgTxt ("invalid input permutation") ;
	}
	/* use only the given permutation */
	cm->nmethods = 1 ;
	cm->method [0].ordering = CHOLMOD_GIVEN ;
	cm->postorder = FALSE ;
    }
    else
    {
	mexErrMsgTxt ("invalid ordering option") ;
    }

    /* ---------------------------------------------------------------------- */
    /* analyze and factorize */
    /* ---------------------------------------------------------------------- */

    L = cholmod_l_analyze_p (A, Perm, NULL, 0, cm) ;
    cholmod_l_free (n, sizeof (Long), Perm, cm) ;
    cholmod_l_factorize (A, L, cm) ;

    rcond = cholmod_l_rcond (L, cm) ;

    if (rcond == 0)
    {
	mexWarnMsgTxt ("Matrix is indefinite or singular to working precision");
    }
    else if (rcond < DBL_EPSILON)
    {
	mexWarnMsgTxt ("Matrix is close to singular or badly scaled.") ;
	mexPrintf ("         Results may be inaccurate. RCOND = %g.\n", rcond) ;
    }

    /* ---------------------------------------------------------------------- */
    /* solve and return solution to MATLAB */
    /* ---------------------------------------------------------------------- */

    if (B_is_sparse)
    {
	/* solve AX=B with sparse X and B; return sparse X to MATLAB */
	Xs = cholmod_l_spsolve (CHOLMOD_A, L, Bs, cm) ;
	pargout [0] = sputil_put_sparse (&Xs, cm) ;
    }
    else
    {
	/* solve AX=B with dense X and B; return dense X to MATLAB */
	X = cholmod_l_solve (CHOLMOD_A, L, B, cm) ;
	pargout [0] = sputil_put_dense (&X, cm) ;
    }

    /* return statistics, if requested */
    if (nargout > 1)
    {
	pargout [1] = mxCreateDoubleMatrix (1, 5, mxREAL) ;
	p = mxGetPr (pargout [1]) ;
	p [0] = rcond ;
	p [1] = L->ordering ;
	p [2] = cm->lnz ;
	p [3] = cm->fl ;
	p [4] = cm->memory_usage / 1048576. ;
    }

    cholmod_l_free_factor (&L, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /*
    if (cm->malloc_count !=
	(mxIsComplex (pargout [0]) + (mxIsSparse (pargout[0]) ? 3:1)))
	mexErrMsgTxt ("memory leak!") ;
    */
}
