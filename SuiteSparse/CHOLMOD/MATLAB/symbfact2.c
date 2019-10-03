/* ========================================================================== */
/* === CHOLMOD/MATLAB/symbfact2 mexFunction ================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* Usage:
 *
 *  count = symbfact2 (A)		returns row counts of R=chol(A)
 *  count = symbfact2 (A,'col')		returns row counts of R=chol(A'*A)
 *  count = symbfact2 (A,'sym')		same as symbfact2(A)
 *  count = symbfact2 (A,'lo')		same as symbfact2(A'), uses tril(A)
 *  count = symbfact2 (A,'row')		returns row counts of R=chol(A*A')
 *
 *	[count, h, parent, post, R] = symbfact2 (...)
 *
 *	h: height of the elimination tree
 *	parent: the elimination tree itself
 *	post: postordering of the elimination tree
 *	R: a 0-1 matrix whose structure is that of chol(A) or chol(A'*A)
 *		for the 'col' case
 *
 *	symbfact2(A) and symbfact2(A,'sym') uses triu(A).
 *	symbfact2(A,'lo') uses tril(A).
 *
 * These forms return L = R' instead of R.  They are faster and take less
 * memory.  They return the same count, h, parent, and post outputs.
 *
 *  [count, h, parent, post, L] = symbfact2 (A,'col','L')
 *  [count, h, parent, post, L] = symbfact2 (A,'sym','L')
 *  [count, h, parent, post, L] = symbfact2 (A,'lo', 'L')
 *  [count, h, parent, post, L] = symbfact2 (A,'row','L')
 */

#include "cholmod_matlab.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double dummy = 0 ;
    double *Lx, *px ;
    Long *Parent, *Post, *ColCount, *First, *Level, *Rp, *Ri, *Lp, *Li, *W ;
    cholmod_sparse *A, Amatrix, *F, *Aup, *Alo, *R, *A1, *A2, *L, *S ;
    cholmod_common Common, *cm ;
    Long n, i, coletree, j, lnz, p, k, height, c ;
    char buf [LEN] ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set defaults */
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargout > 5 || nargin < 1 || nargin > 3)
    {
	mexErrMsgTxt (
	    "Usage: [count h parent post R] = symbfact2 (A, mode, Lmode)") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get input matrix A */
    /* ---------------------------------------------------------------------- */

    A = sputil_get_sparse_pattern (pargin [0], &Amatrix, &dummy, cm) ;
    S = (A == &Amatrix) ? NULL : A ;

    /* ---------------------------------------------------------------------- */
    /* get A->stype, default is to use triu(A) */
    /* ---------------------------------------------------------------------- */

    A->stype = 1 ;
    n = A->nrow ;
    coletree = FALSE ;
    if (nargin > 1)
    {
	buf [0] = '\0' ;
	if (mxIsChar (pargin [1]))
	{
	    mxGetString (pargin [1], buf, LEN) ;
	}
	c = buf [0] ;
	if (tolower (c) == 'r')
	{
	    /* unsymmetric case (A*A') if string starts with 'r' */
	    A->stype = 0 ;
	}
	else if (tolower (c) == 'c')
	{
	    /* unsymmetric case (A'*A) if string starts with 'c' */
	    n = A->ncol ;
	    coletree = TRUE ;
	    A->stype = 0 ;
	}
	else if (tolower (c) == 's')
	{
	    /* symmetric upper case (A) if string starts with 's' */
	    A->stype = 1 ;
	}
	else if (tolower (c) == 'l')
	{
	    /* symmetric lower case (A) if string starts with 'l' */
	    A->stype = -1 ;
	}
	else
	{
	    mexErrMsgTxt ("symbfact2: unrecognized mode") ;
	}
    }

    if (A->stype && A->nrow != A->ncol)
    {
	mexErrMsgTxt ("symbfact2: A must be square") ;
    }

    /* ---------------------------------------------------------------------- */
    /* compute the etree, its postorder, and the row/column counts */
    /* ---------------------------------------------------------------------- */

    Parent = cholmod_l_malloc (n, sizeof (Long), cm) ;
    Post = cholmod_l_malloc (n, sizeof (Long), cm) ;
    ColCount = cholmod_l_malloc (n, sizeof (Long), cm) ;
    First = cholmod_l_malloc (n, sizeof (Long), cm) ;
    Level = cholmod_l_malloc (n, sizeof (Long), cm) ;

    /* F = A' */
    F = cholmod_l_transpose (A, 0, cm) ;

    if (A->stype == 1 || coletree)
    {
	/* symmetric upper case: find etree of A, using triu(A) */
	/* column case: find column etree of A, which is etree of A'*A */
	Aup = A ;
	Alo = F ;
    }
    else
    {
	/* symmetric lower case: find etree of A, using tril(A) */
	/* row case: find row etree of A, which is etree of A*A' */
	Aup = F ;
	Alo = A ;
    }

    cholmod_l_etree (Aup, Parent, cm) ;

    if (cm->status < CHOLMOD_OK)
    {
	/* out of memory or matrix invalid */
	mexErrMsgTxt ("symbfact2 failed: matrix corrupted!") ;
    }

    if (cholmod_l_postorder (Parent, n, NULL, Post, cm) != n)
    {
	/* out of memory or Parent invalid */
	mexErrMsgTxt ("symbfact2 postorder failed!") ;
    }

    /* symmetric upper case: analyze tril(F), which is triu(A) */
    /* column case: analyze F*F', which is A'*A */
    /* symmetric lower case: analyze tril(A) */
    /* row case: analyze A*A' */
    cholmod_l_rowcolcounts (Alo, NULL, 0, Parent, Post, NULL, ColCount,
		First, Level, cm) ;

    if (cm->status < CHOLMOD_OK)
    {
	/* out of memory or matrix invalid */
	mexErrMsgTxt ("symbfact2 failed: matrix corrupted!") ;
    }

    /* ---------------------------------------------------------------------- */
    /* return results to MATLAB: count, h, parent, and post */
    /* ---------------------------------------------------------------------- */

    pargout [0] = sputil_put_int (ColCount, n, 0) ;
    if (nargout > 1)
    {
	/* compute the elimination tree height */
	height = 0 ;
	for (i = 0 ; i < n ; i++)
	{
	    height = MAX (height, Level [i]) ;
	}
	height++ ;
	pargout [1] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
	px = mxGetPr (pargout [1]) ;
	px [0] = height ;
    }
    if (nargout > 2)
    {
	pargout [2] = sputil_put_int (Parent, n, 1) ;
    }
    if (nargout > 3)
    {
	pargout [3] = sputil_put_int (Post, n, 1) ;
    }

    /* ---------------------------------------------------------------------- */
    /* construct L, if requested */
    /* ---------------------------------------------------------------------- */

    if (nargout > 4)
    {

	if (A->stype == 1)
	{
	    /* symmetric upper case: use triu(A) only, A2 not needed */
	    A1 = A ;
	    A2 = NULL ;
	}
	else if (A->stype == -1)
	{
	    /* symmetric lower case: use tril(A) only, A2 not needed */
	    A1 = F ;
	    A2 = NULL ;
	}
	else if (coletree)
	{
	    /* column case: analyze F*F' */
	    A1 = F ;
	    A2 = A ;
	}
	else
	{
	    /* row case: analyze A*A' */
	    A1 = A ;
	    A2 = F ;
	}

	/* count the total number of entries in L */
	lnz = 0 ;
	for (j = 0 ; j < n ; j++)
	{
	    lnz += ColCount [j] ;
	}

	/* allocate the output matrix L (pattern-only) */
	L = cholmod_l_allocate_sparse (n, n, lnz, TRUE, TRUE, 0,
	    CHOLMOD_PATTERN, cm) ;
	Lp = L->p ;
	Li = L->i ;

	/* initialize column pointers */
	lnz = 0 ;
	for (j = 0 ; j < n ; j++)
	{
	    Lp [j] = lnz ;
	    lnz += ColCount [j] ;
	}
	Lp [j] = lnz ;

	/* create a copy of the column pointers */
	W = First ;
	for (j = 0 ; j < n ; j++)
	{
	    W [j] = Lp [j] ;
	}

	/* get workspace for computing one row of L */
	R = cholmod_l_allocate_sparse (n, 1, n, FALSE, TRUE, 0, CHOLMOD_PATTERN,
		cm) ;
	Rp = R->p ;
	Ri = R->i ;

	/* compute L one row at a time */
	for (k = 0 ; k < n ; k++)
	{
	    /* get the kth row of L and store in the columns of L */
	    cholmod_l_row_subtree (A1, A2, k, Parent, R, cm) ;
	    for (p = 0 ; p < Rp [1] ; p++)
	    {
		Li [W [Ri [p]]++] = k ;
	    }
	    /* add the diagonal entry */
	    Li [W [k]++] = k ;
	}

	/* free workspace */
	cholmod_l_free_sparse (&R, cm) ;

	/* transpose L to get R, or leave as is */
	if (nargin < 3)
	{
	    /* R = L' */
	    R = cholmod_l_transpose (L, 0, cm) ;
	    cholmod_l_free_sparse (&L, cm) ;
	    L = R ;
	}

	/* fill numerical values of L with one's (only MATLAB needs this...) */
	L->x = cholmod_l_malloc (lnz, sizeof (double), cm) ;
	Lx = L->x ;
	for (p = 0 ; p < lnz ; p++)
	{
	    Lx [p] = 1 ;
	}
	L->xtype = CHOLMOD_REAL ;

	/* return L (or R) to MATLAB */
	pargout [4] = sputil_put_sparse (&L, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free (n, sizeof (Long), Parent, cm) ;
    cholmod_l_free (n, sizeof (Long), Post, cm) ;
    cholmod_l_free (n, sizeof (Long), ColCount, cm) ;
    cholmod_l_free (n, sizeof (Long), First, cm) ;
    cholmod_l_free (n, sizeof (Long), Level, cm) ;
    cholmod_l_free_sparse (&F, cm) ;
    cholmod_l_free_sparse (&S, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /*
    if (cm->malloc_count != ((nargout == 5) ? 3:0)) mexErrMsgTxt ("!") ;
    */
}
