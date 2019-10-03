/* ========================================================================== */
/* === CHOLMOD/MATLAB/etree2 mexFunction ==================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* Usage:
 *
 *	parent = etree2 (A)	      returns etree of A, uses triu(A)
 *	parent = etree2 (A, 'col')    returns etree of A'*A
 *	parent = etree2 (A, 'sym')    same as etree2 (A)
 *	parent = etree2 (A, 'row')    same as etree2 (A', 'col')
 *	parent = etree2 (A, 'lo')     returns etree of A, uses tril(A)
 *
 *	[parent post] = etree2(...)   also returns a postorder of the tree
 *
 * etree2 (A, 'col') does not form A'*A and is thus faster than etree2 (A'*A)
 * and takes less memory.  Likewise, etree (A, 'row') does not
 * form A*A', and is thus faster than etree2 (A*A') and takes less memory.
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
    Long *Parent ;
    cholmod_sparse *A, Amatrix, *S ;
    cholmod_common Common, *cm ;
    Long n, coletree, c ;
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

    if (nargout > 2 || nargin < 1 || nargin > 2)
    {
	mexErrMsgTxt ("Usage: [parent post] = etree2 (A, mode)") ;
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
	    mexErrMsgTxt ("etree2: unrecognized mode") ;
	}
    }

    if (A->stype && A->nrow != A->ncol)
    {
	mexErrMsgTxt ("etree2: A must be square") ;
    }

    /* ---------------------------------------------------------------------- */
    /* compute the etree */
    /* ---------------------------------------------------------------------- */

    Parent = cholmod_l_malloc (n, sizeof (Long), cm) ;
    if (A->stype == 1 || coletree)
    {
	/* symmetric case: find etree of A, using triu(A) */
	/* column case: find column etree of A, which is etree of A'*A */
	cholmod_l_etree (A, Parent, cm) ;
    }
    else
    {
	/* symmetric case: find etree of A, using tril(A) */
	/* row case: find row etree of A, which is etree of A*A' */
	/* R = A' */
	cholmod_sparse *R ;
	R = cholmod_l_transpose (A, 0, cm) ;
	cholmod_l_etree (R, Parent, cm) ;
	cholmod_l_free_sparse (&R, cm) ;
    }

    if (cm->status < CHOLMOD_OK)
    {
	/* out of memory or matrix invalid */
	mexErrMsgTxt ("etree2 failed: matrix corrupted!") ;
    }

    /* ---------------------------------------------------------------------- */
    /* return Parent to MATLAB */
    /* ---------------------------------------------------------------------- */

    pargout [0] = sputil_put_int (Parent, n, 1) ;

    /* ---------------------------------------------------------------------- */
    /* postorder the tree and return results to MATLAB */
    /* ---------------------------------------------------------------------- */

    if (nargout > 1)
    {
	Long *Post ;
	Post = cholmod_l_malloc (n, sizeof (Long), cm) ;
	if (cholmod_l_postorder (Parent, n, NULL, Post, cm) != n)
	{
	    /* out of memory or Parent invalid */
	    mexErrMsgTxt ("etree2 postorder failed!") ;
	}
	pargout [1] = sputil_put_int (Post, n, 1) ;
	cholmod_l_free (n, sizeof (Long), Post, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free (n, sizeof (Long), Parent, cm) ;
    cholmod_l_free_sparse (&S, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /* if (cm->malloc_count != 0) mexErrMsgTxt ("!") ; */
}
