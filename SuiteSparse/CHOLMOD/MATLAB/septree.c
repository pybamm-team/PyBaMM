/* ========================================================================== */
/* === CHOLMOD/MATLAB/septree mexFunction =================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * METIS is Copyrighted by G. Karypis
 * -------------------------------------------------------------------------- */

/* Prune a separator tree.
 *
 * Usage:
 *
 *	[cp_new, cmember_new] = septree (cp, cmember, nd_oksep, nd_small) ;
 *
 * cp and cmember are outputs of the nesdis mexFunction.
 *
 * cmember(i)=c means that node i is in component
 * c, where c is in the range of 1 to the number of components.  length(cp) is
 * the number of components found.  cp is the separator tree; cp(c) is the
 * parent of component c, or 0 if c is a root.  There can be anywhere from
 * 1 to n components, where n is the number of rows of A, A*A', or A'*A.
 *
 * On output, cp_new and cmember_new are the new tree and graph-to-tree mapping.
 * A subtree is collapsed into a single node if the number of nodes in the
 * separator is > nd_oksep times the total size of the subtree, or if the
 * subtree has fewer than nd_small nodes.
 *
 * Requires the CHOLMOD Partition Module.
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
#ifndef NPARTITION
    double *p ;
    Long *Cmember, *CParent ;
    cholmod_common Common, *cm ;
    double nd_oksep ;
    Long nd_small, nc, n, c, j, nc_new ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set defaults */
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargout > 2 || nargin != 4)
    {
	mexErrMsgTxt ("Usage: [cp_new, cmember_new] = "
		"septree (cp, cmember, nd_oksep, nd_small)") ;
    }

    nc = mxGetNumberOfElements (pargin [0]) ;
    n  = mxGetNumberOfElements (pargin [1]) ;
    nd_oksep = mxGetScalar (pargin [2]) ;
    nd_small = mxGetScalar (pargin [3]) ;

    if (n < nc)
    {
	mexErrMsgTxt ("invalid inputs") ;
    }

    CParent = cholmod_l_malloc (nc, sizeof (Long), cm) ;
    Cmember = cholmod_l_malloc (n, sizeof (Long), cm) ;

    p = mxGetPr (pargin [0]) ;
    for (c = 0 ; c < nc ; c++)
    {
	CParent [c] = p [c] - 1 ;
	if (CParent [c] < EMPTY || CParent [c] > nc)
	{
	    mexErrMsgTxt ("cp invalid") ;
	}
    }

    p = mxGetPr (pargin [1]) ;
    for (j = 0 ; j < n ; j++)
    {
	Cmember [j] = p [j] - 1 ;
	if (Cmember [j] < 0 || Cmember [j] > nc)
	{
	    mexErrMsgTxt ("cmember invalid") ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* collapse the tree */
    /* ---------------------------------------------------------------------- */

    nc_new = cholmod_l_collapse_septree (n, nc, nd_oksep, nd_small, CParent,
	Cmember, cm) ; 
    if (nc_new < 0)
    {
	mexErrMsgTxt ("septree failed") ;
	return ;
    }

    /* ---------------------------------------------------------------------- */
    /* return CParent and Cmember */
    /* ---------------------------------------------------------------------- */

    pargout [0] = sputil_put_int (CParent, nc_new, 1) ;
    if (nargout > 1)
    {
	pargout [1] = sputil_put_int (Cmember, n, 1) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free (nc, sizeof (Long), CParent, cm) ;
    cholmod_l_free (n, sizeof (Long), Cmember, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /*
    if (cm->malloc_count != 0) mexErrMsgTxt ("!") ;
    */
#else
    mexErrMsgTxt ("CHOLMOD Partition Module not installed\n") ;
#endif
}
