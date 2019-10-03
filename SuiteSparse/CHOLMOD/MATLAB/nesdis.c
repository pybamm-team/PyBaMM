/* ========================================================================== */
/* === CHOLMOD/MATLAB/nesdis mexFunction ==================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * METIS is Copyrighted by G. Karypis
 * -------------------------------------------------------------------------- */

/* CHOLMOD's nested dissection, based on METIS_ComputeVertexSeparator, CAMD, and
 * CCOLAMD.
 *
 * Usage:
 *
 *	[p, cp, cmember] = nesdis (A)		orders A, using tril(A)
 *	[p, cp, cmember] = nesdis (A,'sym')	orders A, using tril(A)
 *	[p, cp, cmember] = nesdis (A,'row')	orders A*A'
 *	[p, cp, cmember] = nesdis (A,'col')	orders A'*A
 *
 * Nested dissection ordering.  Returns a permutation p such that the Cholesky
 * factorization of A(p,p), A(p,:)*A(p,:)', or A(:,p)'*A(:,p) is sparser than
 * the unpermuted system.  'mode' defaults to 'sym'.
 *
 * An optional 3rd input argument:
 *
 *	nesdis (A,mode,opts)
 *
 * specifies control parameters.  opts(1) is the smallest subgraph that should
 * not be partitioned (default is 200), opts(2) is 1 if connected components are
 * to be split independently (default is 0).  opts(3) controls when a separator
 * is kept; it is kept if nsep < opts(3)*n, where nsep is the number of nodes in
 * the separator and n is the number of nodes in the graph being cut (default is
 * 1).
 *
 * opts(4) is 0 if the smallest subgraphs are not to be ordered.  For the 'sym'
 * case, or if mode is not present: 1 if to be ordered by CAMD, or 2 if to be
 * ordered with CSYMAMD (default is 1).  For the other cases: 0 for natural
 * ordering, 1 if to be ordered by CCOLAMD.
 *
 * cp and cmember are optional.  cmember(i)=c means that node i is in component
 * c, where c is in the range of 1 to the number of components.  length(cp) is
 * the number of components found.  cp is the separator tree; cp(c) is the
 * parent of component c, or 0 if c is a root.  There can be anywhere from
 * 1 to n components, where n is the number of rows of A, A*A', or A'*A.
 *
 * Requires METIS and the CHOLMOD Partition Module.
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
    double dummy = 0 ;
    Long *Perm, *Cmember, *CParent ;
    cholmod_sparse *A, Amatrix, *C, *S ;
    cholmod_common Common, *cm ;
    Long n, transpose, c, ncomp ;
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

    if (nargout > 3 || nargin < 1 || nargin > 3)
    {
	mexErrMsgTxt ("Usage: [p cp cmember] = nesdis (A, mode, opts)") ;
    }
    if (nargin > 2)
    {
	double *x = mxGetPr (pargin [2]) ;
	n = mxGetNumberOfElements (pargin [2]) ;
	if (n > 0) cm->method [0].nd_small = x [0] ;
	if (n > 1) cm->method [0].nd_components = x [1] ;
	if (n > 2) cm->method [0].nd_oksep = x [2] ;
	if (n > 3) cm->method [0].nd_camd = x [3] ;
    }

    /* ---------------------------------------------------------------------- */
    /* get input matrix A */
    /* ---------------------------------------------------------------------- */

    A = sputil_get_sparse_pattern (pargin [0], &Amatrix, &dummy, cm) ;
    S = (A == &Amatrix) ? NULL : A ;

    /* ---------------------------------------------------------------------- */
    /* get A->stype, default is to use tril(A) */
    /* ---------------------------------------------------------------------- */

    A->stype = -1 ;
    transpose = FALSE ;

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
	    transpose = FALSE ;
	    A->stype = 0 ;
	}
	else if (tolower (c) == 'c')
	{
	    /* unsymmetric case (A'*A) if string starts with 'c' */
	    transpose = TRUE ;
	    A->stype = 0 ;
	}
	else if (tolower (c) == 's')
	{
	    /* symmetric case (A) if string starts with 's' */
	    transpose = FALSE ;
	    A->stype = -1 ;
	}
	else
	{
	    mexErrMsgTxt ("nesdis: unrecognized mode") ;
	}
    }

    if (A->stype && A->nrow != A->ncol)
    {
	mexErrMsgTxt ("nesdis: A must be square") ;
    }

    C = NULL ;
    if (transpose)
    {
	/* C = A', and then order C*C' with cholmod_l_nested_dissection */
	C = cholmod_l_transpose (A, 0, cm) ;
	if (C == NULL)
	{
	    mexErrMsgTxt ("nesdis failed") ;
	}
	A = C ;
    }

    n = A->nrow ;

    /* ---------------------------------------------------------------------- */
    /* get workspace */
    /* ---------------------------------------------------------------------- */

    CParent = cholmod_l_malloc (n, sizeof (Long), cm) ;
    Cmember = cholmod_l_malloc (n, sizeof (Long), cm) ;
    Perm = cholmod_l_malloc (n, sizeof (Long), cm) ;

    /* ---------------------------------------------------------------------- */
    /* order the matrix with CHOLMOD's nested dissection */
    /* ---------------------------------------------------------------------- */

    ncomp = cholmod_l_nested_dissection (A, NULL, 0, Perm, CParent, Cmember,cm);
    if (ncomp < 0)
    {
	mexErrMsgTxt ("nesdis failed") ;
	return ;
    }

    /* ---------------------------------------------------------------------- */
    /* return Perm, CParent, and Cmember */
    /* ---------------------------------------------------------------------- */

    pargout [0] = sputil_put_int (Perm, n, 1) ;
    if (nargout > 1)
    {
	pargout [1] = sputil_put_int (CParent, ncomp, 1) ;
    }
    if (nargout > 2)
    {
	pargout [2] = sputil_put_int (Cmember, n, 1) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free (n, sizeof (Long), Perm, cm) ;
    cholmod_l_free (n, sizeof (Long), CParent, cm) ;
    cholmod_l_free (n, sizeof (Long), Cmember, cm) ;
    cholmod_l_free_sparse (&C, cm) ;
    cholmod_l_free_sparse (&S, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /*
    if (cm->malloc_count != 0) mexErrMsgTxt ("!") ;
    */
#else
    mexErrMsgTxt ("METIS and the CHOLMOD Partition Module not installed\n") ;
#endif
}
