/* ========================================================================== */
/* === CHOLMOD/MATLAB/metis mexFunction ===================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * METIS is Copyrighted by G. Karypis
 * -------------------------------------------------------------------------- */

/* Nested dissection using METIS_NodeND
 *
 * Usage:
 *
 *	p = metis (A)		orders A, using just tril(A)
 *	p = metis (A,'sym')	orders A, using just tril(A)
 *	p = metis (A,'row')	orders A*A'
 *	p = metis (A,'col')	orders A'*A
 *
 * METIS_NodeND's ordering is followed by CHOLMOD's etree or column etree
 * postordering.  Requires METIS and the CHOLMOD Partition Module.
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
    Long *Perm ;
    cholmod_sparse *A, Amatrix, *C, *S ;
    cholmod_common Common, *cm ;
    Long n, transpose, c, postorder ;
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

    if (nargout > 1 || nargin < 1 || nargin > 3)
    {
	mexErrMsgTxt ("Usage: p = metis (A, mode)") ;
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
	    mexErrMsgTxt ("metis: p=metis(A,mode) ; unrecognized mode") ;
	}
    }

    if (A->stype && A->nrow != A->ncol)
    {
	mexErrMsgTxt ("metis: A must be square") ;
    }

    C = NULL ;
    if (transpose)
    {
	/* C = A', and then order C*C' with METIS */
	C = cholmod_l_transpose (A, 0, cm) ;
	if (C == NULL)
	{
	    mexErrMsgTxt ("metis failed") ;
	}
	A = C ;
    }

    n = A->nrow ;

    /* ---------------------------------------------------------------------- */
    /* get workspace */
    /* ---------------------------------------------------------------------- */

    Perm = cholmod_l_malloc (n, sizeof (Long), cm) ;

    /* ---------------------------------------------------------------------- */
    /* order the matrix with CHOLMOD's interface to METIS_NodeND */ 
    /* ---------------------------------------------------------------------- */

    postorder = (nargin < 3) ;
    if (!cholmod_l_metis (A, NULL, 0, postorder, Perm, cm))
    {
	mexErrMsgTxt ("metis failed") ;
	return ;
    }

    /* ---------------------------------------------------------------------- */
    /* return Perm */
    /* ---------------------------------------------------------------------- */

    pargout [0] = sputil_put_int (Perm, n, 1) ;

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free (n, sizeof (Long), Perm, cm) ;
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
