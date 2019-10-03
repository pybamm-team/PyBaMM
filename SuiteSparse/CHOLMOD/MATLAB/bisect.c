/* ========================================================================== */
/* === CHOLMOD/MATLAB/bisect mexFunction ==================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * METIS is Copyrighted by G. Karypis
 * -------------------------------------------------------------------------- */

/* Find an node separator of an undirected graph, using
 * METIS_ComputeVertexSeparator.
 *
 * Usage:
 *
 *	s = bisect (A)		bisects A, uses tril(A)
 *	s = bisect (A, 'sym')	bisects A, uses tril(A)
 *	s = bisect (A, 'row')	bisects A*A'
 *	s = bisect (A, 'col')	bisects A'*A
 *
 * Node i of the graph is in the left graph if s(i)=0, the right graph if
 * s(i)=1, and in the separator if s(i)=2.
 *
 * Requirse METIS and the CHOLMOD Partition Module.
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
    Long *Partition ;
    cholmod_sparse *A, Amatrix, *C, *S ;
    cholmod_common Common, *cm ;
    Long n, transpose, c ;
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

    if (nargout > 1 || nargin < 1 || nargin > 2)
    {
	mexErrMsgTxt ("Usage: p = bisect (A, mode)") ;
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
	c = buf [0];
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
	    mexErrMsgTxt ("bisect: p=bisect(A,mode) ; unrecognized mode") ;
	}
    }

    if (A->stype && A->nrow != A->ncol)
    {
	mexErrMsgTxt ("bisect: A must be square") ;
    }

    C = NULL ;
    if (transpose)
    {
	/* C = A', and then bisect C*C' */
	C = cholmod_l_transpose (A, 0, cm) ;
	if (C == NULL)
	{
	    mexErrMsgTxt ("bisect failed") ;
	}
	A = C ;
    }

    n = A->nrow ;

    /* ---------------------------------------------------------------------- */
    /* get workspace */
    /* ---------------------------------------------------------------------- */

    Partition = cholmod_l_malloc (n, sizeof (Long), cm) ;

    /* ---------------------------------------------------------------------- */
    /* order the matrix with CHOLMOD's interface to METIS_NodeND */ 
    /* ---------------------------------------------------------------------- */

    if (cholmod_l_bisect (A, NULL, 0, TRUE, Partition, cm) < 0)
    {
	mexErrMsgTxt ("bisect failed") ;
    }

    /* ---------------------------------------------------------------------- */
    /* return Partition */
    /* ---------------------------------------------------------------------- */

    pargout [0] = sputil_put_int (Partition, n, 0) ;

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    cholmod_l_free (n, sizeof (Long), Partition, cm) ;
    cholmod_l_free_sparse (&C, cm) ;
    cholmod_l_free_sparse (&S, cm) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /* if (cm->malloc_count != 0) mexErrMsgTxt ("!") ; */

#else
    mexErrMsgTxt ("METIS and the CHOLMOD Partition Module not installed\n") ;
#endif
}
