/* ========================================================================== */
/* === CHOLMOD/MATLAB/mwrite mexFunction ==================================== */
/* ========================================================================== */

/* Write a matrix to a file in Matrix Market form.
 *
 *	symmetry = mwrite (filename, A, Z, comments_filename)
 *
 * A can be sparse or full.
 *
 * If present and non-empty, A and Z must have the same dimension.  Z contains
 * the explicit zero entries in the matrix (which MATLAB drops).  The entries
 * of Z appear as explicit zeros in the output file.  Z is optional.  If it is
 * an empty matrix it is ignored.  Z must be sparse or empty, if present.
 * It is ignored if A is full.
 *
 * filename is the name of the output file.  comments_file is file whose
 * contents are include after the Matrix Market header and before the first
 * data line.  Ignored if an empty string or not present.
 */

#include "cholmod_matlab.h"

#define MAXLEN 1030

/* -------------------------------------------------------------------------- */
/* mwrite mexFunction */
/* -------------------------------------------------------------------------- */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double dummy = 0 ;
    cholmod_sparse Amatrix, Zmatrix, *A, *Z ;
    cholmod_dense Xmatrix, *X ;
    cholmod_common Common, *cm ;
    Long arg_z, arg_comments, sym ;
    char filename [MAXLEN], comments [MAXLEN] ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */ 
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 2 || nargin > 4 || nargout > 1)
    {
	mexErrMsgTxt ("Usage: mwrite (filename, A, Z, comments_filename)") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the output filename */
    /* ---------------------------------------------------------------------- */

    if (!mxIsChar (pargin [0]))
    {
	mexErrMsgTxt ("first parameter must be a filename") ;
    }
    mxGetString (pargin [0], filename, MAXLEN) ;

    /* ---------------------------------------------------------------------- */
    /* get the A matrix (sparse or dense) */
    /* ---------------------------------------------------------------------- */

    if (mxIsSparse (pargin [1]))
    {
	A = sputil_get_sparse (pargin [1], &Amatrix, &dummy, 0) ;
	X = NULL ;
    }
    else
    {
	X = sputil_get_dense (pargin [1], &Xmatrix, &dummy) ;
	A = NULL ;
    }

    /* ---------------------------------------------------------------------- */
    /* determine if the Z matrix and comments_file are present */
    /* ---------------------------------------------------------------------- */

    if (nargin == 3)
    {
	if (mxIsChar (pargin [2]))
	{
	    /* mwrite (file, A, comments) */
	    arg_z = -1 ;
	    arg_comments = 2 ;
	}
	else
	{
	    /* mwrite (file, A, Z).  Ignore Z if A is full */
	    arg_z = (A == NULL) ? -1 : 2 ;
	    arg_comments = -1 ;
	}
    }
    else if (nargin == 4)
    {
	/* mwrite (file, A, Z, comments).  Ignore Z is A is full */
	arg_z = (A == NULL) ? -1 : 2 ;
	arg_comments = 3 ;
    }
    else
    {
	arg_z = -1 ;
	arg_comments = -1 ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the Z matrix */
    /* ---------------------------------------------------------------------- */

    if (arg_z == -1 ||
	mxGetM (pargin [arg_z]) == 0 || mxGetN (pargin [arg_z]) == 0)
    {
	/* A is dense, Z is not present, or Z is empty.  Ignore Z. */
	Z = NULL ;
    }
    else
    {
	/* A is sparse and Z is present and not empty */
	if (!mxIsSparse (pargin [arg_z]))
	{
	    mexErrMsgTxt ("Z must be sparse") ;
	}
	Z = sputil_get_sparse (pargin [arg_z], &Zmatrix, &dummy, 0) ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the comments filename */
    /* ---------------------------------------------------------------------- */

    comments [0] = '\0' ;
    if (arg_comments != -1)
    {
	if (!mxIsChar (pargin [arg_comments]))
	{
	    mexErrMsgTxt ("comments filename must be a string") ;
	}
	mxGetString (pargin [arg_comments], comments, MAXLEN) ;
    }

    /* ---------------------------------------------------------------------- */
    /* write the matrix to the file */
    /* ---------------------------------------------------------------------- */

    sputil_file = fopen (filename, "w") ;
    if (sputil_file == NULL)
    {
	mexErrMsgTxt ("error opening file") ;
    }
    if (A != NULL)
    {
	sym = cholmod_l_write_sparse (sputil_file, A, Z, comments, cm) ;
    }
    else
    {
	sym = cholmod_l_write_dense (sputil_file, X, comments, cm) ;
    }
    fclose (sputil_file) ;
    sputil_file = NULL ;
    if (sym < 0)
    {
	mexErrMsgTxt ("mwrite failed") ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace and return symmetry */
    /* ---------------------------------------------------------------------- */

    pargout [0] = sputil_put_int (&sym, 1, 0) ;
    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
}
