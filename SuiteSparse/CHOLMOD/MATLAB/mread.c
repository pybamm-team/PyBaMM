/* ========================================================================== */
/* === CHOLMOD/MATLAB/mread mexFunction ===================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/MATLAB Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * MATLAB(tm) is a Trademark of The MathWorks, Inc.
 * -------------------------------------------------------------------------- */

/* [A Z] = mread (filename, prefer_binary)
 *
 * Read a sparse or dense matrix from a file in Matrix Market format.
 *
 * All MatrixMarket formats are supported.
 * The Matrix Market "integer" format is converted into real, but the values
 * are preserved.  The "pattern" format is converted into real.  If a pattern
 * matrix is unsymmetric, all of its values are equal to one.  If a pattern is
 * symmetric, the kth diagonal entry is set to one plus the number of
 * off-diagonal nonzeros in row/column k, and off-diagonal entries are set to
 * -1.
 *
 * Explicit zero entries are returned as the binary pattern of the matrix Z.
 */

#include "cholmod_matlab.h"

/* maximum file length */
#define MAXLEN 1030

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    void *G ;
    cholmod_dense *X = NULL ;
    cholmod_sparse *A = NULL, *Z = NULL ;
    cholmod_common Common, *cm ;
    Long *Ap = NULL, *Ai ;
    double *Ax, *Az = NULL ;
    char filename [MAXLEN] ;
    Long nz, k, is_complex = FALSE, nrow = 0, ncol = 0, allzero ;
    int mtype ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set parameters */ 
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 1 || nargin > 2 || nargout > 2)
    {
	mexErrMsgTxt ("usage: [A Z] = mread (filename, prefer_binary)") ;
    }
    if (!mxIsChar (pargin [0]))
    {
	mexErrMsgTxt ("mread requires a filename") ;
    }
    mxGetString (pargin [0], filename, MAXLEN) ;
    sputil_file = fopen (filename, "r") ;
    if (sputil_file == NULL)
    {
	mexErrMsgTxt ("cannot open file") ;
    }
    if (nargin > 1)
    {
	cm->prefer_binary = (mxGetScalar (pargin [1]) != 0) ;
    }

    /* ---------------------------------------------------------------------- */
    /* read the matrix, as either a dense or sparse matrix */
    /* ---------------------------------------------------------------------- */

    G = cholmod_l_read_matrix (sputil_file, 1, &mtype, cm) ;
    fclose (sputil_file) ;
    sputil_file = NULL ;
    if (G == NULL)
    {
	mexErrMsgTxt ("could not read file") ;
    }

    /* get the specific matrix (A or X), and change to ZOMPLEX if needed */
    if (mtype == CHOLMOD_SPARSE)
    {
	A = (cholmod_sparse *) G ;
	nrow = A->nrow ;
	ncol = A->ncol ;
	is_complex = (A->xtype == CHOLMOD_COMPLEX) ;
	Ap = A->p ;
	Ai = A->i ;
	if (is_complex)
	{
	    /* if complex, ensure A is ZOMPLEX */
	    cholmod_l_sparse_xtype (CHOLMOD_ZOMPLEX, A, cm) ;
	}
	Ax = A->x ;
	Az = A->z ;
    }
    else if (mtype == CHOLMOD_DENSE)
    {
	X = (cholmod_dense *) G ;
	nrow = X->nrow ;
	ncol = X->ncol ;
	is_complex = (X->xtype == CHOLMOD_COMPLEX) ;
	if (is_complex)
	{
	    /* if complex, ensure X is ZOMPLEX */
	    cholmod_l_dense_xtype (CHOLMOD_ZOMPLEX, X, cm) ;
	}
	Ax = X->x ;
	Az = X->z ;
    }
    else
    {
	mexErrMsgTxt ("invalid file") ;
    }

    /* ---------------------------------------------------------------------- */
    /* if requested, extract the zero entries and place them in Z */
    /* ---------------------------------------------------------------------- */

    if (nargout > 1)
    {
	if (mtype == CHOLMOD_SPARSE)
	{
	    /* A is a sparse real/zomplex double matrix */
	    Z = sputil_extract_zeros (A, cm) ;
	}
	else
	{
	    /* input is full; just return an empty Z matrix */
	    Z = cholmod_l_spzeros (nrow, ncol, 0, CHOLMOD_REAL, cm) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* prune the zero entries from A and set nzmax(A) to nnz(A) */
    /* ---------------------------------------------------------------------- */

    if (mtype == CHOLMOD_SPARSE)
    {
	sputil_drop_zeros (A) ;
	cholmod_l_reallocate_sparse (cholmod_l_nnz (A, cm), A, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* change a complex matrix to real if its imaginary part is all zero */
    /* ---------------------------------------------------------------------- */

    if (is_complex)
    {
	if (mtype == CHOLMOD_SPARSE)
	{
	    nz = Ap [ncol] ;
	}
	else
	{
	    nz = nrow * ncol ;
	}
	allzero = TRUE ;
	for (k = 0 ; k < nz ; k++)
	{
	    if (Az [k] != 0)
	    {
		allzero = FALSE ;
		break ;
	    }
	}
	if (allzero)
	{
	    /* discard the all-zero imaginary part */
	    if (mtype == CHOLMOD_SPARSE)
	    {
		cholmod_l_sparse_xtype (CHOLMOD_REAL, A, cm) ;
	    }
	    else
	    {
		cholmod_l_dense_xtype (CHOLMOD_REAL, X, cm) ;
	    }
	}
    }

    /* ---------------------------------------------------------------------- */
    /* return results to MATLAB */
    /* ---------------------------------------------------------------------- */

    if (mtype == CHOLMOD_SPARSE)
    {
	pargout [0] = sputil_put_sparse (&A, cm) ;
    }
    else
    {
	pargout [0] = sputil_put_dense (&X, cm) ;
    }
    if (nargout > 1)
    {
	pargout [1] = sputil_put_sparse (&Z, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
}
