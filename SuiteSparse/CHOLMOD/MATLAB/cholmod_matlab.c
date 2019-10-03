/* ========================================================================== */
/* === MATLAB/cholmod_matlab ================================================ */
/* ========================================================================== */

/* Utility routines for the CHOLMOD MATLAB mexFunctions.
 *
 * If CHOLMOD runs out of memory, MATLAB will terminate the mexFunction
 * immediately since it uses mxMalloc (see sputil_config, below).  Likewise,
 * if mxCreate* or mxMalloc (as called in this file) fails, MATLAB will also
 * terminate the mexFunction.  When this occurs, MATLAB frees all allocated
 * memory, so we don't have to worry about memory leaks.  If this were not the
 * case, the routines in this file would suffer from memory leaks whenever an
 * error occurred.
 */

#include "cholmod_matlab.h"

#ifndef INT64_T
#define INT64_T long long
#endif

/* This file pointer is used for the mread and mwrite mexFunctions.  It must
 * be a global variable, because the file pointer is not passed to the
 * sputil_error_handler function when an error occurs. */
FILE *sputil_file = NULL ;

/* ========================================================================== */
/* === sputil_config ======================================================== */
/* ========================================================================== */

/* Define function pointers and other parameters for a mexFunction */

void sputil_config (Long spumoni, cholmod_common *cm)
{
    /* cholmod_l_solve must return a real or zomplex X for MATLAB */
    cm->prefer_zomplex = TRUE ;

    /* printing and error handling */
    if (spumoni == 0)
    {
	/* do not print anything from within CHOLMOD */
	cm->print = -1 ;
        SuiteSparse_config.printf_func = NULL ;
    }
    else
    {
	/* spumoni = 1: print warning and error messages.  cholmod_l_print_*
	 *	routines will print a one-line summary of each object printed.
	 * spumoni = 2: also print a short summary of each object.
	 */
	cm->print = spumoni + 2 ;
        /* SuiteSparse_config.printf_func = mexPrintf : */
    }

    /* error handler */
    cm->error_handler  = sputil_error_handler ;

    /* Turn off METIS memory guard.  It is not needed, because mxMalloc will
     * safely terminate the mexFunction and free any workspace without killing
     * all of MATLAB.  This assumes cholmod_make was used to compile CHOLMOD
     * for MATLAB. */
    cm->metis_memory = 0.0 ;
}


/* ========================================================================== */
/* === sputil_error_handler ================================================= */
/* ========================================================================== */

void sputil_error_handler (int status, const char *file, int line,
    const char *message)
{
    if (status < CHOLMOD_OK)
    {
	/*
	mexPrintf ("ERROR: file %s line %d, status %d\n", file, line, status) ;
	*/
	if (sputil_file != NULL)
	{
	    fclose (sputil_file) ;
	    sputil_file = NULL ;
	}
	mexErrMsgTxt (message) ;
    }
    /*
    else
    {
	mexPrintf ("Warning: file %s line %d, status %d\n", file, line, status);
    }
    */
}


/* ========================================================================== */
/* === sputil_get_sparse ==================================================== */
/* ========================================================================== */

/* Create a shallow CHOLMOD copy of a MATLAB sparse matrix.  No memory is
 * allocated.  The resulting matrix A must not be modified.
 */

cholmod_sparse *sputil_get_sparse
(
    const mxArray *Amatlab, /* MATLAB version of the matrix */
    cholmod_sparse *A,	    /* CHOLMOD version of the matrix */
    double *dummy,	    /* a pointer to a valid scalar double */
    Long stype		    /* -1: lower, 0: unsymmetric, 1: upper */
)
{
    Long *Ap ;
    A->nrow = mxGetM (Amatlab) ;
    A->ncol = mxGetN (Amatlab) ;
    A->p = (Long *) mxGetJc (Amatlab) ;
    A->i = (Long *) mxGetIr (Amatlab) ;
    Ap = A->p ;
    A->nzmax = Ap [A->ncol] ;
    A->packed = TRUE ;
    A->sorted = TRUE ;
    A->nz = NULL ;
    A->itype = CHOLMOD_LONG ;       /* was CHOLMOD_INT in v1.6 and earlier */
    A->dtype = CHOLMOD_DOUBLE ;
    A->stype = stype ;

#ifndef MATLAB6p1_OR_EARLIER

    if (mxIsLogical (Amatlab))
    {
	A->x = NULL ;
	A->z = NULL ;
	A->xtype = CHOLMOD_PATTERN ;
    }
    else if (mxIsEmpty (Amatlab))
    {
	/* this is not dereferenced, but the existence (non-NULL) of these
	 * pointers is checked in CHOLMOD */
	A->x = dummy ;
	A->z = dummy ;
	A->xtype = mxIsComplex (Amatlab) ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL ;
    }
    else if (mxIsDouble (Amatlab))
    {
	A->x = mxGetPr (Amatlab) ;
	A->z = mxGetPi (Amatlab) ;
	A->xtype = mxIsComplex (Amatlab) ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL ;
    }
    else
    {
	/* only logical and complex/real double matrices supported */
	sputil_error (ERROR_INVALID_TYPE, 0) ;
    }

#else

    if (mxIsEmpty (Amatlab))
    {
	/* this is not dereferenced, but the existence (non-NULL) of these
	 * pointers is checked in CHOLMOD */
	A->x = dummy ;
	A->z = dummy ;
	A->xtype = mxIsComplex (Amatlab) ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL ;
    }
    else
    {
	/* in MATLAB 6.1, the matrix is sparse, so it must be double */
	A->x = mxGetPr (Amatlab) ;
	A->z = mxGetPi (Amatlab) ;
	A->xtype = mxIsComplex (Amatlab) ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL ;
    }

#endif 

    return (A) ;
}


/* ========================================================================== */
/* === sputil_get_dense ===================================================== */
/* ========================================================================== */

/* Create a shallow CHOLMOD copy of a MATLAB dense matrix.  No memory is
 * allocated.  Only double (real and zomplex) matrices are supported.  The
 * resulting matrix B must not be modified.
 */

cholmod_dense *sputil_get_dense
(
    const mxArray *Amatlab, /* MATLAB version of the matrix */
    cholmod_dense *A,	    /* CHOLMOD version of the matrix */
    double *dummy	    /* a pointer to a valid scalar double */
)
{
    A->nrow = mxGetM (Amatlab) ;
    A->ncol = mxGetN (Amatlab) ;
    A->d = A->nrow ;
    A->nzmax = A->nrow * A->ncol ;
    A->dtype = CHOLMOD_DOUBLE ;

    if (mxIsEmpty (Amatlab))
    {
	A->x = dummy ;
	A->z = dummy ;
    }
    else if (mxIsDouble (Amatlab))
    {
	A->x = mxGetPr (Amatlab) ;
	A->z = mxGetPi (Amatlab) ;
    }
    else
    {
	/* only full double matrices supported by sputil_get_dense */
	sputil_error (ERROR_INVALID_TYPE, 0) ;
    }
    A->xtype = mxIsComplex (Amatlab) ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL ;

    return (A) ;
}


/* ========================================================================== */
/* === sputil_get_sparse_pattern ============================================ */
/* ========================================================================== */

/* Create a CHOLMOD_PATTERN sparse matrix for a MATLAB matrix, depending on the
 * type:
 *
 *  (1) MATLAB full real double:	duplicate CHOLMOD_REAL sparse matrix.
 *  (2) MATLAB full complex double:	duplicate CHOLMOD_ZOMPLEX sparse matrix.
 *  (3) MATLAB full logical:		duplicate CHOLMOD_PATTERN sparse matrix.
 *  (4) MATLAB sparse real double:	shallow CHOLMOD_REAL copy.
 *  (5) MATLAB sparse complex double:	shallow CHOLMOD_ZOMPLEX copy.
 *  (6) MATLAB sparse logical:		shallow CHOLMOD_PATTERN copy.
 *
 * A shallow copy or duplicate is returned; the shallow copy must not be freed.
 * For a shallow copy, the return value A is the same as Ashallow.  For a
 * complete duplicate, A and Ashallow will differ.
 */

cholmod_sparse *sputil_get_sparse_pattern
(
    const mxArray *Amatlab,	/* MATLAB version of the matrix */
    cholmod_sparse *Ashallow,	/* shallow CHOLMOD version of the matrix */
    double *dummy,		/* a pointer to a valid scalar double */
    cholmod_common *cm
)
{
    cholmod_sparse *A = NULL ;

    if (!mxIsSparse (Amatlab))
    {

	/* ------------------------------------------------------------------ */
	/* A = sparse (X) where X is full */
	/* ------------------------------------------------------------------ */

	if (mxIsDouble (Amatlab))
	{

	    /* -------------------------------------------------------------- */
	    /* convert full double X into sparse matrix A (pattern only) */
	    /* -------------------------------------------------------------- */

	    cholmod_dense Xmatrix, *X ;
	    X = sputil_get_dense (Amatlab, &Xmatrix, dummy) ;
	    A = cholmod_l_dense_to_sparse (X, FALSE, cm) ;

	}

#ifndef MATLAB6p1_OR_EARLIER

	else if (mxIsLogical (Amatlab))
	{

	    /* -------------------------------------------------------------- */
	    /* convert full logical MATLAB matrix into CHOLMOD_PATTERN */
	    /* -------------------------------------------------------------- */

	    /* (this is copied and modified from t_cholmod_dense.c) */

	    char *x ;
	    Long *Ap, *Ai ;
	    Long nrow, ncol, i, j, nz, nzmax, p ;

	    /* -------------------------------------------------------------- */
	    /* count the number of nonzeros in the result */
	    /* -------------------------------------------------------------- */

	    nrow = mxGetM (Amatlab) ;
	    ncol = mxGetN (Amatlab) ;
	    x = (char *) mxGetData (Amatlab) ;
	    nzmax = nrow * ncol ;
	    for (nz = 0, j = 0 ; j < nzmax ; j++)
	    {
		if (x [j])
		{
		    nz++ ;
		}
	    }

	    /* -------------------------------------------------------------- */
	    /* allocate the result A */
	    /* -------------------------------------------------------------- */

	    A = cholmod_l_allocate_sparse (nrow, ncol, nz, TRUE, TRUE, 0,
		    CHOLMOD_PATTERN, cm) ;

	    if (cm->status < CHOLMOD_OK)
	    {
		return (NULL) ;	    /* out of memory */
	    }
	    Ap = A->p ;
	    Ai = A->i ;

	    /* -------------------------------------------------------------- */
	    /* copy the full logical matrix into the sparse matrix A */
	    /* -------------------------------------------------------------- */

	    p = 0 ;
	    for (j = 0 ; j < ncol ; j++)
	    {
		Ap [j] = p ;
		for (i = 0 ; i < nrow ; i++)
		{
		    if (x [i+j*nrow])
		    {
			Ai [p++] = i ;
		    }
		}
	    }
	    /* ASSERT (p == nz) ; */
	    Ap [ncol] = nz ;
	}

#endif

	else
	{
	    /* only double and logical matrices supported */
	    sputil_error (ERROR_INVALID_TYPE, 0) ;
	}

    }
    else
    {

	/* ------------------------------------------------------------------ */
	/* create a shallow copy of sparse matrix A (default stype is zero) */
	/* ------------------------------------------------------------------ */

	A = sputil_get_sparse (Amatlab, Ashallow, dummy, 0) ;
	A->x = NULL ;
	A->z = NULL ;
	A->xtype = CHOLMOD_PATTERN ;
    }

    return (A) ;
}


/* ========================================================================== */
/* === sputil_put_sparse ==================================================== */
/* ========================================================================== */

/* Creates a true MATLAB version of a CHOLMOD sparse matrix.  The CHOLMOD sparse
 * matrix is destroyed.  Both real and zomplex matrices are supported.
 */

mxArray *sputil_put_sparse
(
    cholmod_sparse **Ahandle,	/* CHOLMOD version of the matrix */
    cholmod_common *cm
)
{
    mxArray *Amatlab ;
    cholmod_sparse *A ;
    A = *Ahandle ;
    if (!A || A->x == NULL) mexErrMsgTxt ("invalid sparse matrix") ;
    Amatlab = mxCreateSparse (0, 0, 0, 
	    (A->xtype != CHOLMOD_REAL) ? mxCOMPLEX: mxREAL) ;
    mxSetM (Amatlab, A->nrow) ;
    mxSetN (Amatlab, A->ncol) ;
    mxSetNzmax (Amatlab, A->nzmax) ;
    MXFREE (mxGetJc (Amatlab)) ;
    MXFREE (mxGetIr (Amatlab)) ;
    MXFREE (mxGetPr (Amatlab)) ;
    mxSetJc (Amatlab, A->p) ;
    mxSetIr (Amatlab, A->i) ;
    mxSetPr (Amatlab, A->x) ;
    mexMakeMemoryPersistent (A->p) ;
    mexMakeMemoryPersistent (A->i) ;
    mexMakeMemoryPersistent (A->x) ;
    if (A->xtype != CHOLMOD_REAL)
    {
        if (A->z == NULL) mexErrMsgTxt ("invalid complex sparse matrix") ;
	MXFREE (mxGetPi (Amatlab)) ;
	mxSetPi (Amatlab, A->z) ;
	mexMakeMemoryPersistent (A->z) ;
    }
    A->p = NULL ;
    A->i = NULL ;
    A->x = NULL ;
    A->z = NULL ;
    cholmod_l_free_sparse (Ahandle, cm) ;
    return (Amatlab) ;
}


/* ========================================================================== */
/* === sputil_put_dense ===================================================== */
/* ========================================================================== */

/* Creates a true MATLAB version of a CHOLMOD dense matrix.  The CHOLMOD dense
 * matrix is destroyed.  Both real and zomplex matrices are supported.
 */

mxArray *sputil_put_dense
(
    cholmod_dense **Ahandle,	/* CHOLMOD version of the matrix */
    cholmod_common *cm
)
{
    mxArray *Amatlab ;
    cholmod_dense *A ;
    A = *Ahandle ;
    if (!A || A->x == NULL) mexErrMsgTxt ("invalid dense matrix") ;
    Amatlab = mxCreateDoubleMatrix (0, 0,
	    (A->xtype != CHOLMOD_REAL) ? mxCOMPLEX: mxREAL) ;
    mxSetM (Amatlab, A->nrow) ;
    mxSetN (Amatlab, A->ncol) ;
    MXFREE (mxGetPr (Amatlab)) ;
    mxSetPr (Amatlab, A->x) ;
    mexMakeMemoryPersistent (A->x) ;
    if (A->xtype != CHOLMOD_REAL)
    {
        if (A->z == NULL) mexErrMsgTxt ("invalid complex dense matrix") ;
	MXFREE (mxGetPi (Amatlab)) ;
	mxSetPi (Amatlab, A->z) ;
	mexMakeMemoryPersistent (A->z) ;
    }
    A->x = NULL ;
    A->z = NULL ;
    cholmod_l_free_dense (Ahandle, cm) ;
    return (Amatlab) ;
}


/* ========================================================================== */
/* === sputil_put_int ======================================================= */
/* ========================================================================== */

/* Convert a Long vector into a double mxArray */

mxArray *sputil_put_int
(
    Long *P,		/* vector to convert */
    Long n,		/* length of P */
    Long one_based	/* 1 if convert from 0-based to 1-based, 0 otherwise */
)
{
    double *p ;
    mxArray *Q ;
    Long i ;
    Q = mxCreateDoubleMatrix (1, n, mxREAL) ;
    p = mxGetPr (Q) ;
    for (i = 0 ; i < n ; i++)
    {
	p [i] = (double) (P [i] + one_based) ;
    }
    return (Q) ;
}


/* ========================================================================== */
/* === sputil_error ========================================================= */
/* ========================================================================== */

/* An integer is out of range, or other error has occurred. */

void sputil_error
(
    Long error,	    /* kind of error */
    Long is_index    /* TRUE if a matrix index, FALSE if a matrix dimension */
)
{
    if (error == ERROR_TOO_SMALL)
    {
	mexErrMsgTxt (is_index ?
	    "sparse: index into matrix must be positive" :
	    "sparse: sparse matrix sizes must be non-negative integers") ;
    }
    else if (error == ERROR_HUGE)
    {
	mexErrMsgTxt (is_index ?
	    "sparse: index into matrix is too large" :
	    "sparse: sparse matrix size is too large") ;
    }
    else if (error == ERROR_NOT_INTEGER)
    {
	mexErrMsgTxt (is_index ?
	    "sparse: index into matrix must be an integer" :
	    "sparse: sparse matrix size must be an integer") ;
    }
    else if (error == ERROR_TOO_LARGE)
    {
	mexErrMsgTxt ("sparse: index exceeds matrix dimensions") ;
    }
    else if (error == ERROR_USAGE)
    {
	mexErrMsgTxt (
		"Usage:\n"
		"A = sparse (S)\n"
		"A = sparse (i,j,s,m,n,nzmax)\n"
		"A = sparse (i,j,s,m,n)\n"
		"A = sparse (i,j,s)\n"
		"A = sparse (m,n)\n") ;
    }
    else if (error == ERROR_LENGTH)
    {
	mexErrMsgTxt ("sparse: vectors must be the same lengths") ;
    }
    else if (error == ERROR_INVALID_TYPE)
    {
	mexErrMsgTxt ("matrix class not supported") ;
    }
}


/* ========================================================================== */
/* === sputil_double_to_int ================================================= */
/* ========================================================================== */

/* convert a double into an integer */

Long sputil_double_to_int   /* returns integer value of x */
(
    double x,	    /* double value to convert */
    Long is_index,  /* TRUE if a matrix index, FALSE if a matrix dimension */
    Long n	    /* if a matrix index, x cannot exceed this dimension,
		     * except that -1 is treated as infinity */
)
{
    Long i ;
    if (x > INT_MAX)
    {
	/* x is way too big for an integer */
	sputil_error (ERROR_HUGE, is_index) ;
    }
    else if (x < 0)
    {
	/* x must be non-negative */
	sputil_error (ERROR_TOO_SMALL, is_index) ;
    }
    i = (Long) x ;
    if (x != (double) i)
    {
	/* x must be an integer */
	sputil_error (ERROR_NOT_INTEGER, is_index) ;
    }
    if (is_index)
    {
	if (i < 1)
	{
	    sputil_error (ERROR_TOO_SMALL, is_index) ;
	}
	else if (i > n && n != EMPTY)
	{
	    sputil_error (ERROR_TOO_LARGE, is_index) ;
	}
    }
    return (i) ;
}


/* ========================================================================== */
/* === sputil_nelements ===================================================== */
/* ========================================================================== */

/* return the number of elements in an mxArray.  Trigger an error on integer
 * overflow (in case the argument is sparse) */

Long sputil_nelements (const mxArray *arg)
{
    double size ;
    const Long *dims ;
    Long k, ndims ;
    ndims = mxGetNumberOfDimensions (arg) ;
    dims = (Long *) mxGetDimensions (arg) ;
    size = 1 ;
    for (k = 0 ; k < ndims ; k++)
    {
	size *= dims [k] ;
    }
    return (sputil_double_to_int (size, FALSE, 0)) ;
}


/* ========================================================================== */
/* === sputil_get_double ==================================================== */
/* ========================================================================== */

double sputil_get_double (const mxArray *arg)
{
    if (sputil_nelements (arg) < 1)
    {
	/* [] is not a scalar, but its value is zero so that
	 * sparse ([],[],[]) is a 0-by-0 matrix */
	return (0) ;
    }
    return (mxGetScalar (arg)) ;
}


/* ========================================================================== */
/* === sputil_get_integer =================================================== */
/* ========================================================================== */

/* return an argument as a non-negative integer scalar, or -1 if error */

Long sputil_get_integer
(
    const mxArray *arg,	    /* MATLAB argument to convert */
    Long is_index,	    /* TRUE if an index, FALSE if a matrix dimension */
    Long n		    /* maximum value, if an index */
)
{
    double x = sputil_get_double (arg) ;
    if (mxIsInf (x) || mxIsNaN (x))
    {
	/* arg is Inf or NaN, return -1 */
	return (EMPTY) ;
    }
    return (sputil_double_to_int (x, is_index, n)) ;
}


/* ========================================================================== */
/* === sputil_trim ========================================================== */
/* ========================================================================== */

/* Remove columns k to n-1 from a sparse matrix S, leaving columns 0 to k-1.
 * S must be packed (there can be no S->nz array).  This condition is not
 * checked, since only packed matrices are passed to this routine.  */

void sputil_trim
(
    cholmod_sparse *S,
    Long k,
    cholmod_common *cm
)
{
    Long *Sp ;
    Long ncol ;
    size_t n1, nznew ;

    if (S == NULL)
    {
	return ;
    }

    ncol = S->ncol ;
    if (k < 0 || k >= ncol)
    {
	/* do not modify S */
	return ;
    }

    /* reduce S->p in size.  This cannot fail. */
    n1 = ncol + 1 ;
    S->p = cholmod_l_realloc (k+1, sizeof (Long), S->p, &n1, cm) ;

    /* get the new number of entries in S */
    Sp = S->p ;
    nznew = Sp [k] ;

    /* reduce S->i, S->x, and S->z (if present) to size nznew */
    cholmod_l_reallocate_sparse (nznew, S, cm) ;

    /* S now has only k columns */
    S->ncol = k ;
}


/* ========================================================================== */
/* === sputil_extract_zeros ================================================= */
/* ========================================================================== */

/* Create a sparse binary (real double) matrix Z that contains the pattern
 * of explicit zeros in the sparse real/zomplex double matrix A. */

cholmod_sparse *sputil_extract_zeros
(
    cholmod_sparse *A,
    cholmod_common *cm
)
{
    Long *Ap, *Ai, *Zp, *Zi ;
    double *Ax, *Az, *Zx ;
    Long j, p, nzeros = 0, is_complex, pz, nrow, ncol ;
    cholmod_sparse *Z ;

    if (A == NULL || A->xtype == CHOLMOD_PATTERN || A->xtype == CHOLMOD_COMPLEX)
    {
	/* only sparse real/zomplex double matrices supported */
	sputil_error (ERROR_INVALID_TYPE, 0) ;
    }

    Ap = A->p ;
    Ai = A->i ;
    Ax = A->x ;
    Az = A->z ;
    ncol = A->ncol ;
    nrow = A->nrow ;
    is_complex = (A->xtype == CHOLMOD_ZOMPLEX) ;

    /* count the number of zeros in a sparse matrix A */
    for (j = 0 ; j < ncol ; j++)
    {
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    if (CHOLMOD_IS_ZERO (Ax [p]) &&
		((is_complex) ? CHOLMOD_IS_ZERO (Az [p]) : TRUE))
	    {
		nzeros++ ;
	    }
	}
    }

    /* allocate the Z matrix with space for all the zero entries */
    Z = cholmod_l_spzeros (nrow, ncol, nzeros, CHOLMOD_REAL, cm) ;

    /* extract the zeros from A and store them in Z as binary values */
    if (nzeros > 0)
    {
	Zp = Z->p ;
	Zi = Z->i ;
	Zx = Z->x ;
	pz = 0 ;
	for (j = 0 ; j < ncol ; j++)
	{
	    Zp [j] = pz ;
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		if (CHOLMOD_IS_ZERO (Ax [p]) &&
		    ((is_complex) ? CHOLMOD_IS_ZERO (Az [p]) : TRUE))
		{
		    Zi [pz] = Ai [p] ;
		    Zx [pz] = 1 ;
		    pz++ ;
		}
	    }
	}
	Zp [ncol] = pz ;
    }

    return (Z) ;
}


/* ========================================================================== */
/* === sputil_drop_zeros ==================================================== */
/* ========================================================================== */

/* Drop zeros from a packed CHOLMOD sparse matrix (zomplex or real).  This is
 * very similar to CHOLMOD/MatrixOps/cholmod_drop, except that this routine has
 * no tolerance parameter and it can handle zomplex matrices.  NaN's are left
 * in the matrix.  If this is used on the sparse matrix version of the factor
 * L, then the update/downdate methods cannot be applied to L (ldlupdate).
 * Returns the number of entries dropped.
 */

Long sputil_drop_zeros
(
    cholmod_sparse *S
)
{
    double sik, zik ;
    Long *Sp, *Si ;
    double *Sx, *Sz ;
    Long pdest, k, ncol, p, pend, nz ;

    if (S == NULL)
    {
	return (0) ;
    }

    Sp = S->p ;
    Si = S->i ;
    Sx = S->x ;
    Sz = S->z ;
    pdest = 0 ;
    ncol = S->ncol ;
    nz = Sp [ncol] ;

    if (S->xtype == CHOLMOD_ZOMPLEX)
    {
	for (k = 0 ; k < ncol ; k++)
	{
	    p = Sp [k] ;
	    pend = Sp [k+1] ;
	    Sp [k] = pdest ;
	    for ( ; p < pend ; p++)
	    {
		sik = Sx [p] ;
		zik = Sz [p] ;
		if (CHOLMOD_IS_NONZERO (sik) || CHOLMOD_IS_NONZERO (zik))
		{
		    if (p != pdest)
		    {
			Si [pdest] = Si [p] ;
			Sx [pdest] = sik ;
			Sz [pdest] = zik ;
		    }
		    pdest++ ;
		}
	    }
	}
    }
    else
    {
	for (k = 0 ; k < ncol ; k++)
	{
	    p = Sp [k] ;
	    pend = Sp [k+1] ;
	    Sp [k] = pdest ;
	    for ( ; p < pend ; p++)
	    {
		sik = Sx [p] ;
		if (CHOLMOD_IS_NONZERO (sik))
		{
		    if (p != pdest)
		    {
			Si [pdest] = Si [p] ;
			Sx [pdest] = sik ;
		    }
		    pdest++ ;
		}
	    }
	}
    }
    Sp [ncol] = pdest ;
    return (nz - pdest) ;
}


/* ========================================================================== */
/* === sputil_copy_ij ======================================================= */
/* ========================================================================== */

/* copy i or j arguments into an Long vector.  For small integer types, i and
 * and j can be returned with negative entries; this error condition is caught
 * later, in cholmod_triplet_to_sparse.
 *
 * Future work: if the mxClassID matches the default integer (INT32 for 32-bit
 * MATLAB and INT64 for 64-bit), then it would save memory to patch in the
 * vector with a pointer copy, rather than making a copy of the whole vector.
 * This would require that the 1-based i and j vectors be converted on the fly
 * to 0-based vectors in cholmod_triplet_to_sparse.
 */

Long sputil_copy_ij		/* returns the dimension, n */
(
    Long is_scalar,	/* TRUE if argument is a scalar, FALSE otherwise */
    Long scalar,        /* scalar value of the argument */
    void *vector,	/* vector value of the argument */
    mxClassID category,	/* type of vector */
    Long nz,		/* length of output vector I */
    Long n,		/* maximum dimension, EMPTY if not yet known */
    Long *I		/* vector of length nz to copy into */
)
{
    Long i, k, ok, ok2, ok3, n2 ;

    if (is_scalar)
    {
	n2 = scalar ;
	if (n == EMPTY)
	{
	    n = scalar ;
	}
	i = scalar - 1 ;
	for (k = 0 ; k < nz ; k++)
	{
	    I [k] = i ;
	}
    }
    else
    {
	/* copy double input into Long vector (convert to 0-based) */
	ok = TRUE ;
	ok2 = TRUE ;
	ok3 = TRUE ;
	n2 = 0 ;

	switch (category)
	{

#ifndef MATLAB6p1_OR_EARLIER

	    /* MATLAB 6.1 or earlier do not have mxLOGICAL_CLASS */
	    case mxLOGICAL_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    i = (Long) (((mxLogical *) vector) [k]) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

#endif

	    case mxCHAR_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    i = (Long) (((mxChar *) vector) [k]) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

	    case mxINT8_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    i = (Long) (((INT8_T *) vector) [k]) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

	    case mxUINT8_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    i = (Long) (((UINT8_T *) vector) [k]) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

	    case mxINT16_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    i = (Long) (((INT16_T *) vector) [k]) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

	    case mxUINT16_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    i = (Long) (((UINT16_T *) vector) [k]) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

	    case mxINT32_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    i = (Long) (((INT32_T *) vector) [k]) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

	    case mxUINT32_CLASS:

		for (k = 0 ; ok3 && k < nz ; k++)
		{
		    double y = (((UINT32_T *) vector) [k]) ;
		    i = (Long) y ;
		    ok3 = (y < Long_max) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

	    case mxINT64_CLASS:

		for (k = 0 ; ok2 && ok3 && k < nz ; k++)
		{
		    INT64_T y = ((INT64_T *) vector) [k] ;
		    i = (Long) y ;
		    ok2 = (y > 0) ;
		    ok3 = (y < Long_max) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

	    case mxUINT64_CLASS:

		for (k = 0 ; ok2 && ok3 && k < nz ; k++)
		{
		    unsigned INT64_T y = ((unsigned INT64_T *) vector) [k] ;
		    i = (Long) y ;
		    ok2 = (y > 0) ;
		    ok3 = (y < Long_max) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

	    case mxSINGLE_CLASS:

		for (k = 0 ; ok && ok2 && ok3 && k < nz ; k++)
		{
		    float y = ((float *) vector) [k] ;
		    i = (Long) y ;
		    ok = (y == (float) i) ;
		    ok2 = (y > 0) ;
		    ok3 = (y < Long_max) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}
		break ;

	    case mxDOUBLE_CLASS:

		for (k = 0 ; ok && ok2 && ok3 && k < nz ; k++)
		{
		    double y = ((double *) vector) [k] ;
		    i = (Long) y ;
		    ok = (y == (double) i) ;
		    ok2 = (y > 0) ;
		    ok3 = (y < Long_max) ;
		    I [k] = i - 1 ;
		    n2 = MAX (n2, i) ;
		}

		break ;

	    default:

		sputil_error (ERROR_INVALID_TYPE, FALSE) ;
		break ;
	}

	if (!ok)
	{
	    sputil_error (ERROR_NOT_INTEGER, TRUE) ;
	}

	if (!ok2)
	{
	    sputil_error (ERROR_TOO_SMALL, TRUE) ;
	}

	if (!ok3)
	{
	    sputil_error (ERROR_HUGE, TRUE) ;
	}

    }
    return ((n == EMPTY) ? n2 : n) ;
}


/* ========================================================================== */
/* === sputil_dense_to_sparse =============================================== */
/* ========================================================================== */

/* Convert a dense matrix of any numeric type into a
 * sparse double or sparse logical matrix.
 */

#define COUNT_NZ \
{ \
    for (j = 0 ; j < ncol ; j++) \
    { \
	for (i = 0 ; i < nrow ; i++) \
	{ \
	    xij = X [i + j*nrow] ; \
	    if (CHOLMOD_IS_NONZERO (xij)) \
	    { \
		nz++ ; \
	    } \
	} \
    } \
}

#define COPY_DENSE_TO_SPARSE(stype) \
{ \
    stype *Sx ; \
    Sp = (Long *) mxGetJc (S) ; \
    Si = (Long *) mxGetIr (S) ; \
    Sx = (stype *) mxGetData (S) ; \
    nz = 0 ; \
    for (j = 0 ; j < ncol ; j++) \
    { \
	Sp [j] = nz ; \
	for (i = 0 ; i < nrow ; i++) \
	{ \
	    xij = X [i + j*nrow] ; \
	    if (CHOLMOD_IS_NONZERO (xij)) \
	    { \
		Si [nz] = i ; \
		Sx [nz] = (stype) xij ; \
		nz++ ; \
	    } \
	} \
    } \
    Sp [ncol] = nz ; \
}

#define DENSE_TO_SPARSE(type) \
{ \
    type *X, xij ; \
    X = (type *) mxGetData (arg) ; \
    COUNT_NZ ; \
    S = mxCreateSparse (nrow, ncol, nz, mxREAL) ; \
    COPY_DENSE_TO_SPARSE (double) ; \
}

mxArray *sputil_dense_to_sparse (const mxArray *arg)
{
    mxArray *S = NULL ;
    Long *Sp, *Si ;
    Long nrow, ncol, nz, i, j ;

    nrow = mxGetM (arg) ;
    ncol = mxGetN (arg) ;
    nz = 0 ;

    if (mxIsComplex (arg))
    {

	/* ------------------------------------------------------------------ */
	/* convert a complex dense matrix into a complex sparse matrix */
	/* ------------------------------------------------------------------ */

	double xij, zij ;
	double *X, *Z, *Sx, *Sz ;

	if (mxGetClassID (arg) != mxDOUBLE_CLASS)
	{
	    /* A complex matrix can have any class (int8, int16, single, etc),
	     * but this function only supports complex double.  This condition
	     * is not checked in the caller. */
	    sputil_error (ERROR_INVALID_TYPE, FALSE) ;
	}

	X = mxGetPr (arg) ;
	Z = mxGetPi (arg) ;
	for (j = 0 ; j < ncol ; j++)
	{
	    for (i = 0 ; i < nrow ; i++)
	    {
		xij = X [i + j*nrow] ;
		zij = Z [i + j*nrow] ;
		if (CHOLMOD_IS_NONZERO (xij) || CHOLMOD_IS_NONZERO (zij))
		{
		    nz++ ;
		}
	    }
	} 
	S = mxCreateSparse (nrow, ncol, nz, mxCOMPLEX) ;
	Sp = (Long *) mxGetJc (S) ;
	Si = (Long *) mxGetIr (S) ;
	Sx = mxGetPr (S) ;
	Sz = mxGetPi (S) ;
	nz = 0 ;
	for (j = 0 ; j < ncol ; j++)
	{
	    Sp [j] = nz ;
	    for (i = 0 ; i < nrow ; i++)
	    {
		xij = X [i + j*nrow] ;
		zij = Z [i + j*nrow] ;
		if (CHOLMOD_IS_NONZERO (xij) || CHOLMOD_IS_NONZERO (zij))
		{
		    Si [nz] = i ;
		    Sx [nz] = xij ;
		    Sz [nz] = zij ;
		    nz++ ;
		}
	    }
	}
	Sp [ncol] = nz ;

    }
    else
    {

	/* ------------------------------------------------------------------ */
	/* convert real matrix (any class) to sparse double or logical */
	/* ------------------------------------------------------------------ */

	switch (mxGetClassID (arg))
	{

#ifndef MATLAB6p1_OR_EARLIER

	    /* MATLAB 6.1 or earlier do not have mxLOGICAL_CLASS */
	    case mxLOGICAL_CLASS:
		{
		    mxLogical *X, xij ;
		    X = (mxLogical *) mxGetData (arg) ;
		    COUNT_NZ ;
		    S = mxCreateSparseLogicalMatrix (nrow, ncol, nz) ;
		    COPY_DENSE_TO_SPARSE (mxLogical) ;
		}
		break ;
#endif

	    case mxCHAR_CLASS:

		DENSE_TO_SPARSE (mxChar) ;
		break ;

	    case mxINT8_CLASS:

		DENSE_TO_SPARSE (char) ;
		break ;

	    case mxUINT8_CLASS:

		DENSE_TO_SPARSE (unsigned char) ;
		break ;

	    case mxINT16_CLASS:

		DENSE_TO_SPARSE (short) ;
		break ;

	    case mxUINT16_CLASS:

		DENSE_TO_SPARSE (unsigned short) ;
		break ;

	    case mxINT32_CLASS:

		DENSE_TO_SPARSE (INT32_T) ;
		break ;

	    case mxUINT32_CLASS:

		DENSE_TO_SPARSE (unsigned INT32_T) ;
		break ;

	    case mxINT64_CLASS:

		DENSE_TO_SPARSE (INT64_T) ;
		break ;

	    case mxUINT64_CLASS:

		DENSE_TO_SPARSE (unsigned INT64_T) ;
		break ;

	    case mxSINGLE_CLASS:

		DENSE_TO_SPARSE (float) ;
		break ;

	    case mxDOUBLE_CLASS:

		DENSE_TO_SPARSE (double) ;
		break ;

	    default:

		sputil_error (ERROR_INVALID_TYPE, FALSE) ;
		break ;
	}
    }

    return (S) ;
}


/* ========================================================================== */
/* === sputil_triplet_to_sparse ============================================= */
/* ========================================================================== */

/* Convert a triplet form into a sparse matrix.  If complex, s must be double.
 * If real, s can be of any class.
 */

cholmod_sparse *sputil_triplet_to_sparse
(
    Long nrow, Long ncol, Long nz, Long nzmax,
    Long i_is_scalar, Long i, void *i_vector, mxClassID i_class,
    Long j_is_scalar, Long j, void *j_vector, mxClassID j_class,
    Long s_is_scalar, double x, double z, void *x_vector, double *z_vector,
    mxClassID s_class, Long s_complex,
    cholmod_common *cm
)
{
    double dummy = 0 ;
    cholmod_triplet *T ;
    cholmod_sparse *S ;
    double *Tx, *Tz ;
    Long *Ti, *Tj ;
    Long k, x_patch ;

    /* ---------------------------------------------------------------------- */
    /* allocate the triplet form */
    /* ---------------------------------------------------------------------- */

    /* Note that nrow and ncol may be EMPTY; this is not an error condition.
     * Allocate the numerical part of T only if s is a scalar. */
    x_patch = (!s_is_scalar && (s_class == mxDOUBLE_CLASS || s_complex)) ;

    T = cholmod_l_allocate_triplet (MAX (0,nrow), MAX (0,ncol), nz, 0,
	    x_patch ? CHOLMOD_PATTERN : 
	    (s_complex ? CHOLMOD_ZOMPLEX : CHOLMOD_REAL), cm) ;
    Ti = T->i ;
    Tj = T->j ;
    Tx = T->x ;
    Tz = T->z ;

    /* ---------------------------------------------------------------------- */
    /* fill the triplet form */
    /* ---------------------------------------------------------------------- */

    if (s_is_scalar)
    {

	/* ------------------------------------------------------------------ */
	/* fill T->x and T->z with a scalar value */
	/* ------------------------------------------------------------------ */

	for (k = 0 ; k < nz ; k++)
	{
	    Tx [k] = x ;
	}
	if (s_complex)
	{
	    for (k = 0 ; k < nz ; k++)
	    {
		Tz [k] = z ;
	    }
	}

    }
    else
    {

	/* ------------------------------------------------------------------ */
	/* copy x/z_vector into T->x and T->z, and convert to double */
	/* ------------------------------------------------------------------ */

	if (s_complex)
	{

	    /* Patch in s as the numerical values of the triplet matrix.
	     * Note that T->x and T->z must not be free'd when done. */
	    T->x = (x_vector == NULL) ? &dummy : x_vector ;
	    T->z = (z_vector == NULL) ? &dummy : z_vector ;
	    T->xtype = CHOLMOD_ZOMPLEX ;

	}
	else switch (s_class)
	{

#ifndef MATLAB6p1_OR_EARLIER

	    /* MATLAB 6.1 or earlier do not have mxLOGICAL_CLASS */
	    case mxLOGICAL_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((mxLogical *) x_vector) [k]) ;
		}
		break ;

#endif

	    case mxCHAR_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((mxChar *) x_vector) [k]) ;
		}
		break ;

	    case mxINT8_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((INT8_T *) x_vector) [k]) ;
		}
		break ;

	    case mxUINT8_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((UINT8_T *) x_vector) [k]) ;
		}
		break ;

	    case mxINT16_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((INT16_T *) x_vector) [k]) ;
		}
		break ;

	    case mxUINT16_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((UINT16_T *) x_vector) [k]) ;
		}
		break ;

	    case mxINT32_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((INT32_T *) x_vector) [k]) ;
		}
		break ;

	    case mxUINT32_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((UINT32_T *) x_vector) [k]) ;
		}
		break ;

	    case mxINT64_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((INT64_T *) x_vector) [k]) ;
		}
		break ;

	    case mxUINT64_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((unsigned INT64_T *) x_vector) [k]) ;
		}
		break ;

	    case mxSINGLE_CLASS:

		for (k = 0 ; k < nz ; k++)
		{
		    Tx [k] = (double) (((float *) x_vector) [k]) ;
		}
		break ;

	    case mxDOUBLE_CLASS:

		/* Patch in s as the numerical values of the triplet matrix.
		 * Note that T->x must not be free'd when done. */
		T->x = (x_vector == NULL) ? &dummy : x_vector ;
		T->xtype = CHOLMOD_REAL ;
		break ;

	    default:

		sputil_error (ERROR_INVALID_TYPE, FALSE) ;
		break ;
	}
    }

    /* copy i in to the integer vector T->i */
    nrow = sputil_copy_ij (i_is_scalar, i, i_vector, i_class, nz, nrow, Ti) ;

    /* copy j in to the integer vector T->j */
    ncol = sputil_copy_ij (j_is_scalar, j, j_vector, j_class, nz, ncol, Tj) ;

    /* nrow and ncol are known */
    T->nrow = nrow ;
    T->ncol = ncol ;
    T->nnz = nz ;

    /* ---------------------------------------------------------------------- */
    /* convert triplet to sparse matrix */
    /* ---------------------------------------------------------------------- */

    /* If the triplet matrix T is invalid, or if CHOLMOD runs out of memory,
     * then S is NULL. */
    S = cholmod_l_triplet_to_sparse (T, nzmax, cm) ;

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    /* do not free T->x or T->z if it points to input x_vector */
    if (x_patch)
    {
	T->x = NULL ;
	T->z = NULL ;
	T->xtype = CHOLMOD_PATTERN ;
    }
    cholmod_l_free_triplet (&T, cm) ;
    return (S) ;
}


/* ========================================================================== */
/* === sputil_copy_sparse =================================================== */
/* ========================================================================== */

/* copy a sparse matrix, S = sparse(A), dropping any zero entries and ensuring
 * the nzmax(S) == nnz(S).   Explicit zero entries in A "cannot" occur, in
 * the current version of MATLAB ... but a user mexFunction might generate a
 * matrix with explicit zeros.  This function ensures S=sparse(A) drops those
 * explicit zeros. */

mxArray *sputil_copy_sparse (const mxArray *A)
{
    double aij, zij ;
    mxArray *S ;
    double *Ax, *Az, *Sx, *Sz ;
    Long *Ap, *Ai, *Sp, *Si ;
    Long anz, snz, p, j, nrow, ncol, pend ;

#ifndef MATLAB6p1_OR_EARLIER

    /* MATLAB 6.1 or earlier : all sparse matrices are OK */
    if (! (mxGetClassID (A) == mxLOGICAL_CLASS ||
	   mxGetClassID (A) == mxDOUBLE_CLASS))
    {
	/* Only sparse logical and real/complex double matrices supported.
	 * This condition is not checked in the caller. */
	sputil_error (ERROR_INVALID_TYPE, 0) ;
    }

#endif

    nrow = mxGetM (A) ;
    ncol = mxGetN (A) ;
    Ap = (Long *) mxGetJc (A) ;
    Ai = (Long *) mxGetIr (A) ;
    anz = Ap [ncol] ;

    snz = 0 ;

#ifndef MATLAB6p1_OR_EARLIER

    /* MATLAB 6.1 or earlier do not have mxLOGICAL_CLASS */
    if (mxIsLogical (A))
    {

	/* ------------------------------------------------------------------ */
	/* copy a sparse logical matrix */
	/* ------------------------------------------------------------------ */

	/* count the number of nonzeros in A */
	mxLogical *Al, *Sl ;
	Al = mxGetLogicals (A) ;
	for (p = 0 ; p < anz ; p++)
	{
	    if (Al [p])
	    {
		snz++ ;
	    }
	}

	/* allocate S */
	S = mxCreateSparseLogicalMatrix (nrow, ncol, snz) ;
	Sp = (Long *) mxGetJc (S) ;
	Si = (Long *) mxGetIr (S) ;
	Sl = mxGetLogicals (S) ;

	/* copy A into S, dropping zero entries */
	snz = 0 ;
	for (j = 0 ; j < ncol ; j++)
	{
	    Sp [j] = snz ;
	    pend = Ap [j+1] ;
	    for (p = Ap [j] ; p < pend ; p++)
	    {
		if (Al [p])
		{
		    Si [snz] = Ai [p] ;
		    Sl [snz] = 1 ;
		    snz++ ;
		}
	    }
	}

    }
    else

#endif

    if (mxIsComplex (A))
    {

	/* ------------------------------------------------------------------ */
	/* copy a sparse complex double matrix */
	/* ------------------------------------------------------------------ */

	/* count the number of nonzeros in A */
	Ax = mxGetPr (A) ;
	Az = mxGetPi (A) ;
	for (p = 0 ; p < anz ; p++)
	{
	    aij = Ax [p] ;
	    zij = Az [p] ;
	    if (CHOLMOD_IS_NONZERO (aij) || CHOLMOD_IS_NONZERO (zij))
	    {
		snz++ ;
	    }
	}

	/* allocate S */
	S = mxCreateSparse (nrow, ncol, snz, mxCOMPLEX) ;
	Sp = (Long *) mxGetJc (S) ;
	Si = (Long *) mxGetIr (S) ;
	Sx = mxGetPr (S) ;
	Sz = mxGetPi (S) ;

	/* copy A into S, dropping zero entries */
	snz = 0 ;
	for (j = 0 ; j < ncol ; j++)
	{
	    Sp [j] = snz ;
	    pend = Ap [j+1] ;
	    for (p = Ap [j] ; p < pend ; p++)
	    {
		aij = Ax [p] ;
		zij = Az [p] ;
		if (CHOLMOD_IS_NONZERO (aij) || CHOLMOD_IS_NONZERO (zij))
		{
		    Si [snz] = Ai [p] ;
		    Sx [snz] = aij ;
		    Sz [snz] = zij ;
		    snz++ ;
		}
	    }
	}

    }
    else
    {

	/* ------------------------------------------------------------------ */
	/* copy a sparse real double matrix */
	/* ------------------------------------------------------------------ */

	/* count the number of nonzeros in A */
	Ax = mxGetPr (A) ;
	for (p = 0 ; p < anz ; p++)
	{
	    aij = Ax [p] ;
	    if (CHOLMOD_IS_NONZERO (aij))
	    {
		snz++ ;
	    }
	}

	/* allocate S */
	S = mxCreateSparse (nrow, ncol, snz, mxREAL) ;
	Sp = (Long *) mxGetJc (S) ;
	Si = (Long *) mxGetIr (S) ;
	Sx = mxGetPr (S) ;

	/* copy A into S, dropping zero entries */
	snz = 0 ;
	for (j = 0 ; j < ncol ; j++)
	{
	    Sp [j] = snz ;
	    pend = Ap [j+1] ;
	    for (p = Ap [j] ; p < pend ; p++)
	    {
		aij = Ax [p] ;
		if (CHOLMOD_IS_NONZERO (aij))
		{
		    Si [snz] = Ai [p] ;
		    Sx [snz] = aij ;
		    snz++ ;
		}
	    }
	}
    }

    Sp [ncol] = snz ;
    return (S) ;
}


/* ========================================================================== */
/* === sputil_sparse_to_dense =============================================== */
/* ========================================================================== */

/* convert a sparse double or logical array to a dense double array */

mxArray *sputil_sparse_to_dense (const mxArray *S)
{
    mxArray *X ;
    double *Sx, *Sz, *Xx, *Xz ;
    Long *Sp, *Si ;
    Long nrow, ncol, i, j, p, pend, j2 ;

#ifndef MATLAB6p1_OR_EARLIER

    /* MATLAB 6.1 or earlier : all sparse matrices are OK */
    if (! (mxGetClassID (S) == mxLOGICAL_CLASS ||
	   mxGetClassID (S) == mxDOUBLE_CLASS))
    {
	/* only sparse logical and real/complex double matrices supported */
	sputil_error (ERROR_INVALID_TYPE, 0) ;
    }

#endif

    nrow = mxGetM (S) ;
    ncol = mxGetN (S) ;
    Sp = (Long *) mxGetJc (S) ;
    Si = (Long *) mxGetIr (S) ;

#ifndef MATLAB6p1_OR_EARLIER

    /* MATLAB 6.1 or earlier do not have mxLOGICAL_CLASS */
    if (mxIsLogical (S))
    {
	/* logical */
	mxLogical *Sl ;
	Sl = (mxLogical *) mxGetData (S) ;
	X = mxCreateDoubleMatrix (nrow, ncol, mxREAL) ;
	Xx = mxGetPr (X) ;
	for (j = 0 ; j < ncol ; j++)
	{
	    pend = Sp [j+1] ;
	    j2 = j*nrow ;
	    for (p = Sp [j] ; p < pend ; p++)
	    {
		Xx [Si [p] + j2] = (double) (Sl [p]) ;
	    }
	}
    }
    else

#endif 

    if (mxIsComplex (S))
    {
	/* complex */
	Sx = mxGetPr (S) ;
	Sz = mxGetPi (S) ;
	X = mxCreateDoubleMatrix (nrow, ncol, mxCOMPLEX) ;
	Xx = mxGetPr (X) ;
	Xz = mxGetPi (X) ;
	for (j = 0 ; j < ncol ; j++)
	{
	    pend = Sp [j+1] ;
	    j2 = j*nrow ;
	    for (p = Sp [j] ; p < pend ; p++)
	    {
		i = Si [p] ;
		Xx [i + j2] = Sx [p] ;
		Xz [i + j2] = Sz [p] ;
	    }
	}
    }
    else
    {
	/* real */
	Sx = mxGetPr (S) ;
	X = mxCreateDoubleMatrix (nrow, ncol, mxREAL) ;
	Xx = mxGetPr (X) ;
	for (j = 0 ; j < ncol ; j++)
	{
	    pend = Sp [j+1] ;
	    j2 = j*nrow ;
	    for (p = Sp [j] ; p < pend ; p++)
	    {
		Xx [Si [p] + j2] = Sx [p] ;
	    }
	}
    }

    return (X) ;
}


/* ========================================================================== */
/* === sputil_check_ijvector ================================================ */
/* ========================================================================== */

/* Check a sparse i or j input argument */

void sputil_check_ijvector (const mxArray *arg)
{
    if (mxIsComplex (arg))
    {
	/* i and j cannot be complex */
	sputil_error (ERROR_NOT_INTEGER, TRUE) ;
    }
    if (mxIsSparse (arg))
    {
	/* the i and j arguments for sparse(i,j,s,...) can be sparse, but if so
	 * they must have no zero entries. */
	double mn, m, nz ;
	Long *p, n ;
	m = (double) mxGetM (arg) ;
	n =  mxGetN (arg) ;
	mn = m*n ;
	p = (Long *) mxGetJc (arg) ;
	nz = p [n] ;
	if (mn != nz)
	{
	    /* i or j contains at least one zero, which is invalid */
	    sputil_error (ERROR_TOO_SMALL, TRUE) ;
	}
    }
}


/* ========================================================================== */
/* === sputil_sparse ======================================================== */
/* ========================================================================== */

/* Implements the sparse2 mexFunction */

void sputil_sparse
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    double x, z ;
    double *z_vector ;
    void *i_vector, *j_vector, *x_vector ;
    mxArray *s_array ;
    cholmod_sparse *S, *Z ;
    cholmod_common Common, *cm ;
    Long nrow, ncol, k, nz, i_is_scalar, j_is_scalar, s_is_sparse,
	s_is_scalar, ilen, jlen, slen, nzmax, i, j, s_complex, ndropped ;
    mxClassID i_class, j_class, s_class ;

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD and set defaults */
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil_config (SPUMONI, cm) ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargout > 2 || nargin > 6 || nargin == 4 || nargin == 0)
    {
	sputil_error (ERROR_USAGE, FALSE) ;
    }

    /* ---------------------------------------------------------------------- */
    /* convert inputs into a sparse matrix S */
    /* ---------------------------------------------------------------------- */

    S = NULL ;
    Z = NULL ;

    if (nargin == 1)
    {

	/* ------------------------------------------------------------------ */
	/* S = sparse (A) where A is sparse or full */
	/* ------------------------------------------------------------------ */

	nrow = mxGetM (pargin [0]) ;
	ncol = mxGetN (pargin [0]) ;

	if (mxIsSparse (pargin [0]))
	{

	    /* -------------------------------------------------------------- */
	    /* S = sparse (A) where A is sparse (double, complex, or logical) */
	    /* -------------------------------------------------------------- */

	    pargout [0] = sputil_copy_sparse (pargin [0]) ;

	}
	else
	{

	    /* -------------------------------------------------------------- */
	    /* S = sparse (A) where A is full (real or complex) */
	    /* -------------------------------------------------------------- */

	    /* A can be of any numeric type (mxLogical, int8, ..., double),
	     * except that if A is complex, it must also be double. */
	    pargout [0] = sputil_dense_to_sparse (pargin [0]) ;
	}

    }
    else if (nargin == 2)
    {

	/* ------------------------------------------------------------------ */
	/* S = sparse (m,n) */
	/* ------------------------------------------------------------------ */

	Long *Sp ;
	nrow = sputil_get_integer (pargin [0], FALSE, 0) ;
	ncol = sputil_get_integer (pargin [1], FALSE, 0) ;
	pargout [0] = mxCreateSparse (nrow, ncol, 1, mxREAL) ;
	Sp = (Long *) mxGetJc (pargout [0]) ;
	Sp [0] = 0 ;

    }
    else
    {

	/* ------------------------------------------------------------------ */
	/* S = sparse (i,j,s), sparse (i,j,s,m,n) or sparse (i,j,s,m,n,nzmax) */
	/* ------------------------------------------------------------------ */

	/* i, j, and s can be of any numeric type */

	/* ensure i and j are valid (i and j cannot be complex) */
	sputil_check_ijvector (pargin [0]) ;
	sputil_check_ijvector (pargin [1]) ;

	/* convert s from sparse to dense (double), if necessary */
	s_is_sparse = mxIsSparse (pargin [2]) ;
	if (s_is_sparse)
	{
	    /* s must be double (real/complex) or logical */
	    s_array = sputil_sparse_to_dense (pargin [2]) ;
	}
	else
	{
	    s_array = (mxArray *) pargin [2] ;
	}

	/* s is now full.  It can be any class, except if complex it must also
	 * be double */
	s_class = mxGetClassID (s_array) ;
	s_complex = mxIsComplex (s_array) ;
	if (s_complex && s_class != mxDOUBLE_CLASS)
	{
	    /* for complex case, only double class is supported */
	    sputil_error (ERROR_INVALID_TYPE, 0) ;
	}

	/* get sizes of inputs */
	ilen = sputil_nelements (pargin [0]) ;
	jlen = sputil_nelements (pargin [1]) ;
	slen = sputil_nelements (s_array) ;

	/* if i, j, s are scalars, they "float" to sizes of non-scalar args */
	i_is_scalar = (ilen == 1) ;
	j_is_scalar = (jlen == 1) ;
	s_is_scalar = (slen == 1) ;

	/* find the length */
	if (!i_is_scalar)
	{
	    /* if i is not a scalar, let it determine the length */
	    nz = ilen ;
	}
	else if (!j_is_scalar)
	{
	    /* otherwise, if j is not a scalar, let it determine the length */
	    nz = jlen ;
	}
	else
	{
	    /* finally, i and j are both scalars, so let s determine length */
	    nz = slen ;
	}

	/* make sure the sizes are compatible */
	if (!((i_is_scalar || ilen == nz) &&
	      (j_is_scalar || jlen == nz) &&
	      (s_is_scalar || slen == nz)))
	{
	    sputil_error (ERROR_LENGTH, FALSE) ;
	}

	if (nargin > 4)
	{
	    nrow = sputil_get_integer (pargin [3], FALSE, 0) ;
	    ncol = sputil_get_integer (pargin [4], FALSE, 0) ;
	}
	else
	{
	    /* nrow and ncol will be discovered by scanning i and j */
	    nrow = EMPTY ;
	    ncol = EMPTY ;
	}

	if (nargin > 5)
	{
	    nzmax = sputil_get_integer (pargin [5], FALSE, 0) ;
	    nzmax = MAX (nzmax, nz) ;
	}
	else
	{
	    nzmax = nz ;
	}

	/* ------------------------------------------------------------------ */
	/* convert triplet form to sparse form */
	/* ------------------------------------------------------------------ */

	i = i_is_scalar ? sputil_get_integer (pargin [0], TRUE, nrow) : 0 ;
	i_vector = mxGetData (pargin [0]) ;
	i_class = mxGetClassID (pargin [0]) ;

	j = j_is_scalar ? sputil_get_integer (pargin [1], TRUE, ncol) : 0 ;
	j_vector = mxGetData (pargin [1]) ;
	j_class = mxGetClassID (pargin [1]) ;

	x_vector = mxGetData (s_array) ;
	z_vector = mxGetPi (s_array) ;
	x = sputil_get_double (s_array) ;
	z = (s_complex && z_vector != NULL) ? (z_vector [0]) : 0 ;

	S = sputil_triplet_to_sparse (nrow, ncol, nz, nzmax,
		i_is_scalar, i, i_vector, i_class,
		j_is_scalar, j, j_vector, j_class,
		s_is_scalar, x, z, x_vector, z_vector,
		s_class, s_complex,
		cm) ;

	/* set nzmax(S) to nnz(S), unless nzmax is specified on input */
	if (nargin <= 5 && S != NULL)
	{
	    cholmod_l_reallocate_sparse (cholmod_l_nnz (S, cm), S, cm) ;
	}

	if (nargout > 1)
	{
	    /* return a binary pattern of the explicit zero entries, for the
	     * [S Z] = sparse(i,j,x, ...) form. */
	    Z = sputil_extract_zeros (S, cm) ;
	}

	/* drop explicit zeros from S */
	ndropped = sputil_drop_zeros (S) ;

	/* if entries dropped, set nzmax(S) to nnz(S), unless nzmax specified */
	if (ndropped > 0 && nargin <= 5 && S != NULL)
	{
	    cholmod_l_reallocate_sparse (cholmod_l_nnz (S, cm), S, cm) ;
	}

	if (s_is_sparse)
	{
	    mxDestroyArray (s_array) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* convert S into a MATLAB sparse matrix */
    /* ---------------------------------------------------------------------- */

    k = 0 ;
    if (S != NULL)
    {

#ifndef MATLAB6p1_OR_EARLIER

	/* MATLAB 6.1 or earlier do not have mxLOGICAL_CLASS */
	if (mxIsLogical (pargin [2]))
	{
	    /* copy S into a MATLAB sparse logical matrix */
	    mxLogical *s_logical ;
	    pargout [0] = mxCreateSparseLogicalMatrix (0, 0, 0) ;
	    s_logical = cholmod_l_malloc (S->nzmax, sizeof (mxLogical), cm) ;
	    for (k = 0 ; k < (Long) (S->nzmax) ; k++)
	    {
		s_logical [k] = 1 ;
	    }
	    MXFREE (mxGetData (pargout [0])) ;
	    mxSetData (pargout [0], s_logical) ;
	    mexMakeMemoryPersistent (s_logical) ;
	    k++ ;
	}
	else

#endif	
        if (S->x == NULL) mexErrMsgTxt ("invalid sparse matrix") ;

	if (mxIsComplex (pargin [2]))
	{
	    /* copy S into a MATLAB sparse complex double matrix */
            if (S->z == NULL) mexErrMsgTxt ("invalid complex sparse matrix") ;
	    pargout [0] = mxCreateSparse (0, 0, 0, mxCOMPLEX) ;
	    MXFREE (mxGetPr (pargout [0])) ;
	    MXFREE (mxGetPi (pargout [0])) ;
	    mxSetPr (pargout [0], S->x) ;
	    mxSetPi (pargout [0], S->z) ;
	    mexMakeMemoryPersistent (S->x) ;
	    mexMakeMemoryPersistent (S->z) ;
	    k += 2 ;
	    S->x = NULL ;
	    S->z = NULL ;
	}
	else
	{
	    /* copy S into a MATLAB sparse real double matrix */
	    pargout [0] = mxCreateSparse (0, 0, 0, mxREAL) ;
	    mxSetPr (pargout [0], S->x) ;
	    mexMakeMemoryPersistent (S->x) ;
	    k++ ;
	    S->x = NULL ;
	}

	mxSetM (pargout [0], S->nrow) ;
	mxSetN (pargout [0], S->ncol) ;
	mxSetNzmax (pargout [0], S->nzmax) ;
	MXFREE (mxGetJc (pargout [0])) ;
	MXFREE (mxGetIr (pargout [0])) ;
	mxSetJc (pargout [0], S->p) ;
	mxSetIr (pargout [0], S->i) ;
	mexMakeMemoryPersistent (S->p) ;
	mexMakeMemoryPersistent (S->i) ;
	k += 2 ;

	/* free cholmod_sparse S, except for what has been given to MATLAB */
	S->p = NULL ;
	S->i = NULL ;
	cholmod_l_free_sparse (&S, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* return Z to MATLAB, if requested */
    /* ---------------------------------------------------------------------- */

    if (nargout > 1)
    {
	if (Z == NULL)
	{
	    /* Z not computed; return an empty matrix */
	    Z = cholmod_l_spzeros (nrow, ncol, 0, CHOLMOD_REAL, cm) ;
	}
	pargout [1] = sputil_put_sparse (&Z, cm) ;
    }

    cholmod_l_finish (cm) ;
    cholmod_l_print_common (" ", cm) ;
    /*
    if (cm->malloc_count != k) mexErrMsgTxt ("!") ;
    */
}
