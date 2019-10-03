/* ========================================================================== */
/* === SuiteSparseCollection/ssfull_write.c ================================= */
/* ========================================================================== */

/* SuiteSparseCollection: a MATLAB toolbox for managing the SuiteSparse Matrix
 * Collection.
 */

/* Copyright 2007-2018, Timothy A. Davis, http://www.suitesparse.com */

/* ========================================================================== */

/* ssfull_write (filename, X): write a full matrix to a file.  A small subset of
 * the Matrix Market format is used.  The first line is one of:
 *
 *	%%MatrixMarket matrix array real general
 *	%%MatrixMarket matrix array complex general
 * 
 * The 2nd line contains two numbers: m and n, where X is m-by-n.  The next
 * m*n lines contain the numerical values (one per line if real, two per line
 * if complex, containing the real and imaginary parts).  The values are
 * listed in column-major order.  The resulting file can be read by any
 * Matrix Market reader, or by ssfull_read.  No comments or blank lines are
 * used.
 */

#ifndef NLARGEFILE
#include "io64.h"
#endif

#include "mex.h"
#include <math.h>
#define MAXLINE 1030
#define BIG 1e308

/* -------------------------------------------------------------------------- */
/* print_value */
/* -------------------------------------------------------------------------- */

static void print_value (FILE *f, double x, char *s)
{
    double y ;
    int k, width ;

    /* change -inf to -BIG, and change +inf and nan to +BIG */
    if (x != x || x >= BIG)
    {
	x = BIG ;
    }
    else if (x <= -BIG)
    {
	x = -BIG ;
    }

    /* convert to int and back again */
    k = (int) x ;
    y = (double) k ;
    if (y == x)
    {
	/* x is a small integer */
	fprintf (f, "%d", k) ;
    }
    else
    {
	/* x is not an integer, use the smallest width possible */
	for (width = 6 ; width < 20 ; width++)
	{
	    /* write the value to a string, read it back in, and check */
	    sprintf (s, "%.*g", width, x) ;
	    sscanf (s, "%lg", &y) ;
	    if (x == y) break ;
	}
	fprintf (f, "%s", s) ;
    }
}


/* -------------------------------------------------------------------------- */
/* ssfull */
/* -------------------------------------------------------------------------- */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    int iscomplex ;
    mwSignedIndex nrow, ncol, i, j ;
    double *Ax, *Az ;
    char filename [MAXLINE], s [MAXLINE] ;
    FILE *f ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (nargout > 0 || nargin != 2)
    {
	mexErrMsgTxt ("usage: ssfull (filename,A)") ;
    }
    if (mxIsSparse (pargin [1]) || !mxIsClass (pargin [1], "double"))
    {
	mexErrMsgTxt ("A must be full and double") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get filename and open the file */
    /* ---------------------------------------------------------------------- */

    if (!mxIsChar (pargin [0]))
    {
	mexErrMsgTxt ("first parameter must be a filename") ;
    }
    mxGetString (pargin [0], filename, MAXLINE) ;
    f = fopen (filename, "w") ;
    if (f == NULL)
    {
	mexErrMsgTxt ("error openning file") ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the matrix */
    /* ---------------------------------------------------------------------- */

    iscomplex = mxIsComplex (pargin [1]) ;
    nrow = mxGetM (pargin [1]) ;
    ncol = mxGetN (pargin [1]) ;
    Ax = mxGetPr (pargin [1]) ;
    Az = mxGetPi (pargin [1]) ;

    /* ---------------------------------------------------------------------- */
    /* write the matrix */
    /* ---------------------------------------------------------------------- */

    if (iscomplex)
    {
	fprintf (f, "%%%%MatrixMarket matrix array complex general\n") ;
    }
    else
    {
	fprintf (f, "%%%%MatrixMarket matrix array real general\n") ;
    }
    fprintf (f, "%.0f %.0f\n", (double) nrow, (double) ncol) ;
    for (j = 0 ; j < ncol ; j++)
    {
	for (i = 0 ; i < nrow ; i++)
	{
	    print_value (f, Ax [i + j*nrow], s) ;
	    if (iscomplex)
	    {
		fprintf (f, " ") ;
		print_value (f, Az [i + j*nrow], s) ;
	    }
	    fprintf (f, "\n") ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* close the file */
    /* ---------------------------------------------------------------------- */

    fclose (f) ;
}
