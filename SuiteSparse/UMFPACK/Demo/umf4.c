/* ========================================================================== */
/* === umf4 ================================================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* UMFPACK Copyright (c) 2005-2012 by Timothy A. Davis,                       */
/* http://www.suitesparse.com. All Rights Reserved.                           */
/* See ../Doc/License.txt for License.                                        */
/* -------------------------------------------------------------------------- */

/* Demo program for UMFPACK.  Reads in a triplet-form matrix in the
 * directory tmp/A, whose size and # of nonzeros are in the file tmp/Asize.
 * Then calls UMFPACK to analyze, factor, and solve the system.
 *
 * Syntax:
 *
 *	umf4		default "auto" strategy, 1-norm row scaling
 *	umf4 a		default "auto" strategy, 1-norm row scaling
 *	umf4 u		unsymmetric strategy, 1-norm row scaling
 *	umf4 s		symmetric strategy, 1-norm row scaling
 *	umf4 A		default "auto" strategy, maxnorm row scaling
 *	umf4 U		unsymmetric strategy, maxnorm row scaling
 *	umf4 S		symmetric strategy, maxnorm row scaling
 *
 * To test a matrix in the Harwell/Boeing format, do the following:
 *
 *	readhb < HB/arc130.rua > tmp/A
 *	readhb_size < HB/arc130.rua > tmp/Asize
 *	umf4
 *
 * The above options do not drop any nonzero entry in L or U.  To compute an
 * incomplete factorization, you can add a second argument to give the drop
 * tolerance.  Entries less than or equal to the drop tolerance are then
 * removed from L and U during factorization, unless dropping those entries
 * does not save any memory space.  For example:
 *
 *	umf4 a 1e-6	default "auto" strategy, 1-norm row scaling,
 *			drop tolerance of 1e-6.
 *
 * Note that adding a drop tolerance can lead to an apparent (but not real)
 * increase in peak memory usage.  This is illustrated in the arc130.rua
 * matrix.  With a drop tolerance, garbage collection happens to be avoided
 * for this matrix.  During garbage collection, both internal and external
 * fragmentation in the memory space is removed.  Peak memory usage includes
 * all internal memory fragmentation, even though this can be removed via
 * garbage collection.
 *
 * Control parameters can also be set in the optional tmp/control.umf4 file.
 * The right-hand-side can be provided in the optional tmp/b file.  The solution
 * is written to tmp/x, and the output statistics are written to tmp/info.umf4.
 *
 * After the matrix is factorized, solved, and the LU factors deallocated,
 * this program then test the AMD ordering routine.  This call to AMD is NOT
 * part of the UMFPACK analysis, factorize, or solve steps.  It is just a
 * separate test of the AMD ordering routine.  If the matrix is unsymmetric,
 * AMD orders the pattern of A+A'.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "umfpack.h"
#include "amd.h"

#define SMAX 256
#define ABS(x) ((x) >= 0 ? (x) : -(x))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define ISNAN(x) ((x) != (x))

#define XTRUE(i,n) (1.0 + ((double) i) / ((double) n))

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

/* -------------------------------------------------------------------------- */
/* err: compute the relative error, ||x-xtrue||/||xtrue|| */
/* -------------------------------------------------------------------------- */

static double err
(
    int n,
    double x [ ]
)
{
    int i  ;
    double enorm, e, abse, absxtrue, xnorm ;
    enorm = 0 ;
    xnorm = 0 ;

    for (i = 0 ; i < n ; i++)
    {
	if (ISNAN (x [i]))
	{
	    enorm = x [i] ;
	    break ;
	}
	e = x [i] - XTRUE (i,n) ;
	abse = ABS (e) ;
	enorm = MAX (enorm, abse) ;
    }

    for (i = 0 ; i < n ; i++)
    {
	/* XTRUE is positive, but do this in case XTRUE is redefined */
	absxtrue = ABS (XTRUE (i,n)) ;
	xnorm = MAX (xnorm, absxtrue) ;
    }

    if (xnorm == 0)
    {
	xnorm = 1 ;
    }
    return (enorm / xnorm) ;
}


/* -------------------------------------------------------------------------- */
/* resid: compute the relative residual, ||Ax-b||/||b|| or ||A'x-b||/||b|| */
/* -------------------------------------------------------------------------- */

static double resid
(
    int n,
    int Ap [ ],
    int Ai [ ],
    double Ax [ ],
    double x [ ],
    double r [ ],
    double b [ ],
    int transpose
)
{
    int i, j, p ;
    double rnorm, absr, absb, bnorm ;
    for (i = 0 ; i < n ; i++)
    {
	r [i] = 0 ;
    }

    if (transpose)
    {
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		r [j] += Ax [p] * x [i] ;
	    }
	}
    }
    else
    {
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		r [i] += Ax [p] * x [j] ;
	    }
	}
    }

    for (i = 0 ; i < n ; i++)
    {
	r [i] -= b [i] ;
    }
    rnorm = 0. ;
    bnorm = 0. ;
    for (i = 0 ; i < n ; i++)
    {
	if (ISNAN (r [i]))
	{
	    rnorm = r [i] ;
	    break ;
	}
	absr = ABS (r [i]) ;
	rnorm = MAX (rnorm, absr) ;
    }
    for (i = 0 ; i < n ; i++)
    {
	if (ISNAN (b [i]))
	{
	    bnorm = b [i] ;
	    break ;
	}
	absb = ABS (b [i]) ;
	bnorm = MAX (bnorm, absb) ;
    }
    if (bnorm == 0)
    {
	bnorm = 1 ;
    }
    return (rnorm / bnorm) ;
}


/* -------------------------------------------------------------------------- */
/* Atimesx: compute y = A*x  or A'*x, where x (i) = 1 + i/n */
/* -------------------------------------------------------------------------- */

static void Atimesx
(
    int n,
    int Ap [ ],
    int Ai [ ],
    double Ax [ ],
    double y [ ],
    int transpose
)
{
    int i, j, p ;
    for (i = 0 ; i < n ; i++)
    {
	y [i] = 0 ;
    }
    if (transpose)
    {
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		y [j] += Ax [p] * XTRUE (i,n) ;
	    }
	}
    }
    else
    {
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		y [i] += Ax [p] * XTRUE (j,n) ;
	    }
	}
    }
}

/* -------------------------------------------------------------------------- */
/* main program */
/* -------------------------------------------------------------------------- */

int main (int argc, char **argv)
{
    int i, j, k, n, nz, *Ap, *Ai, *Ti, *Tj, status, *Pamd, nrow, ncol, rhs ;
    double *Ax, *b, *x, Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], aij,
	*Tx, *r, amd_Control [AMD_CONTROL], amd_Info [AMD_INFO], tamd [2],
	stats [2], droptol ;
    void *Symbolic, *Numeric ;
    FILE *f, *f2 ;
    char s [SMAX] ;

    /* ---------------------------------------------------------------------- */
    /* set controls */
    /* ---------------------------------------------------------------------- */

    printf ("\n===========================================================\n"
	    "=== UMFPACK v%d.%d.%d ========================================\n"
	    "===========================================================\n",
	    UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION, UMFPACK_SUBSUB_VERSION) ;

    umfpack_di_defaults (Control) ;
    Control [UMFPACK_PRL] = 3 ;
    Control [UMFPACK_BLOCK_SIZE] = 32 ;

    f = fopen ("tmp/control.umf4", "r") ;
    if (f != (FILE *) NULL)
    {
	printf ("Reading control file tmp/control.umf4\n") ;
	for (i = 0 ; i < UMFPACK_CONTROL ; i++)
	{
	    fscanf (f, "%lg\n", & Control [i]) ;
	}
	fclose (f) ;
    }

    if (argc > 1)
    {
	char *t = argv [1] ;

	/* get the strategy */
	if (t [0] == 'u')
	{
	    Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC ;
	}
	else if (t [0] == 'a')
	{
	    Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_AUTO ;
	}
	else if (t [0] == 's')
	{
	    Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC ;
	}
	else if (t [0] == '2')
	{
	    printf ("unrecognized strategy: %s\n", argv [1]) ;
	}
	else if (t [0] == 'U')
	{
	    Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC ;
	    Control [UMFPACK_SCALE] = UMFPACK_SCALE_MAX ;
	}
	else if (t [0] == 'A')
	{
	    Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_AUTO ;
	    Control [UMFPACK_SCALE] = UMFPACK_SCALE_MAX ;
	}
	else if (t [0] == 'S')
	{
	    Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC ;
	    Control [UMFPACK_SCALE] = UMFPACK_SCALE_MAX ;
	}
	else if (t [0] == 'T')
	{
	    printf ("unrecognized strategy: %s\n", argv [1]) ;
	}
	else
	{
	    printf ("unrecognized strategy: %s\n", argv [1]) ;
	}

	if (t [1] == 'n')
	{
	    /* no aggressive absorption */
	    Control [UMFPACK_AGGRESSIVE] = FALSE ;
	}
    }

    if (argc > 2)
    {
	/* get the drop tolerance */
	sscanf (argv [2], "%lg", &droptol) ;
	printf ("droptol %g\n", droptol) ;
	Control [UMFPACK_DROPTOL] = droptol ;
    }

    umfpack_di_report_control (Control) ;

    /* ---------------------------------------------------------------------- */
    /* open the matrix file (tmp/A) */
    /* ---------------------------------------------------------------------- */

    printf ("File: tmp/A\n") ;
    f = fopen ("tmp/A", "r") ;
    if (!f)
    {
	printf ("Unable to open file\n") ;
	exit (1) ;
    }

    /* ---------------------------------------------------------------------- */
    /* get n and nz */
    /* ---------------------------------------------------------------------- */

    printf ("File: tmp/Asize\n") ;
    f2 = fopen ("tmp/Asize", "r") ;
    if (f2)
    {
	fscanf (f2, "%d %d %d\n", &nrow, &ncol, &nz) ;
	fclose (f2) ;
    }
    else
    {
	nrow = 1 ;
	ncol = 1 ;
    }
    nz = 0 ;
    while (fgets (s, SMAX, f) != (char *) NULL)
    {
	sscanf (s, "%d %d %lg", &i, &j, &aij) ;
#ifdef ZERO_BASED
	/* matrix is zero based */
	i++ ;
	j++ ;
#endif
	nrow = MAX (nrow, i) ;
	ncol = MAX (ncol, j) ;
	nz++ ;
    }
    fclose (f) ;
    n = MAX (nrow, ncol) ;

    printf ("n %d nrow %d ncol %d nz %d\n", n, nrow, ncol, nz) ;

    /* ---------------------------------------------------------------------- */
    /* allocate space for the input triplet form */
    /* ---------------------------------------------------------------------- */

    Ti = (int *) malloc (nz * sizeof (int)) ;
    Tj = (int *) malloc (nz * sizeof (int)) ;
    Tx = (double *) malloc (nz * sizeof (double)) ;
    if (!Ti || !Tj || !Tx)
    {
	printf ("out of memory for input matrix\n") ;
	exit (1) ;
    }

    /* ---------------------------------------------------------------------- */
    /* read in the triplet form */
    /* ---------------------------------------------------------------------- */

    f2 = fopen ("tmp/A", "r") ;
    if (!f2)
    {
	printf ("Unable to open file\n") ;
	exit (1) ;
    }

    k = 0 ;
    while (fgets (s, SMAX, f2) != (char *) NULL)
    {
	sscanf (s, "%d %d %lg", &i, &j, &aij) ;
#ifndef ZERO_BASED
	i-- ;	/* convert to 0-based */
	j-- ;
#endif
	if (k >= nz)
	{
	    printf ("Error!  Matrix size is wrong\n") ;
	    exit (1) ;
	}
	Ti [k] = i ;
	Tj [k] = j ;
	Tx [k] = aij ;
	k++ ;
    }
    fclose (f2) ;

    (void) umfpack_di_report_triplet (nrow, ncol, nz, Ti, Tj, Tx, Control) ;

    /* ---------------------------------------------------------------------- */
    /* convert to column form */
    /* ---------------------------------------------------------------------- */

    /* convert to column form */
    Ap = (int *) malloc ((n+1) * sizeof (int)) ;
    Ai = (int *) malloc (nz * sizeof (int)) ;
    Ax = (double *) malloc (nz * sizeof (double)) ;
    b = (double *) malloc (n * sizeof (double)) ;
    r = (double *) malloc (n * sizeof (double)) ;
    x = (double *) malloc (n * sizeof (double)) ;

    if (!Ap || !Ai || !Ax || !b || !r)
    {
	printf ("out of memory") ;
	exit (1) ;
    }

    umfpack_tic (stats) ;
    status = umfpack_di_triplet_to_col (nrow, ncol, nz, Ti, Tj, Tx, Ap, Ai, Ax,
	(int *) NULL) ;
    umfpack_toc (stats) ;
    printf ("triplet-to-col time: wall %g cpu %g\n", stats [0], stats [1]) ;
    if (status != UMFPACK_OK)
    {
	umfpack_di_report_status (Control, status) ;
	printf ("umfpack_di_triplet_to_col failed") ;
	exit (1) ;
    }

    /* print the column-form of A */
    (void) umfpack_di_report_matrix (nrow, ncol, Ap, Ai, Ax, 1, Control) ;

    /* b = A * xtrue */
    rhs = FALSE ;
    if (nrow == ncol)
    {
	f = fopen ("tmp/b", "r") ;
	if (f != (FILE *) NULL)
	{
	    printf ("Reading tmp/b\n") ;
	    rhs = TRUE ;
	    for (i = 0 ; i < n ; i++)
	    {
		fscanf (f, "%lg\n", &b [i]) ;
	    }
	    fclose (f) ;
	}
	else
	{
	    Atimesx (n, Ap, Ai, Ax, b, FALSE) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* free the triplet form */
    /* ---------------------------------------------------------------------- */

    free (Ti) ;
    free (Tj) ;
    free (Tx) ;

    /* ---------------------------------------------------------------------- */
    /* symbolic factorization */
    /* ---------------------------------------------------------------------- */

    status = umfpack_di_symbolic (nrow, ncol, Ap, Ai, Ax, &Symbolic,
	    Control, Info) ;

    umfpack_di_report_info (Control, Info) ;
    if (status != UMFPACK_OK)
    {
	umfpack_di_report_status (Control, status) ;
	printf ("umfpack_di_symbolic failed") ;
	exit (1) ;
    }

    /* print the symbolic factorization */
    (void) umfpack_di_report_symbolic (Symbolic, Control) ;

    /* ---------------------------------------------------------------------- */
    /* numeric factorization */
    /* ---------------------------------------------------------------------- */

    status = umfpack_di_numeric (Ap, Ai, Ax, Symbolic, &Numeric, Control, Info);
    if (status < UMFPACK_OK)
    {
	umfpack_di_report_info (Control, Info) ;
	umfpack_di_report_status (Control, status) ;
	fprintf (stderr, "umfpack_di_numeric failed: %d\n", status) ;
	printf ("umfpack_di_numeric failed\n") ;
	exit (1) ;
    }

    /* print the numeric factorization */
    (void) umfpack_di_report_numeric (Numeric, Control) ;

    /* ---------------------------------------------------------------------- */
    /* solve Ax=b */
    /* ---------------------------------------------------------------------- */

    if (nrow == ncol && status == UMFPACK_OK)
    {
	status = umfpack_di_solve (UMFPACK_A, Ap, Ai, Ax, x, b, Numeric,
		Control, Info) ;

	umfpack_di_report_info (Control, Info) ;
	umfpack_di_report_status (Control, status) ;
	if (status < UMFPACK_OK)
	{
	    printf ("umfpack_di_solve failed\n") ;
	    exit (1) ;
	}
	(void) umfpack_di_report_vector (n, x, Control) ;
	printf ("relative maxnorm of residual, ||Ax-b||/||b||: %g\n",
	    resid (n, Ap, Ai, Ax, x, r, b, FALSE)) ;
	if (!rhs)
	{
	    printf ("relative maxnorm of error, ||x-xtrue||/||xtrue||: %g\n\n",
		err (n, x)) ;
	}

	f = fopen ("tmp/x", "w") ;
	if (f != (FILE *) NULL)
	{
	    printf ("Writing tmp/x\n") ;
	    for (i = 0 ; i < n ; i++)
	    {
		fprintf (f, "%30.20e\n", x [i]) ;
	    }
	    fclose (f) ;
	}
	else
	{
	    printf ("Unable to write output x!\n") ;
	    exit (1) ;
	}

	f = fopen ("tmp/info.umf4", "w") ;
	if (f != (FILE *) NULL)
	{
	    printf ("Writing tmp/info.umf4\n") ;
	    for (i = 0 ; i < UMFPACK_INFO ; i++)
	    {
		fprintf (f, "%30.20e\n", Info [i]) ;
	    }
	    fclose (f) ;
	}
	else
	{
	    printf ("Unable to write output info!\n") ;
	    exit (1) ;
	}
    }
    else
    {
	/* don't solve, just report the results */
	umfpack_di_report_info (Control, Info) ;
	umfpack_di_report_status (Control, status) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free the Symbolic and Numeric factorization */
    /* ---------------------------------------------------------------------- */

    umfpack_di_free_symbolic (&Symbolic) ;
    umfpack_di_free_numeric (&Numeric) ;

    printf ("umf4 done, strategy: %g\n", Control [UMFPACK_STRATEGY]) ;

    /* ---------------------------------------------------------------------- */
    /* test just AMD ordering (not part of UMFPACK, but a separate test) */
    /* ---------------------------------------------------------------------- */

    /* first make the matrix square */
    if (ncol < n)
    {
	for (j = ncol+1 ; j <= n ; j++)
	{
	    Ap [j] = Ap [ncol] ;
	}
    }

    printf (
	"\n\n===========================================================\n"
	"=== AMD ===================================================\n"
	"===========================================================\n") ;
    printf ("\n\n------- Now trying the AMD ordering.  This not part of\n"
	"the UMFPACK analysis or factorization, above, but a separate\n"
	"test of just the AMD ordering routine.\n") ;
	Pamd = (int *) malloc (n * sizeof (int)) ;
    if (!Pamd)
    {
	printf ("out of memory\n") ;
	exit (1) ;
    }
    amd_defaults (amd_Control) ;
    amd_control (amd_Control) ;
    umfpack_tic (tamd) ;
    status = amd_order (n, Ap, Ai, Pamd, amd_Control, amd_Info) ;
    umfpack_toc (tamd) ;
    printf ("AMD ordering time: cpu %10.2f wall %10.2f\n",
	tamd [1], tamd [0]) ;
    if (status != AMD_OK)
    {
	printf ("amd failed: %d\n", status) ;
	exit (1) ;
    }
    amd_info (amd_Info) ;
    free (Pamd) ;
    printf ("AMD test done\n") ;

    free (Ap) ;
    free (Ai) ;
    free (Ax) ;
    free (b) ;
    free (r) ;
    free (x) ;

    return (0) ;
}
