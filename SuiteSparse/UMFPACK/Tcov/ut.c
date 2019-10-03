/* ========================================================================== */
/* === umfpack tcov ========================================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/* (Nearly) exhaustive statement-coverage testing for UMFPACK.  */

/* #define DEBUGGING */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include "umfpack.h"
#include "amd.h"
#include "umf_internal.h"
#include "umf_is_permutation.h"
/*
#include "umf_free.h"
#include "umf_malloc.h"
*/
#include "umf_report_perm.h"
#include "umf_realloc.h"
#include "umf_free.h"
#include "umf_malloc.h"

/*
#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
#include "umf_malloc.h"
#endif
*/

#define TOL 1e-3

#define INULL ((Int *) NULL)
#define DNULL ((double *) NULL)

#ifdef COMPLEX
#define CARG(real,imag) real,imag
#define C1ARG(a) ,a
#else
#define CARG(real,imag) real
#define C1ARG(a)
#endif


int check_tol ;

static double divide (double x, double y)
{
    return (x/y) ;
}

/* ========================================================================== */
/* my_ordering */
/* ========================================================================== */

int my_ordering
(
    Int nrow,
    Int ncol,
    Int sym,
    Int *Ap,
    Int *Ai,
    Int *P,         /* size ncol */
    void *params,
    double *info
)
{
    /* return a valid permutation for use by UMFPACK ... using AMD/COLAMD */
    return (UMF_cholmod (nrow, ncol, sym, Ap, Ai, P, params, info)) ;
}

int my_bad_ordering
(
    Int nrow,
    Int ncol,
    Int sym,
    Int *Ap,
    Int *Ai,
    Int *P,         /* size ncol */
    void *params,
    double *info
)
{
    /* return an invalid permutation, for testing */
    Int k, *p ;
    for (k = 0 ; k < ncol ; k++) P [k] = EMPTY ;
    p = (Int *) params ;
    return (p [0]) ;
}

/* ========================================================================== */
/* inv_umfpack_dense: inverse of UMFPACK_DENSE_COUNT */
/* ========================================================================== */

/* the inverse of UMFPACK_DENSE_COUNT: given a col count, find alpha */

static double inv_umfpack_dense (Int d, Int n)
{
    if (d <= 16)
    {
	return (0.0) ;
    }
    else
    {
	return (((double) d) / (16 * sqrt ((double) n))) ;
    }
}


/* ========================================================================== */


static void dump_mat (char *name, Int m, Int n, Int Ap [ ], Int Ai [ ], double Ax [ ]
#ifdef COMPLEX
	, double Az [ ]
#endif
	)
{
    Entry aa ;
    Int j, p ;

    printf ("\n%s = sparse ("ID", "ID") ;\n", name, m, n) ;
    for (j = 0 ; j < n ; j++)
    {
    	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
#ifdef COMPLEX
	    ASSIGN (aa, Ax, Az, p, SPLIT (Az)) ;
	    printf ("%s ("ID","ID") = %30.20g + (1i * %30.20g);\n",
			name, 1+Ai [p], j+1, REAL_COMPONENT(aa), IMAG_COMPONENT(aa)) ;
#else
	    printf ("%s ("ID","ID") = %30.20g ;\n",
			name, 1+Ai [p], j+1, Ax [p]) ;
#endif
	}
    }

}

/* ========================================================================== */

static void dump_vec (char *name, Int n, double X [ ], double Xz[ ])
{
    Int j ;
    printf ("\n%s = [\n", name) ;
    for (j = 0 ; j < n ; j++)
    {
	printf ("%30.20g", X [j]) ;
	if (Xz) printf (" + (1i*%30.20g)", Xz[j]) ;
	printf ("\n") ;
    }
    printf ("] ; \n") ;
}

/* ========================================================================== */

static void dump_perm (char *name, Int n, Int P [ ])
{
    Int j ;
    printf ("\n%s = [\n", name) ;
    for (j = 0 ; j < n ; j++)
    {
	printf (""ID"\n", 1+P [j]) ;
    }
    printf ("] ; \n") ;
    printf ("%s = %s' ;\n", name, name) ;
}

/* ========================================================================== */
/* error: display message and exit */
/* ========================================================================== */

static void error (char *s, double x)
{
    printf ("TEST FAILURE: %s %g   ", s, x) ;

#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
    printf (" umf_malloc_count "ID"\n", UMF_malloc_count) ;
#endif

    printf ("\n") ;
    exit (1) ;
}

/* ========================================================================== */
/* resid: compute the (possibly permuted) residual.  return maxnorm of resid */
/* ========================================================================== */

static double resid
(
    Int n,
    Int Ap [ ],
    Int Ai [ ],
    double Ax [ ],	double Az [ ],
    double x [ ],	double xz [ ],
    double b [ ],	double bz [ ],
    double r [ ],	double rz [ ],
    Int transpose,
    Int P [ ],
    Int Q [ ],
    double Wx [ ]	/* size 2*n double workspace */
)
{
    Int i, j, k, p ;
    double norm, ra, *wx, *wz ;
    Entry bb, xx, aa ;

    wx = Wx ;
    wz = wx + n ;

/*
    transpose: UMFPACK_A

	r = P'AQ'x - b
	Pr = AQ'x - Pb
	we compute and return Pr, not r.

    transpose: UMFPACK_At

	r = QA'Px - b
	Q'r = A'Px - Q'b
	we compute and return Q'r, not r.

    transpose: UMFPACK_Aat

	r = QA.'Px - b
	Q'r = A.'Px - Q'b
	we compute and return Q'r, not r.

*/

    if (transpose == UMFPACK_A)
    {

	if (!P)		/* r = -b */
	{
	    for (i = 0 ; i < n ; i++)
	    {
		ASSIGN (bb, b, bz, i, SPLIT(bz)) ;
	        r [i] = -REAL_COMPONENT (bb) ;
	        rz[i] = -IMAG_COMPONENT (bb) ;
	    }
	}
	else		/* r = -Pb */
	{
	    for (k = 0 ; k < n ; k++)
	    {
		ASSIGN (bb, b, bz, P [k], SPLIT(bz)) ;
	        r [k] = -REAL_COMPONENT (bb) ;
	        rz[k] = -IMAG_COMPONENT (bb) ;
	    }
	}

	if (!Q)		/* w = x */
	{
	    for (j = 0 ; j < n ; j++)
	    {
		ASSIGN (xx, x, xz, j, SPLIT(xz)) ;
	        wx[j] = REAL_COMPONENT (xx) ;
	        wz[j] = IMAG_COMPONENT (xx) ;
	    }
	}
	else		/* w = Q'x */
	{
	    for (k = 0 ; k < n ; k++)
	    {
		ASSIGN (xx, x, xz, Q [k], SPLIT(xz)) ;
	        wx[k] = REAL_COMPONENT (xx) ;
	        wz[k] = IMAG_COMPONENT (xx) ;
	    }
	}

	/* r = r + Aw */
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		ASSIGN (aa, Ax, Az, p, SPLIT(Az)) ;
		r [i] += REAL_COMPONENT(aa) * wx[j] ;
                r [i] -= IMAG_COMPONENT(aa) * wz[j] ;
                rz[i] += IMAG_COMPONENT(aa) * wx[j] ;
                rz[i] += REAL_COMPONENT(aa) * wz[j] ;
	    }
	}

	/* note that we just computed Pr, not r */

    }
    else if (transpose == UMFPACK_At)
    {

	if (!Q)		/* r = -b */
	{
	    for (i = 0 ; i < n ; i++)
	    {
		ASSIGN (bb, b, bz, i, SPLIT(bz)) ;
	        r [i] = -REAL_COMPONENT (bb) ;
	        rz[i] = -IMAG_COMPONENT (bb) ;
	    }
	}
	else		/* r = -Q'b */
	{
	    for (k = 0 ; k < n ; k++)
	    {
		ASSIGN (bb, b, bz, Q [k], SPLIT(bz)) ;
	        r [k] = -REAL_COMPONENT (bb) ;
	        rz[k] = -IMAG_COMPONENT (bb) ;
	    }
	}

	if (!P)		/* w = x */
	{
	    for (j = 0 ; j < n ; j++)
	    {
		ASSIGN (xx, x, xz, j, SPLIT(xz)) ;
	        wx[j] = REAL_COMPONENT (xx) ;
	        wz[j] = IMAG_COMPONENT (xx) ;
	    }
	}
	else		/* w = Px */
	{
	    for (k = 0 ; k < n ; k++)
	    {
		ASSIGN (xx, x, xz, P [k], SPLIT(xz)) ;
	        wx[k] = REAL_COMPONENT (xx) ;
	        wz[k] = IMAG_COMPONENT (xx) ;
	    }
	}

	/* r = r + A'w */
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		ASSIGN (aa, Ax, Az, p, SPLIT(Az)) ;
		/* complex conjugate */
		r [j] += REAL_COMPONENT(aa) * wx[i] ;
                r [j] += IMAG_COMPONENT(aa) * wz[i] ;
                rz[j] -= IMAG_COMPONENT(aa) * wx[i] ;
                rz[j] += REAL_COMPONENT(aa) * wz[i] ;
	    }
	}

	/* note that we just computed Q'r, not r */

    }
    else if (transpose == UMFPACK_Aat)
    {

	if (!Q)		/* r = -b */
	{
	    for (i = 0 ; i < n ; i++)
	    {
		ASSIGN (bb, b, bz, i, SPLIT(bz)) ;
	        r [i] = -REAL_COMPONENT (bb) ;
	        rz[i] = -IMAG_COMPONENT (bb) ;
	    }
	}
	else		/* r = -Q'b */
	{
	    for (k = 0 ; k < n ; k++)
	    {
		ASSIGN (bb, b, bz, Q [k], SPLIT(bz)) ;
	        r [k] = -REAL_COMPONENT (bb) ;
	        rz[k] = -IMAG_COMPONENT (bb) ;
	    }
	}

	if (!P)		/* w = x */
	{
	    for (j = 0 ; j < n ; j++)
	    {
		ASSIGN (xx, x, xz, j, SPLIT(xz)) ;
	        wx[j] = REAL_COMPONENT (xx) ;
	        wz[j] = IMAG_COMPONENT (xx) ;
	    }
	}
	else		/* w = Px */
	{
	    for (k = 0 ; k < n ; k++)
	    {
		ASSIGN (xx, x, xz, P [k], SPLIT(xz)) ;
	        wx[k] = REAL_COMPONENT (xx) ;
	        wz[k] = IMAG_COMPONENT (xx) ;
	    }
	}

	/* r = r + A.'w */
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		ASSIGN (aa, Ax, Az, p, SPLIT(Az)) ;
		/* not complex conjugate */
		r [j] += REAL_COMPONENT(aa) * wx[i] ;
                r [j] -= IMAG_COMPONENT(aa) * wz[i] ;
                rz[j] += IMAG_COMPONENT(aa) * wx[i] ;
                rz[j] += REAL_COMPONENT(aa) * wz[i] ;
	    }
	}

	/* note that we just computed Q'r, not r */

    }

    norm = 0. ;
    for (i = 0 ; i < n ; i++)
    {
	Entry rr ;

	/* --- */
	/* ASSIGN (rr, r [i], rz [i]) ; */
	ASSIGN (rr, r, rz, i, TRUE) ;
	/* --- */

	ABS (ra, rr) ;
	norm = MAX (norm, ra) ;
    }
    return (norm) ;
}


/* ========================================================================== */
/* irand:  return a random Integer in the range 0 to n-1 */
/* ========================================================================== */

static Int irand (Int n)
{
    return (rand ( ) % n) ;
}

/* ========================================================================== */
/* xrand:  return a random double, > 0 and <= 1 */
/* ========================================================================== */

/* rand ( ) returns an Integer in the range 0 to RAND_MAX */

static double xrand ( )
{
    return ((1.0 + (double) rand ( )) / (1.0 + (double) RAND_MAX)) ;
}


/* ========================================================================== */
/* randperm:  generate a random permutation of 0..n-1 */
/* ========================================================================== */

static void randperm (Int n, Int P [ ])
{
    Int i, t, k ;
    for (i = 0 ; i < n ; i++)
    {
	P [i] = i ;
    }
    for (i = n-1 ; i > 0 ; i--)
    {
	k = irand (i) ;
	/* swap positions i and k */
	t = P [k] ;
	P [k] = P [i] ;
	P [i] = t ;
    }
}

/* ========================================================================== */
/* do_solvers:  test Ax=b, etc */
/* ========================================================================== */

static double do_solvers
(
    Int n_row,
    Int n_col,
    Int Ap [ ],
    Int Ai [ ],
    double Ax [ ],		double Az [ ],
    double b [ ],		double bz [ ],
    double Control [ ],
    double Info [ ],
    void *Numeric,
    Int Lp [ ],
    Int Li [ ],
    double Lx [ ],		double Lz [ ],
    Int Up [ ],
    Int Ui [ ],
    double Ux [ ],		double Uz [ ],
    Int P [ ],
    Int Q [ ],
    double x [ ],		double xz [ ],
    double r [ ],		double rz [ ],
    Int W [ ],
    double Wx [ ],
    Int split	    /* TRUE if complex variables split, FALSE if merged */
)
{
    double maxrnorm = 0.0, rnorm, xnorm, xa, xaz, *Rb, *Rbz,
	*y, *yz, *Rs, *Cx, *Cz ;
    double Con [UMFPACK_CONTROL] ;
    Int *noP = INULL, *noQ = INULL, irstep, orig, i, prl, status, n,
	s1, s2, do_recip, *Cp, *Ci, nz, scale ;
    Entry bb, xx, xtrue ;
    NumericType *Num ;

#ifdef COMPLEX
    if (split)
    {
	if (!Az || !bz || !xz || !Lz || !Uz || !xz) error ("bad split\n", 0.) ;
    }
    else
    {
	if ( Az ||  bz ||  xz ||  Lz ||  Uz ||  xz) error ("bad merge\n", 0.) ;
    }
    /* rz is never passed to umfpack, and is always split in ut.c */
    if (!rz) error ("bad rz\n", 0.) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* get parameters */
    /* ---------------------------------------------------------------------- */

    n = MAX (n_row, n_col) ;
    if (n == 0) error ("n zero", 0.) ;
    /* n = MAX (n,1) ; */
    /* n_inner = MIN (n_row, n_col) ; */

    if (Control)
    {
	orig = Control [UMFPACK_IRSTEP] ;
	prl = Control [UMFPACK_PRL] ;
    }
    else
    {
	prl = UMFPACK_DEFAULT_PRL ;
    }

    if (n_row == n_col)
    {
	nz = Ap [n_col] ;
	nz = MAX (nz, Lp [n_col]) ;
	nz = MAX (nz, Up [n_col]) ;
	Cp = (Int *) malloc ((n_col+1) * sizeof (Int)) ;
	Ci = (Int *) malloc ((nz+1) * sizeof (Int)) ;
	Cx = (double *) calloc (2*(nz+1) , sizeof (double)) ;
	if (split)
	{
	    Cz = Cx + nz ;
	}
	else
	{
	    Cz = DNULL ;
	}
	if (!Cp || !Ci || !Cx) error ("out of memory (0)", 0.) ;
    }
    else
    {
	Cp = INULL ;
	Ci = INULL ;
	Cx = DNULL ;
    }

    Num = (NumericType *) Numeric ;
    scale = (Num->Rs != DNULL) ;

    /* ---------------------------------------------------------------------- */
    /* error handling */
    /* ---------------------------------------------------------------------- */

    if (n_row != n_col)
    {
	status = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az) , CARG(x,xz), CARG(b,bz), Numeric, DNULL, DNULL) ;
	if (status != UMFPACK_ERROR_invalid_system) error ("rectangular Ax=b should have failed\n", 0.) ;
    }
    else
    {
	status = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az) , CARG(DNULL,xz), CARG(b,bz), Numeric, DNULL, DNULL) ;
	if (status != UMFPACK_ERROR_argument_missing) error ("missing x should have failed\n", 0.) ;
	status = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(DNULL,Az) , CARG(DNULL,xz), CARG(b,bz), Numeric, DNULL, DNULL) ;
	if (status != UMFPACK_ERROR_argument_missing) error ("missing Ax should have failed\n", 0.) ;
    }

    /* ---------------------------------------------------------------------- */
    /* Ax=b */
    /* ---------------------------------------------------------------------- */

    for (irstep = -1 ; irstep <= 3 ; irstep++)
    {
	if (Control)
	{
	    for (i = 0 ; i < UMFPACK_CONTROL ; i++) Con [i] = Control [i] ;
	}
	else
	{
	    UMFPACK_defaults (Con) ;
	}
	Con [UMFPACK_PRL] = prl ;
	Con [UMFPACK_IRSTEP] = MAX (0, irstep) ;

	if (prl >= 2) printf ("1: do solve: Ax=b: "ID"\n", irstep) ;
	if (irstep == -1)
	{
	    status = UMFPACK_solve (UMFPACK_A, INULL, INULL, CARG(DNULL,DNULL) , CARG(x,xz), CARG(b,bz), Numeric, Con, Info) ;
	}
	else
	{
	    status = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az) , CARG(x,xz), CARG(b,bz), Numeric, Con, Info) ;
	}
	UMFPACK_report_status (Con, status) ;
	UMFPACK_report_info (Con, Info) ;
	if (n_row != n_col)
	{
	    if (status != UMFPACK_ERROR_invalid_system)
	    {
		dump_mat ("A", n_row, n_col, Ap, Ai, CARG(Ax,Az)) ;
	    	error ("rectangular Ax=b should have failed\n", 0.) ;
	    }
	    /* return immediately if the matrix is rectangular */
	    return (0.) ;
	}
	if (status == UMFPACK_WARNING_singular_matrix)
	{
	    if (prl >= 2) printf ("Ax=b singular\n") ;
	}
	else if (status != UMFPACK_OK)
	{
	    dump_mat ("A", n, n, Ap, Ai, CARG(Ax,Az)) ;
	    error ("Ax=b failed\n", 0.) ;
	}
	else
	{
	    rnorm = resid (n, Ap, Ai, Ax, Az, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	    if (prl >= 2) printf ("1: rnorm Ax=b is %g\n", rnorm) ;
	    maxrnorm = MAX (rnorm, maxrnorm) ;

	    /* compare x with xtrue */
	    xnorm = 0. ;
	    for (i = 0 ; i < n ; i++)
	    {
	        REAL_COMPONENT(xtrue) = 1.0 + ((double) i) / ((double) n) ;
#ifdef COMPLEX
	        IMAG_COMPONENT(xtrue) = 1.3 - ((double) i) / ((double) n) ;
#endif
		/* --- */
		/* ASSIGN (xx, x [i] - xtrue, xz[i]-xtruez) ; */
		ASSIGN (xx, x, xz, i, SPLIT(xz)) ;
		DECREMENT (xx, xtrue) ;
		/* --- */

	        ABS (xa, xx) ;
	        xnorm = MAX (xnorm, xa) ;
	    }

#if 0
	    { FILE *f ;
		char s [200] ;
		sprintf (s, "b_XXXXXX") ;
		mkstemp (s) ;
		f = fopen (s, "w") ;
		for (i = 0 ; i < n ; i++) fprintf (f, "%40.25e %40.25e\n", b [i], bz [i]) ;
		fclose (f) ;
		s [0] = 'x' ;
		f = fopen (s, "w") ;
		for (i = 0 ; i < n ; i++) fprintf (f, "%40.25e %40.25e\n", x [i], xz [i]) ;
		fclose (f) ;
	    }
#endif

	    if (check_tol && (status == UMFPACK_OK && (rnorm > TOL || xnorm > TOL)))
	    {
		Con [UMFPACK_PRL] = 5 ;
		UMFPACK_report_control (Con) ; 
	        printf ("Ax=b inaccurate %g %g\n", rnorm, xnorm) ;
		dump_mat ("A", n, n, Ap, Ai, CARG(Ax,Az)) ;
	        printf ("\nb: ") ;
	        UMFPACK_report_vector (n, CARG(b,bz), Con) ;
	        printf ("\nx: ") ;
	        UMFPACK_report_vector (n, CARG(x,xz), Con) ;
		error ("Ax=b inaccurate", MAX (rnorm, xnorm)) ;
	    }

	    maxrnorm = MAX (xnorm, maxrnorm) ;
	}

#ifdef DEBUGGING
	printf ("\n") ;
#endif
	if (prl >= 2) printf ("Ax=b irstep "ID" attempted %g\n", irstep, Info [UMFPACK_IR_ATTEMPTED]) ;
	if (irstep > Info [UMFPACK_IR_ATTEMPTED])
	{
	    break ;
	}
    }

    if (n != n_row && n != n_col && n <= 0) error ("huh?", 0.) ;

    /* ---------------------------------------------------------------------- */
    /* A'x=b */
    /* ---------------------------------------------------------------------- */

    for (irstep = 0 ; irstep <= 3 ; irstep++)
    {
	if (Control)
	{
	    for (i = 0 ; i < UMFPACK_CONTROL ; i++) Con [i] = Control [i] ;
	}
	else
	{
	    UMFPACK_defaults (Con) ;
	}
	Con [UMFPACK_PRL] = prl ;
	Con [UMFPACK_IRSTEP] = irstep ;

	if (prl >= 2) printf ("do solve: A'x=b: "ID"\n", irstep) ;
	status = UMFPACK_solve (UMFPACK_At, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info) ;
	UMFPACK_report_status (Con, status) ;
	/* UMFPACK_report_info (Con, Info) ; */
	if (status == UMFPACK_WARNING_singular_matrix)
	{
	    if (prl >= 2) printf ("A'x=b singular\n") ;
	}
	else if (status != UMFPACK_OK)
	{
	    dump_mat ("A", n, n, Ap, Ai, CARG(Ax,Az)) ;
	    error ("A'x=b failed\n", 0.) ;
	}
	else
	{
	    rnorm = resid (n, Ap, Ai, Ax, Az, x, xz, b, bz, r, rz, UMFPACK_At, noP, noQ, Wx) ;

	    if (prl  >= 2) printf ("2: rnorm A'x=b is %g\n", rnorm) ;
	    if (check_tol && rnorm > TOL)
	    {
		Con [UMFPACK_PRL] = 99 ;
	        printf ("A'x=b inaccurate %g\n", rnorm) ;
		dump_mat ("A", n, n, Ap, Ai, CARG(Ax,Az)) ;
		/*
	        printf ("\nA: ") ;
	        UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	        printf ("\nb: ") ;
	        UMFPACK_report_vector (n, CARG(b,bz), Con) ;
	        printf ("\nx: ") ;
	        UMFPACK_report_vector (n, CARG(x,xz), Con) ;
		error ("A'x=b inaccurate", MAX (rnorm, xnorm)) ;
		*/
	    }
	    maxrnorm = MAX (rnorm, maxrnorm) ;

	    /* also check using UMFPACK_transpose */
	    status = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), noP, noQ, Cp, Ci, CARG(Cx,Cz) C1ARG(1)) ;
	    if (status != UMFPACK_OK)
	    {
		error ("transposed A'x=b failed\n", 0.) ;
	    }

	    rnorm = resid (n, Cp, Ci, Cx, Cz, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;

	    if (prl  >= 2) printf ("2b: rnorm A'x=b is %g\n", rnorm) ;
	    if (check_tol && rnorm > TOL)
	    {
		Con [UMFPACK_PRL] = 99 ;
	        printf ("transpose A'x=b inaccurate %g\n", rnorm) ;
		/*
	        printf ("\nA: ") ;
	        UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	        printf ("\nb: ") ;
	        UMFPACK_report_vector (n, CARG(b,bz), Con) ;
	        printf ("\nx: ") ;
	        UMFPACK_report_vector (n, CARG(x,xz), Con) ;
		error ("A'x=b inaccurate", MAX (rnorm, xnorm)) ;
		*/
	    }
	    maxrnorm = MAX (rnorm, maxrnorm) ;

	}
	if (prl >= 2) printf ("A'x=b irstep "ID" attempted %g\n", irstep, Info [UMFPACK_IR_ATTEMPTED]) ;
	if (irstep > Info [UMFPACK_IR_ATTEMPTED])
	{
	    break ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* A.'x=b */
    /* ---------------------------------------------------------------------- */

    for (irstep = 0 ; irstep <= 3 ; irstep++)
    {
	if (Control)
	{
	    for (i = 0 ; i < UMFPACK_CONTROL ; i++) Con [i] = Control [i] ;
	}
	else
	{
	    UMFPACK_defaults (Con) ;
	}
	Con [UMFPACK_PRL] = prl ;
	Con [UMFPACK_IRSTEP] = irstep ;

	if (prl >= 2) printf ("do solve: A.'x=b: "ID"\n", irstep) ;
	status = UMFPACK_solve (UMFPACK_Aat, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info) ;
	UMFPACK_report_status (Con, status) ;
	/* UMFPACK_report_info (Con, Info) ; */
	if (status == UMFPACK_WARNING_singular_matrix)
	{
	    if (prl >= 2) printf ("A.'x=b singular\n") ;
	}
	else if (status != UMFPACK_OK)
	{
	    dump_mat ("A", n, n, Ap, Ai, CARG(Ax,Az)) ;
	    error ("A.'x=b failed\n", 0.) ;
	}
	else
	{
	    rnorm = resid (n, Ap, Ai, Ax, Az, x, xz, b, bz, r, rz, UMFPACK_Aat, noP, noQ, Wx) ;
	    if (prl >= 2) printf ("3: rnorm A.'x=b is %g\n", rnorm) ;
	    if (check_tol && rnorm > TOL)
	    {
		Con [UMFPACK_PRL] = 99 ;
	        printf ("A.'x=b inaccurate %g\n", rnorm) ;
		/*
		dump_mat ("A", n, n, Ap, Ai, CARG(Ax,Az)) ;
	        printf ("\nA: ") ;
	        UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	        printf ("\nb: ") ;
	        UMFPACK_report_vector (n, CARG(b,bz), Con) ;
	        printf ("\nx: ") ;
	        UMFPACK_report_vector (n, CARG(x,xz), Con) ;
	        error ("A.'x=b inaccurate %g\n", MAX (rnorm, xnorm)) ;
		*/
	    }
	    maxrnorm = MAX (rnorm, maxrnorm) ;

	    /* also check using UMFPACK_transpose */
	    status = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), noP, noQ, Cp, Ci, CARG(Cx,Cz) C1ARG(0)) ;
	    if (status != UMFPACK_OK)
	    {
		error ("transposed A.'x=b failed\n", 0.) ;
	    }

	    rnorm = resid (n, Cp, Ci, Cx, Cz, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;

	    if (prl  >= 2) printf ("2b: rnorm A'x=b is %g\n", rnorm) ;
	    if (check_tol && rnorm > TOL)
	    {
		Con [UMFPACK_PRL] = 99 ;
	        printf ("transpose A'x=b inaccurate %g\n", rnorm) ;
		/*
	        printf ("\nA: ") ;
	        UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	        printf ("\nb: ") ;
	        UMFPACK_report_vector (n, CARG(b,bz), Con) ;
	        printf ("\nx: ") ;
	        UMFPACK_report_vector (n, CARG(x,xz), Con) ;
		error ("A'x=b inaccurate", MAX (rnorm, xnorm)) ;
		*/
	    }
	    maxrnorm = MAX (rnorm, maxrnorm) ;
	}
	if (prl >= 2) printf ("A.'x=b irstep "ID" attempted %g\n", irstep, Info [UMFPACK_IR_ATTEMPTED]) ;
	if (irstep > Info [UMFPACK_IR_ATTEMPTED])
	{
	    break ;
	}
    }

    if (Control)
    {
	for (i = 0 ; i < UMFPACK_CONTROL ; i++) Con [i] = Control [i] ;
    }
    else
    {
	UMFPACK_defaults (Con) ;
    }

    /* ---------------------------------------------------------------------- */
    /* wsolve Ax=b */
    /* ---------------------------------------------------------------------- */

    /* printf ("do wsolve: Ax=b:\n") ; */
    if (Control) Control [UMFPACK_IRSTEP] = 1 ;
    if (prl >= 2) printf ("2: do solve: Ax=b: "ID" (wsolve) \n", irstep) ;
    status = UMFPACK_wsolve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info, W, Wx) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	if (prl >= 2) printf ("Ax=b wsolve singular\n") ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("A", n, n, Ap, Ai, CARG(Ax,Az)) ;
	error ("Ax=b wsolve failure\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Ap, Ai, Ax, Az, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	if (prl >= 2) printf ("4: rnorm Ax=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("A", n, n, Ap, Ai, CARG(Ax,Az)) ;
    	    error ("wsolve inaccurate %g\n", rnorm) ;
        }
	maxrnorm = MAX (rnorm, maxrnorm) ;
    }

    if (Control) Control [UMFPACK_IRSTEP] = orig ;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace */
    /* ---------------------------------------------------------------------- */

    /*
    prl = 999 ;
    */

    Rs  = (double *) malloc (n * sizeof (double)) ;  /* [ */
    Rb  = (double *) calloc (2*n , sizeof (double)) ;  /* [ */
    y   = (double *) calloc (2*n , sizeof (double)) ;  /* [ */

    /* ---------------------------------------------------------------------- */
    /* Ax=b with individual calls */
    /* ---------------------------------------------------------------------- */

    if (split)
    {
	yz = y + n ;
	Rbz = Rb + n ;
    }
    else
    {
	yz = DNULL ;
	Rbz = DNULL ;
    }

    /* status = UMFPACK_get_scale (Rs, Numeric) ; */
    status = UMFPACK_get_numeric (
	    INULL, INULL, CARG(DNULL,DNULL),
	    INULL, INULL, CARG(DNULL,DNULL),
	    INULL, INULL, CARG (DNULL,DNULL), &do_recip, Rs, Numeric) ;
    if (status != UMFPACK_OK) error ("get Rs failed", (double) status) ; 

    /*
    printf ("Rs:\n") ;
    for (i = 0 ; i < n ; i++)
    {
	printf ("   Rs [%d] = %g\n", i, Rs [i]) ;
    }
    */

    if (prl >= 2) printf ("3: do solve: Ax=b in different steps:\n") ;
    /* Rb = R*b */
    /* dump_vec ("b", n, b, bz) ; */
    status = UMFPACK_scale (CARG (Rb, Rbz), CARG (b,bz), Numeric) ;
    if (status != UMFPACK_OK) error ("Rb failed", (double) status) ; 
    /* dump_vec ("R*b", n, Rb, Rbz) ; */

    /*
    UMFPACK_defaults (Con) ;
    Con [UMFPACK_PRL] = 999 ;
    printf ("Rb:\n") ;
    UMFPACK_report_vector (n, CARG(Rb,Rbz), Con) ;
    printf ("b:\n") ;
    UMFPACK_report_vector (n, CARG(b,bz), Con) ;
    error ("finish early\n", rnorm) ;
    */

    /* solve Ly = P*(Rb) */
    s1 = UMFPACK_solve (UMFPACK_Pt_L, Ap, Ai, CARG(Ax,Az), CARG(y,yz), CARG(Rb,Rbz), Numeric, Control, Info) ;
    if (! (s1 == UMFPACK_OK || s1 == UMFPACK_WARNING_singular_matrix))
    {
	error ("P'Ly=Rb failed", (double) status) ; 
    }
    /* solve UQ'x=y */
    s2 = UMFPACK_solve (UMFPACK_U_Qt, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(y,yz), Numeric, Control, Info) ;
    if (! (s2 == UMFPACK_OK || s2 == UMFPACK_WARNING_singular_matrix))
    {
	error ("UQ'x=y  failed", (double) status) ; 
    }
    if (s1 == UMFPACK_OK && s2 == UMFPACK_OK)
    {
	rnorm = resid (n, Ap, Ai, Ax, Az, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	if (prl >= 2) printf ("5: rnorm Ax=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    /* error ("Ax=b (different steps) inaccurate ", rnorm) ; */
	    printf ("Ax=b (different steps) inaccurate  %g !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", rnorm) ; 
        }
	maxrnorm = MAX (rnorm, maxrnorm) ;
    }

    if (prl >= 2) printf ("4: do solve: Ax=b, different steps, own scale:\n") ;
    /* Rb = R*b */

    if (do_recip)
    {
	for (i = 0 ; i < n ; i++)
	{
	    ASSIGN (bb, b, bz, i, SPLIT(bz)) ;
	    SCALE (bb, Rs [i]) ;

	    if (split)
	    {
		Rb  [i] = REAL_COMPONENT(bb) ;
		Rbz [i] = IMAG_COMPONENT(bb) ;
	    }
	    else
	    {
		Rb [2*i] = REAL_COMPONENT(bb) ;
		Rb [2*i+1] = IMAG_COMPONENT(bb) ;
	    }

	    /*
	    Rb  [i] = REAL_COMPONENT(bb) * Rs [i] ;
	    Rbz [i] = IMAG_COMPONENT(bb) * Rs [i] ;
	    */
	}
    }
    else
    {
	for (i = 0 ; i < n ; i++)
	{
	    ASSIGN (bb, b, bz, i, SPLIT(bz)) ;
	    SCALE_DIV (bb, Rs [i]) ;

	    if (split)
	    {
		Rb  [i] = REAL_COMPONENT(bb) ;
		Rbz [i] = IMAG_COMPONENT(bb) ;
	    }
	    else
	    {
		Rb [2*i] = REAL_COMPONENT(bb) ;
		Rb [2*i+1] = IMAG_COMPONENT(bb) ;
	    }

	    /*
	    Rb  [i] = REAL_COMPONENT(bb) / Rs [i] ;
	    Rbz [i] = IMAG_COMPONENT(bb) / Rs [i] ;
	    */
	}
    }

    /* solve Ly = P*(Rb) */
    s1 = UMFPACK_solve (UMFPACK_Pt_L, Ap, Ai, CARG(Ax,Az), CARG(y,yz), CARG(Rb,Rbz), Numeric, Control, Info) ;
    if (! (s1 == UMFPACK_OK || s1 == UMFPACK_WARNING_singular_matrix))
    {
	error ("P'Ly=Rb failed", (double) status) ; 
    }
    /* solve UQ'x=y */
    s2 = UMFPACK_solve (UMFPACK_U_Qt, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(y,yz), Numeric, Control, Info) ;
    if (! (s2 == UMFPACK_OK || s2 == UMFPACK_WARNING_singular_matrix))
    {
	error ("UQ'x=y  failed", (double) status) ; 
    }
    if (s1 == UMFPACK_OK && s2 == UMFPACK_OK)
    {
	rnorm = resid (n, Ap, Ai, Ax, Az, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	if (prl >= 2) printf ("6: rnorm Ax=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    error ("Ax=b (different steps, own scale) inaccurate ", rnorm) ;
        }
	maxrnorm = MAX (rnorm, maxrnorm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* (PAQ)'x=b with individual calls, no scaling */
    /* ---------------------------------------------------------------------- */

    if (!scale)
    {
	int k ;

	s1 = UMFPACK_solve (UMFPACK_Ut, Ap, Ai, CARG(Ax,Az), CARG(y,yz), CARG(b,bz), Numeric, Control, Info) ;
	if (! (s1 == UMFPACK_OK || s1 == UMFPACK_WARNING_singular_matrix))
	{
	    error ("U'y=b failed", (double) status) ; 
	}
	s2 = UMFPACK_solve (UMFPACK_Lt, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(y,yz), Numeric, Control, Info) ;
	if (! (s2 == UMFPACK_OK || s2 == UMFPACK_WARNING_singular_matrix))
	{
	    error ("L'x=y failed", (double) status) ; 
	}

	/* check using UMFPACK_transpose */
	if (s1 == UMFPACK_OK && s2 == UMFPACK_OK)
	{
	    status = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), P, Q, Cp, Ci, CARG(Cx,Cz) C1ARG(1)) ;
	    if (status != UMFPACK_OK)
	    {
		error ("transposed (PAQ)'x=b failed\n", 0.) ;
	    }
	    rnorm = resid (n, Cp, Ci, Cx, Cz, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	    if (prl >= 2) printf ("99b: rnorm (PAQ)'x=b is %g\n", rnorm) ;
	    if (check_tol && rnorm > TOL)
	    {
		dump_mat ("A", n, n, Ap, Ai, CARG(Ax,Az)) ;
		dump_mat ("C", n, n, Cp, Ci, CARG(Cx,Cz)) ;
		dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
		dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;

		printf ("P = [ ") ;
                for (k = 0 ; k < n ; k++)
                {
                    printf (ID" ", P [k]) ;
                }
                printf ("]\n") ;
		printf ("Q = [ ") ;
                for (k = 0 ; k < n ; k++)
                {
                    printf (ID" ", Q [k]) ;
                }
                printf ("]\n") ;

		error ("transposed (PAQ)'x=b inaccurate\n",rnorm) ;
	    }
	    maxrnorm = MAX (rnorm, maxrnorm) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* (PAQ).'x=b with individual calls, no scaling */
    /* ---------------------------------------------------------------------- */

    if (!scale)
    {
	int k ;

	s1 = UMFPACK_solve (UMFPACK_Uat, Ap, Ai, CARG(Ax,Az), CARG(y,yz), CARG(b,bz), Numeric, Control, Info) ;
	if (! (s1 == UMFPACK_OK || s1 == UMFPACK_WARNING_singular_matrix))
	{
	    error ("U'y=b failed", (double) status) ; 
	}
	s2 = UMFPACK_solve (UMFPACK_Lat, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(y,yz), Numeric, Control, Info) ;
	if (! (s2 == UMFPACK_OK || s2 == UMFPACK_WARNING_singular_matrix))
	{
	    error ("L'x=y failed", (double) status) ; 
	}

	/* check using UMFPACK_transpose */
	if (s1 == UMFPACK_OK && s2 == UMFPACK_OK)
	{
	    status = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), P, Q, Cp, Ci, CARG(Cx,Cz) C1ARG(0)) ;
	    if (status != UMFPACK_OK)
	    {
		error ("transposed (PAQ).'x=b failed\n", 0.) ;
	    }
	    rnorm = resid (n, Cp, Ci, Cx, Cz, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	    /* printf ("98b: rnorm (PAQ)'x=b is %g\n", rnorm) ; */
	    if (check_tol && rnorm > TOL)
	    {
		dump_mat ("A", n, n, Ap, Ai, CARG(Ax,Az)) ;
		dump_mat ("C", n, n, Cp, Ci, CARG(Cx,Cz)) ;
		dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
		dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
		printf ("P = [ ") ;
                for (k = 0 ; k < n ; k++)
                {
                    printf (ID" ", P [k]) ;
                }
                printf ("]\n") ;
		printf ("Q = [ ") ;
                for (k = 0 ; k < n ; k++)
                {
                    printf (ID" ", Q [k]) ;
                }
                printf ("]\n") ;
		error ("transposed (PAQ).'x=b inaccurate\n",rnorm) ;
	    }
	    maxrnorm = MAX (rnorm, maxrnorm) ;
	}
    }


    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    free (y) ;	    /* ] */
    free (Rb) ;	    /* ] */
    free (Rs) ;	    /* ] */

    /* ---------------------------------------------------------------------- */
    /* Lx=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: Lx=b:\n") ;
    status = UMFPACK_solve (UMFPACK_L, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	error ("Lx=b solve singular!", 0.) ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	error ("Lx=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Lp, Li, Lx, Lz, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	if (prl >= 2) printf ("7: rnorm Lx=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	    error ("Lx=b inaccurate %g", rnorm) ;
        }
	maxrnorm = MAX (rnorm, maxrnorm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* L'x=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: L'x=b:\n") ;
    status = UMFPACK_solve (UMFPACK_Lt, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	error ("L'x=b solve singular!", 0.) ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	error ("L'x=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Lp, Li, Lx, Lz, x, xz, b, bz, r, rz, UMFPACK_At, noP, noQ, Wx) ;
	if (prl >= 2) printf ("7: rnorm L'x=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
    	    error ("L'x=b inaccurate %g\n",rnorm) ;
	}
	maxrnorm = MAX (rnorm, maxrnorm) ;

	/* also check using UMFPACK_transpose */
	status = UMFPACK_transpose (n, n, Lp, Li, CARG(Lx,Lz), noP, noQ, Cp, Ci, CARG(Cx,Cz) C1ARG(1)) ;
	if (status != UMFPACK_OK)
	{
	    error ("transposed L'x=b failed\n", 0.) ;
	}
        rnorm = resid (n, Cp, Ci, Cx, Cz, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	if (prl >= 2) printf ("7b: rnorm L'x=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
    	    error ("transposed L'x=b inaccurate %g\n",rnorm) ;
	}
	maxrnorm = MAX (rnorm, maxrnorm) ;

    }

    /* ---------------------------------------------------------------------- */
    /* L.'x=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: L.'x=b:\n") ;
    status = UMFPACK_solve (UMFPACK_Lat, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	error ("L.'x=b solve singular!", 0.) ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	error ("L.'x=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Lp, Li, Lx, Lz, x, xz, b, bz, r, rz, UMFPACK_Aat, noP, noQ, Wx) ;
	if (prl >= 2) printf ("8: rnorm L.'x=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
    	    error ("L.'x=b inaccurate %g\n",rnorm) ;
	}
	maxrnorm = MAX (rnorm, maxrnorm) ;

	/* also check using UMFPACK_transpose */
	status = UMFPACK_transpose (n, n, Lp, Li, CARG(Lx,Lz), noP, noQ, Cp, Ci, CARG(Cx,Cz) C1ARG(0)) ;
	if (status != UMFPACK_OK)
	{
	    error ("transposed L'x=b failed\n", 0.) ;
	}
        rnorm = resid (n, Cp, Ci, Cx, Cz, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	if (prl >= 2) printf ("8b: rnorm L'x=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
    	    error ("8b transposed L'x=b inaccurate %g\n",rnorm) ;
	}
	maxrnorm = MAX (rnorm, maxrnorm) ;

    }

    /* ---------------------------------------------------------------------- */
    /* Ux=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: Ux=b:\n") ;
    status = UMFPACK_solve (UMFPACK_U, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	if (prl >= 2) printf ("Ux=b solve singular\n") ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	error ("Ux=b failed\n", 0.) ;
    }
    else
    {
	rnorm = resid (n, Up, Ui, Ux, Uz, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	if (prl >= 2) printf ("9: rnorm Ux=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
    	    error ("Ux=b inaccurate %g\n",rnorm) ;
        }
        maxrnorm = MAX (rnorm, maxrnorm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* U'x=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: U'x=b:\n") ;
    status = UMFPACK_solve (UMFPACK_Ut, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	if (prl >= 2) printf ("U'x=b solve singular\n") ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	error ("U'x=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Up, Ui, Ux, Uz, x, xz, b, bz, r, rz, UMFPACK_At, noP, noQ, Wx) ;
	if (prl >= 2) printf ("10: rnorm U'x=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	    error ("U'x=b inaccurate %g\n",rnorm) ;
        }
        maxrnorm = MAX (rnorm, maxrnorm) ;

	/* also check using UMFPACK_transpose */
	status = UMFPACK_transpose (n, n, Up, Ui, CARG(Ux,Uz), noP, noQ, Cp, Ci, CARG(Cx,Cz) C1ARG(1)) ;
	if (status != UMFPACK_OK)
	{
	    error ("transposed U'x=b failed\n", 0.) ;
	}
        rnorm = resid (n, Cp, Ci, Cx, Cz, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	if (prl >= 2) printf ("10b: rnorm U'x=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
    	    error ("10b transposed U'x=b inaccurate %g\n",rnorm) ;
	}
	maxrnorm = MAX (rnorm, maxrnorm) ;

    }

    /* ---------------------------------------------------------------------- */
    /* U.'x=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: U.'x=b:\n") ;
    status = UMFPACK_solve (UMFPACK_Uat, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	if (prl >= 2) printf ("U.'x=b solve singular\n") ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	error ("U.'x=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Up, Ui, Ux, Uz, x, xz, b, bz, r, rz, UMFPACK_Aat, noP, noQ, Wx) ;
	if (prl >= 2) printf ("11: rnorm U.'x=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	    error ("U.'x=b inaccurate %g\n",rnorm) ;
        }
        maxrnorm = MAX (rnorm, maxrnorm) ;

	/* also check using UMFPACK_transpose */
	status = UMFPACK_transpose (n, n, Up, Ui, CARG(Ux,Uz), noP, noQ, Cp, Ci, CARG(Cx,Cz) C1ARG(0)) ;
	if (status != UMFPACK_OK)
	{
	    error ("11b transposed U.'x=b failed\n", 0.) ;
	}
        rnorm = resid (n, Cp, Ci, Cx, Cz, x, xz, b, bz, r, rz, UMFPACK_A, noP, noQ, Wx) ;
	if (prl >= 2) printf ("11b: rnorm U'x=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
    	    error ("11b transposed U'x=b inaccurate %g\n",rnorm) ;
	}
	maxrnorm = MAX (rnorm, maxrnorm) ;

    }

    /* ---------------------------------------------------------------------- */
    /* P'Lx=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: P'Lx=b:\n") ;
    status = UMFPACK_solve (UMFPACK_Pt_L, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	error ("P'Lx=b solve singular!", 0.) ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	error ("P'Lx=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Lp, Li, Lx, Lz, x, xz, b, bz, r, rz, UMFPACK_A, P, noQ, Wx) ;
	if (prl >= 2) printf ("12: rnorm P'Lx=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	    error ("P'Lx=b inaccurate: %g\n",rnorm) ;
        }
        maxrnorm = MAX (rnorm, maxrnorm) ;

    }

    /* ---------------------------------------------------------------------- */
    /* L'Px=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: L'Px=b:\n") ;
    status = UMFPACK_solve (UMFPACK_Lt_P, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	error ("L'Px=b solve singular!", 0.) ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	error ("L'Px=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Lp, Li, Lx, Lz, x, xz, b, bz, r, rz, UMFPACK_At, P, noQ, Wx) ;
	if (prl >= 2) printf ("13: rnorm L'Px=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	    error ("L'Px=b inaccurate %g\n",rnorm) ;
        }
        maxrnorm = MAX (rnorm, maxrnorm) ;

    }

    /* ---------------------------------------------------------------------- */
    /* L.'Px=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: L.'Px=b:\n") ;
    status = UMFPACK_solve (UMFPACK_Lat_P, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	error ("L.'Px=b solve singular!", 0.) ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	error ("L.'Px=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Lp, Li, Lx, Lz, x, xz, b, bz, r, rz, UMFPACK_Aat, P, noQ, Wx) ;
	if (prl >= 2) printf ("14: rnorm L.'Px=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("L", n, n, Lp, Li, CARG(Lx,Lz)) ;
	    error ("L.'Px=b inaccurate %g\n",rnorm) ;
        }
        maxrnorm = MAX (rnorm, maxrnorm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* UQ'x=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: UQ'x=b:\n") ;
    status = UMFPACK_solve (UMFPACK_U_Qt, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	if (prl >= 2) printf ("UQ'x=b solve singular\n") ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	error ("UQ'x=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Up, Ui, Ux, Uz, x, xz, b, bz, r, rz, UMFPACK_A, noP, Q, Wx) ;
	if (prl >= 2) printf ("15: rnorm UQ'x=b is %g\n", rnorm) ;
        if (check_tol && status == UMFPACK_OK && rnorm > TOL)
        {
	    dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	    error ("UQ'x=b inaccurate %g\n",rnorm) ;
        }
        maxrnorm = MAX (rnorm, maxrnorm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* QU'x=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: QU'x=b:\n") ;
    status = UMFPACK_solve (UMFPACK_Q_Ut, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	if (prl >= 2) printf ("QU'x=b solve singular\n") ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	error ("QU'x=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Up, Ui, Ux, Uz, x, xz, b, bz, r, rz, UMFPACK_At, noP, Q, Wx) ;
	if (prl >= 2) printf ("16: rnorm QU'x=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	    error ("QU'x=b inaccurate %g\n",rnorm) ;
        }
        maxrnorm = MAX (rnorm, maxrnorm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* QU.'x=b */
    /* ---------------------------------------------------------------------- */

    if (prl >= 2) printf ("do solve: QU.'x=b:\n") ;
    status = UMFPACK_solve (UMFPACK_Q_Uat, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
    /* UMFPACK_report_info (Control, Info) ; */
    if (status == UMFPACK_WARNING_singular_matrix)
    {
	if (prl >= 2) printf ("QU.'x=b solve singular\n") ;
    }
    else if (status != UMFPACK_OK)
    {
	dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	error ("QU.'x=b failed\n", 0.) ;
    }
    else
    {
        rnorm = resid (n, Up, Ui, Ux, Uz, x, xz, b, bz, r, rz, UMFPACK_Aat, noP, Q, Wx) ;
	if (prl >= 2) printf ("17: rnorm QU.'x=b is %g\n", rnorm) ;
        if (check_tol && rnorm > TOL)
        {
	    dump_mat ("U", n, n, Up, Ui, CARG(Ux,Uz)) ;
	    error ("QU.'x=b inaccurate %g\n",rnorm) ;
        }
        maxrnorm = MAX (rnorm, maxrnorm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* done */
    /* ---------------------------------------------------------------------- */

    if (n_row == n_col)
    {
	free (Cp) ;
	free (Ci) ;
	free (Cx) ;
    }
    return (maxrnorm) ;
}


/* ========================================================================== */
/* do_symnum:  factor A once, and then test the solves - return if error */
/* ========================================================================== */

static double do_symnum
(
    Int n_row,
    Int n_col,
    Int Ap [ ],
    Int Ai [ ],
    double Ax [ ],		double Az [ ],
    double b [ ],		double bz [ ],
    double Control [ ],
    Int Qinit [ ],
    double x [ ],		double xz [ ],
    double r [ ],		double rz [ ],
    double Wx [ ],
    Int P [ ],
    Int Q [ ],
    Int Qtree [ ],
    Int Ptree [ ],
    Int W [ ],
    Int Lp [ ],
    Int Up [ ],
    Int save_and_load,
    Int split,	    /* TRUE if complex variables split, FALSE if merged */
    Int det_check, double det_x, double det_z
)
{
    void *Symbolic, *Numeric ;
    double *Lx, *Ux, *Lz, *Uz, rnorm, *Rs ;
    Int *noP = INULL, *noQ = INULL, *Li, *Ui, n, n_inner, n1, do_recip ;
    Int lnz, unz, nz, nn, nfr, nchains, nsparse_col, status ;
    Int *Front_npivots, *Front_parent, *Chain_start, *Chain_maxrows ;
    Int *Chain_maxcols, *Lrowi, *Lrowp, is_singular ;
    double Info [UMFPACK_INFO], *Lrowx, *Lrowz, *Dx, *Dz ;
    Int nnrow, nncol, nzud, *Front_1strow, *Front_leftmostdesc, prl, i ;
    double mind, maxd, rcond ;
    Entry d ;
    double da, deterr ;
    NumericType *Num ;
    SymbolicType *Sym ;
    double Mx [2], Mz, Exp ;

#ifdef COMPLEX
    if (split)
    {
	if (!Az || !bz || !xz) error ("bad split\n", 0.) ;
    }
    else
    {
	if ( Az ||  bz ||  xz) error ("bad merge\n", 0.) ;
    }
    if (!rz) error ("bad rz\n", 0.) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* do the symbolic factorization */
    /* ---------------------------------------------------------------------- */

    prl = Control ? Control [UMFPACK_PRL] : UMFPACK_DEFAULT_PRL ; 

    n = MAX (n_row, n_col) ;
    n = MAX (n,1) ;
    n_inner = MIN (n_row, n_col) ;

    if (prl > 2)
    {
	printf ("\nA::\n") ;
	status = UMFPACK_report_matrix (n_row, n_col, Ap, Ai, CARG(Ax,Az), 1, Control) ;
	printf ("\nb::\n") ;
	if (n_row == n_col) UMFPACK_report_vector (n, CARG(b,bz), Control) ;
    }

    if (Qinit)
    {
	/* dump_perm ("Qinit", n_col, Qinit) ; */
	status = UMFPACK_qsymbolic (n_row, n_col, Ap, Ai, CARG(Ax,Az), Qinit,
            &Symbolic, Control, Info) ; /* ( */
    }
    else if (Control != NULL &&
        Control [UMFPACK_ORDERING] == UMFPACK_ORDERING_USER)
    {
        Int params [3] ;
        params [0] = UMFPACK_ORDERING_AMD ;
        params [1] = prl-1 ;
        params [2] = EMPTY ;
        status = UMFPACK_fsymbolic (n_row, n_col, Ap, Ai, CARG(Ax,Az),
            &my_ordering, (void *) params, &Symbolic, Control, Info) ;
    }
    else
    {
	status = UMFPACK_symbolic (n_row, n_col, Ap, Ai, CARG(Ax,Az),
            &Symbolic, Control, Info) ;
    }

    UMFPACK_report_status (Control, status) ;
    UMFPACK_report_info (Control, Info) ;

    if (!Symbolic)
    {
	UMFPACK_report_info (Control, Info) ;
	error ("symbolic invalid\n", 0.) ;
    }

    /* ---------------------------------------------------------------------- */
    /* test save and load */
    /* ---------------------------------------------------------------------- */

    status = UMFPACK_save_symbolic (Symbolic, "s.umf") ;
    if (status != UMFPACK_OK)
    {
	error ("save symbolic failed\n", 0.) ;
    }
    UMFPACK_free_symbolic (&Symbolic) ;
    status = UMFPACK_load_symbolic (&Symbolic, "s.umf") ;
    if (status != UMFPACK_OK)
    {
	error ("load symbolic failed\n", 0.) ;
    }

    if (n < 15)
    {

	int umf_fail_save [3], memcnt ;

	status = UMFPACK_save_symbolic (Symbolic, (char *) NULL) ;
	if (status != UMFPACK_OK)
	{
	    error ("save symbolic failed\n", 0.) ;
	}
	UMFPACK_free_symbolic (&Symbolic) ;
	status = UMFPACK_load_symbolic (&Symbolic, (char *) NULL) ;
	if (status != UMFPACK_OK)
	{
	    error ("load symbolic failed\n", 0.) ;
	}

	/* test memory handling */
	umf_fail_save [0] = umf_fail ;
	umf_fail_save [1] = umf_fail_lo ;
	umf_fail_save [2] = umf_fail_hi ;

	umf_fail = -1 ;
	umf_fail_lo = 0 ;
	umf_fail_hi = 0 ;

	UMFPACK_free_symbolic (&Symbolic) ;
	status = UMFPACK_load_symbolic (&Symbolic, (char *) NULL) ;
	if (status != UMFPACK_OK)
	{
	    error ("load symbolic failed\n", 0.) ;
	}

	Sym = (SymbolicType *) Symbolic ;

	memcnt = 12 ;
	if (Sym->esize > 0)
	{
	    memcnt++ ;
	}
	if (Sym->prefer_diagonal > 0)
	{
	    memcnt++ ;
	}

	for (i = 1 ; i <= memcnt ; i++)
	{
	    umf_fail = i ;
	    UMFPACK_free_symbolic (&Symbolic) ;
	    status = UMFPACK_load_symbolic (&Symbolic, (char *) NULL) ;
	    if (status != UMFPACK_ERROR_out_of_memory)
	    {
		error ("load symbolic should have failed\n", 0.) ;
	    }
	}

	umf_fail = memcnt + 1 ;

	UMFPACK_free_symbolic (&Symbolic) ;
	status = UMFPACK_load_symbolic (&Symbolic, (char *) NULL) ;
	if (status != UMFPACK_OK)
	{
	    error ("load symbolic failed (edge)\n", 0.) ;
	}

	umf_fail    = umf_fail_save [0] ;
	umf_fail_lo = umf_fail_save [1] ;
	umf_fail_hi = umf_fail_save [2] ;

	UMFPACK_free_symbolic (&Symbolic) ;
	status = UMFPACK_load_symbolic (&Symbolic, "s.umf") ;
	if (status != UMFPACK_OK)
	{
	    error ("load symbolic failed\n", 0.) ;
	}

    }

    /* ---------------------------------------------------------------------- */
    /* get the symbolic factorization */
    /* ---------------------------------------------------------------------- */

    if (prl > 2) printf ("\nSymbolic: ") ;
    status = UMFPACK_report_symbolic (Symbolic, Control) ;
    if (status != UMFPACK_OK)
    {
	UMFPACK_free_symbolic (&Symbolic) ;
	error ("bad symbolic report\n", 0.) ;
    }

    Front_npivots = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
    Front_parent = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
    Front_1strow = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
    Front_leftmostdesc = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* [ */
    Chain_start = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
    Chain_maxrows = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
    Chain_maxcols = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */

    if (!Front_npivots || !Front_parent || !Chain_start || !Chain_maxrows
	|| !Front_leftmostdesc || !Front_1strow
	|| !Chain_maxcols || !Qtree || !Ptree) error ("out of memory (1)",0.) ;

    status = UMFPACK_get_symbolic (&nnrow, &nncol, &n1, &nz, &nfr, &nchains,
	Ptree, Qtree, Front_npivots, Front_parent, Front_1strow, Front_leftmostdesc,
	Chain_start, Chain_maxrows, Chain_maxcols, Symbolic) ;

    if (status != UMFPACK_OK)
    {
	UMFPACK_report_info (Control, Info) ;
	error ("get symbolic failed\n", 0.) ;
    }

    free (Chain_maxcols) ;	/* ] */
    free (Chain_maxrows) ;	/* ] */
    free (Chain_start) ;	/* ] */
    free (Front_leftmostdesc) ;	/* ] */
    free (Front_1strow) ;	/* ] */
    free (Front_parent) ;	/* ] */
    free (Front_npivots) ;	/* ] */

    if (!UMF_is_permutation (Qtree, W, n_col, n_col)) { error ("Qtree invalid\n", 0.) ; }
    if (!UMF_is_permutation (Ptree, W, n_row, n_row)) { error ("Ptree invalid\n", 0.) ; }

    /* ---------------------------------------------------------------------- */
    /* do the numerical factorization */
    /* ---------------------------------------------------------------------- */

    status = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;	/* [ */
    if (status != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    is_singular = (status == UMFPACK_WARNING_singular_matrix) ;

    UMFPACK_report_status (Control, status) ;
    UMFPACK_report_info (Control, Info) ;

    UMFPACK_free_symbolic (&Symbolic) ;			/* ) */

    if (!Numeric)
    {
	/* printf ("numeric bad:  %g\n", Control [UMFPACK_ALLOC_INIT]) ; */
	fflush (stdout) ;
	return (9e10) ;
    }

    if (prl > 2) printf ("Numeric: ") ;
    status = UMFPACK_report_numeric (Numeric, Control) ;
    if (status != UMFPACK_OK)
    {
	UMFPACK_free_numeric (&Numeric) ;
	error ("bad numeric report\n", 0.) ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the determinant */
    /* ---------------------------------------------------------------------- */

    Mx [0] = 0. ;
    Mx [1] = 0. ;
    Mz = 0. ;
    Exp = 0. ;

    for (i = 0 ; i <= 3 ; i++)
    {
	if (i == 0)
	{
	    status = UMFPACK_get_determinant (CARG (Mx, &Mz), &Exp, Numeric,
		Info) ;
	}
	else if (i == 1)
	{
	    status = UMFPACK_get_determinant (CARG (Mx, &Mz), &Exp, Numeric,
		DNULL) ;
	}
	else if (i == 2)
	{
	    status = UMFPACK_get_determinant (CARG (Mx, &Mz), DNULL, Numeric,
		Info) ;
	}
	else if (i == 3)
	{
	    status = UMFPACK_get_determinant (CARG (Mx, DNULL), DNULL, Numeric,
		Info) ;
	}
	if (n_row == n_col)
	{
	    if (status != UMFPACK_OK)
	    {
		error ("bad det\n", 0.) ;
	    }
	    if (det_check && SCALAR_ABS (det_x < 1e100))
	    {

		if (i == 0 || i == 1)
		{
		    deterr = SCALAR_ABS (det_x - (Mx [0] * pow (10.0, Exp))) ;
		}
		else if (i == 2 || i == 3)
		{
		    deterr = SCALAR_ABS (det_x - Mx [0]) ;
		}

		if (deterr > 1e-7)
		{
		    printf ("det: real err %g i: "ID"\n", deterr, i) ;
		    error ("det: bad real part", det_x) ;
		}

#ifdef COMPLEX

		if (i == 0 || i == 1)
		{
		    deterr = SCALAR_ABS (det_z - (Mz * pow (10.0, Exp))) ;
		}
		else if (i == 2)
		{
		    deterr = SCALAR_ABS (det_z - Mz) ;
		}
		else if (i == 3)
		{
		    deterr = SCALAR_ABS (det_z - Mx [1]) ;
		}

		if (deterr > 1e-7)
		{
		    printf ("det: imag err %g\n", deterr) ;
		    error ("det: bad imag part", det_z) ;
		}
#endif
	    }
	}
	else
	{
	    if (status != UMFPACK_ERROR_invalid_system)
	    {
		error ("bad det (rectangluar)\n", 0.) ;
	    }
	}
    }

    /* ---------------------------------------------------------------------- */
    /* test save and load */
    /* ---------------------------------------------------------------------- */

    status = UMFPACK_save_numeric (Numeric, "n.umf") ;
    if (status != UMFPACK_OK)
    {
	error ("save numeric failed\n", 0.) ;
    }
    UMFPACK_free_numeric (&Numeric) ;
    status = UMFPACK_load_numeric (&Numeric, "n.umf") ;
    if (status != UMFPACK_OK)
    {
	error ("load numeric failed\n", 0.) ;
    }

    if (n < 15)
    {
	int umf_fail_save [3], memcnt ;

	status = UMFPACK_save_numeric (Numeric, (char *) NULL) ;
	if (status != UMFPACK_OK)
	{
	    error ("save numeric failed\n", 0.) ;
	}
	UMFPACK_free_numeric (&Numeric) ;
	status = UMFPACK_load_numeric (&Numeric, (char *) NULL) ;
	if (status != UMFPACK_OK)
	{
	    error ("load numeric failed\n", 0.) ;
	}

	/* test memory handling */
	umf_fail_save [0] = umf_fail ;
	umf_fail_save [1] = umf_fail_lo ;
	umf_fail_save [2] = umf_fail_hi ;

	umf_fail = -1 ;
	umf_fail_lo = 0 ;
	umf_fail_hi = 0 ;

	UMFPACK_free_numeric (&Numeric) ;
	status = UMFPACK_load_numeric (&Numeric, (char *) NULL) ;
	if (status != UMFPACK_OK)
	{
	    error ("load numeric failed\n", 0.) ;
	}

	Num = (NumericType *) Numeric ;

	memcnt = 11 ;
	if (Num->scale != UMFPACK_SCALE_NONE)
	{
	    memcnt++ ;
	}
	if (Num->ulen > 0)
	{
	    memcnt++ ;
	}

	for (i = 1 ; i <= memcnt ; i++)
	{
	    umf_fail = i ;
	    UMFPACK_free_numeric (&Numeric) ;
	    status = UMFPACK_load_numeric (&Numeric, (char *) NULL) ;
	    if (status != UMFPACK_ERROR_out_of_memory)
	    {
		error ("load numeric should have failed\n", 0.) ;
	    }
	}

	umf_fail = memcnt + 1 ;

	UMFPACK_free_numeric (&Numeric) ;
	status = UMFPACK_load_numeric (&Numeric, (char *) NULL) ;
	if (status != UMFPACK_OK)
	{
	    printf ("memcnt %d\n", memcnt) ;
	    error ("load numeric failed (edge)\n", 0.) ;
	}

	umf_fail    = umf_fail_save [0] ;
	umf_fail_lo = umf_fail_save [1] ;
	umf_fail_hi = umf_fail_save [2] ;

	UMFPACK_free_numeric (&Numeric) ;
	status = UMFPACK_load_numeric (&Numeric, "n.umf") ;
	if (status != UMFPACK_OK)
	{
	    error ("load numeric failed\n", 0.) ;
	}

    }

    /* ---------------------------------------------------------------------- */
    /* get the LU factorization */
    /* ---------------------------------------------------------------------- */

    status = UMFPACK_get_lunz (&lnz, &unz, &nnrow, &nncol, &nzud, Numeric) ;
    if (status != UMFPACK_OK)
    {
	UMFPACK_report_info (Control, Info) ;
	error ("get lunz failure\n", 0.) ;
	UMFPACK_free_numeric (&Numeric) ;
    }

    /* guard against malloc of zero-sized arrays */
    lnz = MAX (lnz,1) ;
    unz = MAX (unz,1) ;

    Rs = (double *) malloc ((n+1) * sizeof (double)) ;	/* [ */
    Li = (Int *) malloc (lnz * sizeof (Int)) ;		/* [ */
    Lx = (double *) calloc (2*lnz , sizeof (double)) ;	/* [ */
    Ui = (Int *) malloc (unz * sizeof (Int)) ;		/* [ */
    Ux = (double *) calloc (2*unz , sizeof (double)) ;	/* [ */
    Dx = (double *) calloc (2*n , sizeof (double)) ;	/* [ */

    Lrowp = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* [ */
    Lrowi = (Int *) malloc (lnz * sizeof (Int)) ;	/* [ */
    Lrowx = (double *) calloc (2*lnz , sizeof (double)) ;	/* [ */

    if (!Li || !Lx || !Ui || !Ux || !Lrowp || !Lrowi || !Lrowx) error ("out of memory (2)\n",0.) ;

    if (split)
    {
	Dz = Dx + n ;
	Lrowz = Lrowx + lnz ;
	Lz = Lx + lnz ;
	Uz = Ux + unz ;
    }
    else
    {
	Dz = DNULL ;
	Lrowz = DNULL ;
	Lz = DNULL ;
	Uz = DNULL ;
    }

    status = UMFPACK_get_numeric (Lrowp, Lrowi, CARG(Lrowx,Lrowz), Up, Ui, CARG(Ux,Uz), P, Q, CARG(Dx,Dz), &do_recip, Rs, Numeric) ;
    if (status != UMFPACK_OK)
    {
	UMFPACK_report_info (Control, Info) ;
	error ("get LU failed\n", 0.) ;
    }

    if (!UMF_is_permutation (P, W, n_row, n_row)) { error ("P invalid\n", 0.) ; }
    if (!UMF_is_permutation (Q, W, n_col, n_col)) { error ("Q invalid\n", 0.) ; }
    if (prl > 2) printf ("\nP: ") ;
    status = UMFPACK_report_perm (n_row, P, Control) ;
    if (status != UMFPACK_OK) { error ("bad P 1\n", 0.) ; }
    if (prl > 2) printf ("\nQ: ") ;
    status = UMFPACK_report_perm (n_col, Q, Control) ;
    if (status != UMFPACK_OK) { error ("bad Q 1\n", 0.) ; }

    if (prl > 2) printf ("\nL row: ") ;
    status = UMFPACK_report_matrix (n_row, n_inner, Lrowp, Lrowi, CARG(Lrowx,Lrowz), 0, Control) ;
    if (status != UMFPACK_OK) { error ("bad Lrow\n", 0.) ; }

    if (prl > 2) printf ("\nD, diag of U: ") ;
    status = UMFPACK_report_vector (n_inner, CARG(Dx,Dz), Control) ;
    if (status != UMFPACK_OK) { error ("bad D\n", 0.) ; }

    /* --- */
    /* ASSIGN (d, Dx [0], Dz [0]) ; */
    ASSIGN (d, Dx, Dz, 0, SPLIT(Dz)) ;
    /* --- */

    ABS (da, d) ;
    mind = da ;
    maxd = da ;
    for (i = 1 ; i < n_inner ; i++)
    {
	/* --- */
	/* ASSIGN (d, Dx [i], Dz [i]) ; */
	ASSIGN (d, Dx, Dz, i, SPLIT(Dz)) ;
	/* --- */

	ABS (da, d) ;
	mind = MIN (mind, da) ;
	maxd = MAX (maxd, da) ;
    }
    if (maxd == 0.)
    {
	rcond = 0. ;
    }
    else
    {
	rcond = mind / maxd ;
    }
    if (prl > 2) printf ("mind: %g maxd: %g rcond: %g %g\n", mind, maxd, rcond, Info [UMFPACK_RCOND]) ;
    if (rcond == 0.0 && Info [UMFPACK_RCOND] != 0.0)
    {
	printf ("rcond error %30.20e  %30.20e\n", rcond, Info [UMFPACK_RCOND]) ;
	error ("rcond", 0.) ;
    }
    if (SCALAR_ABS (rcond - Info [UMFPACK_RCOND]) / rcond > 1e-10)
    {
	printf ("rcond error %30.20e  %30.20e\n", rcond, Info [UMFPACK_RCOND]) ;
	error ("rcond", 0.) ;
    }

    /* get L = Lrow' */
    status = UMFPACK_transpose (n_inner, n_row, Lrowp, Lrowi, CARG(Lrowx,Lrowz), noP, noQ, Lp, Li, CARG(Lx,Lz) C1ARG(0)) ;
    if (status != UMFPACK_OK)
    {
	error ("L=Lrow' failed\n", 0.) ;
    }

    if (prl > 2) printf ("\nL col: ") ;
    status = UMFPACK_report_matrix (n_row, n_inner, Lp, Li, CARG(Lx,Lz), 1, Control) ;
    if (status != UMFPACK_OK) { error ("bad Lrow\n", 0.) ; }

    if (prl > 2) printf ("\nU col: ") ;
    status = UMFPACK_report_matrix (n_inner, n_col, Up, Ui, CARG(Ux,Uz), 1, Control) ;
    if (status != UMFPACK_OK) { error ("bad Ucol\n", 0.) ; }

    free (Lrowx) ;	/* ] */
    free (Lrowi) ;	/* ] */
    free (Lrowp) ;	/* ] */

    rnorm = do_solvers (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, Info, Numeric,
    	Lp, Li, Lx,Lz, Up, Ui, Ux,Uz, P, Q, x,xz, r,rz, W, Wx, split) ;

    UMFPACK_report_info (Control, Info) ;

    /* ---------------------------------------------------------------------- */
    /* free everything */
    /* ---------------------------------------------------------------------- */

    free (Dx) ;		/* ] */
    free (Ux) ;		/* ] */
    free (Ui) ;		/* ] */
    free (Lx) ;		/* ] */
    free (Li) ;		/* ] */
    free (Rs) ;		/* ] */

    UMFPACK_free_numeric (&Numeric) ;	/* ] */

    return (rnorm) ;
}


/* ========================================================================== */
/* do_once:  factor A once, and then test the solves - return if error */
/* ========================================================================== */

/* exit if an error occurs. otherwise, return the largest residual norm seen. */

static double do_once
(
    Int n_row,
    Int n_col,
    Int Ap [ ],
    Int Ai [ ],
    double Ax [ ],		double Az [ ],
    double b [ ],		double bz [ ],
    double Control [ ],
    Int Qinit [ ],
    Int MemControl [6],
    Int save_and_load,
    Int split,	    /* TRUE if complex variables split, FALSE if merged */
    Int det_check, double det_x, double det_z
)
{

    double *x, rnorm, *r, *Wx, *xz, *rz ;
    Int *P, *Q, *Lp, *Up, *W, *Qtree, *Ptree, n ;

#ifdef COMPLEX
    if (split)
    {
	if (!Az || !bz) error ("bad split\n", 0.) ;
    }
    else
    {
	if ( Az ||  bz) error ("bad merge\n", 0.) ;
    }
#endif

    /* ---------------------------------------------------------------------- */
    /* malloc and realloc failure control */
    /* ---------------------------------------------------------------------- */

    umf_fail = MemControl [0] ;
    umf_fail_hi = MemControl [1] ;
    umf_fail_lo = MemControl [2] ;

    umf_realloc_fail = MemControl [3] ;
    umf_realloc_hi = MemControl [4] ;
    umf_realloc_lo = MemControl [5] ;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace */
    /* ---------------------------------------------------------------------- */

    n = MAX (n_row, n_col) ;
    n = MAX (n,1) ;

    r = (double *) calloc (2*n , sizeof (double)) ;	/* [ */
    x = (double *) calloc (2*n , sizeof (double)) ;	/* [ */
    rz = r + n ;

    if (split)
    {
	/* real/complex are in r/rz and x/xz */
	xz = x + n ;
    }
    else
    {
	/* r and z are treated as array of size n Entry's */
	xz = DNULL ;
    }

    Wx = (double *) malloc (10*n * sizeof (double)) ;	/* [ */
    P = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
    Q = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
    Qtree = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
    Ptree = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
    W = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
    Lp = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* [ */
    Up = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* [ */
    if (!x || !Wx || !r || !P || !Q || !Qtree || !Ptree || !W || !Lp || !Up) error ("out of memory (3)",0.) ;

    /* ---------------------------------------------------------------------- */
    /* report controls */
    /* ---------------------------------------------------------------------- */

    if (Control)
    {
    	if (Control [UMFPACK_PRL] >= 2)
	{
	    UMFPACK_report_control (Control) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* do the symbolic & numeric factorization, and solvers */
    /* ---------------------------------------------------------------------- */

    rnorm = do_symnum (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, Qinit, x,xz,
		r,rz, Wx, P, Q, Qtree, Ptree, W, Lp, Up, save_and_load, split,
		det_check, det_x, det_z) ;

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    free (Up) ;		/* ] */
    free (Lp) ;		/* ] */
    free (W) ;		/* ] */
    free (Ptree) ;	/* ] */
    free (Qtree) ;	/* ] */
    free (Q) ;		/* ] */
    free (P) ;		/* ] */
    free (Wx) ;		/* ] */
    free (x) ;		/* ] */
    free (r) ;		/* ] */

#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
    if (UMF_malloc_count != 0) error ("umfpack memory leak!!",0.) ;
#endif

    return (rnorm) ;
}

/* ========================================================================== */
/* do_many:  factor A once, and then test the solves - return if error */
/* ========================================================================== */

/* runs do_once with complex variables split, and again with them merged */

static double do_many
(
    Int n_row,
    Int n_col,
    Int Ap [ ],
    Int Ai [ ],
    double Ax [ ],		double Az [ ],
    double b [ ],		double bz [ ],
    double Control [ ],
    Int Qinit [ ],
    Int MemControl [6],
    Int save_and_load,
    Int det_check, double det_x, double det_z
)
{
    double rnorm, r ;
    Entry *A, *B, a ;
    Int p, i, nz ;

    rnorm = 0 ;

#ifdef COMPLEX

    if (!Az || !bz) error ("do_many missing imag parts!\n", 0.) ;

    nz = Ap [n_col] ;
    A = (Entry *) malloc ((nz+1) * sizeof (Entry)) ;
    B = (Entry *) malloc ((n_col+1) * sizeof (Entry)) ;
    if (!A || !B) error ("out of memory (4)",0.) ;

    for (p = 0 ; p < nz ; p++)
    {
	ASSIGN (A [p], Ax, Az, p, TRUE) ;
    }
    for (i = 0 ; i < n_col ; i++)
    {
	ASSIGN (B [i], b, bz, i, TRUE) ;
    }

    /* with complex variables merged */
    r = do_once (n_row, n_col, Ap, Ai, (double *)A,DNULL, (double *)B,DNULL, Control, Qinit, MemControl, save_and_load, FALSE, det_check, det_x, det_z) ;

    free (A) ;
    free (B) ;

    rnorm = MAX (r, rnorm) ;

#endif

    /* with complex variables split */
    r = do_once (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemControl, save_and_load, TRUE, det_check, det_x, det_z) ;
    rnorm = MAX (r, rnorm) ;
    return (rnorm) ;
}


/* ========================================================================== */
/* bgen:  b = A*xtrue, where xtrue (i) = 1 + i/n */
/* ========================================================================== */

static void bgen
(
    Int n,
    Int Ap [ ],
    Int Ai [ ],
    double Ax [ ],	double Az [ ],
    double b [ ],	double bz [ ]
)
{
    Int i, col, p ;
    double xtrue, xtruez ;
    for (i = 0 ; i < n ; i++)
    {
	b [i] = 0.0 ;
	bz[i] = 0.0 ;
    }
    for (col = 0 ; col < n ; col++)
    {
	xtrue = 1.0 + ((double) col) / ((double) n) ;
#ifdef COMPLEX
	xtruez= 1.3 - ((double) col) / ((double) n) ;
#else
	xtruez= 0. ;
#endif
	for (p = Ap [col] ; p < Ap [col+1] ; p++)
	{
	    i = Ai [p] ;
	    b [i] += Ax [p] * xtrue ;
	    b [i] -= Az [p] * xtruez ;
	    bz[i] += Az [p] * xtrue ;
	    bz[i] += Ax [p] * xtruez ;
	}
    }
}


/* ========================================================================== */
/* do_matrix: process one matrix with lots of options - exit if errors */
/* ========================================================================== */

/* return the largest residual norm seen, or exit if error occured */

static double do_matrix
(
    Int n,
    Int Ap [ ],
    Int Ai [ ],
    double Ax [ ],	double Az [ ],

    double Controls [UMFPACK_CONTROL][1000],
    Int Ncontrols [UMFPACK_CONTROL],
    Int MemControl [6],
    Int do_dense
)
{
    double Control [UMFPACK_CONTROL], *b, *bz, maxrnorm, rnorm, tol, init,
        psave ;
    Int *colhist, *rowhist, *rowdeg, *noQinit, *Qinit, *cknob, *rknob,
	c, r, cs, rs, row, col, i, coldeg, p, d, nb, ck, rk, *Head, *Next,
	col1, col2, d1, d2, k, status, n_amd,
	i_tol, i_nb, i_init, n_tol, n_nb, n_init, n_scale, i_scale, scale,
	i_pivot, n_pivot, pivopt, ordering, singletons ;

    /* ---------------------------------------------------------------------- */
    /* initializations */
    /* ---------------------------------------------------------------------- */

    maxrnorm = 0.0 ;

    /* get the default control parameters */
    UMFPACK_defaults (Control) ;

    UMFPACK_report_control (Control) ;

#ifdef DEBUGGING
    Control [UMFPACK_PRL] = 5 ;
#endif

    status = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Control) ;
    if (status != UMFPACK_OK) { error ("bad A do_matrix\n", 0.) ; }

    /* ---------------------------------------------------------------------- */
    /* allocate workspace */
    /* ---------------------------------------------------------------------- */

    rowdeg = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* ( */
    rowhist = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* ( */
    colhist = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* ( */
    cknob = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* [ */
    rknob = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* [ */

    /* ---------------------------------------------------------------------- */
    /* count the dense rows and columns */
    /* ---------------------------------------------------------------------- */

    for (i = 0 ; i < n ; i++)
    {
	colhist [i] = 0 ;
	rowhist [i] = 0 ;
	rowdeg [i] = 0 ;
    }

    for (col = 0 ; col < n ; col++)
    {
	coldeg = Ap [col+1] - Ap [col] ;
	colhist [coldeg]++ ;
	for (p = Ap [col] ; p < Ap [col+1] ; p++)
	{
	    rowdeg [Ai [p]]++ ;
	}
    }

    for (row = 0 ; row < n ; row++)
    {
	rowhist [rowdeg [row]]++ ;
    }

    rs = 0 ;
    if (do_dense)
    {
        if (n < 16)
        {
	    rknob [rs++] = 0 ;  /* all dense rows */
        }
        for (d = 16 ; d < n ; d++)
        {
	    if (rowhist [d] > 0)
	    {
	        rknob [rs++] = d ;
	    }
        }
    }
    rknob [rs++] = n ;	/* no dense rows */

    cs = 0 ;
    if (do_dense)
    {
        if (n < 16)
        {
	    cknob [cs++] = 0 ;  /* all dense columns */
        }
        for (d = 16 ; d < n ; d++)
        {
	    if (colhist [d] > 0)
	    {
	        cknob [cs++] = d ;
	    }
        }
    }
    cknob [cs++] = n ;	/* no dense cols */

    free (colhist) ;	/* ) */
    free (rowhist) ;	/* ) */

    /* ---------------------------------------------------------------------- */
    /* compute b assuming xtrue (i) = 1 + i/n */
    /* ---------------------------------------------------------------------- */

    b = (double *) malloc (n * sizeof (double)) ;	/* [ */
    bz= (double *) calloc (n , sizeof (double)) ;	/* [ */
    if (!b) error ("out of memory (5)",0.) ;
    if (!bz) error ("out of memory (6)",0.) ;
    bgen (n, Ap, Ai, Ax, Az, b, bz) ;

    /* ---------------------------------------------------------------------- */
    /* compute Qinit = sort by colcounts */
    /* ---------------------------------------------------------------------- */

    noQinit = INULL ;
    Qinit = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* [ */
    if (!Qinit) error ("out of memory (7)",0.) ;
    Head = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* [ */
    Next = (Int *) malloc ((n+1) * sizeof (Int)) ;	/* [ */
    for (d = 0 ; d <= n ; d++)
    {
	Head [d] = EMPTY ;
    }
    for (col = n-1 ; col >= 0 ; col--)
    {
	d = Ap [col+1] - Ap [col] ;
	Next [col] = Head [d] ;
	Head [d] = col ;
    }
    k = 0 ;
    for (d = 0 ; d <= n ; d++)
    {
	for (col = Head [d] ; col != EMPTY ; col = Next [col])
	{
	    Qinit [k++] = col ;
	}
    }
    free (Next) ;	/* ] */
    free (Head) ;	/* ] */

    if (k != n) error ("Qinit error\n",0.) ;

    /* use rowdeg for workspace */
    if (!UMF_is_permutation (Qinit, rowdeg, n, n)) error ("Qinit, colsort, not a permutation vector",0.) ;

    for (k = 1 ; k < n ; k++)
    {
	col1 = Qinit [k-1] ;
	col2 = Qinit [k] ;
	d1 = Ap [col1+1] - Ap [col1] ;
	d2 = Ap [col2+1] - Ap [col2] ;
	if (d1 > d2) error ("Qinit error = not sorted\n",0.) ;
    }

    free (rowdeg) ;					/* ) */

    /* ---------------------------------------------------------------------- */
    /* exhaustive test */
    /* ---------------------------------------------------------------------- */

    n_tol   = Ncontrols [UMFPACK_PIVOT_TOLERANCE] ;
    n_nb    = Ncontrols [UMFPACK_BLOCK_SIZE] ;
    n_init  = Ncontrols [UMFPACK_ALLOC_INIT] ;
    n_scale = Ncontrols [UMFPACK_SCALE] ;
    n_amd   = Ncontrols [UMFPACK_AMD_DENSE] ;

    printf (" cs "ID" rs "ID" : "ID"  ", cs, rs,
	2 + (cs * rs * n_tol * n_nb * n_init )) ;
    fflush (stdout) ;

    /* with defaults - null Control array, Qinit */
    printf ("with qinit:\n") ;
    rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, DNULL, Qinit, MemControl, FALSE, FALSE, 0., 0.) ;
    maxrnorm = MAX (rnorm, maxrnorm) ;

    /* with defaults - null Control array */
    printf ("with null control array:\n") ;
    rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, DNULL, noQinit, MemControl, FALSE, FALSE, 0., 0.) ;
    maxrnorm = MAX (rnorm, maxrnorm) ;

    /* with defaults */
    printf ("with defaults:\n") ;
    rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, noQinit, MemControl, FALSE, FALSE, 0., 0.) ;
    maxrnorm = MAX (rnorm, maxrnorm) ;

    printf ("with defaults and qinit:\n") ;
    rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemControl, FALSE, FALSE, 0., 0.) ;
    maxrnorm = MAX (rnorm, maxrnorm) ;

    /* with lots of ordering options */
    psave = Control [UMFPACK_PRL] ;
    if (n < 15) Control [UMFPACK_PRL] = 4 ;
    for (singletons = 0 ; singletons <= 1 ; singletons++)
    {
        for (ordering = EMPTY ; ordering <= UMFPACK_ORDERING_USER ; ordering++)
        {
            Control [UMFPACK_ORDERING] = ordering ;
            Control [UMFPACK_SINGLETONS] = singletons ;
            printf ("with ordering method "ID" singletons "ID"\n", ordering,
                singletons) ;
            UMFPACK_report_control (Control) ;
            rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, noQinit, MemControl, FALSE, FALSE, 0., 0.) ;
            maxrnorm = MAX (rnorm, maxrnorm) ;
        }
    }
    Control [UMFPACK_ORDERING] = UMFPACK_DEFAULT_ORDERING ;
    Control [UMFPACK_SINGLETONS] = UMFPACK_DEFAULT_SINGLETONS ;
    Control [UMFPACK_PRL] = psave ;

    printf ("starting lengthy tests\n") ;

    for (c = 0 ; c <= cs ; c++)
    {
	if (c == cs)
	{
	    rs = n_amd ;
	}

	for (r = 0 ; r < rs ; r++)
	{
	    if (c == cs)
	    {
		Control [UMFPACK_DENSE_COL] = UMFPACK_DEFAULT_DENSE_COL ;
		Control [UMFPACK_DENSE_ROW] = UMFPACK_DEFAULT_DENSE_ROW ;
		Control [UMFPACK_AMD_DENSE] = Controls [UMFPACK_AMD_DENSE][r] ;
	    }
	    else
	    {
		ck = cknob [c] ;
		rk = rknob [r] ;
		/* ignore columns with degree > ck and rows with degree > rk */
		Control [UMFPACK_DENSE_COL] = inv_umfpack_dense (ck, n) ;
		Control [UMFPACK_DENSE_ROW] = inv_umfpack_dense (rk, n) ;
		Control [UMFPACK_AMD_DENSE] = UMFPACK_DEFAULT_AMD_DENSE ;
	    }

	    for (i_tol = 0 ; i_tol < n_tol ; i_tol++)
	    {
		tol = Controls [UMFPACK_PIVOT_TOLERANCE][i_tol] ;
		Control [UMFPACK_PIVOT_TOLERANCE] = tol ;

		for (i_nb = 0 ; i_nb < n_nb ; i_nb++)
		{
		    nb = (Int) Controls [UMFPACK_BLOCK_SIZE][i_nb] ;
		    Control [UMFPACK_BLOCK_SIZE] = (double) nb ;

		    for (i_init = 0 ; i_init < n_init ; i_init++)
		    {
			init = Controls [UMFPACK_ALLOC_INIT][i_init] ;
			Control [UMFPACK_ALLOC_INIT] = init ;

			for (i_scale = 0 ; i_scale < n_scale ; i_scale++)
			{
			    Int strategy, fixQ ;
			    scale = Controls [UMFPACK_SCALE][i_scale] ;
			    Control [UMFPACK_SCALE] = scale ;

			    for (strategy = UMFPACK_STRATEGY_AUTO ; strategy <= UMFPACK_STRATEGY_SYMMETRIC ; strategy ++)
			    {
                                if (strategy == UMFPACK_STRATEGY_OBSOLETE)
                                {
                                    continue ;
                                }
				Control [UMFPACK_STRATEGY] = strategy ;
				{
				    rnorm = do_once (n, n, Ap, Ai, Ax,Az, b,bz, Control, noQinit, MemControl, FALSE, TRUE, FALSE, 0., 0.) ;
				    maxrnorm = MAX (rnorm, maxrnorm) ;
				    rnorm = do_once (n, n, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemControl, FALSE, TRUE, FALSE, 0., 0.) ;
				    maxrnorm = MAX (rnorm, maxrnorm) ;
				}
			    }
			}
		    }
		}
	    }
	}
    }

    free (Qinit) ;	/* ] */
    free (bz) ;		/* ] */
    free (b) ;		/* ] */
    free (rknob) ;	/* ] */
    free (cknob) ;	/* ] */
    
    return (maxrnorm) ;
}


/* ========================================================================== */
/* matgen_dense: generate a dense matrix */
/* ========================================================================== */

/* allocates Ap, Ai, Ax, and Az */

static void matgen_dense
(
    Int n,
    Int **Ap,
    Int **Ai,
    double **Ax,	double **Az
)
{
    Int nz, *Bp, *Bi, *Ti, *Tj, k, i, j, *P, status ;
    double *Bx, *Tx, *Bz, *Tz ;

    nz = n*n + n ;

    /* allocate Bp, Bi, and Bx - but do not free them */
    Bp = (Int *) malloc ((n+1) * sizeof (Int)) ;
    Bi = (Int *) malloc ((nz+1) * sizeof (Int)) ;
    Bx = (double *) malloc ((nz+1) * sizeof (double)) ;
    Bz = (double *) calloc ((nz+1) , sizeof (double)) ;

    Ti = (Int *) malloc ((nz+1) * sizeof (Int)) ;		/* [ */
    Tj = (Int *) malloc ((nz+1) * sizeof (Int)) ;		/* [ */
    Tx = (double *) malloc ((nz+1) * sizeof (double)) ;	/* [ */
    Tz = (double *) calloc ((nz+1) , sizeof (double)) ;	/* [ */
    P = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */

    if (!Bp || !Bi || !Bx || !Ti || !Tj || !Tx || !P) error ("out of memory (8)",0.) ;
    if (!Bz || !Tz) error ("outof memory",0.) ;

    k = 0 ;
    for (i = 0 ; i < n ; i++)
    {
	for (j = 0 ; j < n ; j++)
	{
	    Ti [k] = i ;
	    Tj [k] = j ;
	    Tx [k] = 2.0 * (xrand ( ) - 1.0) ;
#ifdef COMPLEX
	    Tz [k] = 2.0 * (xrand ( ) - 1.0) ;
#else
	    Tz [k] = 0. ;
#endif
	    k++ ;
	}
    }

    /* beef up each column and row */
    randperm (n, P) ;
    for (i = 0 ; i < n ; i++)
    {
	Ti [k] = i ;
	Tj [k] = P [i] ;
	Tx [k] = xrand ( ) ;
#ifdef COMPLEX
	Tz [k] = xrand ( ) ;
#else
	Tz [k] = 0. ;
#endif
	k++ ;
    }

    if (k != nz) error ("matgen_dense error",0.) ;

    /* convert to column form */
    status = UMFPACK_triplet_to_col (n, n, k, Ti, Tj, CARG(Tx,Tz), Bp, Bi, CARG(Bx,Bz), (Int *) NULL) ;
    if (status != UMFPACK_OK) error ("matgen_dense triplet_to_col failed 2",0.) ;

    /* return the allocated column-form */
    *Ap = Bp ;
    *Ai = Bi ;
    *Ax = Bx ;
    *Az = Bz ;

    free (P) ;		/* ] */
    free (Tz) ;		/* ] */
    free (Tx) ;		/* ] */
    free (Tj) ;		/* ] */
    free (Ti) ;		/* ] */

}


/* ========================================================================== */
/* matgen_funky: generate a kind of arrowhead matrix */
/* ========================================================================== */

/* allocates Ap, Ai, Ax, and Az

   A = speye (n) ;
   A (n, 2:3) = rand (1,2) ;
   A (3, 3:n) = rand (1, n-2) ;
   A (3:n, 1) = rand (n-2, 1) ;

 */

static void matgen_funky
(
    Int n,
    Int **Ap,
    Int **Ai,
    double **Ax,	double **Az
)
{
    Int nz, *Bp, *Bi, *Ti, *Tj, k, i, j, *P, status ;
    double *Bx, *Tx, *Bz, *Tz ;

    nz = 4*n + 5 ;

    /* allocate Bp, Bi, and Bx - but do not free them */
    Bp = (Int *) malloc ((n+1) * sizeof (Int)) ;
    Bi = (Int *) malloc (nz * sizeof (Int)) ;
    Bx = (double *) malloc (nz * sizeof (double)) ;
    Bz = (double *) calloc (nz , sizeof (double)) ;

    Ti = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
    Tj = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
    Tx = (double *) malloc (nz * sizeof (double)) ;	/* [ */
    Tz = (double *) calloc (nz , sizeof (double)) ;	/* [ */
    P = (Int *) malloc (n * sizeof (Int)) ;		/* [ */

    if (!Bp || !Bi || !Bx || !Ti || !Tj || !Tx || !P) error ("out of memory (9)",0.) ;
    if (!Bz || !Tz) error ("outof memory",0.) ;

    k = 0 ;

    /* A = speye (n) ; */
    for (i = 0 ; i < n ; i++)
    {
	Ti [k] = i ;
	Tj [k] = i ;
	Tx [k] = 1.0 ;
#ifdef COMPLEX
	Tz [k] = 1.0 ;
#endif
	k++ ;
    }

    /* A (n, 2:3) = rand (1,2) ; */
    for (j = 1 ; j <= 2 ; j++)
    {
	Ti [k] = n-1 ;
	Tj [k] = j ;
	Tx [k] = 1.0 ;
#ifdef COMPLEX
	Tz [k] = 1.0 ;
#endif
	k++ ;
    }

    /* A (3, 3:n) = rand (1, n-2) ; */
    for (j = 2 ; j < n ; j++)
    {
	Ti [k] = 2 ;
	Tj [k] = j ;
	Tx [k] = 1.0 ;
#ifdef COMPLEX
	Tz [k] = 1.0 ;
#endif
	k++ ;
    }

    /* A (3:n, 1) = rand (n-2, 1) ; */
    for (i = 2 ; i < n ; i++)
    {
	Ti [k] = i ;
	Tj [k] = 0 ;
	Tx [k] = 1.0 ;
#ifdef COMPLEX
	Tz [k] = 1.0 ;
#endif
	k++ ;
    }

    /* convert to column form */
    status = UMFPACK_triplet_to_col (n, n, k, Ti, Tj, CARG(Tx,Tz), Bp, Bi, CARG(Bx,Bz), (Int *) NULL) ;
    if (status != UMFPACK_OK) error ("matgen_dense triplet_to_col failed 1",0.) ;

    /* return the allocated column-form */
    *Ap = Bp ;
    *Ai = Bi ;
    *Ax = Bx ;
    *Az = Bz ;

    free (P) ;		/* ] */
    free (Tz) ;		/* ] */
    free (Tx) ;		/* ] */
    free (Tj) ;		/* ] */
    free (Ti) ;		/* ] */

}


/* ========================================================================== */
/* matgen_band: generate a banded matrix */
/* ========================================================================== */

/* allocates Ap, Ai, and Ax */

static void matgen_band
(
    Int n,
    Int lo,	/* lo = 0:  upper triangular only */
    Int up,	/* up = 0:  lower triangular only */
    Int ndrow,	/* plus ndrow dense rows, each of degree rdeg */
    Int rdeg,
    Int ndcol,	/* plus ndcol dense cols, each of degree cdeg */
    Int cdeg,
    Int **Ap,
    Int **Ai,
    double **Ax,	double **Az
)
{
    Int nz, *Bp, *Bi, *Ti, *Tj, k, i, j, jlo, jup, j1, i1, k1, status ;
    double *Bx, *Bz, *Tx, *Tz ;

    /* an upper bound */
    nz = n * (lo + 1 + up) + n + ndrow*rdeg + ndcol*cdeg ;

    /* allocate Bp, Bi, and Bx - but do not free them */
    Bp = (Int *) malloc ((n+1) * sizeof (Int)) ;
    Bi = (Int *) malloc (nz * sizeof (Int)) ;
    Bx = (double *) malloc (nz * sizeof (double)) ;
    Bz = (double *) calloc (nz , sizeof (double)) ;

    Ti = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
    Tj = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
    Tx = (double *) malloc (nz * sizeof (double)) ;	/* [ */
    Tz = (double *) calloc (nz , sizeof (double)) ;	/* [ */

    if (!Bp || !Bi || !Bx || !Ti || !Tj || !Tx) error ("out of memory (10)",0.) ;
    if (!Bz || !Tz) error ("out ofmemory", 0.) ;

    k = 0 ;
    for (i = 0 ; i < n ; i++)
    {
	jlo = MAX (0,   i - lo) ;
	jup = MIN (n-1, i + up) ;
	for (j = jlo ; j <= jup ; j++)
	{
	    Ti [k] = i ;
	    Tj [k] = j ;
	    Tx [k] = 2.0 * (xrand ( ) - 1.0) ;
#ifdef COMPLEX
	    Tz [k] = 2.0 * (xrand ( ) - 1.0) ;
#else
	    Tz [k] = 0. ;
#endif
	    k++ ;
	}
    }

    /* beef up the diagonal */
    for (i = 0 ; i < n ; i++)
    {
	Ti [k] = i ;
	Tj [k] = i ;
	Tx [k] = xrand ( ) ;
#ifdef COMPLEX
	Tz [k] = xrand ( ) ;
#else
	Tz [k] = 0. ;
#endif
	k++ ;
    }

    /* add ndrow rows of degree rdeg */
    for (i1 = 0 ; i1 < ndrow ; i1++)
    {
	i = irand (n) ;
	for (k1 = 0 ; k1 < rdeg ; k1++)
	{
	    Ti [k] = i ;
	    Tj [k] = irand (n) ;
	    Tx [k] = 0.1 * xrand ( ) ;
#ifdef COMPLEX
	    Tz [k] = 0.1 * xrand ( ) ;
#else
	    Tz [k] = 0. ;
#endif
	    k++ ;
	}
    }

    /* add ndcol rows of degree cdeg */
    for (j1 = 0 ; j1 < ndcol ; j1++)
    {
	j = irand (n) ;
	for (k1 = 0 ; k1 < cdeg ; k1++)
	{
	    Ti [k] = irand (n) ;
	    Tj [k] = j ;
	    Tx [k] = 0.1 * xrand ( ) ;
#ifdef COMPLEX
	    Tz [k] = 0.1 * xrand ( ) ;
#else
	    Tz [k] = 0. ;
#endif
	    k++ ;
	}
    }

    if (k > nz) error ("matgen_band error\n",0.) ;

    /* convert to column form */
    status = UMFPACK_triplet_to_col (n, n, k, Ti, Tj, CARG(Tx,Tz), Bp, Bi, CARG(Bx,Bz), (Int *) NULL) ;
    if (status != UMFPACK_OK) error ("matgen_band triplet_to_col failed\n",0.) ;

    /* return the allocated column-form */
    *Ap = Bp ;
    *Ai = Bi ;
    *Ax = Bx ;
    *Az = Bz ;

    free (Tz) ;	    /* ] */
    free (Tx) ;	    /* ] */
    free (Tj) ;	    /* ] */
    free (Ti) ;	    /* ] */

}


/* ========================================================================== */
/* test_col */
/* ========================================================================== */

static void test_col
(
    Int n,
    Int Bp [ ],
    Int Bi [ ],
    double Bx [ ],	double Bz [ ],
    Int prl
)
{
    Int *Ci, *Cj, *Ep, *Ei, k, s, t, i, j, nz, p, status, *Map, *noMap ;
    double *Cx, *Cz, *Ex, *Ez, z, Control [UMFPACK_CONTROL] ;
    noMap = (Int *) NULL ;

    printf ("\n\n===== test col -> triplet and triplet -> col\n") ;

    UMFPACK_defaults (Control) ;

    Control [UMFPACK_PRL] = prl ;
    UMFPACK_report_control (Control) ;

    nz = Bp [n] ;

    Ci = (Int *) malloc ((2*nz+1) * sizeof (Int)) ;		/* [ */
    Cj = (Int *) malloc ((2*nz+1) * sizeof (Int)) ;		/* [ */
    Cx = (double *) calloc (2*(2*nz+1) , sizeof (double)) ;	/* [ */
    Cz = Cx + (2*nz+1) ;
    Ep = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
    Ei = (Int *) malloc ((2*nz+1) * sizeof (Int)) ;		/* [ */
    Ex = (double *) calloc (2*(2*nz+1) , sizeof (double)) ;	/* [ */
    Ez = Ex + (2*nz+1) ;
    Map = (Int *) malloc ((2*nz+1) * sizeof (Int)) ;		/* [ */

    if (!Ci || !Cj || !Cx || !Ep || !Ei || !Ex) error ("out of memory (11)",0.) ;

    /* ---------------------------------------------------------------------- */
    /* test with split complex values */
    /* ---------------------------------------------------------------------- */

    /* convert B (col) to C (triplet) */
    status = UMFPACK_col_to_triplet (n, Bp, Cj) ;
    if (status != UMFPACK_OK) error ("col->triplet",0.) ;
    for (k = 0 ; k < nz ; k++)
    {
	Ci [k] = Bi [k] ;
	Cx [k] = Bx [k] ;
	Cz [k] = Bz [k] ;
    }

    /* convert C (triplet) to E (col) */
    status = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(Cx,Cz), Ep, Ei, CARG(Ex,Ez), Map) ;
    if (status != UMFPACK_OK) error ("t->col failed (1)",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (1)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [p]) > 1e-15) error ("Ex",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ez [p]) > 1e-15) error ("Ez (1)",0.) ;
	}
    }

    /* convert C (triplet) to E (col), no map */
    status = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(Cx,Cz), Ep, Ei, CARG(Ex,Ez), noMap) ;
    if (status != UMFPACK_OK) error ("t->col failed (1)",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (2)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [p]) > 1e-15) error ("Ex",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ez [p]) > 1e-15) error ("Ez",0.) ;
	}
    }

    /* jumble C a little bit */
    for (k = 0 ; k < MIN (nz,4) ; k++)
    {
	s = irand (nz) ;
	/* swap positions s and k */
	t = Ci [k] ; Ci [k] = Ci [s] ; Ci [s] = t ;
	t = Cj [k] ; Cj [k] = Cj [s] ; Cj [s] = t ;
	z = Cx [k] ; Cx [k] = Cx [s] ; Cx [s] = z ;
	z = Cz [k] ; Cz [k] = Cz [s] ; Cz [s] = z ;
    }

    /* convert C (triplet) to E (col) */
    status = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(Cx,Cz), Ep, Ei, CARG(Ex,Ez), Map) ;
    if (status != UMFPACK_OK) error ("t->col failed (1)",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (3)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [p]) > 1e-15) error ("Ex",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ez [p]) > 1e-15) error ("Ez",0.) ;
	}
    }

    /* convert C (triplet) to E (col), no map */
    status = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(Cx,Cz), Ep, Ei, CARG(Ex,Ez), noMap) ;
    if (status != UMFPACK_OK) error ("t->col failed (1)",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (4)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [p]) > 1e-15) error ("Ex",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ez [p]) > 1e-15) error ("Ez",0.) ;
	}
    }

    /* jumble C a lot */
    for (k = 0 ; k < nz ; k++)
    {
	s = irand (nz) ;
	/* swap positions s and k */
	t = Ci [k] ; Ci [k] = Ci [s] ; Ci [s] = t ;
	t = Cj [k] ; Cj [k] = Cj [s] ; Cj [s] = t ;
	z = Cx [k] ; Cx [k] = Cx [s] ; Cx [s] = z ;
	z = Cz [k] ; Cz [k] = Cz [s] ; Cz [s] = z ;
    }

    /* add duplicates to C, but preserve pattern and values */
    for (k = nz ; k < 2*nz ; k++)
    {
	/* add a duplicate */
	s = irand (k) ;
	Ci [k] = Ci [s] ;
	Cj [k] = Cj [s] ;
	z = Cx [s] ;
	Cx [s] = z/2 ;
	Cx [k] = z/2 ;
	z = Cz [s] ;
	Cz [s] = z/2 ;
	Cz [k] = z/2 ;
    }

    if (prl > 2) printf ("\ntest c->t,t->c: ") ;
    status = UMFPACK_report_triplet (n, n, 2*nz, Ci, Cj, CARG(Cx,Cz), Control) ;
    if (status != UMFPACK_OK) error ("report col->triplet",0.) ;

    /* convert C (triplet) to E (col), no Map */
    status = UMFPACK_triplet_to_col (n, n, 2*nz, Ci, Cj, CARG(Cx,Cz), Ep, Ei, CARG(Ex,Ez), noMap) ;
    if (status != UMFPACK_OK) error ("t->col failed",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (5)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [p]) > 1e-15) error ("Ex",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ez [p]) > 1e-15)
	    {
		printf ("%30.18e %30.18e %g\n", Bz[p], Ez[p], Bz[p]-Ez[p]) ;
		error ("Ez (5)",0.) ;
	    }
	}
    }

    /* convert C (triplet) to E (col) */
    status = UMFPACK_triplet_to_col (n, n, 2*nz, Ci, Cj, CARG(Cx,Cz), Ep, Ei, CARG(Ex,Ez), Map) ;
    if (status != UMFPACK_OK) error ("t->col failed",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (6)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [p]) > 1e-15) error ("Ex",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ez [p]) > 1e-15) error ("Ez",0.) ;
	}
    }

    /* convert C (triplet) to E (col), using Map */
    for (p = 0 ; p < Ep [n] ; p++)
    {
	Ex [p] = 0. ;
	Ez [p] = 0. ;
    }
    for (k = 0 ; k < 2*nz ; k++)
    {
	p = Map [k] ;
	i = Ci [k] ;
	j = Cj [k] ;
	if (i != Ei [p]) error ("Map", 0.) ;
	if (!(Ep [j] <= p && p < Ep [j+1])) error ("Map Ep", 0.) ;
	Ex [p] += Cx [k] ;
	Ez [p] += Cz [k] ;
    }

    /* compare E and B, they should be identical */
    for (j = 0 ; j < n ; j++)
    {
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    z = SCALAR_ABS (Bx [p] - Ex [p]) ;
	    if (z > 1e-15) error ("Ex",z) ;
	    z = SCALAR_ABS (Bz [p] - Ez [p]) ;
	    if (z > 1e-15) error ("Ez (7)",z) ;
	}
    }

#ifdef COMPLEX
    /* ---------------------------------------------------------------------- */
    /* repeat, but with merged complex values */
    /* ---------------------------------------------------------------------- */

    /* convert B (col) to C (triplet) */
    status = UMFPACK_col_to_triplet (n, Bp, Cj) ;
    if (status != UMFPACK_OK) error ("col->triplet",0.) ;
    for (k = 0 ; k < nz ; k++)
    {
	Ci [k] = Bi [k] ;
	Cx [2*k  ] = Bx [k] ;
	Cx [2*k+1] = Bz [k] ;
    }

    /* convert C (triplet) to E (col) */
    status = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(Cx,DNULL), Ep, Ei, CARG(Ex,DNULL), Map) ;
    if (status != UMFPACK_OK) error ("t->col failed (1)",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (7)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [2*p  ]) > 1e-15) error ("Ex (merge)",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ex [2*p+1]) > 1e-15) error ("Ez (merge)",0.) ;
	}
    }

    /* convert C (triplet) to E (col) no Map */
    status = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(Cx,DNULL), Ep, Ei, CARG(Ex,DNULL), noMap) ;
    if (status != UMFPACK_OK) error ("t->col failed (1)",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (8)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [2*p  ]) > 1e-15) error ("Ex (merge)",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ex [2*p+1]) > 1e-15) error ("Ez (merge)",0.) ;
	}
    }

    /* jumble C a little bit */
    for (k = 0 ; k < MIN (nz,4) ; k++)
    {
	s = irand (nz) ;
	/* swap positions s and k */
	t = Ci [k] ; Ci [k] = Ci [s] ; Ci [s] = t ;
	t = Cj [k] ; Cj [k] = Cj [s] ; Cj [s] = t ;
	z = Cx [2*k  ] ; Cx [2*k  ] = Cx [2*s  ] ; Cx [2*s  ] = z ;
	z = Cx [2*k+1] ; Cx [2*k+1] = Cx [2*s+1] ; Cx [2*s+1] = z ;
    }

    /* convert C (triplet) to E (col) */
    status = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(Cx,DNULL), Ep, Ei, CARG(Ex,DNULL), Map) ;
    if (status != UMFPACK_OK) error ("t->col failed (1)",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (9)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [2*p  ]) > 1e-15) error ("Ex (merge)",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ex [2*p+1]) > 1e-15) error ("Ez (merge)",0.) ;
	}
    }

    /* convert C (triplet) to E (col) no Map */
    status = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(Cx,DNULL), Ep, Ei, CARG(Ex,DNULL), noMap) ;
    if (status != UMFPACK_OK) error ("t->col failed (1)",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (10)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [2*p  ]) > 1e-15) error ("Ex (merge)",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ex [2*p+1]) > 1e-15) error ("Ez (merge)",0.) ;
	}
    }

    /* jumble C */
    for (k = 0 ; k < nz ; k++)
    {
	s = irand (nz) ;
	/* swap positions s and k */
	t = Ci [k] ; Ci [k] = Ci [s] ; Ci [s] = t ;
	t = Cj [k] ; Cj [k] = Cj [s] ; Cj [s] = t ;
	z = Cx [2*k  ] ; Cx [2*k  ] = Cx [2*s  ] ; Cx [2*s  ] = z ;
	z = Cx [2*k+1] ; Cx [2*k+1] = Cx [2*s+1] ; Cx [2*s+1] = z ;
    }

    /* add duplicates to C, but preserve pattern and values */
    for (k = nz ; k < 2*nz ; k++)
    {
	/* add a duplicate */
	s = irand (k) ;
	Ci [k] = Ci [s] ;
	Cj [k] = Cj [s] ;
	z = Cx [2*s] ;
	Cx [2*s] = z/2 ;
	Cx [2*k] = z/2 ;
	z = Cx [2*s+1] ;
	Cx [2*s+1] = z/2 ;
	Cx [2*k+1] = z/2 ;
    }

    if (prl > 2) printf ("\ntest c->t,t->c: ") ;
    status = UMFPACK_report_triplet (n, n, 2*nz, Ci, Cj, CARG(Cx,DNULL), Control) ;
    if (status != UMFPACK_OK) error ("report col->triplet",0.) ;

    /* convert C (triplet) to E (col) */
    status = UMFPACK_triplet_to_col (n, n, 2*nz, Ci, Cj, CARG(Cx,DNULL), Ep, Ei, CARG(Ex,DNULL), noMap) ;
    if (status != UMFPACK_OK) error ("t->col failed",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (11)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [2*p  ]) > 1e-15) error ("Ex (merge)",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ex [2*p+1]) > 1e-15) error ("Ez (merge)",0.) ;
	}
    }

    /* convert C (triplet) to E (col) */
    status = UMFPACK_triplet_to_col (n, n, 2*nz, Ci, Cj, CARG(Cx,DNULL), Ep, Ei, CARG(Ex,DNULL), Map) ;
    if (status != UMFPACK_OK) error ("t->col failed",0.) ;

    /* compare E and B, they should be identical */
    if (nz != Ep [n]) error ("E nz (12)",0.) ;
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Ep [j]) error ("Ep j",0.) ;
	if (Bp [j+1] != Ep [j+1]) error ("Ep j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ei [p]) error ("Ei",0.) ;
	    if (SCALAR_ABS (Bx [p] - Ex [2*p  ]) > 1e-15) error ("Ex (merge)",0.) ;
	    if (SCALAR_ABS (Bz [p] - Ex [2*p+1]) > 1e-15) error ("Ez (merge)",0.) ;
	}
    }

    /* convert C (triplet) to E (col), using Map */
    for (p = 0 ; p < Ep [n] ; p++)
    {
	Ex [2*p  ] = 0. ;
	Ex [2*p+1] = 0. ;
    }
    for (k = 0 ; k < 2*nz ; k++)
    {
	p = Map [k] ;
	i = Ci [k] ;
	j = Cj [k] ;
	if (i != Ei [p]) error ("Map", 0.) ;
	if (!(Ep [j] <= p && p < Ep [j+1])) error ("Map Ep", 0.) ;
	Ex [2*p  ] += Cx [2*k  ] ;
	Ex [2*p+1] += Cx [2*k+1] ;
    }

    /* compare E and B, they should be identical */
    for (j = 0 ; j < n ; j++)
    {
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    z = SCALAR_ABS (Bx [p] - Ex [2*p  ]) ;
	    if (z > 1e-15) error ("Ex merged",z) ;
	    z = SCALAR_ABS (Bz [p] - Ex [2*p+1]) ;
	    if (z > 1e-15) error ("Ez merged 7",z) ;
	}
    }

#endif

    printf ("\n =============== test OK\n\n") ;

    free (Map) ;	/* ] */
    free (Ex) ;		/* ] */
    free (Ei) ;		/* ] */
    free (Ep) ;		/* ] */
    free (Cx) ;		/* ] */
    free (Cj) ;		/* ] */
    free (Ci) ;		/* ] */
}

/* ========================================================================== */
/* matgen_compaction: generate a matrix to test umf_symbolic compaction */
/* ========================================================================== */

static void matgen_compaction
(
   Int n,
   Int **Ap,
   Int **Ai,
   double **Ax,		double **Az
)
{
    Int nz, *Bp, *Bi, *Ti, *Tj, k, i, j, prl, status ;
    double *Bx, *Tx, *Bz, *Tz, Control [UMFPACK_INFO] ;

    prl = Control ? Control [UMFPACK_PRL] : UMFPACK_DEFAULT_PRL ; 
    UMFPACK_defaults (Control) ;

    UMFPACK_report_control (Control) ;

    nz = 5*n ;

    /* allocate Bp, Bi, and Bx - but do not free them */
    Bp = (Int *) malloc ((n+1) * sizeof (Int)) ;
    Bi = (Int *) malloc (nz * sizeof (Int)) ;
    Bx = (double *) malloc (nz * sizeof (double)) ;
    Bz = (double *) calloc (nz , sizeof (double)) ;

    Ti = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
    Tj = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
    Tx = (double *) malloc (nz * sizeof (double)) ;	/* [ */ 
    Tz = (double *) calloc (nz , sizeof (double)) ;	/* [ */ 
    if (!Bp || !Bi || !Bx || !Ti || !Tj || !Tx) error ("out of memory (12)",0.) ;
    if (!Bz || !Tz) error ("out of mery",0.) ;

    k = 0 ;
    for (i = 0 ; i < n ; i++)
    {
	if (i > 0)
	{
	    Ti [k] = i ;
	    Tj [k] = i-1 ;
	    Tx [k] = xrand ( ) ;
#ifdef COMPLEX
	    Tz [k] = xrand ( ) ;
#else
	    Tz [k] = 0. ;
#endif
	    k++ ;
	}
	Ti [k] = i ;
	Tj [k] = i ;
	Tx [k] = xrand ( ) ;
	k++ ;
	if (i < n-1)
	{
	    Ti [k] = i ;
	    Tj [k] = i+1 ;
	    Tx [k] = xrand ( ) ;
#ifdef COMPLEX
	    Tz [k] = xrand ( ) ;
#else
	    Tz [k] = 0. ;
#endif
	    k++ ;
	}
    }

    for (j = 0 ; j < n ; j += 2)
    {
	Ti [k] = 0 ;
	Tj [k] = j ;
	Tx [k] = xrand ( ) ;
#ifdef COMPLEX
	Tz [k] = xrand ( ) ;
#else
	Tz [k] = 0. ;
#endif
	k++ ;
    }

    for (j = 1 ; j < n ; j += 2)
    {
	Ti [k] = 1 ;
	Tj [k] = j ;
	Tx [k] = xrand ( ) ;
#ifdef COMPLEX
	Tz [k] = xrand ( ) ;
#else
	Tz [k] = 0. ;
#endif
	k++ ;
    }


    if (prl > 2) printf ("\nmatgen_compact: ") ;
    status = UMFPACK_report_triplet (n, n, k, Ti, Tj, CARG(Tx,Tz), Control) ;
    if (status != UMFPACK_OK) error ("bad triplet report",0.) ;

    /* convert to column form */
    status = UMFPACK_triplet_to_col (n, n, k, Ti, Tj, CARG(Tx,Tz), Bp, Bi, CARG(Bx,Bz), (Int *) NULL) ;
    if (status != UMFPACK_OK) error ("matgen_compact triplet_to_col failed",0.) ;

    if (prl > 2) printf ("\nmatgen_compact: ") ;
    status = UMFPACK_report_matrix (n, n, Bp, Bi, CARG(Bx,Bz), 1, Control) ;
    if (status != UMFPACK_OK) error ("bad A matget sparse",0.) ;

    /* return the allocated column-form */
    *Ap = Bp ;
    *Ai = Bi ;
    *Ax = Bx ;
    *Az = Bz ;

    free (Tz) ;	/* ] */
    free (Tx) ;	/* ] */
    free (Tj) ;	/* ] */
    free (Ti) ;	/* ] */
}


/* ========================================================================== */
/* matgen_sparse: generate a matrix with random pattern */
/* ========================================================================== */

/* allocates Ap, Ai, Ax, and Az */

static void matgen_sparse
(
    Int n,
    Int s,	/* s random entries, and one more in each row and column */
    Int ndrow,	/* plus ndrow dense rows, each of degree rdeg */
    Int rdeg,
    Int ndcol,	/* plus ndcol dense cols, each of degree cdeg */
    Int cdeg,
    Int **Ap,
    Int **Ai,
    double **Ax,	double **Az,
    Int prl,
    Int has_nans
)
{
    Int nz, *Bp, *Bi, *Ti, *Tj, k, i, j, j1, i1, k1, *P, status, *Map, p, *Cp, *Ci, *noMap ;
    double *Bx, *Tx, *Bz, *Tz, Control [UMFPACK_CONTROL], xnan, xinf, x, *Txx, *Cx ;
    noMap = (Int *) NULL ;

    if (has_nans)
    {
	xnan = divide (0., 0.) ;
	xinf = divide (1., 0.) ;
    }

    UMFPACK_defaults (Control) ;
    Control [UMFPACK_PRL] = prl ;
    UMFPACK_report_control (Control) ;

    nz = s + n + rdeg*ndrow + cdeg*ndcol ; 
    if (nz == 0) nz++ ;

    /* allocate Bp, Bi, and Bx - but do not free them */
    Bp = (Int *) malloc ((n+1) * sizeof (Int)) ;
    Bi = (Int *) malloc ((nz+1) * sizeof (Int)) ;
    Bx = (double *) malloc ((nz+1) * sizeof (double)) ;
    Bz = (double *) calloc ((nz+1) , sizeof (double)) ;

    Ti = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
    Tj = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
    Tx = (double *) malloc (nz * sizeof (double)) ;	/* [ */
    Tz = (double *) calloc (nz , sizeof (double)) ;	/* [ */ 
    P = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
    Map = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */

    Txx = (double *) calloc (2*(nz+1) , sizeof (double)) ;	/* [ */ 
    Cp = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
    Ci = (Int *) malloc ((nz+1) * sizeof (Int)) ;		/* [ */
    Cx = (double *) calloc (2*(nz+1) , sizeof (double)) ;	/* [ */ 

    if (!Bp || !Bi || !Bx || !Ti || !Tj || !Tx || !P) error ("out of memory (13)",0.) ;
    if (!Bz || !Tz) error ("out of m",0.) ;
    if (!Txx || !Cx || !Cp || !Ci) error ("out of mem xx",0.) ;

    for (k = 0 ; k < s ; k++)
    {
	Ti [k] = irand (n) ;
	Tj [k] = irand (n) ;
	if (has_nans)
	{
	    x = xrand ( ) ;
	    Tx [k] = (x > 0.8) ? ((x > 0.9) ? xnan : xinf) : (2*x-1) ;
#ifdef COMPLEX
	    x = xrand ( ) ;
	    Tz [k] = (x > 0.8) ? ((x > 0.9) ? xnan : xinf) : (2*x-1) ;
#else
	    Tz [k] = 0. ;
#endif
	}
	else
	{
	    Tx [k] = 2.0 * (xrand ( ) - 1.0) ;
#ifdef COMPLEX
	    Tz [k] = 2.0 * (xrand ( ) - 1.0) ;
#else
	    Tz [k] = 0. ;
#endif
	}
    }

    /* beef up each column and row */
    randperm (n, P) ;
    for (i = 0 ; i < n ; i++)
    {
	Ti [k] = i ;
	Tj [k] = P [i] ;
	Tx [k] = xrand ( ) ;
#ifdef COMPLEX
	Tz [k] = xrand ( ) ;
#else
	Tz [k] = 0. ;
#endif
	k++ ;
    }

    /* add ndrow rows of degree rdeg */
    for (i1 = 0 ; i1 < ndrow ; i1++)
    {
	i = irand (n) ;
	for (k1 = 0 ; k1 < rdeg ; k1++)
	{
	    Ti [k] = i ;
	    Tj [k] = irand (n) ;
	    Tx [k] = 0.1 * xrand ( ) ;
#ifdef COMPLEX
	    Tz [k] = 0.1 * xrand ( ) ;
#else
	    Tz [k] = 0. ;
#endif
	    k++ ;
	}
    }

    /* add ndcol rows of degree cdeg */
    for (j1 = 0 ; j1 < ndcol ; j1++)
    {
	j = irand (n) ;
	for (k1 = 0 ; k1 < cdeg ; k1++)
	{
	    Ti [k] = irand (n) ;
	    Tj [k] = j ;
	    Tx [k] = 0.1 * xrand ( ) ;
#ifdef COMPLEX
	    Tz [k] = 0.1 * xrand ( ) ;
#else
	    Tz [k] = 0. ;
#endif
	    k++ ;
	}
    }

    if (k != nz) error ("matgen_sparse error",0.) ;

    if (prl > 2) printf ("\nmatgen_sparse: ") ;
    status = UMFPACK_report_triplet (n, n, k, Ti, Tj, CARG(Tx,Tz), Control) ;
    if (status != UMFPACK_OK) error ("bad triplet report",0.) ;

    /* convert to column form */
    status = UMFPACK_triplet_to_col (n, n, k, Ti, Tj, CARG(Tx,Tz), Bp, Bi, CARG(Bx,Bz), Map) ;
    if (status != UMFPACK_OK) error ("matgen_sparse triplet_to_col failed",0.) ;

    /* convert to column form, no values and no map */
    status = UMFPACK_triplet_to_col (n, n, k, Ti, Tj, CARG(DNULL,DNULL), Cp, Ci, CARG(DNULL,DNULL), noMap) ;
    if (status != UMFPACK_OK) error ("matgen_sparse triplet_to_col failed",0.) ;

    /* compare C and B, they should be identical */
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Cp [j]) error ("Cp j",0.) ;
	if (Bp [j+1] != Cp [j+1]) error ("Cp j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ci [p]) error ("Ci",0.) ;
	}
    }

    /* convert to column form, no values and with map */
    status = UMFPACK_triplet_to_col (n, n, k, Ti, Tj, CARG(DNULL,DNULL), Cp, Ci, CARG(DNULL,DNULL), Map) ;
    if (status != UMFPACK_OK) error ("matgen_sparse triplet_to_col failed",0.) ;

    /* compare C and B, they should be identical */
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Cp [j]) error ("Cp j",0.) ;
	if (Bp [j+1] != Cp [j+1]) error ("Cp j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ci [p]) error ("Ci",0.) ;
	}
    }

#ifdef COMPLEX
    /* test with merged case too */
    for (p = 0 ; p < k ; p++)
    {
	Txx [2*p  ] = Tx [p] ;
	Txx [2*p+1] = Tz [p] ;
    }

    /* convert to column form, no map */
    status = UMFPACK_triplet_to_col (n, n, k, Ti, Tj, CARG(Txx,DNULL), Cp, Ci, CARG(Cx,DNULL), noMap) ;
    if (status != UMFPACK_OK) error ("matgen_sparse triplet_to_col failed",0.) ;

    /* compare C and B, they should be identical */
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Cp [j]) error ("Cp j",0.) ;
	if (Bp [j+1] != Cp [j+1]) error ("Cp j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ci [p]) error ("Ci",0.) ;
	    if (SCALAR_ABS (Bx [p] - Cx [2*p  ]) > 1e-15) error ("Cx",0.) ;
	    if (SCALAR_ABS (Bz [p] - Cx [2*p+1]) > 1e-15) error ("Cz (1)",0.) ;
	}
    }

    /* convert to column form */
    status = UMFPACK_triplet_to_col (n, n, k, Ti, Tj, CARG(Txx,DNULL), Cp, Ci, CARG(Cx,DNULL), Map) ;
    if (status != UMFPACK_OK) error ("matgen_sparse triplet_to_col failed",0.) ;

    /* compare C and B, they should be identical */
    for (j = 0 ; j < n ; j++)
    {
	if (Bp [j] != Cp [j]) error ("Cp j",0.) ;
	if (Bp [j+1] != Cp [j+1]) error ("Cp j",0.) ;
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    if (Bi [p] != Ci [p]) error ("Ci",0.) ;
	    if (SCALAR_ABS (Bx [p] - Cx [2*p  ]) > 1e-15) error ("Cx",0.) ;
	    if (SCALAR_ABS (Bz [p] - Cx [2*p+1]) > 1e-15) error ("Cz (1)",0.) ;
	}
    }

#endif

    if (prl > 2) printf ("\nmatgen_sparse: ") ;
    status = UMFPACK_report_matrix (n, n, Bp, Bi, CARG(Bx,Bz), 1, Control) ;
    if (status != UMFPACK_OK) error ("bad A matgen sparse",0.) ;

    /* check the Map */
    for (k = 0 ; k < nz ; k++)
    {
	p = Map [k] ;
	i = Ti [k] ;
	j = Tj [k] ;
	if (i != Bi [p]) error ("Map Bi", 0.) ;
	if (!(Bp [j] <= p && p < Bp [j+1])) error ("Map Bp", 0.) ;
    }

    /* test triplet->col and col->triplet */
    test_col (n, Bp, Bi, Bx,Bz, prl) ;

    /* return the allocated column-form */
    *Ap = Bp ;
    *Ai = Bi ;
    *Ax = Bx ;
    *Az = Bz ;

    free (Cx) ;    /* ] */
    free (Ci) ;    /* ] */
    free (Cp) ;    /* ] */
    free (Txx) ;    /* ] */

    free (Map) ;    /* ] */
    free (P) ;	    /* ] */
    free (Tz) ;	    /* ] */
    free (Tx) ;	    /* ] */
    free (Tj) ;	    /* ] */
    free (Ti) ;	    /* ] */

}

/* ========================================================================== */
/* matgen_transpose:  B = A(P,Q)', where P and Q are random */
/* ========================================================================== */

static void matgen_transpose
(
    Int n,
    Int Ap [ ],
    Int Ai [ ],
    double Ax [ ],	double Az [ ],
    Int **Bp,
    Int **Bi,
    double **Bx,	double **Bz
)
{
    Int nz, *P, *Q, *Cp, *Ci, status ;
    double *Cx, *Cz ;

#ifdef DEBUGGING
    double Control [UMFPACK_CONTROL] ;
#endif

    nz = Ap [n] ;
    P = (Int *) malloc (n * sizeof (Int)) ;	/* [ */
    Q = (Int *) malloc (n * sizeof (Int)) ;	/* [ */

    Cp = (Int *) malloc ((n+1) * sizeof (Int)) ;
    Ci = (Int *) malloc ((nz+1) * sizeof (Int)) ;
    Cx = (double *) malloc ((nz+1) * sizeof (double)) ;
    Cz = (double *) calloc ((nz+1) , sizeof (double)) ;

    if (!P || !Q || !Bp || !Bi || !Bx) error ("out of memory (14)",0.) ;
    if (!Cz) error ("out mem", 0.) ;

    randperm (n, P) ;
    randperm (n, Q) ;

#ifdef DEBUGGING
    UMFPACK_defaults (Control) ;
    Control [UMFPACK_PRL] = 5 ;
    printf ("\nA: ") ;
    status = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Control) ;
    if (status != UMFPACK_OK) error ("bad A",0.) ;
    printf ("Random P: ") ;
    UMFPACK_report_perm (n, P, Control) ;
    if (status != UMFPACK_OK) error ("bad random P",0.) ;
    printf ("Random Q: ") ;
    status = UMFPACK_report_perm (n, Q, Control) ;
    if (status != UMFPACK_OK) error ("bad random Q",0.) ;
#endif

    /* do complex conjugate transpose */
    status = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), P, Q, Cp, Ci, CARG(Cx,Cz) C1ARG(1)) ;
    if (status != UMFPACK_OK) error ("transpose failed",0.) ;

#ifdef DEBUGGING
    printf ("\nC: ") ;
    status = UMFPACK_report_matrix (n, n, Cp, Ci, CARG(Cx,Cz), 1, Control) ;
    if (status != UMFPACK_OK) error ("bad C",0.) ;
#endif

    /* do not free Cp, Ci, and Cx */
    *Bp = Cp ;
    *Bi = Ci ;
    *Bx = Cx ;
    *Bz = Cz ;

    free (P) ;	/* ] */
    free (Q) ;	/* ] */
}


/* ========================================================================== */
/* matgen_file:  read a (1-based) matrix and a Qinit from a file */
/* ========================================================================== */

/* File syntax:
 *	1st line:	    nrows ncols nnz isreal
 *	next nnz lines:	    i j x   ... or  ...  i j xreal ximag
 *	next ncols lines:   Qk
 *	one line            determinant (real and imag. part if A is complex)
 *	last line	    determinant of real part of A only
 */

static void matgen_file
(
    char *filename,
    Int *n_row,
    Int *n_col,
    Int **Ap,
    Int **Ai,
    double **Ax,	double **Az,
    Int **Qinit,
    Int prl,
    double *det_x,
    double *det_z
)
{
    FILE *f ;
    Int i, j, k, *Ti, *Tj, nr, nc, nz, *Bp, *Bi, *Q, status, isreal, nz1, n ;
    double x, *Tx, *Bx, *Tz, *Bz, ximag, Control [UMFPACK_CONTROL],
	    d_x, d_z, d_real ;

    printf ("\nFile: %s\n", filename) ;
    f = fopen (filename, "r") ;
    if (!f) error ("bad file", 0.) ;

    fscanf (f, ""ID" "ID" "ID" "ID"\n", &nr, &nc, &nz, &isreal) ;
    n = MAX (nr, nc) ;
    n = MAX (n,1) ;

    nz1 = MAX (nz,1) ;
    Ti = (Int *) malloc (nz1 * sizeof (Int)) ;		/* [ */
    Tj = (Int *) malloc (nz1 * sizeof (Int)) ;		/* [ */
    Tx = (double *) malloc (nz1 * sizeof (double)) ;	/* [ */
    Tz = (double *) calloc (nz1 , sizeof (double)) ;	/* [ */

    /* allocate Bp, Bi, Bx, and Q - but do not free them */
    Bp = (Int *) malloc ((n+1) * sizeof (Int)) ;
    Bi = (Int *) malloc (nz1 * sizeof (Int)) ;
    Bx = (double *) malloc (nz1 * sizeof (double)) ;
    Q = (Int *) malloc (n * sizeof (Int)) ;
    Bz = (double *) calloc (nz1 , sizeof (double)) ;

    for (k = 0 ; k < nz ; k++) 
    {
	if (isreal)
	{
	     fscanf (f, ""ID" "ID" %lg\n", &i, &j, &x) ;
	     ximag = 0. ;
	}
	else
	{
	     fscanf (f, ""ID" "ID" %lg %lg\n", &i, &j, &x, &ximag) ;
	}
	Ti [k] = i-1 ;	/* convert to 0-based */ 
	Tj [k] = j-1 ;
	Tx [k] = x ;
#ifdef COMPLEX
	Tz [k] = ximag ;
#else
	/* the file may have a complex part, but set it to zero */
	Tz [k] = 0. ;
#endif
    }

    for (k = 0 ; k < nc ; k++)
    {
	fscanf (f, ""ID"\n", &i) ;
	Q [k] = i-1 ;	/* convert to 0-based */
    }

    if (isreal)
    {
	fscanf (f, "%lg\n", &d_x) ;
	d_z = 0 ;
    }
    else
    {
	fscanf (f, "%lg %lg\n", &d_x, &d_z) ;
    }
    fscanf (f, "%lg\n", &d_real) ;
    printf ("%s det: %g + (%g)i, real(A): %g\n", filename, d_x, d_z, d_real) ;

#ifdef COMPLEX
    *det_x = d_x ;
    *det_z = d_z ;
#else
    /* imaginary part of matrix is ignored */
    *det_x = d_real ;
    *det_z = 0 ;
#endif

    UMFPACK_defaults (Control) ;
    Control [UMFPACK_PRL] = prl ;
    if (prl > 2) printf ("\nmatgen_file: ") ;
    status = UMFPACK_report_triplet (nr, nc, nz, Ti, Tj, CARG(Tx,Tz), Control) ;
    if (status != UMFPACK_OK) error ("bad triplet report",0.) ;

    /* convert to column form */
    status = UMFPACK_triplet_to_col (nr, nc, nz, Ti, Tj, CARG(Tx,Tz), Bp, Bi, CARG(Bx,Bz), (Int *) NULL) ;
    if (status != UMFPACK_OK) error ("matgen_file triplet_to_col failed",0.) ;

    if (prl > 2) printf ("\nmatgen_file: ") ;
    status = UMFPACK_report_matrix (nr, nc, Bp, Bi, CARG(Bx,Bz), 1, Control) ;
    if (status != UMFPACK_OK) error ("bad A matgen_file",0.) ;

    /* return the allocated column-form */
    *n_row = nr ;
    *n_col = nc ;
    *Ap = Bp ;
    *Ai = Bi ;
    *Ax = Bx ;
    *Az = Bz ;
    *Qinit = Q ;

    free (Tz) ;	/* ] */
    free (Tx) ;	/* ] */
    free (Tj) ;	/* ] */
    free (Ti) ;	/* ] */

    fclose (f) ;
}

/* ========================================================================== */
/* matgen_arrow:  create an arrowhead matrix */
/* ========================================================================== */

static Int matgen_arrow
(
    Int n,
    Int **Ap,
    Int **Ai,
    Int **Q
)
{
    Int nz, *Bp, *Bi, i, j, p, *Qp ;

    nz = n + 2*(n-1) ;
    printf ("matgen_arrow: n = "ID" nz = "ID"\n", n, nz) ;

    Bp = (Int *) malloc ((n+1) * sizeof (Int)) ;
    Bi = (Int *) malloc (nz * sizeof (Int)) ;
    Qp = (Int *) malloc (n * sizeof (Int)) ;

    if (!Bp || !Bi || !Qp)
    {
	free (Bp) ;
	free (Bi) ;
	free (Qp) ;
	printf ("arrow failed\n") ;
    	return (FALSE) ;
    }

    /* row and column 0, and diagonal, are dense */

    /* column 0 */
    p = 0 ;
    Bp [0] = p ;
    for (i = 0 ; i < n ; i++)
    {
	Bi [p] = i ;
	Qp [p] = i ;
	p++ ;
    }

    /* columns 1 to n-1 */
    for (j = 1 ; j < n ; j++)
    {
	Bp [j] = p ;
	Bi [p] = 0 ;		/* row 0 */
	p++ ;
	Bi [p] = j ;		/* row j (diagonal) */
    }

    Bp [n] = p ;

    *Ap = Bp ;
    *Ai = Bi ;
    *Q = Qp ;

    printf ("matgen_arrow: n = "ID" nz = "ID" done.\n", n, nz) ;
    return (TRUE) ;

}


/* ========================================================================== */
/* do_and_free: do a matrix, its random transpose, and then free it */
/* ========================================================================== */

static double do_and_free
(
    Int n,
    Int Ap [ ],
    Int Ai [ ],
    double Ax [ ],	double Az [ ],

    double Controls [UMFPACK_CONTROL][1000],
    Int Ncontrols [UMFPACK_CONTROL],
    Int MemControl [6],
    Int do_dense
)
{
    Int *Bp, *Bi ;
    double *Bx, *Bz, rnorm1, rnorm2 ;

    /* A */
    rnorm1 = do_matrix (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemControl, do_dense) ;

    /* B = A (P,Q), P and Q random */
    matgen_transpose (n, Ap, Ai, Ax, Az, &Bp, &Bi, &Bx, &Bz) ;

    free (Ap) ;
    free (Ai) ;
    free (Ax) ;
    free (Az) ;

    rnorm2 = do_matrix (n, Bp, Bi, Bx, Bz, Controls, Ncontrols, MemControl, do_dense) ;

    free (Bp) ;
    free (Bi) ;
    free (Bx) ;
    free (Bz) ;

    return (MAX (rnorm1, rnorm2)) ;
}

/* ========================================================================== */
/* AMD */
/* ========================================================================== */

static int do_amd
(
    Int n,
    Int Ap [],
    Int Ai [],
    Int P []
)
{

#if 0
#ifndef NDEBUG

    FILE *f ;
    f = fopen ("apx.m", "w") ;
    Int j, p, nz ;
    if (Ap && Ai && P)
    {
	nz = Ap [n] ;
	fprintf (f, "ApX = [ ") ;
	for (j = 0 ; j <= n ; j++) fprintf (f, ID" ", Ap [j]) ;
	fprintf (f, "] ; \n nzx = "ID" ;\n Ax = [\n", nz) ;
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		fprintf (f, ID" "ID" 1\n", 1+j, 1+Ai [p]) ;
	    }
	}
	fprintf (f, ID" "ID" 0] ;\n", n, n) ;
	fclose (f) ;
    }

#endif
#endif

#if (defined (DINT) || defined (ZINT))
    return (amd_order   (n, Ap, Ai, P, DNULL, DNULL)) ;
#else
    return (amd_l_order (n, Ap, Ai, P, DNULL, DNULL)) ;
#endif
}


static int do_amd_transpose
(
    Int n,
    Int Ap [],
    Int Ai [],
    Int Rp [],
    Int Ri []
)
{
    Int *W, *Flag ;

#if (defined (DINT) || defined (ZINT))
    if (amd_valid (n, n, Ap, Ai) < AMD_OK || !Ri || !Rp)
    {
	return (AMD_INVALID) ;
    }
#else
    if (amd_l_valid (n, n, Ap, Ai) < AMD_OK || !Ri || !Rp)
    {
	return (AMD_INVALID) ;
    }
#endif

    W = SuiteSparse_malloc (n, sizeof (Int)) ;
    Flag = SuiteSparse_malloc (n, sizeof (Int)) ;
    if (!W || !Flag)
    {
	SuiteSparse_free (W) ;
	SuiteSparse_free (Flag) ;
	return (AMD_OUT_OF_MEMORY) ;
    }

#if (defined (DINT) || defined (ZINT))
    amd_preprocess (n, Ap, Ai, Rp, Ri, W, Flag) ;
#else
    amd_l_preprocess (n, Ap, Ai, Rp, Ri, W, Flag) ;
#endif

    SuiteSparse_free (W) ;
    SuiteSparse_free (Flag) ;
    return (AMD_OK) ;

}



/* ========================================================================== */
/* do_file:  read a matrix from a matrix and call do_many */
/* ========================================================================== */

static double do_file
(
    char *filename,
    Int prl,
    Int MemControl [6]
)
{
    Int n_row, n_col, *Ap, *Ai, *Qinit, n, *P, s, *W, scale, row, col, p,
	strategy, fixQ ;
    double *Ax, *Az, Control [UMFPACK_CONTROL], *b, rnorm, *bz, maxrnorm, bad,
	det_x, det_z ;

    UMFPACK_defaults (Control) ;
    Control [UMFPACK_PRL] = prl ;
    maxrnorm = 0 ;

    /* get the matrix A and preordering Qinit */
    matgen_file (filename, &n_row, &n_col, &Ap, &Ai, &Ax, &Az, &Qinit, prl,
	    &det_x, &det_z) ;	/* [[[[[ */

    check_tol = SCALAR_ABS (det_x < 1e100) ;

    /* test amd, on A and A transpose */
    if (n_row == n_col)
    {
	Int k, *Rp, *Ri ;
	P = (Int *) malloc (n_row * sizeof (Int)) ;	/* [ */
	W = (Int *) malloc (n_row * sizeof (Int)) ;	/* [ */
	Rp = (Int *) malloc ((n_row+1) * sizeof (Int)) ;	/* [ */
	Ri = (Int *) malloc ((Ap [n_row]) * sizeof (Int)) ;	/* [ */
	s = do_amd (n_row, Ap, Ai, P) ;
	if (s != AMD_OK) error ("amd2", (double) s) ;
	s = UMF_report_perm (n_row, P, W, 3, 0) ;
	if (s != UMFPACK_OK) error ("amd3", (double) s) ;
	s = do_amd_transpose (n_row, Ap, Ai, Rp, Ri) ;
	if (s != AMD_OK) error ("amd4", (double) s) ;
	s = do_amd (n_row, Rp, Ri, P) ;
	if (s != AMD_OK) error ("amd5", (double) s) ;
	s = UMF_report_perm (n_row, P, W, 3, 0) ;
	if (s != UMFPACK_OK) error ("amd6", (double) s) ;
	free (Ri) ;  /* ] */
	free (Rp) ;  /* ] */
	free (W) ;  /* ] */
	free (P) ;  /* ] */
    }

    /* do the matrix */
    n = MAX (n_row, n_col) ;
    n = MAX (n,1) ;
    b = (double *) calloc (n, sizeof (double)) ;	/* [ */
    bz= (double *) calloc (n, sizeof (double)) ;	/* [ */

    if (n_row == n_col)
    {
	bgen (n, Ap, Ai, Ax,Az, b,bz) ;
    }

    if (prl == 5 && MAX (n_row, n_col) > 600)
    {
	/* do nothing */
	;
    }
    else if (prl >= 3 || MAX (n_row, n_col) > 15)
    {

	/* quick test */
	printf ("Control strategy auto Q prl "ID"\n", prl) ;
	printf ("quick test..\n") ;
	rnorm = do_many (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemControl, TRUE, TRUE, det_x, det_z) ; 
	printf ("quick test.. done\n") ;
	printf ("Control strategy auto Q prl "ID" :: rnorm %g\n", prl, rnorm) ;
	if (check_tol)
	{
	    if (rnorm >= TOL) error ("bad do_file", rnorm) ;
	    maxrnorm = rnorm ;
	}

	/* quick test - no aggressive absorption */
	printf ("Control strategy auto Q prl "ID" no aggressive\n", prl) ;
	Control [UMFPACK_AGGRESSIVE] = 0 ;
	rnorm = do_many (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemControl, TRUE, TRUE, det_x, det_z) ; 
	printf ("Control strategy auto Q prl "ID" no aggressive:: rnorm %g\n", prl, rnorm) ;
	if (check_tol)
	{
	    if (rnorm >= TOL) error ("bad do_file", rnorm) ;
	    maxrnorm = rnorm ;
	}

	/* quick test - symmetric strategy, no aggressive absorption */
	printf ("Control strategy auto Q prl "ID" no aggressive, symmetric\n", prl) ;
	Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC ;
	UMFPACK_report_control (Control) ;
	rnorm = do_many (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemControl, TRUE, TRUE, det_x, det_z) ; 
	printf ("Control strategy auto Q prl "ID" no aggressive, symmetric:: rnorm %g\n", prl, rnorm) ;
	if (check_tol)
	{
	    if (rnorm >= TOL) error ("bad do_file", rnorm) ;
	    maxrnorm = rnorm ;
	}

    }
    else
    {
	/* full test */
	for (strategy=-1 ; strategy <= UMFPACK_STRATEGY_SYMMETRIC ; strategy++)
	{
            if (strategy == UMFPACK_STRATEGY_OBSOLETE) continue ;
	    Control [UMFPACK_STRATEGY] = strategy ;
	    if (strategy == UMFPACK_STRATEGY_SYMMETRIC)
	    {
		if (n < 5) UMFPACK_report_control (Control) ;
		
		for (fixQ = -1 ; fixQ <= 1 ; fixQ++)
		{
		    Control [UMFPACK_FIXQ] = fixQ ;
		    
	    
		    for (scale = -1 ; scale <= UMFPACK_SCALE_MAX ; scale++)
		    {
			Control [UMFPACK_SCALE] = scale ;
			printf ("Control strategy "ID" "ID" noQ prl "ID" scale "ID"\n",
			    strategy, fixQ, prl, scale) ;
			rnorm = do_many (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, INULL, MemControl, TRUE, TRUE, det_x, det_z) ;
			printf ("Control strategy "ID" "ID" noQ prl "ID" scale "ID" :: rnorm %g\n",
			    strategy, fixQ, prl, scale, rnorm) ;
			if (check_tol)
			{
			    maxrnorm = MAX (maxrnorm, rnorm) ;
			    if (rnorm >= TOL) error ("bad do_file", rnorm) ;
			}
		    }

		    printf ("Control strategy "ID" "ID" Q\n", strategy, fixQ) ;
		    rnorm = do_many (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemControl, TRUE, TRUE, det_x, det_z) ;
		    printf ("Control strategy "ID" "ID" Q :: rnorm %g\n",
			strategy, fixQ, rnorm) ;
		    if (check_tol)
		    {
			maxrnorm = MAX (maxrnorm, rnorm) ;
			if (rnorm >= TOL) error ("bad do_file", rnorm) ;
		    }
		}
		Control [UMFPACK_FIXQ] = UMFPACK_DEFAULT_FIXQ ;
	    }
	    else
	    {
		printf ("Control strategy "ID" Q prl "ID"\n", strategy, prl) ;
		rnorm = do_many (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemControl, TRUE, TRUE, det_x, det_z) ;
		printf ("Control strategy "ID" Q prl "ID" :: rnorm %g\n", strategy, prl, rnorm) ;
		if (check_tol)
		{
		    maxrnorm = MAX (maxrnorm, rnorm) ;
		    if (rnorm >= TOL) error ("bad do_file", rnorm) ;
		}

		printf ("Control strategy "ID" noQ\n", strategy) ;
		rnorm = do_many (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, INULL, MemControl, TRUE, TRUE, det_x, det_z) ;
		printf ("Control strategy "ID" noQ :: rnorm %g\n", strategy, rnorm) ;
		if (check_tol)
		{
		    maxrnorm = MAX (maxrnorm, rnorm) ;
		    if (rnorm >= TOL) error ("bad do_file", rnorm) ;
		}

#ifdef UMFPACK_DROPTOL
		Control [UMFPACK_DROPTOL] = 1e-25 ;
		printf ("Control strategy "ID" noQ droptol 1e-25\n", strategy) ;
		rnorm = do_many (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, INULL, MemControl, TRUE, TRUE, det_x, det_z) ;
		printf ("Control strategy "ID" noQ droptol 1e-25 :: rnorm %g\n", strategy, rnorm) ;
		if (check_tol)
		{
		    maxrnorm = MAX (maxrnorm, rnorm) ;
		    if (rnorm >= TOL) error ("bad do_file", rnorm) ;
		}
		Control [UMFPACK_DROPTOL] = 0 ;
#endif

	    }
	}
    }

    UMFPACK_defaults (Control) ;
    Control [UMFPACK_PRL] = 3 ;

    /* scale row 0 */
    for (col = 0 ; col < n_col ; col++)
    {
	for (p = Ap [col] ; p < Ap [col+1] ; p++)
	{
	    row = Ai [p] ;
	    if (row == 0)
	    {
		Ax [p] *= 1e-20 ;
		Az [p] *= 1e-20 ;
	    }
	}
    }
    b  [0] *= 1e-20 ;
    bz [0] *= 1e-20 ;

    printf ("Control defaults noQ tiny row 0\n") ;
    rnorm = do_many (n_row, n_col, Ap, Ai, Ax,Az, b,bz, Control, INULL, MemControl, TRUE, TRUE, 1e-20*det_x, 1e-20*det_z) ;
    printf ("Control defaults noQ  tiny row 0:: rnorm %g\n", rnorm) ;
    if (check_tol)
    {
	maxrnorm = MAX (maxrnorm, rnorm) ;
	if (rnorm >= TOL) error ("bad do_file", rnorm) ;
    }

    free (bz) ;		/* ] */
    free (b) ;		/* ] */
    free (Ap) ;		/* ] */
    free (Ai) ;		/* ] */
    free (Ax) ;		/* ] */
    free (Az) ;		/* ] */
    free (Qinit) ;	/* ] */

    check_tol = TRUE ;

    return (maxrnorm) ;
}

/* ========================================================================== */
/* main */
/* ========================================================================== */


#if 0
/* compile with -lefence and -DEFENCE */
#ifndef EXTERN
#define EXTERN extern
#endif
EXTERN int EF_PROTECT_FREE ;
EXTERN int EF_PROTECT_BELOW  ;
#endif


int main (int argc, char **argv)
{
    double *Lx, *Ux, *x,  *Cx, *Bx, *Ax, *b, *Ax2,
	   *Lz, *Uz, *xz, *Cz, *Bz, *Az, *bz,*Az2,
	rnorm, maxrnorm, *Con, Info [UMFPACK_INFO], *Wx, *Rs, xnan, xinf, ttt,
	Controls [UMFPACK_CONTROL][1000], Control [UMFPACK_CONTROL],
	alphas [ ] = {-1.0, 0.0, 0.1, 0.5, 10.}, *info, maxrnorm_shl0,
	rnorm_omega2, maxrnorm_arc130, det_x, det_z, Mx, Mz, Exp ;
    Int Ncontrols [UMFPACK_CONTROL], c, i, n, prl, *Qinit, *Qinit2, n1,
	*Ap, *Ai, *Aj, nz, *Ap2, *Ai2, p, j, d, s, s2, *Pinit, k, n2, *Map,
	*Lp, *Li, *P, *Q, *Up, *Ui, lnz, unz, nn, *Cp, *Ci, *Cj, *Bi, *Bp, *Bj,
	*Pa, *Front_npivots, *Front_parent, *Chain_start, *Chain_maxrows, *ip,
	*Chain_maxcols, nfr, nchains, nsparse_col, *Qtree, *Ptree, nnz,
	MemOK [6], MemBad [6], nnrow, nncol, nzud, n_row, n_col, n_row2,
	n_col2, scale, *Front_1strow, *Front_leftmostdesc, strategy,
	t, aggressive, *Pamd, mem1, mem2, do_recip ;
    int ok ;
    void *Symbolic, *Numeric ;
    SymbolicType *Sym ;
    NumericType *Num ;
    DIR *dir ;
    struct dirent *direntp ;
    char filename [200] ;
    FILE *f ;
    Int my_params [3], csave ;
    double my_info [3] ;

    /* turn off debugging */
    { f = fopen ("debug.umf", "w") ; fprintf (f, "-45\n") ; fclose (f) ; }
    { f = fopen ("debug.amd", "w") ; fprintf (f, "-45\n") ; fclose (f) ; }

#if 0
    /* compile with -lefence */
    EF_PROTECT_FREE = 0 ;   /* 1 to test modifications to free'd memory */
    EF_PROTECT_BELOW = 0 ;  /* 1 to test modifications above an obj. */
#endif

    c = UMFPACK_PIVOT_TOLERANCE ;
    Controls [c][0] = UMFPACK_DEFAULT_PIVOT_TOLERANCE ;
    Ncontrols [c] = 1 ;

    c = UMFPACK_SCALE ;
    Controls [c][0] = UMFPACK_SCALE_SUM ;   /* also the default */
    Ncontrols [c] = 1 ;

    c = UMFPACK_BLOCK_SIZE ;
    Controls [c][0] = 32 ;
    Ncontrols [c] = 1 ;

    c = UMFPACK_ALLOC_INIT ;
    Controls [c][0] = 1.0 ;
    Ncontrols [c] = 1 ;

    c = UMFPACK_AMD_DENSE ;
    Controls [c][0] = UMFPACK_DEFAULT_AMD_DENSE ;
    Ncontrols [c] = 1 ;

    /* ---------------------------------------------------------------------- */
    /* test malloc, realloc, and free */
    /* ---------------------------------------------------------------------- */

    /*
    P = (Int *) UMF_malloc (Int_MAX, 2) ;
    if (P) error ("large malloc should have failed\n", 0.) ;

    printf ("reallocing...\n") ;
    P = (Int *) UMF_realloc (P, 1, 4) ;
    if (!P) error ("should have succeeded\n", 0.) ;
#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
    if (UMF_malloc_count != 1) error ("should be 1", 0.) ;
#endif
    printf ("ok here...\n") ;

    P = UMF_free (P) ;
    if (P) error ("should have free'd it\n", 0.) ;
#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
    if (UMF_malloc_count != 0) error ("should be 0", 0.) ;
#endif
    */

    xnan = divide (0., 0.) ;
    xinf = divide (1., 0.) ;

    /* ---------------------------------------------------------------------- */
    /* malloc and realloc control */
    /* ---------------------------------------------------------------------- */

    MemOK [0] = -1 ;
    MemOK [1] = 0 ;
    MemOK [2] = 0 ;

    MemOK [3] = -1 ;
    MemOK [4] = 0 ;
    MemOK [5] = 0 ;

    /* malloc always succeeds */
    umf_fail = -1 ;
    umf_fail_lo = 0 ;
    umf_fail_hi = 0 ;

    /* realloc always succeeds */
    umf_realloc_fail = -1 ;
    umf_realloc_lo = 0 ;
    umf_realloc_hi = 0 ;

    UMFPACK_defaults (Control) ;

    maxrnorm_shl0 = 0.0 ;	/* for shl0 only */
    maxrnorm = 0.0 ;	/* for all other matrices */
    check_tol = TRUE ;

    /* ---------------------------------------------------------------------- */

    printf ("load/save error handling tests:\n") ;

    /* load a bad symbolic object */
    s = UMFPACK_load_symbolic (&Symbolic, "badsym.umf") ;
    if (s == UMFPACK_OK)
    {
	error ("load symbolic failed\n", 0.) ;
    }
    s = UMFPACK_load_symbolic (&Symbolic, "badsym2.umf") ;
    if (s == UMFPACK_OK)
    {
	error ("load symbolic failed (2)\n", 0.) ;
    }

    /* load a bad numeric object */
    s = UMFPACK_load_numeric (&Numeric, "badnum.umf") ;
    if (s == UMFPACK_OK)
    {
	error ("load numeric failed\n", 0.) ;
    }
    s = UMFPACK_load_numeric (&Numeric, "badnum2.umf") ;
    if (s == UMFPACK_OK)
    {
	error ("load numeric failed (2)\n", 0.) ;
    }

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* test a tiny matrix */
    /* ---------------------------------------------------------------------- */

    n = 2 ;
    printf ("\n tiny\n") ;
    check_tol = TRUE ;
    matgen_dense (n, &Ap, &Ai, &Ax, &Az) ;
    rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 0) ;
    printf ("rnorm tiny %g\n", rnorm) ;
    if (rnorm > 1e-12)
    {
	error ("bad rnorm for tiny matrix", rnorm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* test a tiny matrix with a NaN in it */
    /* ---------------------------------------------------------------------- */

    c = UMFPACK_SCALE ;
    Controls [c][0] = UMFPACK_SCALE_SUM ;   /* also the default */
    Controls [c][1] = UMFPACK_SCALE_NONE ;
    Ncontrols [c] = 2 ;

    n = 2 ;
    printf ("\n tiny\n") ;
    check_tol = TRUE ;
    matgen_dense (n, &Ap, &Ai, &Ax, &Az) ;
    Ax [0] = xnan ;
    Az [0] = 0 ;
    rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 0) ;
    printf ("rnorm tiny %g with NaN\n", rnorm) ;

    n = 2 ;
    printf ("\n tiny\n") ;
    check_tol = TRUE ;
    matgen_dense (n, &Ap, &Ai, &Ax, &Az) ;
    Ax [1] = 1e-20 ;
    Az [1] = 0 ;
    Ax [2] = 2e-20 ;
    Az [2] = 0 ;
    Ax [3] = 3e-20 ;
    Az [3] = 0 ;
    rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 0) ;
    printf ("rnorm tiny %g with NaN and small row\n", rnorm) ;

    n = 2 ;
    printf ("\n tiny\n") ;
    check_tol = TRUE ;
    matgen_dense (n, &Ap, &Ai, &Ax, &Az) ;
    Ax [0] = 0 ;
    rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 0) ;
    printf ("rnorm tiny %g with small row\n", rnorm) ;

    c = UMFPACK_SCALE ;
    Controls [c][0] = UMFPACK_SCALE_SUM ;   /* also the default */
    Ncontrols [c] = 1 ;

    /* ---------------------------------------------------------------------- */
    /* test omega2 */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    n = 500 ;
    printf ("\n omega 2 test\n") ;
    matgen_sparse (n, 2*n, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az, 0, 0) ;

    f = fopen ("A500", "w") ;
    fprintf (f, ID" "ID" 0 0\n", n, n) ;
    for (j = 0 ; j < n ; j++)
    {
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    fprintf (f, ID" "ID" %40.25e %40.25e\n", Ai [p], j, Ax [p], Az [p]) ;
	}
    }
    fclose (f) ;

    rnorm_omega2 = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 0) ;
    printf ("rnorm %g omega-2 test\n", rnorm_omega2) ;

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* this is not solved very accurately (about 1e-7) */
    maxrnorm_shl0 = do_file ("TestMat/shl0", 4, MemOK) ;
    printf ("rnorm shl0 %10.4e\n", maxrnorm_shl0) ;

    /* this is not solved very accurately (about 1e-5, because of U'x=b) */
    maxrnorm_arc130 = do_file ("TestMat/arc130", 4, MemOK) ;
    printf ("rnorm arc130 %10.4e\n", maxrnorm_arc130) ;

    /* ---------------------------------------------------------------------- */
    /* test random sparse matrices */ 
    /* ---------------------------------------------------------------------- */

    n = 30 ;

	printf ("sparse "ID" 4*n nz's", n) ;
	matgen_sparse (n, 4*n, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az, 1, 0) ;	/* [[[[ */
	rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 1) ; /* ]]]] */
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf ("rnorm  %10.4e %10.4e\n", rnorm, maxrnorm) ;

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    rnorm = do_file ("TestMat/matrix5", 5, MemOK) ;
    maxrnorm = MAX (rnorm, maxrnorm) ;
    printf ("rnorm matrix 5 %10.4e %10.4e\n", rnorm, maxrnorm) ;

    /* malloc always succeeds */
    umf_fail = -1 ;
    umf_fail_lo = 0 ;
    umf_fail_hi = 0 ;

    /* realloc always fails */
    umf_realloc_fail = 0 ;
    umf_realloc_lo = -9999999 ;
    umf_realloc_hi = 0 ;

    /* ---------------------------------------------------------------------- */
    /* do all test matrices from TestMat directory */
    /* ---------------------------------------------------------------------- */

    /* quick tests */
    prl = 1 ;

    matgen_file ("TestMat/matrix1", &n_row, &n_col, &Ap, &Ai, &Ax, &Az, &Qinit,
	    prl, &det_x, &det_z) ;	/* [[[[[ */

	Control [UMFPACK_ALLOC_INIT] = -211 ;

	/* with no Qinit, out of memory in extend front */
	s = UMFPACK_symbolic (n_row, n_col, Ap, Ai, CARG(Ax,Az), &Symbolic, Control, Info) ;	/* [ */
	if (s != UMFPACK_OK) error ("TestMat matrix1 sym", (double) s) ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ; /* [ */
	if (s != UMFPACK_ERROR_out_of_memory) error ("TestMat matrix1 num", (double) s) ;
	UMFPACK_free_numeric (&Numeric) ;	/* ] */
	UMFPACK_free_symbolic (&Symbolic) ;	/* ] */

	free (Ax) ;	    /* ] */
	free (Ap) ;	    /* ] */
	free (Ai) ;	    /* ] */
	free (Az) ;	    /* ] */
	free (Qinit) ;  /* ] */

    matgen_file ("TestMat/matrix10", &n_row, &n_col, &Ap, &Ai, &Ax, &Az, &Qinit, prl, &det_x, &det_z) ;	/* [[[[[ */

	Control [UMFPACK_ALLOC_INIT] = -1321 ;

	/* with Qinit, out of memory in create front (2) */
	s = UMFPACK_qsymbolic (n_row, n_col, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Control, Info) ;	/* [ */
	if (s != UMFPACK_OK) error ("TestMat matrix10 qsym1", (double) s) ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ; /* [ */
	if (s != UMFPACK_ERROR_out_of_memory) error ("TestMat matrix10 qnum1", (double) s) ;
	UMFPACK_free_numeric (&Numeric) ;	/* ] */
	UMFPACK_free_symbolic (&Symbolic) ;	/* ] */

	Control [UMFPACK_ALLOC_INIT] = -1326 ;

	/* with Qinit, out of memory in init front */
	s = UMFPACK_qsymbolic (n_row, n_col, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Control, Info) ;	/* [ */
	if (s != UMFPACK_OK) error ("TestMat matrix10 qsym", (double) s) ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ; /* [ */
	if (s != UMFPACK_ERROR_out_of_memory) error ("TestMat matrix10 qnum", (double) s) ;
	UMFPACK_free_numeric (&Numeric) ;	/* ] */
	UMFPACK_free_symbolic (&Symbolic) ;	/* ] */

	free (Ax) ;	    /* ] */
	free (Ap) ;	    /* ] */
	free (Ai) ;	    /* ] */
	free (Az) ;	    /* ] */
	free (Qinit) ;  /* ] */

    printf ("\ndone with TestMat memory sizes.\n\n") ;
    UMFPACK_defaults (Control) ;

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* test amd */
    /* ---------------------------------------------------------------------- */

    n = 50 ;
    P = (Int *) malloc (n * sizeof (Int)) ; /* [ */
    for (k = 0 ; k < 10 ; k++)
    {
	matgen_sparse (n, 4*n, 3, 2*n, 0, 0, &Ap, &Ai, &Ax, &Az, 0, 0) ; /* [[[[ */
	for (aggressive = 0 ; aggressive <= 2 ; aggressive++)
	{
	    for (i = 0 ; i < 3 ; i++)
	    {

#if (defined (DINT) || defined (ZINT))

		amd_defaults (Control) ;
		Control [AMD_AGGRESSIVE] = aggressive ;
		Control [AMD_DENSE] = alphas [i] ;
		Con = (aggressive == 2) ? DNULL : Control ;
		info = (aggressive == 2) ? DNULL : Info ;
		amd_control (Con) ;
		s = amd_order (n, Ap, Ai, P, Con, info) ;
		if (s != AMD_OK) error ("amd", (double) s) ;
		amd_info (info) ;

#else

		amd_l_defaults (Control) ;
		Control [AMD_AGGRESSIVE] = aggressive ;
		Control [AMD_DENSE] = alphas [i] ;
		Con = (aggressive == 2) ? DNULL : Control ;
		info = (aggressive == 2) ? DNULL : Info ;
		amd_l_control (Con) ;
		s = amd_l_order (n, Ap, Ai, P, Con, info) ;
		if (s != AMD_OK) error ("amd", (double) s) ;
		amd_l_info (info) ;

#endif

		UMFPACK_defaults (Control) ;
		Control [UMFPACK_PRL] = 3 ;
		UMFPACK_report_perm (n, P, Control) ;
	    }
	}
	free (Ap) ; /* ] */
	free (Ai) ; /* ] */
	free (Ax) ; /* ] */
	free (Az) ; /* ] */
    }
    free (P) ;	/* ] */

    if (AMD_valid (-1, -1, INULL, INULL) >= AMD_OK) error ("amd error", 0.) ;
    Info [AMD_STATUS] = AMD_OUT_OF_MEMORY ;
    AMD_info (Info) ;
    Info [AMD_STATUS] = AMD_INVALID ;
    AMD_info (Info) ;
    Info [AMD_STATUS] = -911 ;
    AMD_info (Info) ;

    /* ---------------------------------------------------------------------- */
    /* malloc and realloc control */
    /* ---------------------------------------------------------------------- */

    MemOK [0] = -1 ;
    MemOK [1] = 0 ;
    MemOK [2] = 0 ;

    MemOK [3] = -1 ;
    MemOK [4] = 0 ;
    MemOK [5] = 0 ;

    umf_fail = -1 ;
    umf_fail_lo = 0 ;
    umf_fail_hi = 0 ;

    umf_realloc_fail = -1 ;
    umf_realloc_lo = 0 ;
    umf_realloc_hi = 0 ;

    for (i = 0 ; i < UMFPACK_CONTROL ; i++)
    {
	Ncontrols [i] = 0 ;
    }

    UMFPACK_defaults (Control) ;

    /* ---------------------------------------------------------------------- */
    /* do three funky sizes to test int overflow cases */
    /* ---------------------------------------------------------------------- */

    {
	/* Int funky_sizes [ ] = { 14402, 16400, 600000 } ; */
	Int funky_sizes [ ] = { 144, 164, 600 } ; 

	UMFPACK_defaults (Control) ;
	Control [UMFPACK_PRL] = 1 ;
	Control [UMFPACK_STRATEGY] = 3 ;
	Control [UMFPACK_FRONT_ALLOC_INIT] = 1 ;

	for (k = 0 ; k < 3 ; k++)
	{
	    n = funky_sizes [k] ;

	    printf ("funky matrix, n = "ID"\n", n) ;
	    matgen_funky (n, &Ap, &Ai, &Ax, &Az) ;	/* [[[[ */

	    b = (double *) malloc (n * sizeof (double)) ;	/* [ */
	    bz= (double *) calloc (n , sizeof (double)) ;	/* [ */
	    Qinit = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
	    if (!b) error ("out of memory (15)",0.) ;
	    if (!bz) error ("out of memory (16)",0.) ;
	    if (!Qinit) error ("out of memory (17)",0.) ;
	    bgen (n, Ap, Ai, Ax, Az, b, bz) ;

	    for (i = 0 ; i < n ; i++) Qinit [i] = i ;
	    fflush (stdout) ;

	    Control [UMFPACK_FIXQ] = 1 ;
	    rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemOK, FALSE, FALSE, 0., 0.) ;
	    printf ("funky matrix rnorm (fixQ): %g\n", rnorm) ;
	    fflush (stdout) ;
	    maxrnorm = MAX (rnorm, maxrnorm) ;

	    Control [UMFPACK_FIXQ] = 0 ;
	    rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemOK, FALSE, FALSE, 0., 0.) ;
	    printf ("funky matrix rnorm (no fixQ): %g\n", rnorm) ;
	    fflush (stdout) ;
	    maxrnorm = MAX (rnorm, maxrnorm) ;

	    free (Qinit) ;  /* ] */
	    free (bz) ;	    /* ] */
	    free (b) ;	    /* ] */
	    free (Ap) ;	    /* ] */
	    free (Ai) ;	    /* ] */
	    free (Ax) ;	    /* ] */
	    free (Az) ;	    /* ] */
	}
    }
    /* maxrnorm = 0 ; */

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* tight controls */
    /* ---------------------------------------------------------------------- */

    c = UMFPACK_PIVOT_TOLERANCE ;
    Controls [c][0] = UMFPACK_DEFAULT_PIVOT_TOLERANCE ;
    Ncontrols [c] = 1 ;

    c = UMFPACK_SCALE ;
    Controls [c][1] = UMFPACK_DEFAULT_SCALE ;
    Ncontrols [c] = 1 ;

    c = UMFPACK_BLOCK_SIZE ;
    Controls [c][0] = UMFPACK_DEFAULT_BLOCK_SIZE ;
    Ncontrols [c] = 1 ;

    c = UMFPACK_ALLOC_INIT ;
    Controls [c][0] = UMFPACK_DEFAULT_ALLOC_INIT ;
    Ncontrols [c] = 1 ;

    c = UMFPACK_AMD_DENSE ;
    Controls [c][0] = UMFPACK_DEFAULT_AMD_DENSE ;
    Ncontrols [c] = 1 ;

    /* ---------------------------------------------------------------------- */
    /* license */
    /* ---------------------------------------------------------------------- */

    Control [UMFPACK_PRL] = 6 ;
    UMFPACK_report_status (Control, UMFPACK_OK) ;

    /* ---------------------------------------------------------------------- */
    /* do all test matrices from TestMat directory */
    /* ---------------------------------------------------------------------- */

    printf ("\nStarting TestMat:\n") ;
    for (prl = 5 ; prl >= 0 ; prl--)
    {
	printf ("=====================TestMat PRL "ID"\n", prl) ;
        dir = opendir ("TestMat") ;
        if (!dir) { printf ("opendir TestMat failed\n") ; exit (1) ; }
        while (TRUE)
        {
	    errno = 0 ;
	    if ((direntp = readdir (dir)) != NULL)
	    {
	        /* skip this */
	        if (direntp->d_name [0] == '.') continue ;
	        sprintf (filename, "TestMat/%s", direntp->d_name) ;
	        rnorm = do_file (filename, prl, MemOK) ;

		if (strcmp (filename, "TestMat/shl0") == 0)
		{
		    printf ("shl0 rnorm: %g\n", rnorm) ;
		    maxrnorm_shl0 = MAX (maxrnorm_shl0, rnorm) ;
		}
		else if (strcmp (filename, "TestMat/arc130") == 0)
		{
		    printf ("arc130 rnorm: %g\n", rnorm) ;
		    maxrnorm_arc130 = MAX (maxrnorm_arc130, rnorm) ;
		}
		else
		{
		    printf ("other testmat rnorm: %g\n", rnorm) ;
		    maxrnorm = MAX (maxrnorm, rnorm) ;
		}
	    }
	    else
	    {
	        if (errno != 0) { printf ("read error\n") ; exit (1) ; }
	        closedir (dir) ;
	        break ;
	    }
        }
        printf ("\n\n@@@@@@ Largest TestMat do_file rnorm: %g shl0: %g @@@@@@ arc130: %g\n\n", maxrnorm, maxrnorm_shl0, maxrnorm_arc130) ;
    }

    printf ("\ndone with TestMat.\n\n") ;

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* test change of pattern */
    /* ---------------------------------------------------------------------- */

    Control [UMFPACK_PRL] = 5 ;
    Control [UMFPACK_ALLOC_INIT] = 0. ;

    matgen_file ("TestMat/matrix1", &n_row, &n_col, &Ap, &Ai, &Ax, &Az, &Qinit, 5, &det_x, &det_z) ;	/* [[[[[ */
    s = UMFPACK_qsymbolic (n_row, n_col, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (!Symbolic || Info [UMFPACK_STATUS] != UMFPACK_OK) error ("p1",0.) ;
    printf ("\nGood symbolic, pattern test: ") ;
    s = UMFPACK_report_symbolic (Symbolic, Control) ;
    UMFPACK_report_status (Control, s) ;
    if (s != UMFPACK_OK) error ("p1c",0.) ;
    UMFPACK_report_control (Control) ;

    s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    printf ("p1b status: "ID" Numeric handle bad %d\n", s, !Numeric) ;
    s2 = UMFPACK_report_numeric (Numeric, Control) ;
    if (!Numeric || s != UMFPACK_OK) error ("p1b",0.) ;
    printf ("Good numeric, pattern test: ") ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    UMFPACK_report_status (Control, s) ;
    if (s2 != UMFPACK_OK) error ("p1d",0.) ;
    UMFPACK_free_numeric (&Numeric) ;

    /* corrupted Ap (negative degree) */
    c = Ap [1] ;
    Ap [1] = -1 ;
    printf ("Bad Ap [1] = -1: \n") ;
    s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (Numeric || s != UMFPACK_ERROR_different_pattern) error ("zzz1",0.) ;
    Ap [1] = c ;

    /* corrupted Ai (out of bounds) */
    c = Ai [1] ;
    Ai [1] = -1 ;
    printf ("Bad Ai [1] = -1: \n") ;
    s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (Numeric || s != UMFPACK_ERROR_different_pattern) error ("zzz2",0.) ;
    Ai [1] = c ;

    /* corrupted Ai (out of bounds) */
    c = Ai [1] ;
    Ai [1] = n_row ;
    printf ("Bad Ai [1] = "ID": \n", n_row) ;
    s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (Numeric || s != UMFPACK_ERROR_different_pattern) error ("zzz3",0.) ;
    Ai [1] = c ;

    free (Ap) ;		/* ] */
    free (Ai) ;		/* ] */
    free (Ax) ;		/* ] */
    free (Az) ;		/* ] */
    free (Qinit) ;	/* ] */

    /* one more entry */
    printf ("one more entry\n") ;
    matgen_file ("TestMat/matrix2", &n_row2, &n_col2, &Ap2, &Ai2, &Ax2, &Az2, &Qinit2, 5, &det_x, &det_z) ;	/* [[[[[ */
    s = UMFPACK_numeric (Ap2, Ai2, CARG(Ax2,Az2), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (Numeric || s != UMFPACK_ERROR_different_pattern) error ("p2",0.) ;
    free (Ap2) ;		/* ] */
    free (Ai2) ;		/* ] */
    free (Ax2) ;		/* ] */
    free (Az2) ;		/* ] */
    free (Qinit2) ;		/* ] */
    printf ("one more entry done\n") ;

    /* one less entry */
    matgen_file ("TestMat/matrix3", &n_row2, &n_col2, &Ap2, &Ai2, &Ax2, &Az2, &Qinit2, 5, &det_x, &det_z) ;	/* [[[[[ */
    s = UMFPACK_numeric (Ap2, Ai2, CARG(Ax2,Az2), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (Numeric || s != UMFPACK_ERROR_different_pattern) error ("p3",0.) ;
    free (Ap2) ;		/* ] */
    free (Ai2) ;		/* ] */
    free (Ax2) ;		/* ] */
    free (Az2) ;		/* ] */
    free (Qinit2) ;		/* ] */

    /* many more entries */
    matgen_file ("TestMat/matrix4", &n_row2, &n_col2, &Ap2, &Ai2, &Ax2, &Az2, &Qinit2, 5, &det_x, &det_z) ;	/* [[[[[ */
    s = UMFPACK_numeric (Ap2, Ai2, CARG(Ax2,Ax2), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (Numeric || s != UMFPACK_ERROR_different_pattern) error ("p4",0.) ;
    free (Ap2) ;		/* ] */
    free (Ai2) ;		/* ] */
    free (Ax2) ;		/* ] */
    free (Az2) ;		/* ] */
    free (Qinit2) ;		/* ] */

    /* some more entries */
    matgen_file ("TestMat/matrix5", &n_row2, &n_col2, &Ap2, &Ai2, &Ax2, &Az2, &Qinit2, 5, &det_x, &det_z) ;	/* [[[[[ */
    s = UMFPACK_numeric (Ap2, Ai2, CARG(Ax2,Az2), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (Numeric || s != UMFPACK_ERROR_different_pattern) error ("p5",0.) ;
    free (Ap2) ;		/* ] */
    free (Ai2) ;		/* ] */
    free (Ax2) ;		/* ] */
    free (Az2) ;		/* ] */
    free (Qinit2) ;		/* ] */

    /* same entries - but different pattern */
    matgen_file ("TestMat/matrix6", &n_row2, &n_col2, &Ap2, &Ai2, &Ax2, &Az2, &Qinit2, 5, &det_x, &det_z) ;	/* [[[[[ */
    s = UMFPACK_numeric (Ap2, Ai2, CARG(Ax2,Az2), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (Numeric || s != UMFPACK_ERROR_different_pattern) error ("p6",0.) ;
    free (Ap2) ;		/* ] */
    free (Ai2) ;		/* ] */
    free (Ax2) ;		/* ] */
    free (Az2) ;		/* ] */
    free (Qinit2) ;		/* ] */

    /* same entries - but different pattern */
    matgen_file ("TestMat/matrix7", &n_row2, &n_col2, &Ap2, &Ai2, &Ax2, &Az2, &Qinit2, 5, &det_x, &det_z) ;	/* [[[[[ */
    s = UMFPACK_numeric (Ap2, Ai2, CARG(Ax2,Az2), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (Numeric || s != UMFPACK_ERROR_different_pattern) error ("p7",0.) ;
    free (Ap2) ;		/* ] */
    free (Ai2) ;		/* ] */
    free (Ax2) ;		/* ] */
    free (Az2) ;		/* ] */
    free (Qinit2) ;		/* ] */

    /* same entries - but different pattern */
    matgen_file ("TestMat/matrix8", &n_row2, &n_col2, &Ap2, &Ai2, &Ax2, &Az2, &Qinit2, 5, &det_x, &det_z) ;	/* [[[[[ */
    s = UMFPACK_numeric (Ap2, Ai2, CARG(Ax2,Az2), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (Numeric || s != UMFPACK_ERROR_different_pattern) error ("p8",0.) ;
    free (Ap2) ;		/* ] */
    free (Ai2) ;		/* ] */
    free (Ax2) ;		/* ] */
    free (Az2) ;		/* ] */
    free (Qinit2) ;		/* ] */

    UMFPACK_free_symbolic (&Symbolic) ;

    /* start over, use a bigger matrix */
    matgen_file ("TestMat/matrix10", &n_row, &n_col, &Ap, &Ai, &Ax, &Az, &Qinit, 5, &det_x, &det_z) ;	/* [[[[[ */
    s = UMFPACK_qsymbolic (n_row, n_col, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    if (prl > 2) printf ("\nGood matrix10 symbolic, pattern test: ") ;
    s = UMFPACK_report_symbolic (Symbolic, Control) ;
    if (!Symbolic || Info [UMFPACK_STATUS] != UMFPACK_OK) error ("p10",0.) ;
    if (s != UMFPACK_OK) error ("p10",0.) ;
    s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;
    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
    if (!Numeric || s != UMFPACK_OK) error ("p10b",0.) ;
    printf ("Good matrix10matrix10  numeric, pattern test:") ;
    s = UMFPACK_report_numeric (Numeric, Control) ;
    if (s != UMFPACK_OK) error ("p10b",0.) ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    UMFPACK_free_numeric (&Numeric) ;

    /* kludge Symbolic to force a huge dmax */
    printf ("\nKludge symbolic to force dmax int overflow:\n") ;
    Control [UMFPACK_PRL] = 3 ;
    Sym = (SymbolicType *) Symbolic ;
    Sym->amd_dmax = 16400 ;
    s = UMFPACK_report_symbolic (Symbolic, Control) ;
    if (!Symbolic || Info [UMFPACK_STATUS] != UMFPACK_OK) error ("p10e",0.) ;
    s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;
    if (!Numeric || s != UMFPACK_OK) error ("p10c",0.) ;
    UMFPACK_report_status (Control, s) ;
    UMFPACK_report_info (Control, Info) ;
    UMFPACK_free_numeric (&Numeric) ;

    free (Ap) ;		/* ] */
    free (Ai) ;		/* ] */
    free (Ax) ;		/* ] */
    free (Az) ;		/* ] */
    free (Qinit) ;	/* ] */

    UMFPACK_free_symbolic (&Symbolic) ;

    /* ---------------------------------------------------------------------- */
    /* reset controls */
    /* ---------------------------------------------------------------------- */

    UMFPACK_defaults (Control) ;

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* test realloc */
    /* ---------------------------------------------------------------------- */

    /* malloc always succeeds */
    MemBad [0] = -1 ;
    MemBad [1] = 0 ;
    MemBad [2] = 0 ;

    /* realloc always fails */
    MemBad [3] = 0 ;
    MemBad [4] = 0 ;
    MemBad [5] = -99999 ;

    c = UMFPACK_ALLOC_INIT ;
    Ncontrols [c] = 1 ;
    Controls [c][0] = 0.001 ;

#ifdef DINT
    printf ("\n all realloc fails sparse + dense rows %7d 4*n nz's\n", n) ;
    matgen_sparse (n, 4*n, 3, 2*n, 0, 0, &Ap, &Ai, &Ax, &Az, 0, 0) ;
    rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemBad, 0) ;
    printf ("rnorm %g should be 9e10\n", rnorm) ;
    if (rnorm != 9e10) error ("MemBad 1 failure", rnorm) ;

    printf ("\n all realloc fails sparse %7d 30*n nz's\n", n) ;
    matgen_sparse (n, 30*n, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az, 0, 0) ;
    rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemBad, 0) ;
    printf ("rnorm %g should be 9e10\n", rnorm) ;
    if (rnorm != 9e10) error ("MemBad 2 failure", rnorm) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* umf_symbolic compaction test */
    /* ---------------------------------------------------------------------- */

    n = 100 ;
    Control [UMFPACK_PRL] = 4 ;
    Qinit = (Int *) malloc (n * sizeof (Int)) ;			/* [ */
    b = (double *) malloc (n * sizeof (double)) ;		/* [ */
    bz= (double *) calloc (n , sizeof (double)) ;		/* [ */
    for (i = 0 ; i < n ; i++) Qinit [i] = i ;
    matgen_compaction (n, &Ap, &Ai, &Ax, &Az) ;					/* [[[[ */
    bgen (n, Ap, Ai, Ax, Az, b, bz) ;
    printf ("\nA compaction: ") ;
    s = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Control) ;
    if (s != UMFPACK_OK) error ("219", 0.) ;
    rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemOK, FALSE, FALSE, 0., 0.) ;
    printf ("rnorm %g A compaction\n", rnorm) ;
    Control [UMFPACK_PRL] = 1 ;

    printf ("do_and_free for compacted matrix:\n") ;
    rnorm = do_and_free (n, Ap, Ai, Ax,Az, Controls, Ncontrols, MemOK, 0) ;	/* ]]]] */
    printf ("rnorm for compacted matrix %g\n", rnorm) ;
    free (b) ; /* ] */
    free (bz) ; /* ] */
    free (Qinit) ; /* ] */

    /* ---------------------------------------------------------------------- */
    /* umf_symbolic compaction test, again (read a file)  */
    /* ---------------------------------------------------------------------- */

    matgen_file ("TestMat/shl0", &n_row, &n_col, &Ap, &Ai, &Ax, &Az, &Qinit, 5, &det_x, &det_z) ;	/* [[[[[ */
    n = n_row ;
    b = (double *) malloc (n * sizeof (double)) ;	/* [ */
    bz= (double *) calloc (n , sizeof (double)) ;	/* [ */
    bgen (n, Ap, Ai, Ax, Az, b, bz) ;
    Control [UMFPACK_PRL] = 5 ;
    Control [UMFPACK_DENSE_ROW] = 0.1 ;
    Control [UMFPACK_DENSE_COL] = 2.0 ;
    printf ("\nshl0 b: ") ;
    UMFPACK_report_vector (n, CARG(b,bz), Control) ;
    printf ("\nshl0 A: ") ;
    UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Control) ;
    rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemOK, FALSE, FALSE, 0., 0.) ;
    printf ("rnorm %g for shl0", rnorm) ;
    maxrnorm_shl0 = MAX (maxrnorm_shl0, rnorm) ;
    free (bz) ;		/* ] */
    free (b) ;		/* ] */
    free (Ap) ;		/* ] */
    free (Ai) ;		/* ] */
    free (Az) ;		/* ] */
    free (Ax) ;		/* ] */
    free (Qinit) ;	/* ] */

    /* ---------------------------------------------------------------------- */
    /* normal Controls */
    /* ---------------------------------------------------------------------- */

    c = UMFPACK_PIVOT_TOLERANCE ;
    Controls [c][0] = UMFPACK_DEFAULT_PIVOT_TOLERANCE ;
    Controls [c][1] = 0.5 ;
    Controls [c][2] = 1.0 ;
    Ncontrols [c] = 3 ;

    c = UMFPACK_SCALE ;
    Controls [c][0] = UMFPACK_SCALE_SUM ;   /* also the default */
    Controls [c][1] = UMFPACK_SCALE_NONE ;
    Controls [c][2] = UMFPACK_SCALE_MAX ;
    Ncontrols [c] = 3 ;

    c = UMFPACK_BLOCK_SIZE ;
    Controls [c][0] = 1 ;
    Controls [c][1] = 8 ;
    Controls [c][2] = 16 ;	/* not the default */
    Ncontrols [c] = 3 ;

    c = UMFPACK_ALLOC_INIT ;
    Controls [c][0] = 0.0 ;
    Controls [c][1] = 0.5 ;
    Controls [c][2] = 1.0 ;	/* not the default */
    Controls [c][3] = -10000 ;
    Ncontrols [c] = 4 ;

    c = UMFPACK_AMD_DENSE ;
    Controls [c][0] = -1 ;
    Controls [c][1] = 0.5 ;
    Controls [c][2] = UMFPACK_DEFAULT_AMD_DENSE ;
    Ncontrols [c] = 3 ;

    UMFPACK_defaults (Control) ;

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* test realloc */
    /* ---------------------------------------------------------------------- */

    /* malloc always succeeds */
    MemBad [0] = -1 ;
    MemBad [1] = 0 ;
    MemBad [2] = 0 ;

    /* realloc always fails */
    MemBad [3] = 0 ;
    MemBad [4] = 0 ;
    MemBad [5] = -99999 ;

    n = 10 ;

    printf ("\n all realloc fails sparse "ID" 4*n nz's\n", n) ;
    matgen_sparse (n, 4*n, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az, 1, 0) ;
    rnorm = do_and_free (n, Ap, Ai, Ax,Az, Controls, Ncontrols, MemBad, 1) ;
    printf ("rnorm %g for all-realloc-fails\n", rnorm) ;

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* reset malloc and realloc failure */
    /* ---------------------------------------------------------------------- */

    umf_fail = -1 ;
    umf_fail_lo = 0 ;
    umf_fail_hi = 0 ;

    umf_realloc_fail = -1 ;
    umf_realloc_lo = 0 ;
    umf_realloc_hi = 0 ;

    /* ---------------------------------------------------------------------- */
    /* test errors */
    /* ---------------------------------------------------------------------- */

    n = 32 ;

    Pamd = (Int *) malloc (2*n * sizeof (Int)) ;		/* [ */
    Qinit = (Int *) malloc (2*n * sizeof (Int)) ;		/* [ */
    Pinit = (Int *) malloc (2*n * sizeof (Int)) ;		/* [ */
    Qinit2 = (Int *) malloc (2*n * sizeof (Int)) ;		/* [ */
    b = (double *) malloc (2*n * sizeof (double)) ;		/* [ */
    bz= (double *) calloc (2*n , sizeof (double)) ;		/* [ */
    x = (double *) malloc (2*n * sizeof (double)) ;		/* [ */
    xz= (double *) calloc (2*n , sizeof (double)) ;		/* [ */
    Ap2 = (Int *) malloc ((2*n+1) * sizeof (Int)) ;		/* [ */

    if (!Qinit || !b || !Ap2 || !Qinit2 || !Pinit)  error ("out of memory (18)",0.) ;
    if (!xz || !bz) error ("memr again",0.) ;

    UMFPACK_defaults (DNULL) ;
    UMFPACK_defaults (Control) ;

    randperm (n, Pinit) ;

    for (prl = 5 ; prl >= -1 ; prl--)
    {
      for (strategy = UMFPACK_STRATEGY_AUTO ; strategy <= UMFPACK_STRATEGY_SYMMETRIC ; strategy++)
      {
        Int *Rp, *Ri ;
        if (strategy == UMFPACK_STRATEGY_OBSOLETE) continue ;

	printf ("\n[[[[ PRL = "ID" strategy = "ID"\n", prl, strategy) ;
	for (k = 0 ; k < n ; k++) Pamd [k] = EMPTY ;
	Control [UMFPACK_PRL] = prl ;
	Control [UMFPACK_STRATEGY] = strategy ;
	UMFPACK_report_control (Control) ;
	i = UMFPACK_DENSE_DEGREE_THRESHOLD (0.2, n) ;
	printf ("(default) dense row/col degree threshold: "ID"\n", i) ;

	/* ------------------------------------------------------------------ */

	matgen_sparse (n, 4*n, 10, 2*n, 10, 2*n, &Ap, &Ai, &Ax, &Az, prl, 0) ;  /* [[[[ */
	bgen (n, Ap, Ai, Ax,Az, b,bz) ;

	nz = Ap [n] ;
	Aj = (Int *) malloc ((nz+n+1) * sizeof (Int)) ;		/* [ */
	Ai2 = (Int *) malloc ((nz+n) * sizeof (Int)) ;		/* [ */
	Ax2 = (double *) malloc ((nz+n) * sizeof (double)) ;	/* [ */
	Az2 = (double *) calloc ((nz+n) , sizeof (double)) ;	/* [ */
	if (!Aj || !Ai2 || !Ax2 || !Az2)  error ("out of memory (19)",0.) ;
	Rp = Aj ;
	Ri = Ai2 ;

	/* ------------------------------------------------------------------ */

	Con = (prl == -1) ? (DNULL) : Control ;
	UMFPACK_report_control (Con) ;

	/* ------------------------------------------------------------------ */

	randperm (n, Qinit) ;
	if (prl > 2) printf ("Qinit OK: ") ;
	s = UMFPACK_report_perm (n, Qinit, Con) ;
	if (s != UMFPACK_OK) error ("Qinit OK", 0.) ;

	randperm (2*n, Qinit2) ;
	if (prl > 2) printf ("Qinit2 OK: ") ;
	s = UMFPACK_report_perm (2*n, Qinit2, Con) ;
	if (s != UMFPACK_OK) error ("Qinit2 OK", 0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nb OK: ") ;
	s = UMFPACK_report_vector (n, CARG(b,bz), Con) ;
	if (s != UMFPACK_OK) error ("0",0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nn=-1: ") ;
	s = UMFPACK_report_vector (-1, CARG(b,bz), Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_n_nonpositive)) error ("2",0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nb null: ") ;
	s = UMFPACK_report_vector (n, CARG(DNULL,DNULL), Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_argument_missing)) error ("2",0.) ;

	/* ------------------------------------------------------------------ */


	if (prl > 2) printf ("\nA OK: ") ;
	s = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != UMFPACK_OK) error ("2a",0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nA pattern OK: ") ;
	s = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(DNULL,DNULL), 1, Con) ;
	if (s != UMFPACK_OK) error ("2c",0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nA OK row: ") ;
	s = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 0, Con) ;
	if (s != UMFPACK_OK) error ("2b",0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nn=zero: ") ;
	s = UMFPACK_report_matrix (0, 0, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_n_nonpositive)) error ("2",0.) ;
	s = UMFPACK_symbolic (0, 0, Ap, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_n_nonpositive) error ("2b",0.) ;
	s = UMFPACK_qsymbolic (0, 0, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_n_nonpositive) error ("2c",0.) ;

	/* ------------------------------------------------------------------ */

	s = do_amd (-1, Ap, Ai, Pamd) ;
	if (s != AMD_INVALID) error ("amd 1", (double) s) ;

	s = do_amd (n, INULL, Ai, Pamd) ;
	if (s != AMD_INVALID) error ("amd 2", (double) s) ;

	s = do_amd (n, Ap, INULL, Pamd) ;
	if (s != AMD_INVALID) error ("amd 3", (double) s) ;

	s = do_amd (n, Ap, Ai, INULL) ;
	if (s != AMD_INVALID) error ("amd 4", (double) s) ;

	s = do_amd (0, Ap, Ai, Pamd) ;
	if (s != AMD_OK) error ("amd 5", (double) s) ;

	s = do_amd_transpose (-1, Ap, Ai, Rp, Ri) ;
	if (s != AMD_INVALID) error ("amd 1t", (double) s) ;

	s = do_amd_transpose (n, INULL, Ai, Rp, Ri) ;
	if (s != AMD_INVALID) error ("amd 2t", (double) s) ;

	s = do_amd_transpose (n, Ap, INULL, Rp, Ri) ;
	if (s != AMD_INVALID) error ("amd 3t", (double) s) ;

	s = do_amd_transpose (n, Ap, Ai, INULL, Ri) ;
	if (s != AMD_INVALID) error ("amd 7t", (double) s) ;

	s = do_amd_transpose (n, Ap, Ai, Rp, INULL) ;
	if (s != AMD_INVALID) error ("amd 8t", (double) s) ;

	s = do_amd_transpose (0, Ap, Ai, Rp, Ri) ;
	if (s != AMD_OK) error ("amd 5t", (double) s) ;

#if 0
{ f = fopen ("debug.amd", "w") ; fprintf (f, "999\n") ; fclose (f) ; }
#endif

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nAp null: ") ;
	s = UMFPACK_report_matrix (n, n, INULL, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_argument_missing)) error ("3",0.) ;
	s = UMFPACK_symbolic (n, n, INULL, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_argument_missing) error ("3b",0.) ;
	s = UMFPACK_qsymbolic (n, n, INULL, Ai, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_argument_missing) error ("3c",0.) ;
	s = UMFPACK_transpose (n, n, INULL, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG(Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_argument_missing) error ("52",0.); 

	s = do_amd (n, Ap, Ai, Pamd) ;
	if (s != AMD_OK) error ("amd 6b", (double) s) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nAi null: ") ;
	s = UMFPACK_report_matrix (n, n, Ap, INULL, CARG(Ax,Az), 1, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_argument_missing)) error ("4",0.) ;
	s = UMFPACK_symbolic (n, n, Ap, INULL, CARG(Ax,Az), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_argument_missing) error ("4b",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, INULL, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_argument_missing) error ("4c",0.) ;

	/* ------------------------------------------------------------------ */

        if (prl >= 0)
        {

            csave = Con [UMFPACK_ORDERING] ;
            my_params [0] = 1 ;
            Con [UMFPACK_ORDERING] = UMFPACK_ORDERING_USER ;
	    s = UMFPACK_fsymbolic (n, n, Ap, Ai, CARG(Ax,Az), 
                &my_bad_ordering, my_params, &Symbolic, Con, Info) ;
            if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
            UMFPACK_report_status (Con, s) ;
            UMFPACK_report_info (Con, Info) ;
            if (Symbolic || s != UMFPACK_ERROR_ordering_failed)error ("6d",0.) ;

            my_params [0] = 0 ;
	    s = UMFPACK_fsymbolic (n, n, Ap, Ai, CARG(Ax,Az), 
                &my_bad_ordering, my_params, &Symbolic, Con, Info) ;
            if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
            UMFPACK_report_status (Con, s) ;
            UMFPACK_report_info (Con, Info) ;
            if (Symbolic || s != UMFPACK_ERROR_ordering_failed)error ("6e",0.) ;

            Con [UMFPACK_ORDERING] = csave ;

        }

	/* ------------------------------------------------------------------ */

        my_params [0] = UMFPACK_ORDERING_AMD ;
        my_params [1] = prl ;
        my_params [2] = EMPTY ;
        my_info [0] = EMPTY ;
        my_info [1] = EMPTY ;
        my_info [2] = EMPTY ;

        ok = UMF_cholmod (n, n, TRUE, Ap, Ai, Pamd, my_params, my_info) ;
        if (!ok) error ("6f", 0.) ;

        ok = UMF_cholmod (n+1, n, TRUE, Ap, Ai, Pamd, my_params, my_info) ;
        if (!ok) error ("6g", 0.) ;

        ok = UMF_cholmod (n, n, TRUE, NULL, Ai, Pamd, my_params, my_info) ;
        if (ok) error ("6h", 0.) ;

	/* ------------------------------------------------------------------ */

	Ap [0] = 1 ;	/* Ap broken [ */
	if (prl > 2) printf ("\nAp [0] != 0: ") ;
	s = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_matrix)) error ("5",0.) ;
	s = UMFPACK_symbolic (n, n, Ap, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("5b",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("5c",0.) ;
	if (prl > 2) printf ("\nCalling umfpack_transpose:\n") ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG(Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("53",0.); 

	s = do_amd (n, Ap, Ai, Pamd) ;
	if (s != AMD_INVALID) error ("amd 6", (double) s) ;

	Ap [0] = 0 ;	/* Ap fixed ] */

	/* ------------------------------------------------------------------ */

	Ap [n] = -1 ;	/* Ap broken [ */
	if (prl > 2) printf ("\nnz < 0: ") ;
	s = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_matrix)) error ("6",0.) ;
	s = UMFPACK_symbolic (n, n, Ap, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("6b",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("6c",0.) ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG(Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("51h",0.); 
	s = UMFPACK_col_to_triplet (n, Ap, Aj) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("52j",0.); 

	s = do_amd (n, Ap, Ai, Pamd) ;
	if (s != AMD_INVALID) error ("amd 6b", (double) s) ;

	Ap [n] = nz ;	/* Ap fixed ] */

	/* ------------------------------------------------------------------ */
#if 0
	Ap [n] = Int_MAX ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != UMFPACK_ERROR_problem_too_large) error ("177a",0.); 
	Ap [n] = nz ;
#endif
	/* ------------------------------------------------------------------ */

	printf ("Ap [2] negative:\n") ;
	UMFPACK_report_control (Con) ;
	c = Ap [2] ;	/* Ap broken [ */
	Ap [2] = -1 ;
	if (prl > 2) printf ("\nAp[2]<0: ") ;
	s = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_matrix)) error ("8",0.) ;
	s = UMFPACK_symbolic (n, n, Ap, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("8b",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("8c",0.) ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG(Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("55",0.); 

	s = do_amd (n, Ap, Ai, Pamd) ;
	if (s != AMD_INVALID) error ("amd 7", (double) s) ;

	Ap [2] = c ;	/* Ap fixed ] */

	/* ------------------------------------------------------------------ */

	c = Ap [2] ;	/* Ap broken [ */
	Ap [2] = nz+1 ;
	if (prl > 2) printf ("\nAp [2] > nz: ") ;
	s = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_matrix)) error ("9",0.) ;
	s = UMFPACK_symbolic (n, n, Ap, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	s = Info [UMFPACK_STATUS] ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("9b",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("9c",0.) ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG(Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("51i",0.); 

	s = do_amd (n, Ap, Ai, Pamd) ;
	if (s != AMD_INVALID) error ("amd 8", (double) s) ;

	Ap [2] = c ;	/* Ap fixed ] */

	/* ------------------------------------------------------------------ */

	c = Ap [4] ;	/* Ap broken [ */
	Ap [4] = Ap [3]-1 ;
	if (prl > 2) printf ("\nAp [4] < Ap [3]-1: ") ;
	s = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK  : UMFPACK_ERROR_invalid_matrix)) error ("10",0.) ;
	s = UMFPACK_symbolic (n, n, Ap, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("8b",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("8c",0.) ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG(Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("51j",0.); 

	s = do_amd (n, Ap, Ai, Pamd) ;
	if (s != AMD_INVALID) error ("amd 9", (double) s) ;

	Ap [4] = c ;	/* Ap fixed ] */

	/* ------------------------------------------------------------------ */

	c = Ai [4] ;	/* Ai broken [ */
	Ai [4] = -1 ;
	if (prl > 2) printf ("\nAi [4] = -1: ") ;
	s = UMFPACK_report_matrix (n , n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_matrix)) error ("12",0.) ;
	s = UMFPACK_symbolic (n, n, Ap, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("12b",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("12c",0.) ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG(Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("51k",0.); 

	s = do_amd (n, Ap, Ai, Pamd) ;
	if (s != AMD_INVALID) error ("amd 10", (double) s) ;

	Ai [4] = c ;	/* Ai fixed ] */

	/* ------------------------------------------------------------------ */

	if (Ap [4] - Ap [3] < 3) error ("col 3 too short",0.) ;
	c = Ai [Ap [3] + 1] ;	/* Ai broken [ */
	Ai [Ap [3] + 1] = 0 ;
	if (prl > 2) printf ("\ncol 3 jumbled: ") ;
	s = UMFPACK_report_matrix (n , n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_matrix)) error ("13",0.) ;
	s = UMFPACK_symbolic (n, n, Ap, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("13b",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_matrix) error ("13c",0.) ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG(Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("51k",0.); 

	s = do_amd (n, Ap, Ai, Pamd) ;
	printf ("amd jumbled: "ID"\n", s) ;
	if (s != AMD_OK_BUT_JUMBLED) error ("amd 11", (double) s) ;

	Ai [Ap [3] + 1] = c ;	/* Ai fixed ] */

	/* ------------------------------------------------------------------ */

#if 0
	{ f = fopen ("debug.amd", "w") ; fprintf (f, "999\n") ; fclose (f) ; }
#endif

	for (i = 0 ; i < n   ; i++) Ap2 [i] = Ap [i] ;
	for (i = n ; i <= 2*n ; i++) Ap2 [i] = nz ;

	s = do_amd (2*n, Ap2, Ai, Pamd) ;
	if (s != AMD_OK) error ("amd 12a", (double) s) ;

	if (prl > 2) printf ("\nhalf empty: ") ;
	s = UMFPACK_report_matrix (2*n, 2*n, Ap2, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != UMFPACK_OK) error ("14",0.) ;
	s = UMFPACK_symbolic (2*n, 2*n, Ap2, Ai, CARG(DNULL,DNULL), &Symbolic, Con, Info) ;

	if (!Symbolic || s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Symbolic || s != UMFPACK_OK) error ("14b",0.) ;
	UMFPACK_free_symbolic (&Symbolic) ;

	s = UMFPACK_symbolic (2*n, 2*n, Ap2, Ai, CARG(DNULL,DNULL), &Symbolic, Con, DNULL) ;
	if (s != UMFPACK_OK) error ("13d2", 0.) ;
	if (!Symbolic) error ("13d",0.) ;

	UMFPACK_free_symbolic (&Symbolic) ;

	s = UMFPACK_qsymbolic (2*n, 2*n, Ap2, Ai, CARG(DNULL,DNULL), Qinit2, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;

        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Symbolic || s != UMFPACK_OK) error ("14c",0.) ;
	UMFPACK_free_symbolic (&Symbolic) ;

	s = do_amd (2*n, Ap2, Ai, Pamd) ;
	if (s != AMD_OK) error ("amd 12", (double) s) ;

	/* ------------------------------------------------------------------ */

	for (i = 0 ; i <= n   ; i++) Ap2 [i] = 0 ;
	if (prl > 2) printf ("\nall empty: ") ;
	s = UMFPACK_report_matrix (n, n, Ap2, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != UMFPACK_OK) error ("141",0.) ;

	s = UMFPACK_col_to_triplet (n, Ap, Aj) ;
	if (s != UMFPACK_OK) error ("151",0.) ;

	s = UMFPACK_symbolic (n, n, Ap2, Ai, CARG(DNULL,DNULL), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Symbolic || s != UMFPACK_OK) error ("142",0.) ;
	UMFPACK_free_symbolic (&Symbolic) ;

	s = UMFPACK_symbolic (n, n, Ap2, Ai, CARG(DNULL,DNULL), &Symbolic, Con, DNULL) ;
	if (s != UMFPACK_OK) error ("142b", 0.) ;
	if (!Symbolic) error ("143",0.) ;
	UMFPACK_free_symbolic (&Symbolic) ;

	s = UMFPACK_qsymbolic (n, n, Ap2, Ai, CARG(DNULL,DNULL), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Symbolic || s != UMFPACK_OK) error ("144",0.) ;
	UMFPACK_free_symbolic (&Symbolic) ;

	s = do_amd (n, Ap, Ai, Pamd) ;
	if (s != AMD_OK) error ("amd 13", (double) s) ;

	/* ------------------------------------------------------------------ */

	for (i = 0 ; i <= n ; i++) Ap2 [i] = Ap [i] ;
	for (p = 0 ; p < nz ; p++)
	{
	    Ai2 [p] = Ai [p] ;
	    Ax2 [p] = Ax [p] ;
	}
	for (i = n ; i < 2*n ; i++)
	{
	    Ap2 [i] = p ;	/* add a dense row 0 */
	    Ai2 [p] = 0 ;
	    Ax2 [p] = 1.0 ;
	    p++ ;
	}
	Ap2 [2*n] = p ;
	if (prl > 2) printf ("\nhalf empty rows: ") ;
	s = UMFPACK_report_matrix (2*n, 2*n, Ap2, Ai2, CARG(Ax2,Az2), 1, Con) ;
	if (s != UMFPACK_OK) error ("30",0.) ;
	s = UMFPACK_symbolic (2*n, 2*n, Ap2, Ai2, CARG(Ax2,Az2), &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Symbolic || s != UMFPACK_OK) error ("30b",0.) ;
	UMFPACK_free_symbolic (&Symbolic) ;

	s = UMFPACK_qsymbolic (2*n, 2*n, Ap2, Ai2, CARG(Ax2,Az2), Qinit2, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Symbolic || s != UMFPACK_OK) error ("30c",0.) ;
	UMFPACK_free_symbolic (&Symbolic) ;

	s = do_amd (2*n, Ap2, Ai2, Pamd) ;
	if (s != AMD_OK) error ("amd 14", (double) s) ;

	/* ------------------------------------------------------------------ */

	for (i = 0 ; i <= 2*n ; i++) Ap2 [i] = 0 ;
	if (prl > 2) printf ("\nall empty: ") ;
	s = UMFPACK_report_matrix (2*n, 2*n, Ap2, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != UMFPACK_OK) error ("15",0.) ;

	s = do_amd (2*n, Ap2, Ai2, Pamd) ;
	if (s != AMD_OK) error ("amd 14b", (double) s) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nold null form was same as col_form: ") ;
	s = UMFPACK_report_matrix (n, n, Ap, Ai, CARG(Ax,Az), 1, Con) ;
	if (s != UMFPACK_OK) error ("16",0.) ;

	/* ================================================================== */
	/* test Numeric [ */
	/* ================================================================== */

	s = UMFPACK_symbolic (n, n, Ap, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;	/* [ */
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
        UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Symbolic || s != UMFPACK_OK) error ("16a",0.) ;

	/* ------------------------------------------------------------------ */

	for (scale = UMFPACK_SCALE_NONE ; scale <= UMFPACK_SCALE_MAX ; scale++)
	{
	    if (Con) Con [UMFPACK_SCALE] = scale ;

	    s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, Info) ;	/* [ */
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    UMFPACK_report_status (Con, s) ;
	    UMFPACK_report_info (Con, Info) ;
	    Info [UMFPACK_FLOPS_ESTIMATE] = -1 ;
	    UMFPACK_report_info (Con, Info) ;
	    if (!Numeric || s != UMFPACK_OK) error ("31",0.) ;

	    if (prl > 2) printf ("good Numeric: ") ;
	    s = UMFPACK_report_numeric ( Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("90",0.) ;

	    /* ------------------------------------------------------------------ */

	    s = UMFPACK_get_lunz (INULL, &unz, &nnrow, &nncol, &nzud, Numeric) ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("57",0.) ;

	    /* ------------------------------------------------------------------ */

	    s = UMFPACK_get_lunz (&lnz, &unz, &nnrow, &nncol, &nzud, Numeric) ;
	    printf ("lnz "ID" unz "ID" nn "ID"\n", lnz, unz, nn) ;
	    if (s != UMFPACK_OK) error ("58",0.) ;

	    /* ------------------------------------------------------------------ */

	    Lp = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	    Li = (Int *) malloc ((lnz+1) * sizeof (Int)) ;		/* [ */
	    Lx = (double *) malloc ((lnz+1) * sizeof (double)) ;	/* [ */
	    Lz = (double *) calloc (lnz , sizeof (double)) ;	/* [ */
	    Up = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	    Ui = (Int *) malloc ((unz+1) * sizeof (Int)) ;		/* [ */
	    Ux = (double *) malloc ((unz+1) * sizeof (double)) ;	/* [ */
	    Uz = (double *) calloc ((unz+1) , sizeof (double)) ;	/* [ */
	    P = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	    Q = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	    Pa = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
	    Wx = (double *) malloc ((10*n) * sizeof (double)) ;	/* [ */
	    Rs = (double *) malloc ((n+1) * sizeof (double)) ;	/* [ */
	    if (!Lp || !Li || !Lx || !Up || !Ui || !Ux || !P || !Q) error ("out of memory (20)",0.) ;
	    if (!Pa || !Wx || !Rs) error ("out of memory (20)",0.) ;
	    if (!Uz || !Lz) error ("out of memory (21)",0.) ;

	    if (prl > 2) printf ("good Numeric again: ") ;
	    s = UMFPACK_report_numeric ( Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("77",0.) ;

	    s = UMFPACK_get_numeric (Lp, Li, CARG(Lx,Lz), Up, Ui, CARG(Ux,Uz), P, Q, CARG(DNULL,DNULL), &do_recip, Rs, Numeric) ;
	    if (s != UMFPACK_OK) error ("59", 0.) ;

	    s = UMFPACK_get_numeric (Lp, Li, CARG(Lx,Lz), Up, Ui, CARG(Ux,Uz), P, Q, CARG(DNULL,DNULL), &do_recip, DNULL, Numeric) ;
	    if (s != UMFPACK_OK) error ("59b", 0.) ;

	    s = UMFPACK_get_numeric (Lp, Li, CARG(Lx,Lz), Up, Ui, CARG(Ux,Uz), P, Q, CARG(DNULL,DNULL), INULL, DNULL, Numeric) ;
	    if (s != UMFPACK_OK) error ("59r", 0.) ;

	    if (prl > 2) printf ("good Numeric yet again: ") ;
	    s = UMFPACK_report_numeric ( Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("75",0.) ;
	    dump_perm ("goodP1", n, Pamd) ;
	    if (prl > 2) printf ("\nL test: ") ;
	    s = UMFPACK_report_matrix (n, n, Lp, Li, CARG(Lx,Lz), 0, Con) ;
	    if (s != UMFPACK_OK) error ("60",0.) ;
	    if (prl > 2) printf ("\nU test: ") ;
	    s = UMFPACK_report_matrix (n, n, Up, Ui, CARG(Ux,Uz), 1, Con) ;
	    if (s != UMFPACK_OK) error ("61",0.) ;
	    dump_perm ("goodP", n, Pamd) ;
	    if (prl > 2) printf ("P test: ") ;
	    s = UMFPACK_report_perm (n, P, Con) ;
	    if (s != UMFPACK_OK) error ("62",0.) ;
	    if (prl > 2) printf ("Q test: ") ;
	    s = UMFPACK_report_perm (n, Q, Con) ;
	    if (s != UMFPACK_OK) error ("63",0.) ;

	    s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_OK) error ("64",0.) ;

	    s = UMFPACK_scale (CARG(DNULL,xz), CARG(b,bz), Numeric) ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("64z",0.) ;

	    s = UMFPACK_scale (CARG(x,xz), CARG(DNULL,bz), Numeric) ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("64y",0.) ;

	    s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(DNULL,xz), CARG(b,bz), Numeric, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("64e",0.) ;

	    s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(DNULL,bz), Numeric, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("64f",0.) ;

	    s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(DNULL,Az), CARG(x,xz), CARG(DNULL,bz), Numeric, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("64g",0.) ;

	    s = UMFPACK_wsolve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info, Pa, Wx) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_OK) error ("64a",0.) ;

	    s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, DNULL) ;
	    if (s != UMFPACK_OK) error ("64b",0.) ;

	    s = UMFPACK_wsolve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, DNULL, Pa, Wx) ;
	    if (s != UMFPACK_OK) error ("64c",0.) ;

	    s = UMFPACK_solve (UMFPACK_A, INULL, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    UMFPACK_report_control (Con) ;
	    UMFPACK_report_status (Con, s) ;
	    UMFPACK_report_info (Con, Info) ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("65a",0.) ;

	    s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(DNULL,xz), CARG(b,bz), Numeric, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("65a",0.) ;

	    s = UMFPACK_wsolve (UMFPACK_A, INULL, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info, Pa, Wx) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("65b",0.) ;

	    s = UMFPACK_solve (UMFPACK_At, INULL, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("65c",0.) ;

	    s = UMFPACK_wsolve (UMFPACK_At, INULL, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info, Pa, Wx) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("66",0.) ;

	    s = UMFPACK_wsolve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info, INULL, Wx) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("67",0.) ;

	    s = UMFPACK_wsolve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info, Pa, DNULL) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("68",0.) ;

	    if (prl > 2) printf ("erroroneous sys arg for umfpack_solve:\n") ;
	    s = UMFPACK_solve (-1, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    UMFPACK_report_status (Con, s) ;
	    UMFPACK_report_info (Con, Info) ;
	    if (s != UMFPACK_ERROR_invalid_system) error ("65d",0.) ;

	    /* check internal error message */
	    UMFPACK_report_status (Con, UMFPACK_ERROR_internal_error) ;

	    /* check unrecognized error code */
	    UMFPACK_report_status (Con, 123123999) ;

	    s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), (void *) NULL, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    UMFPACK_report_status (Con, s) ;
	    UMFPACK_report_info (Con, Info) ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("70",0.) ;

	    s = UMFPACK_wsolve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), (void *) NULL, Con, Info, Pa, Wx) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("71",0.) ;

	    s = UMFPACK_get_determinant (CARG (&Mx, &Mz), &Exp, (void *) NULL, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("71det",0.) ;

	    s = UMFPACK_get_determinant (CARG (DNULL, &Mz), &Exp, Numeric, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh??", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_argument_missing) error ("72det",0.) ;

	    /* corrupt Numeric */
	    Num = (NumericType *) Numeric ;
	    Num->valid = 4909284 ;

	    s = UMFPACK_get_numeric (Lp, Li, CARG(Lx,Lz), Up, Ui, CARG(Ux,Uz), P, Q, CARG(DNULL,DNULL), &do_recip, Rs, Numeric) ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("91",0.) ;

	    s = UMFPACK_save_numeric (Numeric, "nbad.umf") ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("70num",0.) ;

	    s = UMFPACK_get_lunz (&lnz, &unz, &nnrow, &nncol, &nzud, Numeric) ;
	    printf ("s "ID"\n", s) ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("70b",0.) ;

	    s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), (void *) NULL, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("70",0.) ;

	    s = UMFPACK_wsolve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), (void *) NULL, Con, Info, Pa, Wx) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("71",0.) ;

	    if (prl > 2) printf ("bad Numeric: ") ;
	    s = UMFPACK_report_numeric ( Numeric, Con) ;
	    if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("82",0.) ;

	    /* fix numeric */
	    Num->valid = NUMERIC_VALID ;
	    if (prl > 2) printf ("fixed Numeric: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("82",0.) ;

	    /* valid Numeric, but no permissions */
	    s = UMFPACK_save_numeric (Numeric, "/root/nbad.umf") ;
	    if (s != UMFPACK_ERROR_file_IO) error ("72num",0.) ;

	    /* corrupt Numeric again */
	    Num->n_row = -1 ;

	    s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), (void *) NULL, Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("72",0.) ;

	    s = UMFPACK_wsolve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), (void *) NULL, Con, Info, Pa, Wx) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("73",0.) ;

	    s = UMFPACK_scale (CARG(x,xz), CARG(b,bz), (void *) NULL) ;
	    if (s != UMFPACK_ERROR_invalid_Numeric_object) error ("72f",0.) ;

	    /* fix numeric */
	    if (prl > 2) printf ("bad Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("81",0.) ;
	    Num->n_row = n ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("80",0.) ;

	    /* corrupt Numeric again (bad P), then fix it */
	    c = Num->Rperm [0] ;
	    Num->Rperm [0] = -1 ;
	    if (prl > 2) printf ("bad Numeric (P): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("200",0.) ;
	    Num->Rperm [0] = c ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("200b",0.) ;

	    /* corrupt Numeric again (bad Q), then fix it */
	    c = Num->Cperm [0] ;
	    Num->Cperm [0] = -1 ;
	    if (prl > 2) printf ("bad Numeric (Q): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("201",0.) ;
	    Num->Cperm [0] = c ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("201b",0.) ;

	    /* corrupt Numeric again (bad Lpos), then fix it */
	    for (k = 0 ; k < n ; k++)
	    {
		if (Num->Lpos [k] != EMPTY) break ;
	    }
	    c = Num->Lpos [k] ;
	    Num->Lpos [k] = c + 1 ;
	    if (prl > 2) printf ("bad Numeric (Lpos [k]): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s  != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("204",0.) ;
	    Num->Lpos [k] = c ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("204b",0.) ;

	    /* corrupt Numeric again (bad Upos), then fix it */
	    for (k = 0 ; k < n ; k++)
	    {
		if (Num->Upos [k] != EMPTY) break ;
	    }
	    c = Num->Upos [k] ;
	    Num->Upos [k] = 9999999 ;
	    if (prl > 2) printf ("bad Numeric (Upos [0]): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s  != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("204c",0.) ;
	    Num->Upos [k] = c ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("204d",0.) ;

	    /* corrupt Numeric again (bad Lilen), then fix it */
	    c = Num->Lilen [0] ;
	    Num->Lilen [0] = -1 ;
	    if (prl > 2) printf ("bad Numeric (Lilen [0]): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s  != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("205",0.) ;
	    Num->Lilen [0] = c ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("205b",0.) ;

	    /* corrupt Numeric again (bad Lip), then fix it */
	    c = Num->Lip [0] ;
	    Num->Lip [0] = -9999999 ;
	    printf ("Bad numeric (Lip [0])\n") ;
	    fflush (stdout) ;
	    if (prl > 2) printf ("bad Numeric (Lip [0]): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    fflush (stdout) ;
	    if (s  != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("206",0.) ;
	    Num->Lip [0] = c ;
	    printf ("Fixed numeric (Lip [0])\n") ;
	    fflush (stdout) ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("206b",0.) ;
	    fflush (stdout) ;

	    /* corrupt Numeric again (bad LPattern), then fix it */
	    c = Num->Memory [1].header.size ;
	    Num->Memory [1].header.size = -1 ;
	    if (prl > 2) printf ("bad Numeric (Pattern): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s  != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("208",0.) ;
	    Num->Memory [1].header.size = c ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("208b",0.) ;

	    /* corrupt Numeric again (bad UPattern), then fix it */
	    printf ("test 208d:\n") ;
	    for (k = n-1 ; k >= 0 ; k--)
	    {
		if (Num->Uilen [k] > 0) break ;
	    }
	    ip = (Int *) (Num->Memory + SCALAR_ABS (Num->Uip [k])) ;
	    c = *ip ;
	    printf ("Corrupting Num->Uip [k="ID"] = "ID"\n", k, c) ;
	    *ip = -1 ;
	    if (prl > 2) printf ("bad Numeric (UPattern): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s  != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("208c",0.) ;
	    *ip = c ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("208d",0.) ;

	    /* corrupt Numeric again (bad Uilen), then fix it */
	    c = Num->Uilen [k] ;
	    printf ("Corrupting Num->Uilen [k="ID"] = "ID"\n", k, c) ;
	    Num->Uilen [k] = -1 ;
	    if (prl > 2) printf ("bad Numeric (Uilen [k]): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s  != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("205c",0.) ;
	    Num->Uilen [k] = c ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    if (s != UMFPACK_OK) error ("205d",0.) ;

	    /* corrupt Numeric again (bad Uilen), then fix it */
	    c = Num->Uilen [k-1] ;
	    Num->Uilen [k-1] = 99999 ;
	    if (prl > 2) printf ("bad Numeric (Uilen [k-1]): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s  != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("210",0.) ;
	    Num->Uilen [k-1] = c ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (s != UMFPACK_OK) error ("210b",0.) ;

	    /* corrupt Numeric again (bad Uip), then fix it */
	    c = Num->Uip [k] ;
	    Num->Uip [k] = -999999 ;
	    printf ("Bad numeric Uip [k]\n") ;
	    fflush (stdout) ;
	    if (prl > 2) printf ("bad Numeric (Uip [k]): ") ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    fflush (stdout) ;
	    if (s  != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Numeric_object)) error ("206c",0.) ;
	    Num->Uip [k] = c ;
	    printf ("Fixed numeric Uip [k]\n") ;
	    fflush (stdout) ;
	    s = UMFPACK_report_numeric (Numeric, Con) ;
	    if (prl > 2) printf ("fixed Numeric again: ") ;
	    fflush (stdout) ;
	    if (s != UMFPACK_OK) error ("206d",0.) ;

	    free (Rs) ;		/* ] */
	    free (Wx) ;		/* ] */
	    free (Pa) ;		/* ] */
	    free (Q) ;		/* ] */
	    free (P) ;		/* ] */
	    free (Uz) ;		/* ] */
	    free (Ux) ;		/* ] */
	    free (Ui) ;		/* ] */
	    free (Up) ;		/* ] */
	    free (Lz) ;		/* ] */
	    free (Lx) ;		/* ] */
	    free (Li) ;		/* ] */
	    free (Lp) ;		/* ] */

	    UMFPACK_free_numeric (&Numeric) ;	/* ] */

	    if (prl > 2) printf ("Numeric file not found:\n") ;
	    s = UMFPACK_load_numeric (&Numeric, "file_not_found") ;
	    if (s != UMFPACK_ERROR_file_IO) error ("71num",0.) ;

	}

	/* ------------------------------------------------------------------ */

	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, DNULL) ;
	if (!Numeric || s != UMFPACK_OK) error ("31b",0.) ;
	UMFPACK_free_numeric (&Numeric) ;

	/* ------------------------------------------------------------------ */

	/* change the pattern */
	if (Con)
	{
	    Con [UMFPACK_SCALE] = UMFPACK_SCALE_NONE ;
	}

	printf ("change of pattern between symbolic and numeric:\n") ;
	c = Ap [2] ;
	Ap [2] = -1 ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, DNULL) ;
	if (s != UMFPACK_ERROR_different_pattern) error ("97a", (double) s) ;
	Ap [2] = c ;

	c = Ai [2] ;
	Ai [2] = -1 ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, DNULL) ;
	if (s != UMFPACK_ERROR_different_pattern) error ("97b", (double) s) ;
	Ai [2] = c ;

	c = Ai [2] ;
	Ai [2] = 9990099 ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, DNULL) ;
	if (s != UMFPACK_ERROR_different_pattern) error ("97c", (double) s) ;
	Ai [2] = c ;

	c = Ap [n] ;
	Ai [Ap [n]++] = n-1 ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, DNULL) ;
	if (s != UMFPACK_ERROR_different_pattern) error ("97d", (double) s) ;
	Ap [n] = c ;
	printf ("done testing change of pattern between symbolic and numeric.\n") ;

	if (Con)
	{
	    Con [UMFPACK_SCALE] = UMFPACK_DEFAULT_SCALE ;
	}

	/* ------------------------------------------------------------------ */

	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, DNULL) ;
	if (!Numeric || s != UMFPACK_OK) error ("31c",0.) ;
	UMFPACK_free_numeric (&Numeric) ;

	/* ------------------------------------------------------------------ */

	printf ("free nothing:\n") ;
	UMFPACK_free_numeric ((void **) NULL) ;
	UMFPACK_free_symbolic ((void **) NULL) ;
	printf ("free nothing OK\n") ;

	/* ------------------------------------------------------------------ */

	/* test for singular matrix (IN_IN case) */
	for (j = 0 ; j < n ; j++)
	{
	    for (p = 0 ; p < nz ; p++) Ax2 [p] = Ax [p] ;
	    for (p = 0 ; p < nz ; p++) Az2 [p] = Az [p] ;
	    printf ("lastcol = "ID"\n", j) ;
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		Ax2 [p] = 0.0 ;
		Az2 [p] = 0.0 ;
	    }
	    s = UMFPACK_numeric (Ap, Ai, CARG(Ax2,Az2), Symbolic, &Numeric, Con, Info) ;
            UMFPACK_report_status (Con, s) ;
	    UMFPACK_report_info (Con, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    if (!Numeric || s != UMFPACK_WARNING_singular_matrix) error ("120",0.) ;
	    UMFPACK_free_numeric (&Numeric) ;
	}

	/* ------------------------------------------------------------------ */

	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), (void *) NULL, &Numeric, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Numeric || s != UMFPACK_ERROR_invalid_Symbolic_object) error ("32",0.) ;

	/* ------------------------------------------------------------------ */

	s = UMFPACK_numeric (INULL, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Numeric || s != UMFPACK_ERROR_argument_missing) error ("32b",0.) ;

	/* ------------------------------------------------------------------ */

	for (p = 0 ; p < nz ; p++) Ax2 [p] = 0.0 ;
	for (p = 0 ; p < nz ; p++) Az2 [p] = 0.0 ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax2,Az2), Symbolic, &Numeric, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Numeric || s != UMFPACK_WARNING_singular_matrix) error ("33",0.) ;
	UMFPACK_free_numeric (&Numeric) ;

	/* ------------------------------------------------------------------ */

	for (p = 0 ; p < nz ; p++) Ax2 [p] = Ax [p] ;
	for (p = 0 ; p < nz ; p++) Az2 [p] = Ax [p] ;
	i = UMFPACK_DENSE_DEGREE_THRESHOLD (0.2, n) ;
	for (j = 0 ; j < n ; j++)
	{
	    d = Ap [j+1] - Ap [j] ;
	    if (d > i)
	    {
	    	for (p = Ap [j] ; p < Ap [j+1] ; p++)
		{
		    Ax2 [p] = 0.0 ;
		    Az2 [p] = 0.0 ;
		}
	    }
	}
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax2,Az2), Symbolic, &Numeric, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Numeric || s != UMFPACK_WARNING_singular_matrix) error ("33",0.) ;
	UMFPACK_free_numeric (&Numeric) ;

	/* ------------------------------------------------------------------ */

	/* corrupt the Symbolic object */

	Sym = (SymbolicType *) Symbolic ;
	printf ("32c:\n") ;
	fflush (stdout) ;

	Sym->valid = 4040404 ;
	if (prl > 2) printf ("\nSymbolic busted: ") ;
	s = UMFPACK_report_symbolic ((void *) Sym, Con) ;
	if (s  != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_Symbolic_object)) error ("79",0.) ;

	Front_leftmostdesc = (Int *) malloc (n * sizeof (Int)) ;	/* [ */
	Front_1strow = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
	Front_npivots = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
	Front_parent = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
	Chain_start = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	Chain_maxrows = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
	Chain_maxcols = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
	Qtree = (Int *) malloc (n * sizeof (Int)) ;			/* [ */
	Ptree = (Int *) malloc (n * sizeof (Int)) ;			/* [ */
	if (!Front_npivots || !Front_parent || !Chain_start || !Chain_maxrows
	    || !Chain_maxcols || !Qtree) error ("out of memory (22)",0.) ;

	s = UMFPACK_get_symbolic (&nnrow, &nncol, &n1, &nnz, &nfr, &nchains,
	    Ptree, Qtree, Front_npivots, Front_parent, Front_1strow, Front_leftmostdesc,
	    Chain_start, Chain_maxrows, Chain_maxcols, Symbolic) ;
	if (s != UMFPACK_ERROR_invalid_Symbolic_object) error ("93", 0.) ;

	free (Ptree) ;		/* ] */
	free (Qtree) ;		/* ] */
	free (Chain_maxcols) ;	/* ] */
	free (Chain_maxrows) ;	/* ] */
	free (Chain_start) ;	/* ] */
	free (Front_parent) ;	/* ] */
	free (Front_npivots) ;	/* ] */
	free (Front_1strow) ;	/* ] */
	free (Front_leftmostdesc) ;	/* ] */

	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	printf ("32c s: "ID"\n", s) ;
	if (Numeric || s != UMFPACK_ERROR_invalid_Symbolic_object) error ("32c",0.) ;

	Sym->valid = SYMBOLIC_VALID ;
	if (prl > 2) printf ("\nSymbolic fixed: ") ;
	s = UMFPACK_report_symbolic (Symbolic, Con) ;
	if (s != UMFPACK_OK) error ("78",0.) ;

	/* valid Symbolic, but no permissions */
	s = UMFPACK_save_symbolic (Symbolic, "/root/sbad.umf") ;
	if (s != UMFPACK_ERROR_file_IO) error ("72sym",0.) ;

	/* corrupt Symbolic again (bad Qinit) and then fix it */
	c = Sym->Cperm_init [0] ;
	Sym->Cperm_init [0] = -1 ;
	if (prl > 2) printf ("\nSymbolic busted (bad Qinit): ") ;
	s = UMFPACK_report_symbolic (Symbolic, Con) ;
	Sym->Cperm_init [0] = c ;

	/* ------------------------------------------------------------------ */

	/* corrupt the Symbolic object again */

	printf ("32d:\n") ;
	fflush (stdout) ;
	Sym->Cperm_init = (Int *) UMF_free ((void *) Sym->Cperm_init) ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Numeric || s != UMFPACK_ERROR_invalid_Symbolic_object) error ("32d",0.) ;

	s = UMFPACK_save_symbolic (Symbolic, "sbad.umf") ;
	if (s != UMFPACK_ERROR_invalid_Symbolic_object) error ("70sym",0.) ;

	/* ------------------------------------------------------------------ */

	UMFPACK_free_symbolic (&Symbolic) ;		/* ] */

	printf ("Symbolic file not found:\n") ;
	s = UMFPACK_load_symbolic (&Symbolic, "file_not_found") ;
	if (s != UMFPACK_ERROR_file_IO) error ("71sym",0.) ;

#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
	if (UMF_malloc_count != 0) error ("umfpack memory leak!!",0.) ;
#endif

	/* printf (" made it here "ID"\n", umf_fail) ; */
	fflush (stdout) ;

	/* == done ] ======================================================== */

	/* ------------------------------------------------------------------ */

	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(DNULL,DNULL), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	/* printf (" made it here 3 "ID"\n", umf_fail) ; */
	fflush (stdout) ;
	s = Info [UMFPACK_STATUS] ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Symbolic || s != UMFPACK_OK) error ("16b",0.) ;
	/* printf (" made it here too "ID"\n", umf_fail) ; */
	UMFPACK_free_symbolic (&Symbolic) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("Qinit missing: ") ;
	s = UMFPACK_report_perm (n, INULL, Con) ;
	if (s != UMFPACK_OK) error ("17",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(DNULL,DNULL), INULL, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (!Symbolic || s != UMFPACK_OK) error ("17b",0.) ;
	UMFPACK_free_symbolic (&Symbolic) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("Qinit n=0: ") ;
	s = UMFPACK_report_perm (0, Qinit, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_n_nonpositive)) error ("18",0.) ;

	/* ------------------------------------------------------------------ */

	c = Qinit [5] ;
	Qinit [5]++ ;
	if (prl > 2) printf ("Qinit bad: ") ;
	s = UMFPACK_report_perm (n, Qinit, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK  : UMFPACK_ERROR_invalid_permutation)) error ("19",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(DNULL,DNULL), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_permutation) error ("19b",0.) ;
	Qinit [5] = c ;

	/* ------------------------------------------------------------------ */

	c = Qinit [5] ;
	Qinit [5] = -1 ;
	if (prl > 2) printf ("Qinit bad (out of range): ") ;
	s = UMFPACK_report_perm (n, Qinit, Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_permutation)) error ("19c",0.) ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(DNULL,DNULL), Qinit, &Symbolic, Con, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (Symbolic || s != UMFPACK_ERROR_invalid_permutation) error ("19d",0.) ;
	Qinit [5] = c ;

	/* ------------------------------------------------------------------ */

	s = UMFPACK_col_to_triplet (n, Ap, INULL) ;
	if (s != UMFPACK_ERROR_argument_missing) error ("20a",0.) ;

	/* ------------------------------------------------------------------ */

	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG (Ax2,Az2) C1ARG(0)) ;
	if (s != UMFPACK_OK) error ("50",0.); 

	/* ------------------------------------------------------------------ */

	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(DNULL,DNULL), Pinit, Qinit, Ap2, Ai2, CARG (DNULL,DNULL) C1ARG(0)) ;
	if (s != UMFPACK_OK) error ("50e",0.); 

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("UMFPACK transpose test R = A(P,:)'\n") ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, INULL, Ap2, Ai2, CARG (Ax2,Az2) C1ARG(1)) ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (s != UMFPACK_OK) error ("50",0.); 
	if (prl > 2) printf ("\nPinit: ") ;
	s = UMFPACK_report_perm (n, Pinit, Con) ;
	if (prl > 2) printf ("\nR: ") ;
	s = UMFPACK_report_matrix (n , n, Ap2, Ai2, CARG (Ax2,Az2), 1, Con) ;
	if (s != UMFPACK_OK) error ("50e",0.); 
	s = UMFPACK_col_to_triplet (n, Ap2, Aj) ;
	if (s != UMFPACK_OK) error ("50i",0.); 
	if (prl > 2) printf ("\nR, triplet form: ") ;
	s = UMFPACK_report_triplet (n, n, nz, Ai2, Aj, CARG(Ax2,Az2), Con) ;
	if (s != UMFPACK_OK) error ("50y",0.); 

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("UMFPACK transpose test R = pattern of A(P,:).'\n") ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(DNULL,DNULL), Pinit, INULL, Ap2, Ai2, CARG (DNULL,DNULL) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	UMFPACK_report_info (Con, Info) ;
	if (s != UMFPACK_OK) error ("50f",0.); 
	if (prl > 2) printf ("\npattern of R: ") ;
	s = UMFPACK_report_matrix (n , n, Ap2, Ai2, CARG (DNULL,DNULL), 1, Con) ;
	if (s != UMFPACK_OK) error ("50g",0.); 
	s = UMFPACK_col_to_triplet (n, Ap2, Aj) ;
	if (s != UMFPACK_OK) error ("50k",0.); 
	if (prl > 2) printf ("\npattern of R, triplet form: ") ;
	s = UMFPACK_report_triplet (n, n, nz, Ai2, Aj, CARG (DNULL,DNULL), Con) ;
	if (s != UMFPACK_OK) error ("50z",0.); 

	/* ------------------------------------------------------------------ */

	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), INULL, INULL, Ap2, Ai2, CARG (Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_OK) error ("51a",0.); 

	/* ------------------------------------------------------------------ */

	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, INULL, Ap2, Ai2, CARG (Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_OK) error ("51b",0.); 

	/* ------------------------------------------------------------------ */

	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), INULL, Qinit, Ap2, Ai2, CARG (Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_OK) error ("51c",0.); 

	/* ------------------------------------------------------------------ */

	s = UMFPACK_transpose (0, 0, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG (Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_n_nonpositive) error ("54",0.); 

	/* ------------------------------------------------------------------ */

	c = Pinit [5] ;
	Pinit [5] = n-1 ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG (Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_permutation) error ("51e",0.); 
	Pinit [5] = c ;

	/* ------------------------------------------------------------------ */

	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG (Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_OK) error ("51d",0.); 

	/* ------------------------------------------------------------------ */

	c = Pinit [5] ;
	Pinit [5] = -1 ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), Pinit, Qinit, Ap2, Ai2, CARG (Ax2,Az2) C1ARG(0)) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_permutation) error ("56",0.); 
	Pinit [5] = c ;

	/* ------------------------------------------------------------------ */

	s = UMFPACK_col_to_triplet (n, INULL, Aj) ;
	if (s != UMFPACK_ERROR_argument_missing) error ("20b",0.) ;

	/* ------------------------------------------------------------------ */

	s = UMFPACK_col_to_triplet (0, Ap, Aj) ;
	if (s != UMFPACK_ERROR_n_nonpositive) error ("20c",0.) ;

	/* ------------------------------------------------------------------ */

	Ap [0] = 99 ;
	s = UMFPACK_col_to_triplet (n, Ap, Aj) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("20d",0.) ;
	s = do_amd_transpose (n, Ap, Aj, Ap2, Ai2) ;
	if (s != AMD_INVALID) error ("20d_amd",0.) ;
	Ap [0] = 0 ;

	/* ------------------------------------------------------------------ */

	Ap [n] = 0 ;
	s = UMFPACK_col_to_triplet (n, Ap, Aj) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("20e",0.) ;
	Ap [n] = nz ;

	/* ------------------------------------------------------------------ */

	c = Ap [3] ;
	Ap [3] = nz+1 ;
	s = UMFPACK_col_to_triplet (n, Ap, Aj) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("20f",0.) ;
	s = do_amd_transpose (n, Ap, Aj, Ap2, Ai2) ;
	if (s != AMD_INVALID) error ("20f_amd",0.) ;
	Ap [3] = c ;

	/* ------------------------------------------------------------------ */

	c = Aj [0] ;
	Aj [0] = -1 ;
	s = do_amd_transpose (n, Ap, Aj, Ap2, Ai2) ;
	if (s != AMD_INVALID) error ("20z_amd",0.) ;
	Aj [0] = c ;

	/* ------------------------------------------------------------------ */

	c = Ap [4] ;
	Ap [4] = Ap [3]-1 ;
	s = UMFPACK_col_to_triplet (n, Ap, Aj) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("20i",0.) ;
	Ap [4] = c ;

	/* ------------------------------------------------------------------ */

	s = UMFPACK_col_to_triplet (n, Ap, Aj) ;
	if (s != UMFPACK_OK) error ("20",0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nTriples OK: ") ;
	s = UMFPACK_report_triplet (n, n, nz, Ai, Aj, CARG(Ax,Az), Con) ;
	if (s != UMFPACK_OK) error ("21",0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nTriples pattern OK: ") ;
	s = UMFPACK_report_triplet (n, n, nz, Ai, Aj, CARG(DNULL,DNULL), Con) ;
	if (s != UMFPACK_OK) error ("21b",0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nTriples, Ai null: ") ;
	s = UMFPACK_report_triplet (n, n, nz, INULL, Aj, CARG(Ax,Az), Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_argument_missing)) error ("22",0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nTriples, nz=0: ") ;
	s = UMFPACK_report_triplet (n, n, 0, Ai, Aj, CARG(Ax,Az), Con) ;
	if (s != UMFPACK_OK) error ("23a",0.) ;

	if (prl > 2) printf ("\nTriples, nz=-1: ") ;
	s = UMFPACK_report_triplet (n, n, -1, Ai, Aj, CARG(Ax,Az), Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_matrix)) error ("23",0.) ;

	/* ------------------------------------------------------------------ */

	if (prl > 2) printf ("\nTriples, n=0: ") ;
	s = UMFPACK_report_triplet (0, 0, nz, Ai, Aj, CARG(Ax,Az), Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_n_nonpositive)) error ("24",0.) ;

	/* ------------------------------------------------------------------ */

	Map = (Int *) malloc (nz * sizeof (Int)) ;	/* [ */

	c = Aj [1] ;
	Aj [1] = -1 ;
	if (prl > 2) printf ("\nTriples, Aj bad: ") ;
	s = UMFPACK_report_triplet (n, n, nz, Ai, Aj, CARG(Ax,Az), Con) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_invalid_matrix)) error ("41",0.) ;

	s = UMFPACK_triplet_to_col (n, n, nz, Ai, Aj, CARG(Ax,Az), Ap2, Ai2, CARG (Ax2,Az2), (Int *) NULL) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("41",0.) ;

	s = UMFPACK_triplet_to_col (n, n, nz, Ai, Aj, CARG(Ax,Az), Ap2, Ai2, CARG (Ax2,Az2), (Int *) NULL) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("42",0.); 

	s = UMFPACK_triplet_to_col (n, n, nz, Ai, Aj, CARG(DNULL,DNULL), Ap2, Ai2, CARG (DNULL,DNULL), Map) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("42cc",0.); 

	s = UMFPACK_triplet_to_col (n, n, nz, Ai, Aj, CARG(DNULL,DNULL), Ap2, Ai2, CARG (Ax2,Az2), (Int *) NULL) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("42b",0.); 
	Aj [1] = c ;

	/* ------------------------------------------------------------------ */

	s = UMFPACK_triplet_to_col (n, n, nz, Ai, Aj, CARG(Ax,Az), Ap2, Ai2, CARG (Ax2,Az2), (Int *) NULL) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_OK) error ("42c",0.); 

	s = UMFPACK_triplet_to_col (n, n, nz, Ai, Aj, CARG(Ax,Az), Ap2, Ai2, CARG (Ax2,Az2), Map) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_OK) error ("42c.2",0.); 

	/* check the Map */
	for (k = 0 ; k < nz ; k++)
	{
	    p = Map [k] ;
	    i = Ai [k] ;
	    j = Aj [k] ;
	    if (i != Ai2 [p]) error ("Map Ai2.2", 0.) ;
	    if (!(Ap2 [j] <= p && p < Ap2 [j+1])) error ("Map Ap2.2", 0.) ;
	}

	/* ------------------------------------------------------------------ */

	s = UMFPACK_triplet_to_col (n, n, nz, Ai, Aj, CARG(DNULL,DNULL), Ap2, Ai2, CARG (Ax2,Az2), (Int *) NULL) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_OK) error ("42d",0.); 

	s = UMFPACK_triplet_to_col (n, n, nz, Ai, Aj, CARG(DNULL,DNULL), Ap2, Ai2, CARG (Ax2,Az2), Map) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_OK) error ("42d.1",0.); 

	/* check the Map */
	for (k = 0 ; k < nz ; k++)
	{
	    p = Map [k] ;
	    i = Ai [k] ;
	    j = Aj [k] ;
	    if (i != Ai2 [p]) error ("Map Ai2.1", 0.) ;
	    if (!(Ap2 [j] <= p && p < Ap2 [j+1])) error ("Map Ap2.1", 0.) ;
	}

	c = Aj [1] ;
	Aj [1] = -1 ;
	s = UMFPACK_triplet_to_col (n, n, nz, Ai, Aj, CARG(Ax,Az), Ap2, Ai2, CARG (Ax2,Az2), Map) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("42c.3",0.); 
	Aj [1] = c ;

	free (Map) ;	/* ] */

	/* ------------------------------------------------------------------ */

	s = UMFPACK_triplet_to_col (0, 0, nz, Ai, Aj, CARG(Ax,Az), Ap2, Ai2, CARG (Ax2,Az2), (Int *) NULL) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_n_nonpositive) error ("44",0.); 

	/* ------------------------------------------------------------------ */

	s = UMFPACK_triplet_to_col (n, n, 0, Ai, Aj, CARG(Ax,Az), Ap2, Ai2, CARG (Ax2,Az2), (Int *) NULL) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_OK) error ("45a",0.); 

	if (prl > 2) printf ("\nall empty A2: ") ;
	s = UMFPACK_report_matrix (n , n, Ap2, Ai2, CARG (Ax2,Az2), 1, Con) ;
	if (s != UMFPACK_OK) error ("45c",0.) ;

	/* ------------------------------------------------------------------ */

	s = UMFPACK_triplet_to_col (n, n, -1, Ai, Aj, CARG(Ax,Az), Ap2, Ai2, CARG (Ax2,Az2), (Int *) NULL) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_invalid_matrix) error ("45",0.); 

	/* ------------------------------------------------------------------ */

	s = UMFPACK_triplet_to_col (n, n, nz, INULL, Aj, CARG(Ax,Az), Ap2, Ai2, CARG (Ax2,Az2), (Int *) NULL) ;
	UMFPACK_report_status (Con, s) ;
	if (s != UMFPACK_ERROR_argument_missing) error ("46",0.); 

	/* ------------------------------------------------------------------ */

	free (Az2) ;	/* ] */
	free (Ax2) ;	/* ] */
	free (Ai2) ;	/* ] */
	free (Aj) ;	/* ] */

	free (Az) ;	/* ] */
	free (Ax) ;	/* ] */
	free (Ai) ;	/* ] */
	free (Ap) ;	/* ] */


#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
	if (UMF_malloc_count != 0) error ("umfpack memory leak!!",0.) ;
#endif

	printf ("\n]]]]\n\n\n") ;

      }
    }

    free (Ap2) ;	/* ] */
    free (xz) ;		/* ] */
    free (x) ;		/* ] */
    free (bz) ;		/* ] */
    free (b) ;		/* ] */
    free (Qinit2) ;	/* ] */
    free (Pinit) ;	/* ] */
    free (Qinit) ;	/* ] */
    free (Pamd) ;	/* ] */

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* test memory allocation */
    /* ---------------------------------------------------------------------- */

    n = 200 ;

    printf ("memory test\n") ;

#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
    if (UMF_malloc_count != 0) error ("umfpack mem test starts memory leak!!\n",0.) ;
#endif

    matgen_sparse (n, 8*n, 0, 0, 4, 2*n, &Ap, &Ai, &Ax, &Az, 1, 0) ;	/* [[[[ */
    Qinit = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
    b = (double *) malloc (n * sizeof (double)) ;	/* [ */
    bz= (double *) calloc (n , sizeof (double)) ;	/* [ */
    x = (double *) malloc (n * sizeof (double)) ;	/* [ */
    xz= (double *) calloc (n , sizeof (double)) ;	/* [ */
    bgen (n, Ap, Ai, Ax,Az, b,bz) ;

    nz = Ap [n] ;

    randperm (n, Qinit) ;

    UMFPACK_defaults (Control) ;

    for (prl = 5 ; prl >= -1 ; prl--)
    {

	printf ("prl "ID" memtest\n", prl) ;
	fflush (stdout) ;

	umf_realloc_fail = -1 ;
	umf_realloc_hi = 0 ;
	umf_realloc_lo = 0 ;

	umf_fail_hi = 0 ;
	umf_fail_lo = 0 ;
	Control [UMFPACK_PRL] = (Int) prl ;

	umf_fail = 1 ;
	if (prl > 2) printf ("Memfail Qinit: ") ;
	s = UMFPACK_report_perm (n, Qinit, Control) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_out_of_memory)) error ("101",0.) ;

	Cp = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	Cj = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
	Ci = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
	Cx = (double *) malloc (nz * sizeof (double)) ;		/* [ */
	Cz = (double *) calloc (nz , sizeof (double)) ;		/* [ */
	Bp = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	Bj = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
	Bi = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
	Bx = (double *) malloc (nz * sizeof (double)) ;		/* [ */
	Bz = (double *) calloc (nz , sizeof (double)) ;		/* [ */
	Map = (Int *) malloc (nz * sizeof (Int)) ;		/* [ */
	if (!Cp || !Ci || !Cx || !Cj) error ("out of memory (23)",0.) ;
	if (!Bp || !Bi || !Bx || !Bj) error ("out of memory (24)",0.) ;
	if (!Bz || !Cz) error ("out of memory (25)",0.) ;

	umf_fail = 1 ;
	s = UMFPACK_transpose (n, n, Ap, Ai, CARG(Ax,Az), INULL, INULL, Cp, Ci, CARG(Cx,Cz) C1ARG(0)) ;
	if (s != UMFPACK_ERROR_out_of_memory) error ("113", 0.) ;

	for (k = 0 ; k < nz ; k++)
	{
	    Ci [k] = irand (n) ;
	    Cj [k] = irand (n) ;
	    Cx [k] = 2.0 * (xrand ( ) - 1.0) ;
#ifdef COMPLEX
	    Cx [k] = 2.0 * (xrand ( ) - 1.0) ;
#else
	    Cx [k] = 0. ;
#endif
	}

	for (i = 1 ; i <= 4 ; i++)
	{
	    umf_fail = i ;
	    s = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(DNULL,DNULL), Bp, Bi, CARG(DNULL,DNULL), (Int *) NULL) ;
	    UMFPACK_report_status (Con, s) ;
	    if (s != UMFPACK_ERROR_out_of_memory) error ("114", (double) i) ;
	}


	for (i = 1 ; i <= 5 ; i++)
	{
	    umf_fail = i ;
	    s = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(Cx,Cz), Bp, Bi, CARG(Bx,Bz), (Int *) NULL) ;
	    UMFPACK_report_status (Con, s) ;
	    if (s != UMFPACK_ERROR_out_of_memory) error ("115", (double) i) ;
	}

	for (i = 1 ; i <= 5 ; i++)
	{
	    umf_fail = i ;
	    s = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(DNULL,DNULL), Bp, Bi, CARG(DNULL,DNULL), Map) ;
	    UMFPACK_report_status (Con, s) ;
	    if (s != UMFPACK_ERROR_out_of_memory) error ("114", (double) i) ;
	}


	for (i = 1 ; i <= 6 ; i++)
	{
	    umf_fail = i ;
	    s = UMFPACK_triplet_to_col (n, n, nz, Ci, Cj, CARG(Cx,Cz), Bp, Bi, CARG(Bx,Bz), Map) ;
	    UMFPACK_report_status (Con, s) ;
	    if (s != UMFPACK_ERROR_out_of_memory) error ("115", (double) i) ;
	}

	free (Map) ;	/* ] */
	free (Bz) ;	/* ] */
	free (Bx) ;	/* ] */
	free (Bi) ;	/* ] */
	free (Bj) ;	/* ] */
	free (Bp) ;	/* ] */
	free (Cz) ;	/* ] */
	free (Cx) ;	/* ] */
	free (Ci) ;	/* ] */
	free (Cj) ;	/* ] */
	free (Cp) ;	/* ] */

	for (i = 1 ; i <= 24 ; i++)
	{
	    umf_fail = i ;
	    printf ("umf_fail starts at %d\n", umf_fail) ;
	    fflush (stdout) ;
#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
	    if (UMF_malloc_count != 0) error ("umfpack mem test starts memory leak!!\n",0.) ;
#endif
	    s = UMFPACK_symbolic (n, n, Ap, Ai, CARG(Ax,Az), &Symbolic, Control, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    UMFPACK_report_status (Control, s) ;
	    UMFPACK_report_info (Control, Info) ;
	    if (Symbolic || Info [UMFPACK_STATUS] != UMFPACK_ERROR_out_of_memory) error ("104", (double) i) ;
	}

	umf_fail = 25 ;
	s = UMFPACK_qsymbolic (n, n, Ap, Ai, CARG(Ax,Az), Qinit, &Symbolic, Control, Info) ;	/* [ */
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Control, s) ;
	UMFPACK_report_info (Control, Info) ;
	if (!Symbolic || s != UMFPACK_OK) error ("105", 0.) ;

	umf_fail = 1 ;
	if (prl > 2) printf ("\nMemfail Symbolic: ") ;
	s = UMFPACK_report_symbolic (Symbolic, Control) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_out_of_memory)) error ("102",0.) ;

	/* alloc reallocs succeed */
	umf_realloc_fail = -1 ;
	umf_realloc_hi = 0 ;
	umf_realloc_lo = 0 ;

	/* Initial Numeric->Memory allocation fails when umf_fail is 28, and never succeeds.
	 * All mallocs succeed if umf_fail is 16 + 11 + 1 */
	umf_fail_lo = -9999999 ;
	for (i = 1 ; i <= 29 ; i++)
	{
	    umf_fail = i ;
	    printf ("\nDoing numeric, umf_fail = %d\n", umf_fail) ;
	    s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    UMFPACK_report_status (Control, s) ;
	    UMFPACK_report_info (Control, Info) ;
	    if (i < 29)
	    {
		if (Numeric || s != UMFPACK_ERROR_out_of_memory) error ("106", (double) i) ;
	    }
	    else
	    {
		if (!Numeric || s != UMFPACK_OK) error ("106z", (double) umf_fail) ;
		UMFPACK_free_numeric (&Numeric) ;
	    }
	}


	/* everything succeeds, use a small alloc_init */
	printf ("106y:\n") ;
	Control [UMFPACK_ALLOC_INIT] = -30000 ;
	UMFPACK_report_control (Control) ;
	umf_fail = 29 ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Control, s) ;
	UMFPACK_report_info (Control, Info) ;
	if (!Numeric || s != UMFPACK_OK) error ("106y", (double) umf_fail) ;
	UMFPACK_free_numeric (&Numeric) ;

	/* all malloc's succeed - no realloc during factorization */
	umf_fail = -1 ;
	umf_fail_lo = 0 ;
	umf_fail_hi = 0 ;
	umf_realloc_fail = 1 ;
	umf_realloc_hi = 0 ;
	umf_realloc_lo = -9999999 ;

	/* restore Control */
	UMFPACK_defaults (Control) ;
	Control [UMFPACK_PRL] = (Int) prl ;

	/* alloc init the smallest size */
	Control [UMFPACK_ALLOC_INIT] = 0.0 ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Control, s) ;
	UMFPACK_report_info (Control, Info) ;

	if (Numeric)
	{
	    UMFPACK_free_numeric (&Numeric) ;
	    printf ("107 succeeded\n") ;
	}

	/* initial allocation fails once, retry succeeds */
	umf_realloc_fail = 1 ;
	umf_realloc_hi = 0 ;
	umf_realloc_lo = -2 ;
	s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Control, Info) ;	/* ( */
	if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	UMFPACK_report_status (Control, s) ;
	UMFPACK_report_info (Control, Info) ;
	if (!Numeric || s != UMFPACK_OK) error ("110", (double) umf_fail) ;

	/* all reallocs succeed */
	umf_realloc_fail = -1 ;
	umf_realloc_hi = 0 ;
	umf_realloc_lo = 0 ;

	UMFPACK_free_symbolic (&Symbolic) ;		/* ] */

	umf_fail = 1 ;
	if (prl > 2) printf ("Memfail Numeric: ") ;
	s = UMFPACK_report_numeric (Numeric, Control) ;
	if (s != ((prl <= 2) ? UMFPACK_OK : UMFPACK_ERROR_out_of_memory)) error ("108",0.) ;

	for (i = 1 ; i <= 2 ; i++)
	{
	    umf_fail = i ;
	    printf ("\nTest 109, %d\n", umf_fail) ;
	    s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    UMFPACK_report_status (Control, s) ;
	    UMFPACK_report_info (Control, Info) ;
	    if (s != UMFPACK_ERROR_out_of_memory) error ("109", (double) i) ;
	}

	for (i = 1 ; i <= 2 ; i++)
	{
	    umf_fail = i ;
	    s = UMFPACK_solve (UMFPACK_L, Ap, Ai, CARG(Ax,Az), CARG(x,xz), CARG(b,bz), Numeric, Control, Info) ;
	    if (s != Info [UMFPACK_STATUS]) error ("huh", (double) __LINE__)  ;
	    UMFPACK_report_status (Control, s) ;
	    UMFPACK_report_info (Control, Info) ;
	    if (s != UMFPACK_ERROR_out_of_memory) error ("109b", (double) i) ;
	}

	s = UMFPACK_get_lunz (&lnz, &unz, &nnrow, &nncol, &nzud, Numeric) ;
	if (s != UMFPACK_OK) error ("111", 0.) ;

	Rs = (double *) malloc ((n+1) * sizeof (double)) ;	/* [ */
	Lp = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	Li = (Int *) malloc ((lnz+1) * sizeof (Int)) ;		/* [ */
	Lx = (double *) malloc ((lnz+1) * sizeof (double)) ;	/* [ */
	Lz = (double *) calloc ((lnz+1) , sizeof (double)) ;	/* [ */
	Up = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	Ui = (Int *) malloc ((unz+1) * sizeof (Int)) ;		/* [ */
	Ux = (double *) malloc ((unz+1) * sizeof (double)) ;	/* [ */
	Uz = (double *) calloc ((unz+1) , sizeof (double)) ;	/* [ */
	P = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	Q = (Int *) malloc ((n+1) * sizeof (Int)) ;		/* [ */
	if (!Lp || !Li || !Lx || !Up || !Ui || !Ux || !P || !Q) error ("out of memory (26)",0.) ;
	if (!Lz || !Uz) error ("out of memory (27)",0.) ;

	for (i = 1 ; i <= 2 ; i++)
	{
	    umf_fail = i ;
	    s = UMFPACK_get_numeric (Lp, Li, CARG(Lx,Lz), Up, Ui, CARG(Ux,Uz), P, Q, CARG(DNULL,DNULL), &do_recip, Rs, Numeric) ;
	    if (s != UMFPACK_ERROR_out_of_memory) error ("112", (double) i) ;
	}

	umf_fail = 1 ;
	s = UMFPACK_get_determinant (CARG (&Mx, &Mz), &Exp, Numeric, Info) ;
	if (s != Info [UMFPACK_STATUS])
	{
	    printf ("s "ID" %g\n", s, Info [UMFPACK_STATUS]) ;
	    error ("huh", (double) __LINE__)  ;
	}
	if (s != UMFPACK_ERROR_out_of_memory) error ("73det",0.) ;

	UMFPACK_free_numeric (&Numeric) ;		/* ) */

	free (Q) ;		/* ] */
	free (P) ;		/* ] */
	free (Uz) ;		/* ] */
	free (Ux) ;		/* ] */
	free (Ui) ;		/* ] */
	free (Up) ;		/* ] */
	free (Lz) ;		/* ] */
	free (Lx) ;		/* ] */
	free (Li) ;		/* ] */
	free (Lp) ;		/* ] */
	free (Rs) ;		/* ] */

#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
	if (UMF_malloc_count != 0) error ("umfpack memory test leak!!\n",0.) ;
#endif

    }

    free (xz) ;		/* ] */
    free (x) ;		/* ] */
    free (bz) ;		/* ] */
    free (b) ;		/* ] */
    free (Qinit) ;	/* ] */
    free (Ap) ;		/* ] */
    free (Ai) ;		/* ] */
    free (Ax) ;		/* ] */
    free (Az) ;		/* ] */

    /* ---------------------------------------------------------------------- */
    /* NaN/Inf */
    /* ---------------------------------------------------------------------- */

    umf_fail = -1 ;
    umf_fail_lo = 0 ;
    umf_fail_hi = 0 ;
    umf_realloc_fail = -1 ;
    umf_realloc_lo = 0 ;
    umf_realloc_hi = 0 ;

    printf ("matrices with NaN/Infs:\n") ;
    n = 100 ;
    for (k = 0 ; k <= 100 ; k++)
    {
	printf ("NaN/Inf "ID" 4*n nz's, k= "ID"\n", n, k) ;
	matgen_sparse (n, 4*n, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az, 1, 1) ; /* [[[[  */
	if (k == 100)
	{
	    /* make a matrix of all NaN's */
	    for (i = 0 ; i < Ap [n] ; i++)
	    {
		Ax [i] = xnan ;
#ifdef COMPLEX
		Az [i] = xnan ;
#endif
	    }
	}
	b  = (double *) malloc (n * sizeof (double)) ;	/* [ */
	bz = (double *) malloc (n * sizeof (double)) ;	/* [ */
	bgen (n, Ap, Ai, Ax, Az, b, bz) ;
	x  = (double *) malloc (n * sizeof (double)) ;	/* [ */
	xz = (double *) malloc (n * sizeof (double)) ;	/* [ */
	for (prl = 2 ; prl >= -1 ; prl--)
	{
	    printf ("NaN / Inf matrix: \n") ;
	    UMFPACK_defaults (Control) ;
	    Control [UMFPACK_PRL] = prl ;
	    for (scale = UMFPACK_SCALE_NONE ; scale <= UMFPACK_SCALE_MAX ; scale++)
	    {
		Control [UMFPACK_SCALE] = scale ;
		Con = (prl == -1) ? (DNULL) : Control ;
		UMFPACK_report_control (Con) ;

		s = UMFPACK_symbolic (n, n, Ap, Ai, CARG(Ax,Az), &Symbolic, Con, Info) ;	/* [ */
		UMFPACK_report_status (Con, s) ;
		UMFPACK_report_info (Con, Info) ;
		if (!(s == UMFPACK_OK || s == UMFPACK_WARNING_singular_matrix)) error ("887", 0.) ;

		s = UMFPACK_numeric (Ap, Ai, CARG(Ax,Az), Symbolic, &Numeric, Con, Info) ;	/* [ */
		UMFPACK_report_status (Con, s) ;
		UMFPACK_report_info (Con, Info) ;
		if (!(s == UMFPACK_OK || s == UMFPACK_WARNING_singular_matrix)) error ("888", 0.) ;

		s = UMFPACK_solve (UMFPACK_A, Ap, Ai, CARG(Ax,Az) , CARG(x,xz), CARG(b,bz), Numeric, Con, Info) ;
		UMFPACK_report_status (Con, s) ;
		UMFPACK_report_info (Con, Info) ;
		if (!(s == UMFPACK_OK || s == UMFPACK_WARNING_singular_matrix)) error ("889", 0.) ;

		UMFPACK_free_numeric (&Numeric) ;	/* ] */
		UMFPACK_free_symbolic (&Symbolic) ;	/* ] */
	    }
	}
	free (xz) ;	/* ] */
	free (x) ;	/* ] */
	free (bz) ;	/* ] */
	free (b) ;	/* ] */
	free (Az) ;	/* ] */
	free (Ax) ;	/* ] */
	free (Ai) ;	/* ] */
	free (Ap) ;	/* ] */
    }

#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
    if (UMF_malloc_count != 0) error ("umfpack memory leak!!\n",0.) ;
#endif


    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* test report routines */
    /* ---------------------------------------------------------------------- */

    n = 32 ;
    printf ("\n so far: rnorm %10.4e %10.4e\n", rnorm, maxrnorm) ;

    Qinit = (Int *) malloc (n * sizeof (Int)) ;		/* [ */
    b = (double *) malloc (n * sizeof (double)) ;	/* [ */
    bz= (double *) calloc (n , sizeof (double)) ;	/* [ */
    if (!Qinit || !b || !bz) error ("out of memory (28)",0.) ;
    UMFPACK_defaults (Control) ;

    for (prl = 5 ; prl >= 0 ; prl--)
    {
	printf ("\n[[[[ PRL = "ID"\n", prl) ;
	Control [UMFPACK_PRL] = prl ;

	i = UMFPACK_DENSE_DEGREE_THRESHOLD (0.2, n) ;
	printf ("(default) dense row/col degree threshold: "ID"\n", i) ;

	matgen_sparse (n, 12*n, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az, prl, 0) ; /* [[[[ */
	bgen (n, Ap, Ai, Ax,Az, b,bz) ;

	/* also test NaN/Inf handling in solvers */
	b [16] = xnan ;
	b [15] = xinf ;

	/* test col->triplet and triplet->col */
	test_col (n, Ap, Ai, Ax,Az, prl) ;
	rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, INULL, MemOK, FALSE, FALSE, 0., 0.) ;
	printf ("\nrnorm %10.4e %10.4e\n", rnorm, maxrnorm) ;
	randperm (n, Qinit) ;
	rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemOK, FALSE, FALSE, 0., 0.) ;
	printf ("\nrnorm %10.4e %10.4e\n", rnorm, maxrnorm) ;
	free (Ap) ;	/* ] */
	free (Ai) ;	/* ] */
	free (Ax) ;	/* ] */
	free (Az) ;	/* ] */
	printf ("\n]]]]\n\n\n") ;
    }

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* test report with more than 10 dense columns */
    UMFPACK_defaults (Control) ;

    printf ("\nrnorm %10.4e %10.4e\n", rnorm, maxrnorm) ;
    Control [UMFPACK_PRL] = 4 ;
    printf ("\nreport dense matrix with n = "ID"\n", n) ;
    matgen_dense (n, &Ap, &Ai, &Ax, &Az) ; /* [[[[ */
    bgen (n, Ap, Ai, Ax,Az, b,bz) ;
    rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, Qinit, MemOK, FALSE, FALSE, 0., 0.) ;
    maxrnorm = MAX (rnorm, maxrnorm) ;
    printf ("\nrnorm %10.4e %10.4e\n", rnorm, maxrnorm) ;
    free (Ap) ;	/* ] */
    free (Ai) ;	/* ] */
    free (Ax) ;	/* ] */
    free (Az) ;	/* ] */

    free (bz) ;		/* ] */
    free (b) ;		/* ] */
    free (Qinit) ;	/* ] */

    Control [UMFPACK_PRL] = 5 ;

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* test random sparse matrices */ 
    /* ---------------------------------------------------------------------- */

    n = 30 ;

	printf ("sparse "ID" 4*n nz's", n) ;
	matgen_sparse (n, 4*n, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az, 1, 0) ;	/* [[[[ */
	rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 1) ; /* ]]]] */
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;

    n = 200 ;

	printf ("sparse "ID" 4*n nz's", n) ;
	matgen_sparse (n, 4*n, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az, 1, 0) ;	/* [[[[ */

	/*
	rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 1) ;
	*/

	UMFPACK_defaults (Control) ;
	Control [UMFPACK_DENSE_COL] = 0.883883 ;
	Control [UMFPACK_DENSE_ROW] = 0.883883 ; 
	Control [UMFPACK_AMD_DENSE] = 10 ;
	Control [UMFPACK_PIVOT_TOLERANCE] = 0.5 ;
	Control [UMFPACK_BLOCK_SIZE] = 1 ;
	Control [UMFPACK_ALLOC_INIT] = 0 ;
	Control [UMFPACK_SCALE] = 0 ;
	Control [UMFPACK_STRATEGY] = 3 ;
	Control [UMFPACK_FIXQ] = 1 ;

	b = (double *) malloc (n * sizeof (double)) ;	/* [ */
	bz= (double *) calloc (n , sizeof (double)) ;	/* [ */
	if (!b || !bz) error ("out of memory (29)",0.) ;
	bgen (n, Ap, Ai, Ax, Az, b, bz) ;

	rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, INULL, MemOK, FALSE, FALSE, 0., 0.) ;

	UMFPACK_defaults (Control) ;
	Control [UMFPACK_FRONT_ALLOC_INIT] = -10 ;
	Control [UMFPACK_PRL] = 2 ;

	printf ("negative front alloc init\n") ;
	rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, INULL, MemOK, FALSE, FALSE, 0., 0.) ;

	Control [UMFPACK_FRONT_ALLOC_INIT] = -10 ;
	Control [UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC ;
	Control [UMFPACK_AMD_DENSE] = -1 ;

	printf ("symmetric strategy, no dense rows/cols\n") ;
	rnorm = do_many (n, n, Ap, Ai, Ax,Az, b,bz, Control, INULL, MemOK, FALSE, FALSE, 0., 0.) ;

	free (bz) ;	/* ] */
	free (b) ;	/* ] */
	free (Ap) ;	/* ] */
	free (Ai) ;	/* ] */
	free (Ax) ;	/* ] */
	free (Az) ;	/* ] */

	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;

    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

#if 0


    n = 200 ;

	printf ("sparse %7d few nz's", n) ;
	matgen_sparse (n, 20, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az, 1, 0) ;
	rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;


    /* ---------------------------------------------------------------------- */
    /* test random sparse matrices + 4 dense rows */ 
    /* ---------------------------------------------------------------------- */

    n = 100 ;

	printf ("sparse+dense rows %7d ", n) ;
	matgen_sparse (n, 4*n, 4, 2*n, 0, 0, &Ap, &Ai, &Ax, &Az, 1, 0) ;
	rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;


    /* ---------------------------------------------------------------------- */
    /* reset rand ( ) */
    /* ---------------------------------------------------------------------- */

    srand (1) ;

    /* ---------------------------------------------------------------------- */
    /* test random sparse matrices + 4 dense rows & cols */ 
    /* ---------------------------------------------------------------------- */

    n = 100 ;

    /* reduce the number of controls - otherwise this takes too much time */

    c = UMFPACK_BLOCK_SIZE ;
    Controls [c][0] = UMFPACK_DEFAULT_BLOCK_SIZE ;
    Ncontrols [c] = 1 ;

    c = UMFPACK_ALLOC_INIT ;
    Controls [c][0] = 1.0 ;
    Ncontrols [c] = 1 ;

	printf ("sparse+dense rows and cols %7d ", n) ;
	matgen_sparse (n, 4*n, 4, 2*n, 4, 2*n, &Ap, &Ai, &Ax, &Az, 1, 0) ;
	rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;

    c = UMFPACK_BLOCK_SIZE ;
    Controls [c][0] = 1 ;
    Controls [c][1] = 8 ;
    Controls [c][2] = UMFPACK_DEFAULT_BLOCK_SIZE ;
    Ncontrols [c] = 3 ;

    c = UMFPACK_ALLOC_INIT ;
    Controls [c][0] = 0.0 ;
    Controls [c][1] = 0.5 ;
    Controls [c][2] = 1.0 ;	/* not the default */
    Ncontrols [c] = 3 ;

    n = 100 ;

	printf ("very sparse+dense cols %7d ", n) ;
	matgen_sparse (n, 2, 0, 0, 4, 2*n, &Ap, &Ai, &Ax, &Az, 1, 0) ;
	rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;


    /* ---------------------------------------------------------------------- */
    /* test all diagonal matrices */
    /* ---------------------------------------------------------------------- */

    for (n = 1 ; n < 16 ; n++)
    {
	printf ("diagonal %7d ", n) ;
	matgen_band (n, 0, 0, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az) ;
	rnorm = do_and_free (n, Ap, Ai, Ax, Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;
    }

    for (n = 100 ; n <= 500 ; n += 100)
    {
	printf ("diagonal %7d ", n) ;
	matgen_band (n, 0, 0, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az) ;
	rnorm = do_and_free (n, Ap, Ai, Ax,Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* test all tri-diagonal matrices */
    /* ---------------------------------------------------------------------- */

    for (n = 1 ; n < 16 ; n++)
    {
	printf ("tri-diagonal %7d ", n) ;
	matgen_band (n, 1, 1, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az) ;
	rnorm = do_and_free (n, Ap, Ai, Ax,Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;
    }

    for (n = 100 ; n <= 500 ; n += 100)
    {
	printf ("tri-diagonal %7d ", n) ;
	matgen_band (n, 1, 1, 0, 0, 0, 0, &Ap, &Ai, &Ax, &Az) ;
	rnorm = do_and_free (n, Ap, Ai, Ax,Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* test all tri-diagonal matrices + one "dense" row */
    /* ---------------------------------------------------------------------- */

    n = 100 ;

	printf ("tri-diagonal+dense row %7d ", n) ;
	matgen_band (n, 1, 1, 1, n, 0, 0, &Ap, &Ai, &Ax, &Az) ;
	rnorm = do_and_free (n, Ap, Ai, Ax,Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;


    /* ---------------------------------------------------------------------- */
    /* test all tri-diagonal matrices + one "dense" row and col  */
    /* ---------------------------------------------------------------------- */

    n = 100 ;

	printf ("tri-diagonal+dense row and col "ID" ", n) ;
	matgen_band (n, 1, 1, 1, n, 1, n, &Ap, &Ai, &Ax, &Az) ;
	rnorm = do_and_free (n, Ap, Ai, Ax,Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;


    /* ---------------------------------------------------------------------- */
    /* test all small dense matrices */
    /* ---------------------------------------------------------------------- */

    for (n = 1 ; n < 16 ; n++)
    {
	printf ("dense "ID" ", n) ;
	matgen_dense (n, &Ap, &Ai, &Ax, &Az) ;
	rnorm = do_and_free (n, Ap, Ai, Ax,Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;
    }

    for (n = 20 ; n <= 80 ; n += 20 )
    {
	printf ("dense "ID" ", n) ;
	matgen_dense (n, &Ap, &Ai, &Ax, &Az) ;
	rnorm = do_and_free (n, Ap, Ai, Ax,Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;
    }

    n = 130 ;

	printf ("dense "ID" ", n) ;
	matgen_dense (n, &Ap, &Ai, &Ax, &Az) ;
	rnorm = do_and_free (n, Ap, Ai, Ax,Az, Controls, Ncontrols, MemOK, 1) ;
	maxrnorm = MAX (rnorm, maxrnorm) ;
	printf (" %10.4e %10.4e\n", rnorm, maxrnorm) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* done with accurate matrices */
    /* ---------------------------------------------------------------------- */

    ttt = umfpack_timer ( ) ;

    fprintf (stderr,
        "ALL TESTS PASSED: rnorm %8.2e (%8.2e %8.2e %8.2e )\n",
        maxrnorm, maxrnorm_shl0, maxrnorm_arc130, rnorm_omega2) ;

    printf (
        "ALL TESTS PASSED: rnorm %8.2e (%8.2e %8.2e %8.2e)\n",
        maxrnorm, maxrnorm_shl0, maxrnorm_arc130, rnorm_omega2) ;

#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
    if (UMF_malloc_count != 0) error ("umfpack memory leak!!\n",0.) ;
#endif

    return (0) ;
}
