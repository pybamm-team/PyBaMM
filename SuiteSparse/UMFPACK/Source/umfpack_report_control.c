/* ========================================================================== */
/* === UMFPACK_report_control =============================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Prints the control settings.  See umfpack_report_control.h
    for details.
*/

#include "umf_internal.h"

GLOBAL void UMFPACK_report_control
(
    const double Control [UMFPACK_CONTROL]
)
{
    double drow, dcol, relpt, relpt2, alloc_init, front_alloc_init, amd_alpha,
	force_fixQ, droptol, aggr ;
    Int prl, nb, irstep, strategy, scale, s ;
    Int do_singletons, ordering_option ;

    prl = GET_CONTROL (UMFPACK_PRL, UMFPACK_DEFAULT_PRL) ;

    if (prl < 2)
    {
	/* default is to print nothing */
	return ;
    }

    PRINTF  (("UMFPACK V%d.%d.%d (%s), Control:\n", UMFPACK_MAIN_VERSION,
	UMFPACK_SUB_VERSION, UMFPACK_SUBSUB_VERSION, UMFPACK_DATE)) ;

    /* ---------------------------------------------------------------------- */
    /* run-time options */
    /* ---------------------------------------------------------------------- */

    /* This is a "run-time" option because all four umfpack_* versions */
    /* compiled into the UMFPACK library. */

#ifdef DINT
    PRINTF (("    Matrix entry defined as: double\n")) ;
    PRINTF (("    Int (generic integer) defined as: int\n")) ;
#endif
#ifdef DLONG
    PRINTF (("    Matrix entry defined as: double\n")) ;
    PRINTF (("    Int (generic integer) defined as: SuiteSparse_long\n")) ;
#endif
#ifdef ZINT
    PRINTF (("    Matrix entry defined as: double complex\n")) ;
    PRINTF (("    Int (generic integer) defined as: int\n")) ;
#endif
#ifdef ZLONG
    PRINTF (("    Matrix entry defined as: double complex\n")) ;
    PRINTF (("    Int (generic integer) defined as: SuiteSparse_long\n")) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* printing level */
    /* ---------------------------------------------------------------------- */

    PRINTF (("\n    "ID": print level: "ID"\n",
	(Int) INDEX (UMFPACK_PRL), prl)) ;

    /* ---------------------------------------------------------------------- */
    /* dense row/col parameters */
    /* ---------------------------------------------------------------------- */

    drow = GET_CONTROL (UMFPACK_DENSE_ROW, UMFPACK_DEFAULT_DENSE_ROW) ;
    dcol = GET_CONTROL (UMFPACK_DENSE_COL, UMFPACK_DEFAULT_DENSE_COL) ;

    PRINTF (("    "ID": dense row parameter:    %g\n",
	(Int) INDEX (UMFPACK_DENSE_ROW), drow)) ;
    PRINTF (("        \"dense\" rows have    > max (16, (%g)*16*sqrt(n_col)"
	" entries)\n", drow)) ;
    PRINTF (("    "ID": dense column parameter: %g\n",
	(Int) INDEX (UMFPACK_DENSE_COL), dcol)) ;
    PRINTF (("        \"dense\" columns have > max (16, (%g)*16*sqrt(n_row)"
	" entries)\n", dcol)) ;

    /* ---------------------------------------------------------------------- */
    /* pivot tolerance */
    /* ---------------------------------------------------------------------- */

    relpt = GET_CONTROL (UMFPACK_PIVOT_TOLERANCE,
	UMFPACK_DEFAULT_PIVOT_TOLERANCE) ;
    relpt = MAX (0.0, MIN (relpt, 1.0)) ;
    PRINTF (("    "ID": pivot tolerance: %g\n",
	(Int) INDEX (UMFPACK_PIVOT_TOLERANCE), relpt)) ;

    /* ---------------------------------------------------------------------- */
    /* block size */
    /* ---------------------------------------------------------------------- */

    nb = GET_CONTROL (UMFPACK_BLOCK_SIZE, UMFPACK_DEFAULT_BLOCK_SIZE) ;
    nb = MAX (1, nb) ;
    PRINTF (("    "ID": block size for dense matrix kernels: "ID"\n",
	(Int) INDEX (UMFPACK_BLOCK_SIZE), nb)) ;

    /* ---------------------------------------------------------------------- */
    /* strategy */
    /* ---------------------------------------------------------------------- */

    strategy = GET_CONTROL (UMFPACK_STRATEGY, UMFPACK_DEFAULT_STRATEGY) ;
    if (strategy < UMFPACK_STRATEGY_AUTO
     || strategy > UMFPACK_STRATEGY_SYMMETRIC)
    {
	strategy = UMFPACK_STRATEGY_AUTO ;
    }

    PRINTF (("    "ID": strategy: "ID,
	(Int) INDEX (UMFPACK_STRATEGY), strategy)) ;

    if (strategy == UMFPACK_STRATEGY_SYMMETRIC)
    {
	PRINTF ((" (symmetric)\n"
	"        Q = AMD (A+A'), Q not refined during numerical\n"
	"        factorization, and diagonal pivoting (P=Q') attempted.\n")) ;
    }
    else if (strategy == UMFPACK_STRATEGY_UNSYMMETRIC)
    {
	PRINTF ((" (unsymmetric)\n"
	"        Q = COLAMD (A), Q refined during numerical\n"
	"        factorization, and no attempt at diagonal pivoting.\n")) ;
    }
    else /* auto strategy */
    {
	strategy = UMFPACK_STRATEGY_AUTO ;
	PRINTF ((" (auto)\n")) ;
    }

    /* ---------------------------------------------------------------------- */
    /* ordering */
    /* ---------------------------------------------------------------------- */

    ordering_option = GET_CONTROL (UMFPACK_ORDERING, UMFPACK_DEFAULT_ORDERING) ;
    if (ordering_option < 0 || ordering_option > UMFPACK_ORDERING_USER)
    {
        ordering_option = UMFPACK_DEFAULT_ORDERING ;
    }
    PRINTF (("    "ID": ordering: "ID,
	(Int) INDEX (UMFPACK_ORDERING), ordering_option)) ;
    if (ordering_option == UMFPACK_ORDERING_CHOLMOD)
    {
        PRINTF ((" CHOLMOD: AMD/COLAMD, then try METIS, and select best\n")) ;
    }
    else if (ordering_option == UMFPACK_ORDERING_AMD)
    {
        PRINTF ((" AMD/COLAMD\n")) ;
    }
    else if (ordering_option == UMFPACK_ORDERING_GIVEN)
    {
        PRINTF ((" user-provided permutation vector\n")) ;
    }
    else if (ordering_option == UMFPACK_ORDERING_NONE)
    {
        PRINTF ((" none\n")) ;
    }
    else if (ordering_option == UMFPACK_ORDERING_METIS)
    {
        PRINTF ((" METIS\n")) ;
    }
    else if (ordering_option == UMFPACK_ORDERING_BEST)
    {
        PRINTF ((" best effort. Try AMD/COLAMD and METIS, select best\n")) ;
    }
    else if (ordering_option == UMFPACK_ORDERING_USER)
    {
        PRINTF ((" user-provided ordering function\n")) ;
    }

    /* ---------------------------------------------------------------------- */
    /* disable singletons (default is FALSE) */
    /* ---------------------------------------------------------------------- */

    do_singletons = GET_CONTROL (UMFPACK_SINGLETONS,UMFPACK_DEFAULT_SINGLETONS);
    PRINTF (("    "ID": singleton filter:", (Int) INDEX (UMFPACK_SINGLETONS))) ;
    if (do_singletons)
    {
	PRINTF ((" enabled\n")) ;
    }
    else
    {
	PRINTF ((" disabled\n")) ;
    }

    /* ---------------------------------------------------------------------- */
    /* initial allocation parameter */
    /* ---------------------------------------------------------------------- */

    alloc_init = GET_CONTROL (UMFPACK_ALLOC_INIT, UMFPACK_DEFAULT_ALLOC_INIT) ;
    if (alloc_init >= 0)
    {
	PRINTF (("    "ID": initial allocation ratio: %g\n",
	(Int) INDEX (UMFPACK_ALLOC_INIT), alloc_init)) ;
    }
    else
    {
	s = -alloc_init ;
	s = MAX (1, s) ;
	PRINTF (("    "ID": initial allocation (in Units): "ID"\n",
	(Int) INDEX (UMFPACK_ALLOC_INIT), s)) ;
    }

    /* ---------------------------------------------------------------------- */
    /* maximum iterative refinement steps */
    /* ---------------------------------------------------------------------- */

    irstep = GET_CONTROL (UMFPACK_IRSTEP, UMFPACK_DEFAULT_IRSTEP) ;
    irstep = MAX (0, irstep) ;
    PRINTF (("    "ID": max iterative refinement steps: "ID"\n",
	(Int) INDEX (UMFPACK_IRSTEP), irstep)) ;

    /* ---------------------------------------------------------------------- */
    /* force fixQ */
    /* ---------------------------------------------------------------------- */

    force_fixQ = GET_CONTROL (UMFPACK_FIXQ, UMFPACK_DEFAULT_FIXQ) ;
    PRINTF (("    "ID": Q fixed during numerical factorization: %g ",
	(Int) INDEX (UMFPACK_FIXQ), force_fixQ)) ;
    if (force_fixQ > 0)
    {
	PRINTF (("(yes)\n")) ;
    }
    else if (force_fixQ < 0)
    {
	PRINTF (("(no)\n")) ;
    }
    else
    {
	PRINTF (("(auto)\n")) ;
    }

    /* ---------------------------------------------------------------------- */
    /* AMD parameters */
    /* ---------------------------------------------------------------------- */

    amd_alpha = GET_CONTROL (UMFPACK_AMD_DENSE, UMFPACK_DEFAULT_AMD_DENSE) ;
    PRINTF (("    "ID": AMD dense row/col parameter:    %g\n",
	(Int) INDEX (UMFPACK_AMD_DENSE), amd_alpha)) ;
    if (amd_alpha < 0)
    {
	PRINTF (("       no \"dense\" rows/columns\n")) ;
    }
    else
    {
	PRINTF (("       \"dense\" rows/columns have > max (16, (%g)*sqrt(n))"
	    " entries\n", amd_alpha)) ;
    }
    PRINTF (("        Only used if the AMD ordering is used.\n")) ;

    /* ---------------------------------------------------------------------- */
    /* pivot tolerance for symmetric pivoting */
    /* ---------------------------------------------------------------------- */

    relpt2 = GET_CONTROL (UMFPACK_SYM_PIVOT_TOLERANCE,
	UMFPACK_DEFAULT_SYM_PIVOT_TOLERANCE) ;
    relpt2 = MAX (0.0, MIN (relpt2, 1.0)) ;
    PRINTF (("    "ID": diagonal pivot tolerance: %g\n"
	"        Only used if diagonal pivoting is attempted.\n",
	(Int) INDEX (UMFPACK_SYM_PIVOT_TOLERANCE), relpt2)) ;

    /* ---------------------------------------------------------------------- */
    /* scaling */
    /* ---------------------------------------------------------------------- */

    scale = GET_CONTROL (UMFPACK_SCALE, UMFPACK_DEFAULT_SCALE) ;
    if (scale != UMFPACK_SCALE_NONE && scale != UMFPACK_SCALE_MAX)
    {
	scale = UMFPACK_DEFAULT_SCALE ;
    }
    PRINTF (("    "ID": scaling: "ID, (Int) INDEX (UMFPACK_SCALE), scale)) ;
    if (scale == UMFPACK_SCALE_NONE)
    {
	PRINTF ((" (no)")) ;
    }
    else if (scale == UMFPACK_SCALE_SUM)
    {
	PRINTF ((" (divide each row by sum of abs. values in each row)")) ;
    }
    else if (scale == UMFPACK_SCALE_MAX)
    {
	PRINTF ((" (divide each row by max. abs. value in each row)")) ;
    }
    PRINTF (("\n")) ;

    /* ---------------------------------------------------------------------- */
    /* frontal matrix allocation parameter */
    /* ---------------------------------------------------------------------- */

    front_alloc_init = GET_CONTROL (UMFPACK_FRONT_ALLOC_INIT,
	UMFPACK_DEFAULT_FRONT_ALLOC_INIT) ;
    front_alloc_init = MIN (1.0, front_alloc_init) ;
    if (front_alloc_init >= 0)
    {
	PRINTF (("    "ID": frontal matrix allocation ratio: %g\n",
	(Int) INDEX (UMFPACK_FRONT_ALLOC_INIT), front_alloc_init)) ;
    }
    else
    {
	s = -front_alloc_init ;
	s = MAX (1, s) ;
	PRINTF (("    "ID": initial frontal matrix size (# of Entry's): "ID"\n",
	(Int) INDEX (UMFPACK_FRONT_ALLOC_INIT), s)) ;
    }

    /* ---------------------------------------------------------------------- */
    /* drop tolerance */
    /* ---------------------------------------------------------------------- */

    droptol = GET_CONTROL (UMFPACK_DROPTOL, UMFPACK_DEFAULT_DROPTOL) ;
    PRINTF (("    "ID": drop tolerance: %g\n",
	(Int) INDEX (UMFPACK_DROPTOL), droptol)) ;

    /* ---------------------------------------------------------------------- */
    /* aggressive absorption */
    /* ---------------------------------------------------------------------- */

    aggr = GET_CONTROL (UMFPACK_AGGRESSIVE, UMFPACK_DEFAULT_AGGRESSIVE) ;
    PRINTF (("    "ID": AMD and COLAMD aggressive absorption: %g",
	(Int) INDEX (UMFPACK_AGGRESSIVE), aggr)) ;
    if (aggr != 0.0)
    {
	PRINTF ((" (yes)\n")) ;
    }
    else
    {
	PRINTF ((" (no)\n")) ;
    }

    /* ---------------------------------------------------------------------- */
    /* compile-time options */
    /* ---------------------------------------------------------------------- */

    PRINTF ((
	"\n    The following options can only be changed at compile-time:\n")) ;

    PRINTF (("    "ID": BLAS library used:  ",
	(Int) INDEX (UMFPACK_COMPILED_WITH_BLAS))) ;

#ifdef NBLAS
    PRINTF (("none.  UMFPACK will be slow.\n")) ;
#else
    PRINTF (("Fortran BLAS.  size of BLAS integer: "ID"\n",
	(Int) (sizeof (BLAS_INT)))) ;
#endif

#ifdef MATLAB_MEX_FILE
    PRINTF (("    compiled for MATLAB\n")) ;
#else
    PRINTF (("    compiled for ANSI C\n")) ;
#endif

#ifdef SUITESPARSE_TIMER_ENABLED
    PRINTF (("    POSIX C clock_getttime.\n")) ;
#else
    PRINTF (("    no timer used.\n")) ;
#endif

    PRINTF (("    computer/operating system: %s\n", UMFPACK_ARCHITECTURE)) ;
    PRINTF (("    size of int: %g SuiteSparse_long: %g Int: %g pointer: %g"
	" double: %g Entry: %g (in bytes)\n\n", (double) sizeof (int),
	(double) sizeof (SuiteSparse_long), (double) sizeof (Int),
	(double) sizeof (void *), (double) sizeof (double),
	(double) sizeof (Entry))) ;
}
