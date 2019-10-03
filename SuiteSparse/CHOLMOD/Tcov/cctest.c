/* ========================================================================== */
/* === Tcov/cctest ========================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Test for ccolamd v1.0.  Not used if NCAMD defined at compile time. */

#include "cm.h"

#ifndef NCAMD
#include "ccolamd.h"

/* ========================================================================== */
/* === check_constraints ==================================================== */
/* ========================================================================== */

/* Check to see if P obeys the constraints */
Int check_constraints (Int *P, Int *Cmember, Int n)
{
    Int c, clast, k, i ;
    if ((P == NULL) || !CHOLMOD(print_perm) (P, n, n, "ccolamd perm", cm))
    {
	printf ("cctest: Perm is bad\n") ;
	return (FALSE) ;
    }
    clast = EMPTY ;
    for (k = 0 ; k < n ; k++)
    {
	i = P [k] ;
	c = Cmember [i] ;
	if (c < clast)
	{
	    printf ("cctest: constraints are incorrect\n") ;
	    return (FALSE) ;
	}
	clast = c ;
    }
    return (TRUE) ;
}


/* ========================================================================== */
/* === cctest =============================================================== */
/* ========================================================================== */

void cctest (cholmod_sparse *A)
{

    double knobs [CCOLAMD_KNOBS], knobs2 [CCOLAMD_KNOBS] ;
    Int *P, *Cmember, *Cp, *Ci, *Front_npivcol, *Front_nrows, *Front_ncols,
	*Front_parent, *Front_cols, *InFront, *Si, *Sp ;
    cholmod_sparse *C, *A2, *B, *S ;
    Int nrow, ncol, alen, ok, stats [CCOLAMD_STATS], csets, i, nfr, c, p ;
    size_t s ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    my_srand (42) ;						/* RAND reset */

    printf ("\nCCOLAMD test\n") ;

    if (A == NULL)
    {
	return ;
    }

    if (A->stype)
    {
	A2 = CHOLMOD(copy) (A, 0, 0, cm) ;
	B = A2 ;
    }
    else
    {
	A2 = NULL ;
	B = A ;
    }
    S = CHOLMOD(copy_sparse) (A, cm) ;

    nrow = B->nrow ;
    ncol = B->ncol ;
    Si = S->i ;
    Sp = S->p ;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace and Cmember for ccolamd */
    /* ---------------------------------------------------------------------- */

    P = CHOLMOD(malloc) (nrow+1, sizeof (Int), cm) ;
    Cmember = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
    Front_npivcol = CHOLMOD(malloc) (nrow+1, sizeof (Int), cm) ;
    Front_nrows   = CHOLMOD(malloc) (nrow+1, sizeof (Int), cm) ;
    Front_ncols   = CHOLMOD(malloc) (nrow+1, sizeof (Int), cm) ;
    Front_parent  = CHOLMOD(malloc) (nrow+1, sizeof (Int), cm) ;
    Front_cols    = CHOLMOD(malloc) (nrow+1, sizeof (Int), cm) ;
    InFront       = CHOLMOD(malloc) (ncol, sizeof (Int), cm) ;

    csets = MIN (6, nrow) ;
    for (i = 0 ; i < nrow ; i++)
    {
	Cmember [i] = nrand (csets) ;
    }

    CCOLAMD_set_defaults (knobs) ;
    CCOLAMD_set_defaults (knobs2) ;
    CCOLAMD_set_defaults (NULL) ;
    CCOLAMD_report (NULL) ;
    CSYMAMD_report (NULL) ;

    alen = CCOLAMD_recommended (B->nzmax, ncol, nrow) ;
    C = CHOLMOD(allocate_sparse) (ncol, nrow, alen, TRUE, TRUE, 0,
	    CHOLMOD_PATTERN, cm) ;
    Cp = C->p ;
    Ci = C->i ;

    /* ---------------------------------------------------------------------- */
    /* order with ccolamd */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    CHOLMOD(print_sparse) (C, "C for ccolamd", cm) ;
    ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, NULL, stats, Cmember) ;
    CCOLAMD_report (stats) ;
    OK (ok) ;
    ok = stats [CCOLAMD_STATUS] ;
    ok = (ok == CCOLAMD_OK || ok == CCOLAMD_OK_BUT_JUMBLED) ;
    OK (ok) ;

    /* permutation returned in C->p, if the ordering succeeded */
    /* make sure P obeys the constraints */
    OK (check_constraints (Cp, Cmember, nrow)) ;

    /* ---------------------------------------------------------------------- */
    /* order with ccolamd2 */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    ok = CCOLAMD_2 (ncol, nrow, alen, Ci, Cp, NULL, stats,
	    Front_npivcol, Front_nrows, Front_ncols, Front_parent,
	    Front_cols, &nfr, InFront, Cmember) ;
    CCOLAMD_report (stats) ;
    OK (check_constraints (Cp, Cmember, nrow)) ;

    /* ---------------------------------------------------------------------- */
    /* with a small dense-row threshold */
    /* ---------------------------------------------------------------------- */

    knobs2 [CCOLAMD_DENSE_ROW] = 0 ;
    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs2, stats, Cmember) ;
    CCOLAMD_report (stats) ;

    knobs2 [CCOLAMD_DENSE_ROW] = 0.625 ;
    knobs2 [CCOLAMD_DENSE_COL] = 0 ;
    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs2, stats, Cmember) ;
    CCOLAMD_report (stats) ;

    knobs2 [CCOLAMD_DENSE_ROW] = 0.625 ;
    knobs2 [CCOLAMD_DENSE_COL] = -1 ;
    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs2, stats, Cmember) ;
    CCOLAMD_report (stats) ;

    knobs2 [CCOLAMD_DENSE_COL] = 0 ;

    /* ---------------------------------------------------------------------- */
    /* duplicate entries */
    /* ---------------------------------------------------------------------- */

    if (ncol > 2 && nrow > 2)
    {
	ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ;
	OK (ok) ;
	if (Cp [1] - Cp [0] > 2)
	{
	    Ci [0] = Ci [1] ;
	}
	ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs2, stats, Cmember) ;
	CCOLAMD_report (stats) ;
	OK (CHOLMOD(print_perm) (Cp, nrow, nrow, "ccolamd perm", cm)) ;
    }

    /* ---------------------------------------------------------------------- */
    /* csymamd */
    /* ---------------------------------------------------------------------- */

    if (nrow == ncol)
    {
	Int n = nrow ;

	ok = CSYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
                Cmember, A->stype) ;
	OK (ok) ;
	OK (check_constraints (P, Cmember, n)) ;
	CSYMAMD_report (stats) ;

	/* ------------------------------------------------------------------ */
	/* csymamd errors */
	/* ------------------------------------------------------------------ */

	ok = CSYMAMD_MAIN (n, Si, Sp, P, NULL, NULL,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
                Cmember, A->stype) ;		       NOT (ok);

	ok = CSYMAMD_MAIN (n, NULL, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
		Cmember, A->stype) ;		       NOT (ok);
	CSYMAMD_report (stats) ;

	ok = CSYMAMD_MAIN (n, Si, NULL, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
		Cmember, A->stype) ;		       NOT (ok);
	CSYMAMD_report (stats) ;

	ok = CSYMAMD_MAIN (-1, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
		Cmember, A->stype) ;		       NOT (ok);
	CSYMAMD_report (stats) ;

	p = Sp [n] ;
	Sp [n] = -1 ;
	ok = CSYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
		Cmember, A->stype) ;		       NOT (ok);
	CSYMAMD_report (stats) ;
	Sp [n] = p ;

	Sp [0] = -1 ;
	ok = CSYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
		Cmember, A->stype) ;		       NOT (ok);
	CSYMAMD_report (stats) ;
	Sp [0] = 0 ;

	if (n > 2 && Sp [n] > 3)
	{
	    p = Sp [1] ;
	    Sp [1] = -1 ;
	    ok = CSYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
		Cmember, A->stype) ;	       NOT (ok);
	    CSYMAMD_report (stats) ;
	    Sp [1] = p ;

	    i = Si [0] ;
	    Si [0] = -1 ;
	    ok = CSYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
		Cmember, A->stype) ;	       NOT (ok);
	    CSYMAMD_report (stats) ;
	    Si [0] = i ;

	    /* ok, but jumbled */
	    i = Si [0] ;
	    Si [0] = Si [1] ;
	    Si [1] = i ;
	    ok = CSYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
		Cmember, A->stype) ;	       OK (ok);
	    CSYMAMD_report (stats) ;
	    i = Si [0] ;
	    Si [0] = Si [1] ;
	    Si [1] = i ;

	    test_memory_handler ( ) ;
	    ok = CSYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func,
		Cmember, A->stype) ;	       NOT(ok);
	    CSYMAMD_report (stats) ;
	    normal_memory_handler ( ) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* error tests */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ;   OK (ok) ;
    ok = CCOLAMD_MAIN (ncol, nrow, 0, Ci, Cp, knobs, stats, Cmember) ;     NOT (ok) ;
    CCOLAMD_report (stats) ;

    ok = CCOLAMD_MAIN (ncol, nrow, alen, NULL, Cp, knobs, stats, Cmember); NOT (ok) ;
    CCOLAMD_report (stats) ;

    ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, NULL, knobs, stats, Cmember); NOT (ok) ;
    CCOLAMD_report (stats) ;

    ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, NULL, Cmember) ;   NOT (ok) ;
    CCOLAMD_report (stats) ;

    ok = CCOLAMD_MAIN (-1, nrow, alen, Ci, Cp, knobs, stats, Cmember) ;    NOT (ok) ;
    CCOLAMD_report (stats) ;

    ok = CCOLAMD_MAIN (ncol, -1, alen, Ci, Cp, knobs, stats, Cmember) ;    NOT (ok) ;
    CCOLAMD_report (stats) ;

    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    Cp [nrow] = -1 ;
    ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, stats, Cmember) ;  NOT (ok) ;
    CCOLAMD_report (stats) ;

    Cp [0] = 1 ;
    ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, stats, Cmember) ;  NOT (ok) ;
    CCOLAMD_report (stats) ;

    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ;   OK (ok) ;

    if (nrow > 0 && alen > 0 && Cp [1] > 0)
    {
	c = Cmember [0] ;
	Cmember [0] = -1 ;
	ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, stats, Cmember) ;NOT(ok);
	CCOLAMD_report (stats) ;
	Cmember [0] = c ;

	p = Cp [1] ;
	Cp [1] = -1 ;
	ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, stats, Cmember) ;NOT(ok);
	CCOLAMD_report (stats) ;
	Cp [1] = p ;

	i = Ci [0] ;
	Ci [0] = -1 ;
	ok = CCOLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, stats, Cmember) ;NOT(ok);
	CCOLAMD_report (stats) ;
	Ci [0] = i ;
    }

    s = CCOLAMD_recommended (-1, 0, 0) ;
    OK (s == 0) ;

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(free) (nrow+1, sizeof (Int), Front_npivcol, cm) ;
    CHOLMOD(free) (nrow+1, sizeof (Int), Front_nrows, cm) ;
    CHOLMOD(free) (nrow+1, sizeof (Int), Front_ncols, cm) ;
    CHOLMOD(free) (nrow+1, sizeof (Int), Front_parent, cm) ;
    CHOLMOD(free) (nrow+1, sizeof (Int), Front_cols, cm) ;
    CHOLMOD(free) (nrow+1, sizeof (Int), P, cm) ;
    CHOLMOD(free) (nrow, sizeof (Int), Cmember, cm) ;
    CHOLMOD(free) (ncol, sizeof (Int), InFront, cm) ;

    CHOLMOD(free_sparse) (&S, cm) ;
    CHOLMOD(free_sparse) (&A2, cm) ;
    CHOLMOD(free_sparse) (&C, cm) ;
    cm->print = 1 ;
}

#else

void cctest (cholmod_sparse *A)
{
    if (A == NULL)
    {
	return ;
    }

    cm->print = 1 ;
}
#endif
