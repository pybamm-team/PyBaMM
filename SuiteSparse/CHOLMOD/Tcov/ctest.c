/* ========================================================================== */
/* === Tcov/ctest =========================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Test for colamd v2.4 */

#include "cm.h"
#include "colamd.h"


/* ========================================================================== */
/* === ctest ================================================================ */
/* ========================================================================== */

void ctest (cholmod_sparse *A)
{
    double knobs [COLAMD_KNOBS], knobs2 [COLAMD_KNOBS] ;
    Int *P, *Cp, *Ci, *Si, *Sp ;
    cholmod_sparse *C, *A2, *B, *S, *BT ;
    Int nrow, ncol, alen, ok, stats [COLAMD_STATS], i, p, trial ;
    size_t s ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    printf ("\nCOLAMD test\n") ;

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

    nrow = B->nrow ;
    ncol = B->ncol ;
    S = NULL ;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace colamd */
    /* ---------------------------------------------------------------------- */

    P = CHOLMOD(malloc) (nrow+1, sizeof (Int), cm) ;

    COLAMD_set_defaults (knobs) ;
    COLAMD_set_defaults (knobs2) ;
    COLAMD_set_defaults (NULL) ;
    COLAMD_report (NULL) ;
    SYMAMD_report (NULL) ;

    alen = COLAMD_recommended (B->nzmax, ncol, nrow) ;
    C = CHOLMOD(allocate_sparse) (ncol, nrow, alen, TRUE, TRUE, 0,
	    CHOLMOD_PATTERN, cm) ;
    Cp = C->p ;
    Ci = C->i ;

    /* ---------------------------------------------------------------------- */
    /* order with colamd */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    CHOLMOD(print_sparse) (C, "C for colamd", cm) ;
    ok = COLAMD_MAIN (ncol, nrow, alen, Ci, Cp, NULL, stats) ;
    COLAMD_report (stats) ;
    OK (ok) ;
    ok = stats [COLAMD_STATUS] ;
    ok = (ok == COLAMD_OK || ok == COLAMD_OK_BUT_JUMBLED) ;
    OK (ok) ;

    /* permutation returned in C->p, if the ordering succeeded */
    /* make sure P obeys the constraints */
    OK (CHOLMOD(print_perm) (Cp, nrow, nrow, "colamd perm", cm)) ;

    /* ---------------------------------------------------------------------- */
    /* with different dense thresholds */
    /* ---------------------------------------------------------------------- */

    printf ("\nall dense rows:\n") ;
    knobs2 [COLAMD_DENSE_ROW] = 0 ;
    knobs2 [COLAMD_DENSE_COL] = 0.5 ;
    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    ok = COLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs2, stats) ;
    COLAMD_report (stats) ;
    OK (CHOLMOD(print_perm) (Cp, nrow, nrow, "colamd perm", cm)) ;

    printf ("\nall dense cols:\n") ;
    knobs2 [COLAMD_DENSE_ROW] = 0.5 ;
    knobs2 [COLAMD_DENSE_COL] = 0 ;
    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    ok = COLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs2, stats) ;
    COLAMD_report (stats) ;
    OK (CHOLMOD(print_perm) (Cp, nrow, nrow, "colamd perm", cm)) ;

    printf ("\nno dense rows/cols:\n") ;
    knobs2 [COLAMD_DENSE_ROW] = -1 ;
    knobs2 [COLAMD_DENSE_COL] = -1 ;
    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    ok = COLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs2, stats) ;
    COLAMD_report (stats) ;
    OK (CHOLMOD(print_perm) (Cp, nrow, nrow, "colamd perm", cm)) ;

    knobs2 [COLAMD_DENSE_ROW] = 0.5 ;
    knobs2 [COLAMD_DENSE_COL] = 0.5 ;

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
	ok = COLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs2, stats) ;
	COLAMD_report (stats) ;
	OK (CHOLMOD(print_perm) (Cp, nrow, nrow, "colamd perm", cm)) ;
    }

    /* ---------------------------------------------------------------------- */
    /* symamd */
    /* ---------------------------------------------------------------------- */

    if (nrow == ncol)
    {
	Int n = nrow ;

	BT = CHOLMOD(transpose) (B, 0, cm) ;		OKP(BT);
	S = CHOLMOD(add) (B, BT, one, one, FALSE, FALSE, cm) ;
	CHOLMOD(free_sparse) (&BT, cm) ;
	Si = S->i ;
	Sp = S->p ;

	ok = SYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
	OK (ok) ;
	OK (CHOLMOD(print_perm) (P, n, n, "symamd perm", cm)) ;
	SYMAMD_report (stats) ;

	/* ------------------------------------------------------------------ */
	/* symamd errors */
	/* ------------------------------------------------------------------ */

	test_memory_handler ( ) ;
	for (trial = 0 ; trial < 3 ; trial++)
	{
	    my_tries = trial ;
	    ok = SYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
	    NOT (ok) ;
	}
	my_tries = 3 ;
	ok = SYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
	OK (ok) ;
	normal_memory_handler ( ) ;

	ok = SYMAMD_MAIN (n, Si, Sp, P, NULL, NULL,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
		NOT (ok);

	ok = SYMAMD_MAIN (n, NULL, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
		NOT (ok);
	SYMAMD_report (stats) ;

	ok = SYMAMD_MAIN (n, Si, NULL, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
		NOT (ok);
	SYMAMD_report (stats) ;

	ok = SYMAMD_MAIN (-1, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
		NOT (ok);
	SYMAMD_report (stats) ;

	p = Sp [n] ;
	Sp [n] = -1 ;
	ok = SYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
		NOT (ok);
	SYMAMD_report (stats) ;
	Sp [n] = p ;

	Sp [0] = -1 ;
	ok = SYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
		NOT (ok);
	SYMAMD_report (stats) ;
	Sp [0] = 0 ;

	if (n > 2 && Sp [n] > 3)
	{
	    p = Sp [1] ;
	    Sp [1] = -1 ;
	    ok = SYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
		NOT (ok);
	    SYMAMD_report (stats) ;
	    Sp [1] = p ;

	    i = Si [0] ;
	    Si [0] = -1 ;
	    ok = SYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
		NOT (ok);
	    SYMAMD_report (stats) ;
	    Si [0] = i ;

	    /* ok, but jumbled */
	    i = Si [0] ;
	    Si [0] = Si [1] ;
	    Si [1] = i ;
	    ok = SYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
		OK (ok);
	    SYMAMD_report (stats) ;
	    OK (CHOLMOD(print_perm) (P, nrow, nrow, "symamd perm", cm)) ;
	    i = Si [0] ;
	    Si [0] = Si [1] ;
	    Si [1] = i ;

	    test_memory_handler ( ) ;
	    ok = SYMAMD_MAIN (n, Si, Sp, P, NULL, stats,
                SuiteSparse_config.calloc_func,
                SuiteSparse_config.free_func) ;
		NOT (ok);
	    SYMAMD_report (stats) ;
	    normal_memory_handler ( ) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* error tests */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ;   OK (ok) ;
    ok = COLAMD_MAIN (ncol, nrow, 0, Ci, Cp, knobs, stats) ;		NOT (ok) ;
    COLAMD_report (stats) ;

    ok = COLAMD_MAIN (ncol, nrow, alen, NULL, Cp, knobs, stats);	NOT (ok) ;
    COLAMD_report (stats) ;

    ok = COLAMD_MAIN (ncol, nrow, alen, Ci, NULL, knobs, stats);	NOT (ok) ;
    COLAMD_report (stats) ;

    ok = COLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, NULL) ;	NOT (ok) ;
    COLAMD_report (stats) ;

    ok = COLAMD_MAIN (-1, nrow, alen, Ci, Cp, knobs, stats) ;	NOT (ok) ;
    COLAMD_report (stats) ;

    ok = COLAMD_MAIN (ncol, -1, alen, Ci, Cp, knobs, stats) ;	NOT (ok) ;
    COLAMD_report (stats) ;

    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ; OK (ok) ;
    Cp [nrow] = -1 ;
    ok = COLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, stats) ;	NOT (ok) ;
    COLAMD_report (stats) ;

    Cp [0] = 1 ;
    ok = COLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, stats) ;	NOT (ok) ;
    COLAMD_report (stats) ;

    ok = CHOLMOD(transpose_unsym) (B, 0, NULL, NULL, 0, C, cm) ;   OK (ok) ;

    if (nrow > 0 && alen > 0 && Cp [1] > 0)
    {

	p = Cp [1] ;
	Cp [1] = -1 ;
	ok = COLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, stats) ;	    NOT(ok);
	COLAMD_report (stats) ;
	Cp [1] = p ;

	i = Ci [0] ;
	Ci [0] = -1 ;
	ok = COLAMD_MAIN (ncol, nrow, alen, Ci, Cp, knobs, stats) ;	    NOT(ok);
	COLAMD_report (stats) ;
	Ci [0] = i ;
    }

    s = COLAMD_recommended (-1, 0, 0) ;
    OK (s == 0) ;

    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(free) (nrow+1, sizeof (Int), P, cm) ;
    CHOLMOD(free_sparse) (&S, cm) ;
    CHOLMOD(free_sparse) (&A2, cm) ;
    CHOLMOD(free_sparse) (&C, cm) ;
}
