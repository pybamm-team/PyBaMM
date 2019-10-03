/* ========================================================================== */
/* === Tcov/camdtest ======================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Test for camd v2.0 */

#include "cm.h"

#ifndef NCAMD
#include "camd.h"

/* ========================================================================== */
/* === camdtest ============================================================= */
/* ========================================================================== */

void camdtest (cholmod_sparse *A)
{
    double Control [CAMD_CONTROL], Info [CAMD_INFO], alpha ;
    Int *P, *Cp, *Ci, *Sp, *Si, *Bp, *Bi, *Ep, *Ei, *Fp, *Fi,
	*Len, *Nv, *Next, *Head, *Elen, *Deg, *Wi, *W, *Flag, *BucketSet,
	*Constraint ;
    cholmod_sparse *C, *B, *S, *E, *F ;
    Int i, j, n, nrow, ncol, ok, cnz, bnz, p, trial, sorted ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    printf ("\nCAMD test\n") ;

    if (A == NULL)
    {
	return ;
    }

    if (A->stype)
    {
	B = CHOLMOD(copy) (A, 0, 0, cm) ;
    }
    else
    {
	B = CHOLMOD(aat) (A, NULL, 0, 0, cm) ;
    }

    if (A->nrow != A->ncol)
    {
	F = CHOLMOD(copy_sparse) (B, cm) ;
	OK (F->nrow == F->ncol) ;
	CHOLMOD(sort) (F, cm) ;
    }
    else
    {
	/* A is square and unsymmetric, and may have entries in A+A' that
	 * are not in A */
	F = CHOLMOD(copy_sparse) (A, cm) ;
	CHOLMOD(sort) (F, cm) ;
    }

    C = CHOLMOD(copy_sparse) (B, cm) ;

    nrow = C->nrow ;
    ncol = C->ncol ;
    n = nrow ;
    OK (nrow == ncol) ;

    Cp = C->p ;
    Ci = C->i ;

    Bp = B->p ;
    Bi = B->i ;

    /* ---------------------------------------------------------------------- */
    /* S = sorted form of B, using CAMD_preprocess */
    /* ---------------------------------------------------------------------- */

    cnz = CHOLMOD(nnz) (C, cm) ;
    S = CHOLMOD(allocate_sparse) (n, n, cnz, TRUE, TRUE, 0, CHOLMOD_PATTERN,
	    cm);
    Sp = S->p ;
    Si = S->i ;

    W = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Flag = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    CAMD_preprocess (n, Bp, Bi, Sp, Si, W, Flag) ;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace for camd */
    /* ---------------------------------------------------------------------- */

    P = CHOLMOD(malloc) (n+1, sizeof (Int), cm) ;
    Constraint = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    for (i = 0 ; i < n ; i++)
    {
	Constraint [i] = my_rand () % (MIN (n,6)) ;
    }

    ok = CAMD_cvalid (n, Constraint) ; OK (ok) ;
    if (n > 0)
    {
	Constraint [0] = -1 ;
	ok = CAMD_cvalid (n, Constraint) ; OK (!ok) ;
	Constraint [0] = 0 ;
    }
    ok = CAMD_cvalid (n, Constraint) ; OK (ok) ;
    ok = CAMD_cvalid (n, NULL) ; OK (ok) ;

    Len  = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Nv   = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Next = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Head = CHOLMOD(malloc) (n+1, sizeof (Int), cm) ;
    Elen = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Deg  = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Wi   = CHOLMOD(malloc) (n+1, sizeof (Int), cm) ;
    BucketSet = CHOLMOD(malloc) (n, sizeof (Int), cm) ;

    /* ---------------------------------------------------------------------- */

    for (sorted = 0 ; sorted <= 1 ; sorted++)
    {

	if (sorted) CHOLMOD(sort) (C, cm) ;

	Cp = C->p ;
	Ci = C->i ;

	/* ------------------------------------------------------------------ */
	/* order C with CAMD_order */
	/* ------------------------------------------------------------------ */

	CAMD_defaults (Control) ;
	CAMD_defaults (NULL) ;
	CAMD_control (Control) ;
	CAMD_control (NULL) ;
	CAMD_info (NULL) ;

	ok = CAMD_order (n, Cp, Ci, P, Control, Info, Constraint) ;
	printf ("camd return value: "ID"\n", ok) ;
	CAMD_info (Info) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "CAMD permutation", cm)) ;

	ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
	printf ("camd return value: "ID"\n", ok) ;
	CAMD_info (Info) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "CAMD permutation", cm)) ;

	/* no dense rows/cols */
	alpha = Control [CAMD_DENSE] ;
	Control [CAMD_DENSE] = -1 ;
	CAMD_control (Control) ;
	ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
	printf ("camd return value: "ID"\n", ok) ;
	CAMD_info (Info) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "CAMD permutation (alpha=-1)", cm)) ;

	/* many dense rows/cols */
	Control [CAMD_DENSE] = 0 ;
	CAMD_control (Control) ;
	ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
	printf ("camd return value: "ID"\n", ok) ;
	CAMD_info (Info) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "CAMD permutation (alpha=0)", cm)) ;
	Control [CAMD_DENSE] = alpha ;

	/* no aggressive absorption */
	Control [CAMD_AGGRESSIVE] = FALSE ;
	CAMD_control (Control) ;
	ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
	printf ("camd return value: "ID"\n", ok) ;
	CAMD_info (Info) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "CAMD permutation (no agg) ", cm)) ;
	Control [CAMD_AGGRESSIVE] = TRUE ;

	/* ------------------------------------------------------------------ */
	/* order F with CAMD_order */
	/* ------------------------------------------------------------------ */

	Fp = F->p ;
	Fi = F->i ;
	ok = CAMD_order (n, Fp, Fi, P, Control, Info, NULL) ;
	printf ("camd return value: "ID"\n", ok) ;
	CAMD_info (Info) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "F: CAMD permutation", cm)) ;

	/* ------------------------------------------------------------------ */
	/* order S with CAMD_order */
	/* ------------------------------------------------------------------ */

	ok = CAMD_order (n, Sp, Si, P, Control, Info, NULL) ;
	printf ("camd return value: "ID"\n", ok) ;
	CAMD_info (Info) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "CAMD permutation", cm)) ;

	/* ------------------------------------------------------------------ */
	/* order E with CAMD_2, which destroys its contents */
	/* ------------------------------------------------------------------ */

	E = CHOLMOD(copy) (B, 0, -1, cm) ;	/* remove diagonal entries */
	bnz = CHOLMOD(nnz) (E, cm) ;

	/* add the bare minimum extra space to E */
	ok = CHOLMOD(reallocate_sparse) (bnz + n, E, cm) ;
	OK (ok) ;
	Ep = E->p ;
	Ei = E->i ;

	for (j = 0 ; j < n ; j++)
	{
	    Len [j] = Ep [j+1] - Ep [j] ;
	}

	printf ("calling CAMD_2:\n") ;
	if (n > 0)
	{
	    CAMD_2 (n, Ep, Ei, Len, E->nzmax, Ep [n], Nv, Next, P, Head, Elen,
		Deg, Wi, NULL, Info, NULL, BucketSet) ;
	    CAMD_info (Info) ;
	    OK (CHOLMOD(print_perm) (P, n, n, "CAMD2 permutation", cm)) ;
	}

	/* ------------------------------------------------------------------ */
	/* error tests */
	/* ------------------------------------------------------------------ */

	ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	ok = CAMD_order (-1, Cp, Ci, P, Control, Info, NULL) ;
	OK (ok == CAMD_INVALID);
	ok = CAMD_order (0, Cp, Ci, P, Control, Info, NULL) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	ok = CAMD_order (n, NULL, Ci, P, Control, Info, NULL) ;
	OK (ok == CAMD_INVALID);
	ok = CAMD_order (n, Cp, NULL, P, Control, Info, NULL) ;
	OK (ok == CAMD_INVALID);
	ok = CAMD_order (n, Cp, Ci, NULL, Control, Info, NULL) ;
	OK (ok == CAMD_INVALID);

	if (n > 0)
	{
	    printf ("CAMD error tests:\n") ;

	    p = Cp [n] ;
	    Cp [n] = -1 ;
	    ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
	    OK (ok == CAMD_INVALID) ;

	    if (Size_max/2 == Int_max)
	    {
		Cp [n] = Int_max ;
		ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
		printf ("CAMD status is "ID"\n", ok) ;
		OK (ok == CAMD_OUT_OF_MEMORY) ;
	    }

	    Cp [n] = p ;
	    ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
	    OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	    if (Cp [n] > 0)
	    {
		printf ("Mangle column zero:\n") ;
		i = Ci [0] ;
		Ci [0] = -1 ;
		ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
		CAMD_info (Info) ;
		OK (ok == CAMD_INVALID) ;
		Ci [0] = i ;
	    }
	}

	ok = CAMD_valid (n, n, Sp, Si) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	ok = CAMD_valid (-1, n, Sp, Si) ;	    OK (ok == CAMD_INVALID) ;
	ok = CAMD_valid (n, -1, Sp, Si) ;	    OK (ok == CAMD_INVALID) ;
	ok = CAMD_valid (n, n, NULL, Si) ;	    OK (ok == CAMD_INVALID) ;
	ok = CAMD_valid (n, n, Sp, NULL) ;	    OK (ok == CAMD_INVALID) ;

	if (n > 0 && Sp [n] > 0)
	{

	    p = Sp [n] ;
	    Sp [n] = -1 ;
	    ok = CAMD_valid (n, n, Sp, Si) ; OK (ok == CAMD_INVALID) ;
	    Sp [n] = p ;

	    p = Sp [0] ;
	    Sp [0] = -1 ;
	    ok = CAMD_valid (n, n, Sp, Si) ; OK (ok == CAMD_INVALID) ;
	    Sp [0] = p ;

	    p = Sp [1] ;
	    Sp [1] = -1 ;
	    ok = CAMD_valid (n, n, Sp, Si) ; OK (ok == CAMD_INVALID) ;
	    Sp [1] = p ;

	    i = Si [0] ;
	    Si [0] = -1 ;
	    ok = CAMD_valid (n, n, Sp, Si) ; OK (ok == CAMD_INVALID) ;
	    Si [0] = i ;

	}

	ok = CAMD_valid (n, n, Sp, Si) ;
	OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;
	CAMD_preprocess (n, Bp, Bi, Sp, Si, W, Flag) ;
	ok = CAMD_valid (n, n, Sp, Si) ;
	OK (ok == CAMD_OK) ;

	if (n > 0 && Bp [n] > 0)
	{

	    p = Bp [n] ;
	    Bp [n] = -1 ;
	    ok = CAMD_valid (n, n, Bp, Bi) ;	    OK (ok == CAMD_INVALID) ;
	    Bp [n] = p ;


	    p = Bp [1] ;
	    Bp [1] = -1 ;
	    ok = CAMD_valid (n, n, Bp, Bi) ;	    OK (ok == CAMD_INVALID) ;
	    Bp [1] = p ;

	    i = Bi [0] ;
	    Bi [0] = -1 ;
	    ok = CAMD_valid (n, n, Bp, Bi) ;	    OK (ok == CAMD_INVALID) ;
	    Bi [0] = i ;
	}

	CAMD_preprocess (n, Bp, Bi, Sp, Si, W, Flag) ;

	Info [CAMD_STATUS] = 777 ;
	CAMD_info (Info) ;

	/* ------------------------------------------------------------------ */
	/* memory tests */
	/* ------------------------------------------------------------------ */

	if (n > 0)
	{

	    normal_memory_handler ( ) ;

	    ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
	    OK (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK)) ;

	    test_memory_handler ( ) ;

	    for (trial = 0 ; trial < 6 ; trial++)
	    {
		my_tries = trial ;
		printf ("CAMD memory trial "ID"\n", trial) ;
		ok = CAMD_order (n, Cp, Ci, P, Control, Info, NULL) ;
		CAMD_info (Info) ;
		OK (ok == CAMD_OUT_OF_MEMORY
		    || (sorted ? (ok == CAMD_OK) : (ok >= CAMD_OK))) ;
	    }
	    normal_memory_handler ( ) ;
	    OK (CHOLMOD(print_perm) (P, n, n, "CAMD2 permutation", cm)) ;

	}

	CHOLMOD(free_sparse) (&E, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free everything */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(free) (n, sizeof (Int), Len,  cm) ;
    CHOLMOD(free) (n, sizeof (Int), Nv,   cm) ;
    CHOLMOD(free) (n, sizeof (Int), Next, cm) ;
    CHOLMOD(free) (n+1, sizeof (Int), Head, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Elen, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Deg,  cm) ;
    CHOLMOD(free) (n+1, sizeof (Int), Wi, cm) ;
    CHOLMOD(free) (n, sizeof (Int), BucketSet, cm) ;

    CHOLMOD(free) (n+1, sizeof (Int), P, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Constraint, cm) ;

    CHOLMOD(free) (n, sizeof (Int), W, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Flag, cm) ;

    CHOLMOD(free_sparse) (&S, cm) ;
    CHOLMOD(free_sparse) (&B, cm) ;
    CHOLMOD(free_sparse) (&C, cm) ;
    CHOLMOD(free_sparse) (&F, cm) ;
}

#else

void camdtest (cholmod_sparse *A)
{
    if (A == NULL)
    {
	return ;
    }
    cm->print = 1 ;
}
#endif
