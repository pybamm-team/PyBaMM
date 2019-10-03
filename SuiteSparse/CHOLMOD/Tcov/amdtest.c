/* ========================================================================== */
/* === Tcov/amdtest ========================================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Test for amd v2.0 */

#include "cm.h"
#include "amd.h"


/* ========================================================================== */
/* === amdtest ============================================================== */
/* ========================================================================== */

void amdtest (cholmod_sparse *A)
{
    double Control [AMD_CONTROL], Info [AMD_INFO], alpha ;
    Int *P, *Cp, *Ci, *Sp, *Si, *Bp, *Bi, *Ep, *Ei, *Fp, *Fi,
	*Len, *Nv, *Next, *Head, *Elen, *Deg, *Wi, *W, *Flag ;
    cholmod_sparse *C, *B, *S, *E, *F ;
    Int i, j, n, nrow, ncol, ok, cnz, bnz, p, trial, sorted ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    printf ("\nAMD test\n") ;

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
    /* S = sorted form of B, using AMD_preprocess */
    /* ---------------------------------------------------------------------- */

    cnz = CHOLMOD(nnz) (C, cm) ;
    S = CHOLMOD(allocate_sparse) (n, n, cnz, TRUE, TRUE, 0, CHOLMOD_PATTERN,
	    cm);
    Sp = S->p ;
    Si = S->i ;

    W = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Flag = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    AMD_preprocess (n, Bp, Bi, Sp, Si, W, Flag) ;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace for amd */
    /* ---------------------------------------------------------------------- */

    P = CHOLMOD(malloc) (n+1, sizeof (Int), cm) ;

    Len  = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Nv   = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Next = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Head = CHOLMOD(malloc) (n+1, sizeof (Int), cm) ;
    Elen = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Deg  = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Wi   = CHOLMOD(malloc) (n, sizeof (Int), cm) ;

    /* ---------------------------------------------------------------------- */

    for (sorted = 0 ; sorted <= 1 ; sorted++)
    {

	if (sorted) CHOLMOD(sort) (C, cm) ;

	Cp = C->p ;
	Ci = C->i ;

	/* ------------------------------------------------------------------ */
	/* order C with AMD_order */
	/* ------------------------------------------------------------------ */

	AMD_defaults (Control) ;
	AMD_defaults (NULL) ;
	AMD_control (Control) ;
	AMD_control (NULL) ;
	AMD_info (NULL) ;

	ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
	printf ("amd return value: "ID"\n", ok) ;
	AMD_info (Info) ;
	OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "AMD permutation", cm)) ;

	/* no dense rows/cols */
	alpha = Control [AMD_DENSE] ;
	Control [AMD_DENSE] = -1 ;
	AMD_control (Control) ;
	ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
	printf ("amd return value: "ID"\n", ok) ;
	AMD_info (Info) ;
	OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "AMD permutation (alpha=-1)", cm)) ;

	/* many dense rows/cols */
	Control [AMD_DENSE] = 0 ;
	AMD_control (Control) ;
	ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
	printf ("amd return value: "ID"\n", ok) ;
	AMD_info (Info) ;
	OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "AMD permutation (alpha=0)", cm)) ;
	Control [AMD_DENSE] = alpha ;

	/* no aggressive absorption */
	Control [AMD_AGGRESSIVE] = FALSE ;
	AMD_control (Control) ;
	ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
	printf ("amd return value: "ID"\n", ok) ;
	AMD_info (Info) ;
	OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "AMD permutation (no agg) ", cm)) ;
	Control [AMD_AGGRESSIVE] = TRUE ;

	/* ------------------------------------------------------------------ */
	/* order F with AMD_order */
	/* ------------------------------------------------------------------ */

	Fp = F->p ;
	Fi = F->i ;
	ok = AMD_order (n, Fp, Fi, P, Control, Info) ;
	printf ("amd return value: "ID"\n", ok) ;
	AMD_info (Info) ;
	OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "F: AMD permutation", cm)) ;

	/* ------------------------------------------------------------------ */
	/* order S with AMD_order */
	/* ------------------------------------------------------------------ */

	ok = AMD_order (n, Sp, Si, P, Control, Info) ;
	printf ("amd return value: "ID"\n", ok) ;
	AMD_info (Info) ;
	OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	OK (CHOLMOD(print_perm) (P, n, n, "AMD permutation", cm)) ;

	/* ------------------------------------------------------------------ */
	/* order E with AMD_2, which destroys its contents */
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

	printf ("calling AMD_2:\n") ;
	if (n > 0)
	{
	    AMD_2 (n, Ep, Ei, Len, E->nzmax, Ep [n], Nv, Next, P, Head, Elen,
		Deg, Wi, Control, Info) ;
	    AMD_info (Info) ;
	    OK (CHOLMOD(print_perm) (P, n, n, "AMD2 permutation", cm)) ;
	}

	/* ------------------------------------------------------------------ */
	/* error tests */
	/* ------------------------------------------------------------------ */

	ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
	OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	ok = AMD_order (-1, Cp, Ci, P, Control, Info) ;
	OK (ok == AMD_INVALID);
	ok = AMD_order (0, Cp, Ci, P, Control, Info) ;
	OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	ok = AMD_order (n, NULL, Ci, P, Control, Info) ;
	OK (ok == AMD_INVALID);
	ok = AMD_order (n, Cp, NULL, P, Control, Info) ;
	OK (ok == AMD_INVALID);
	ok = AMD_order (n, Cp, Ci, NULL, Control, Info) ;
	OK (ok == AMD_INVALID);

	if (n > 0)
	{
	    printf ("AMD error tests:\n") ;

	    p = Cp [n] ;
	    Cp [n] = -1 ;
	    ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
	    OK (ok == AMD_INVALID) ;

	    if (Size_max/2 == Int_max)
	    {
		Cp [n] = Int_max ;
		ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
		printf ("AMD status is "ID"\n", ok) ;
		OK (ok == AMD_OUT_OF_MEMORY) ;
	    }

	    Cp [n] = p ;
	    ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
	    OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	    if (Cp [n] > 0)
	    {
		printf ("Mangle column zero:\n") ;
		i = Ci [0] ;
		Ci [0] = -1 ;
		ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
		AMD_info (Info) ;
		OK (ok == AMD_INVALID) ;
		Ci [0] = i ;
	    }
	}

	ok = AMD_valid (n, n, Sp, Si) ;
	OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	ok = AMD_valid (-1, n, Sp, Si) ;	    OK (ok == AMD_INVALID) ;
	ok = AMD_valid (n, -1, Sp, Si) ;	    OK (ok == AMD_INVALID) ;
	ok = AMD_valid (n, n, NULL, Si) ;	    OK (ok == AMD_INVALID) ;
	ok = AMD_valid (n, n, Sp, NULL) ;	    OK (ok == AMD_INVALID) ;

	if (n > 0 && Sp [n] > 0)
	{

	    p = Sp [n] ;
	    Sp [n] = -1 ;
	    ok = AMD_valid (n, n, Sp, Si) ; OK (ok == AMD_INVALID) ;
	    Sp [n] = p ;

	    p = Sp [0] ;
	    Sp [0] = -1 ;
	    ok = AMD_valid (n, n, Sp, Si) ; OK (ok == AMD_INVALID) ;
	    Sp [0] = p ;

	    p = Sp [1] ;
	    Sp [1] = -1 ;
	    ok = AMD_valid (n, n, Sp, Si) ; OK (ok == AMD_INVALID) ;
	    Sp [1] = p ;

	    i = Si [0] ;
	    Si [0] = -1 ;
	    ok = AMD_valid (n, n, Sp, Si) ; OK (ok == AMD_INVALID) ;
	    Si [0] = i ;

	}

	ok = AMD_valid (n, n, Sp, Si) ;
	OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;
	AMD_preprocess (n, Bp, Bi, Sp, Si, W, Flag) ;
	ok = AMD_valid (n, n, Sp, Si) ;
	OK (ok == AMD_OK) ;

	if (n > 0 && Bp [n] > 0)
	{

	    p = Bp [n] ;
	    Bp [n] = -1 ;
	    ok = AMD_valid (n, n, Bp, Bi) ;	    OK (ok == AMD_INVALID) ;
	    Bp [n] = p ;


	    p = Bp [1] ;
	    Bp [1] = -1 ;
	    ok = AMD_valid (n, n, Bp, Bi) ;	    OK (ok == AMD_INVALID) ;
	    Bp [1] = p ;

	    i = Bi [0] ;
	    Bi [0] = -1 ;
	    ok = AMD_valid (n, n, Bp, Bi) ;	    OK (ok == AMD_INVALID) ;
	    Bi [0] = i ;
	}

	AMD_preprocess (n, Bp, Bi, Sp, Si, W, Flag) ;

	Info [AMD_STATUS] = 777 ;
	AMD_info (Info) ;

	/* ------------------------------------------------------------------ */
	/* memory tests */
	/* ------------------------------------------------------------------ */

	if (n > 0)
	{

	    normal_memory_handler ( ) ;

	    ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
	    OK (sorted ? (ok == AMD_OK) : (ok >= AMD_OK)) ;

	    test_memory_handler ( ) ;

	    for (trial = 0 ; trial < 6 ; trial++)
	    {
		my_tries = trial ;
		printf ("AMD memory trial "ID"\n", trial) ;
		ok = AMD_order (n, Cp, Ci, P, Control, Info) ;
		AMD_info (Info) ;
		OK (ok == AMD_OUT_OF_MEMORY
		    || (sorted ? (ok == AMD_OK) : (ok >= AMD_OK))) ;
	    }
	    normal_memory_handler ( ) ;
	    OK (CHOLMOD(print_perm) (P, n, n, "AMD2 permutation", cm)) ;

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
    CHOLMOD(free) (n, sizeof (Int), Wi,   cm) ;

    CHOLMOD(free) (n+1, sizeof (Int), P, cm) ;

    CHOLMOD(free) (n, sizeof (Int), W, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Flag, cm) ;

    CHOLMOD(free_sparse) (&S, cm) ;
    CHOLMOD(free_sparse) (&B, cm) ;
    CHOLMOD(free_sparse) (&C, cm) ;
    CHOLMOD(free_sparse) (&F, cm) ;
}
