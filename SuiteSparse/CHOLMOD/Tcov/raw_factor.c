/* ========================================================================== */
/* === Tcov/raw_factor ====================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Factorize A using cholmod_rowfac for the simplicial case, and the 
 * cholmod_super_* routines for the supernodal case, and test the solution to
 * linear systems. */

#include "cm.h"


/* ========================================================================== */
/* === icomp ================================================================ */
/* ========================================================================== */

/* for sorting by qsort */
static int icomp (Int *i, Int *j)
{
    if (*i < *j)
    {
	return (-1) ;
    }
    else
    {
	return (1) ;
    }
}


/* ========================================================================== */
/* === add_gunk ============================================================= */
/* ========================================================================== */

static cholmod_sparse *add_gunk (cholmod_sparse *A)
{
    cholmod_sparse *S ;
    double *Sx, *Sz ;
    Int *Sp, *Si, nz, p, save3, j, n ;

    if (A == NULL) return (NULL) ;

    /* save3 = cm->print ; cm->print = 5 ; */

    A->nzmax++ ;
    S = CHOLMOD(copy_sparse) (A, cm) ;
    A->nzmax-- ;

    /* add a S(n,1)=1 entry to the matrix */
    if (S != NULL)
    {
	S->sorted = FALSE ;
	Sx = S->x ;
	Si = S->i ;
	Sp = S->p ;
	Sz = S->z ;
	n = S->ncol ;
	nz = Sp [n] ;
	for (j = 1 ; j <= n ; j++)
	{
	    Sp [j]++ ;
	}
	if (S->xtype == CHOLMOD_REAL)
	{
	    for (p = nz-1 ; p >= 0 ; p--)
	    {
		Si [p+1] = Si [p] ;
		Sx [p+1] = Sx [p] ;
	    }
	    Si [0] = n-1 ;
	    Sx [0] = 99999 ;
	}
	else if (S->xtype == CHOLMOD_COMPLEX)
	{
	    for (p = nz-1 ; p >= 0 ; p--)
	    {
		Si [p+1] = Si [p] ;
		Sx [2*p+2] = Sx [2*p] ;
		Sx [2*p+3] = Sx [2*p+1] ;
	    }
	    Si [0] = n-1 ;
	    Sx [0] = 99999 ;
	    Sx [1] = 0 ;
	}
	else if (S->xtype == CHOLMOD_ZOMPLEX)
	{
	    for (p = nz-1 ; p >= 0 ; p--)
	    {
		Si [p+1] = Si [p] ;
		Sx [p+1] = Sx [p] ;
		Sz [p+1] = Sz [p] ;
	    }
	    Si [0] = n-1 ;
	    Sx [0] = 99999 ;
	    Sz [0] = 0 ;
	}
    }

    /* CHOLMOD(print_sparse) (A, "A for gunk", cm) ; */
    /* CHOLMOD(print_sparse) (S, "S with gunk", cm) ; */
    /* cm->print = save3 ; */

    return (S) ;
}


/* ========================================================================== */
/* === raw_factor =========================================================== */
/* ========================================================================== */

/* Factor A, without using any fill-reducing permutation.  This may fail due
 * to catastrophic fill-in (which is the desired test result for a large
 * arrowhead matrix).
 */

double raw_factor (cholmod_sparse *A, Int check_errors)
{
    double maxerr = 0, r, anorm ;
    cholmod_sparse *AT, *C, *LT, *Lsparse, *S, *ST, *R, *A1 ;
    cholmod_factor *L, *Lcopy ;
    cholmod_dense *X, *W, *B, *X2 ;
    Int i, k, n, ok, ok1, ok2, trial, rnz, lnz, Lxtype, Axtype, posdef,
	prefer_zomplex, Bxtype ;
    Int *Parent, *Post, *First, *Level, *Ri, *Rp, *LTp = NULL, *LTi = NULL, *P,
	*mask, *RLinkUp ;
    SuiteSparse_long lr ;
    double beta [2] ;
    unsigned SuiteSparse_long save ;

    /* ---------------------------------------------------------------------- */
    /* create the problem */
    /* ---------------------------------------------------------------------- */

    if (A == NULL || A->stype != 1)
    {
	return (0) ;
    }

    W = NULL ;
    X2 = NULL ;
    L = NULL ;
    n = A->nrow ;
    B = rhs (A, 1, n) ;
    AT = CHOLMOD(transpose) (A, 2, cm) ;
    Parent = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Post = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    First = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    Level = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    beta [0] = 0 ;
    beta [1] = 0 ;
    anorm = CHOLMOD(norm_sparse) (A, 1, cm) ;

    prefer_zomplex = (A->xtype == CHOLMOD_ZOMPLEX) ;
    Bxtype = A->xtype ;

    /* ---------------------------------------------------------------------- */
    /* supernodal factorization */
    /* ---------------------------------------------------------------------- */

    L = CHOLMOD(allocate_factor) (n, cm) ;
    ok1 = CHOLMOD(etree) (A, Parent, cm) ;
    lr = CHOLMOD(postorder) (Parent, n, NULL, Post, cm) ;
    ok2 = CHOLMOD(rowcolcounts) (AT, NULL, 0, Parent, Post,
	NULL, (L != NULL) ? (L->ColCount) : NULL, First, Level, cm) ;

    if (ok2)
    {
	printf ("raw_factor: cm->fl %g cm->lnz %g\n", cm->fl, cm->lnz) ;
    }

    if (check_errors)
    {
	OKP (AT) ;
	OKP (Parent) ;
	OKP (Post) ;
	OKP (First) ;
	OKP (Level) ;
	OK (AT->stype == -1) ;
	OKP (L) ;
	OK (ok1) ;
	OK (ok2) ;
	OK (lr >= 0) ;

	/* rowcolcounts requires A in symmetric lower form */
	ok = CHOLMOD(rowcolcounts) (A, NULL, 0, Parent, Post,
	    NULL, L->ColCount, First, Level, cm) ;		    NOT (ok) ;
    }

    /* super_symbolic needs A in upper form, so this will succeed
     * unless the problem is huge */
    ok = CHOLMOD(super_symbolic) (A, NULL, Parent, L, cm) ;

    /* super_symbolic should fail if lnz is too large */
    if (cm->lnz > Size_max / 2)
    {
	printf ("raw_factor: problem is huge\n") ;
	NOT (ok) ;
	OK (L->xtype == CHOLMOD_PATTERN && !(L->is_super)) ;

	/* try changing to LDL packed, which should also fail */
	ok = CHOLMOD(change_factor) (CHOLMOD_REAL, FALSE, FALSE, TRUE, TRUE,
		L, cm) ;
	NOT (ok) ;

	CHOLMOD(free_factor) (&L, cm) ;
	CHOLMOD(free_sparse) (&AT, cm) ;
	CHOLMOD(free_dense) (&B, cm) ;
	CHOLMOD(free) (n, sizeof (Int), First, cm) ;
	CHOLMOD(free) (n, sizeof (Int), Level, cm) ;
	CHOLMOD(free) (n, sizeof (Int), Parent, cm) ;
	CHOLMOD(free) (n, sizeof (Int), Post, cm) ;
	return (0) ;
    }

    if (check_errors)
    {

	if (cm->status == CHOLMOD_OUT_OF_MEMORY
	 || cm->status == CHOLMOD_TOO_LARGE)
	{
	    /* no test case will reach here, but check just to be safe */
	    printf ("raw_factor: out of memory for symbolic case %d\n",
		cm->status) ;
	    CHOLMOD(free_factor) (&L, cm) ;
	    CHOLMOD(free_sparse) (&AT, cm) ;
	    CHOLMOD(free_dense) (&B, cm) ;
	    CHOLMOD(free) (n, sizeof (Int), First, cm) ;
	    CHOLMOD(free) (n, sizeof (Int), Level, cm) ;
	    CHOLMOD(free) (n, sizeof (Int), Parent, cm) ;
	    CHOLMOD(free) (n, sizeof (Int), Post, cm) ;
	    return (0) ;
	}

	OK (ok) ;
	ok = CHOLMOD(super_symbolic)(A, NULL, Parent, L, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(super_symbolic)(AT, NULL, Parent, L, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(super_symbolic)(NULL, NULL, Parent, L, cm) ;   NOT (ok) ;
	ok = CHOLMOD(super_symbolic)(A, NULL, NULL, L, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(super_symbolic)(A, NULL, Parent, NULL, cm) ;   NOT (ok) ;
    }

    /* super_numeric needs A in lower form, so this will succeed unless
     * the problem is huge */
    ok = CHOLMOD(super_numeric) (AT, NULL, Zero, L, cm) ;

    if (check_errors)
    {

	if (cm->status == CHOLMOD_OUT_OF_MEMORY)
	{
	    /* For the 64-bit case, the Matrix/a1 problem will reach here */
	    printf ("raw_factor: out of memory for numeric case\n") ;
	    CHOLMOD(free_factor) (&L, cm) ;
	    CHOLMOD(free_sparse) (&AT, cm) ;
	    CHOLMOD(free_dense) (&B, cm) ;
	    CHOLMOD(free) (n, sizeof (Int), First, cm) ;
	    CHOLMOD(free) (n, sizeof (Int), Level, cm) ;
	    CHOLMOD(free) (n, sizeof (Int), Parent, cm) ;
	    CHOLMOD(free) (n, sizeof (Int), Post, cm) ;
	    return (0) ;
	}

	OK (ok) ;
	ok = CHOLMOD(super_numeric)(A, NULL, Zero, L, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(super_numeric)(NULL, NULL, Zero, L, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(super_numeric)(AT, NULL, Zero, NULL, cm) ;	    NOT (ok) ;
    }

    /* solve */
    Lxtype = (L == NULL) ? CHOLMOD_REAL : L->xtype ;
    W = CHOLMOD(zeros) (n, 1, Lxtype, cm) ;
    X = CHOLMOD(copy_dense) (B, cm) ;
    if (Bxtype == CHOLMOD_ZOMPLEX)
    {
	CHOLMOD(dense_xtype) (CHOLMOD_COMPLEX, X, cm) ;
    }

    CHOLMOD(print_factor) (L, "L for super l/ltsolve", cm) ;
    CHOLMOD(print_dense) (W, "W", cm) ;
    CHOLMOD(print_dense) (X, "X", cm) ;

    ok1 = CHOLMOD(super_lsolve) (L, X, W, cm) ;
    CHOLMOD(print_dense) (X, "X", cm) ;

    ok2 = CHOLMOD(super_ltsolve) (L, X, W, cm) ;
    CHOLMOD(print_dense) (X, "X", cm) ;

    if (Bxtype == CHOLMOD_ZOMPLEX)
    {
	CHOLMOD(dense_xtype) (CHOLMOD_ZOMPLEX, X, cm) ;
    }

    r = resid (A, X, B) ;
    MAXERR (maxerr, r, 1) ;

    if (Bxtype == CHOLMOD_ZOMPLEX)
    {
	CHOLMOD(dense_xtype) (CHOLMOD_COMPLEX, X, cm) ;
    }

    if (check_errors)
    {
	OKP (W) ;
	OKP (X) ;
	OK (ok1) ;
	OK (ok2) ;
	ok = CHOLMOD(super_lsolve) (NULL, X, W, cm) ;		    NOT (ok) ;
	ok = CHOLMOD(super_ltsolve) (NULL, X, W, cm) ;		    NOT (ok) ;
	ok = CHOLMOD(super_lsolve) (L, NULL, W, cm) ;		    NOT (ok) ;
	ok = CHOLMOD(super_ltsolve) (L, NULL, W, cm) ;		    NOT (ok) ;
	ok = CHOLMOD(super_lsolve) (L, X, NULL, cm) ;		    NOT (ok) ;
	ok = CHOLMOD(super_ltsolve) (L, X, NULL, cm) ;		    NOT (ok) ;

	if (L != NULL && L->maxesize > 1)
	{
	    /* W is too small */
	    ok = CHOLMOD(free_dense) (&W, cm) ;			    OK (ok) ;
	    W = CHOLMOD(zeros) (1, 1, Lxtype, cm) ;		    OKP (W) ;
	    ok = CHOLMOD(super_lsolve) (L, X, W, cm) ;		    NOT (ok) ;
	    ok = CHOLMOD(super_ltsolve) (L, X, W, cm) ;		    NOT (ok) ;
	    ok = CHOLMOD(free_dense) (&W, cm) ;			    OK (ok) ;
	    W = CHOLMOD(zeros) (n, 1, Lxtype, cm) ;		    OKP (W) ;
	}

	/* X2 has the wrong dimensions */
	X2 = CHOLMOD(zeros) (n+1, 1, Lxtype, cm) ;		    OKP (X2) ;
	ok = CHOLMOD(super_lsolve) (L, X2, W, cm) ;		    NOT (ok) ;
	ok = CHOLMOD(super_ltsolve) (L, X2, W, cm) ;		    NOT (ok) ;
	CHOLMOD(free_dense) (&X2, cm) ;
    }

    CHOLMOD(free_dense) (&X, cm) ;

    /* X2 is n-by-0, which is OK */
    X2 = CHOLMOD(zeros) (n, 0, Lxtype, cm) ;
    ok1 = CHOLMOD(super_lsolve) (L, X2, W, cm) ;
    ok2 = CHOLMOD(super_ltsolve) (L, X2, W, cm) ;
    CHOLMOD(free_dense) (&W, cm) ;
    CHOLMOD(free_dense) (&X2, cm) ;

    if (check_errors)
    {
	OK (ok1) ;
	OK (ok2) ;
	test_memory_handler ( ) ;
	my_tries = 0 ;
	ok = CHOLMOD(super_symbolic) (A, NULL, Parent, L, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(super_numeric) (AT, NULL, Zero, L, cm) ;	    NOT (ok) ;
	normal_memory_handler ( ) ;
	cm->error_handler = NULL ;
    }

    /* R = space for result of row_subtree and row_lsubtree */
    R = CHOLMOD(allocate_sparse)(n, 1, n, FALSE, TRUE, 0, CHOLMOD_PATTERN, cm) ;

    /* ---------------------------------------------------------------------- */
    /* erroneous factorization */
    /* ---------------------------------------------------------------------- */

    /* cannot use rowfac or row_lsubtree on a supernodal factorization */
    if (check_errors && n > 0)
    {
	ok = CHOLMOD(rowfac) (A, NULL, beta, 0, 0, L, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(row_lsubtree) (A, &i, 0, n-1, L, R, cm) ;	    NOT (ok) ;
    }

    /* ---------------------------------------------------------------------- */
    /* convert to simplicial LDL' */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(change_factor) (Lxtype, FALSE, FALSE, TRUE, TRUE, L, cm) ;

    /* remove entries due to relaxed supernodal amalgamation */
    CHOLMOD(resymbol) (A, NULL, 0, TRUE, L, cm) ;

    /* refactorize a numeric factor */
    posdef = 0 ;   /* unknown */
    if (A != NULL && A->stype >= 0)
    {
	if (A->stype > 0 && A->packed)
	{
	    S = add_gunk (A) ;
	    CHOLMOD(rowfac) (S, NULL, beta, 0, n, L, cm) ;
	    if (S && S->xtype == CHOLMOD_COMPLEX)
	    {
		CHOLMOD(sparse_xtype) (CHOLMOD_ZOMPLEX, S, cm) ;
	    }
	    ok = CHOLMOD(free_sparse) (&S, cm) ;			    OK (ok) ;
	}
	else
	{
	    CHOLMOD(rowfac) (A, NULL, beta, 0, n, L, cm) ;
	}
	posdef = (cm->status == CHOLMOD_OK) ;
    }

    /* convert to a sparse matrix, and transpose L */
    Lcopy = CHOLMOD(copy_factor)(L, cm) ;
    Lsparse = CHOLMOD(factor_to_sparse) (L, cm) ;

    LT = CHOLMOD(transpose) (Lsparse, 0, cm) ;
    CHOLMOD(free_sparse) (&Lsparse, cm) ;

    if (LT != NULL)
    {
	LTp = LT->p ;
	LTi = LT->i ;
	OK (LT->packed) ;
    }

    /* remove the unit diagonal of LT */
    CHOLMOD(band_inplace) (1, n, -1, LT, cm) ;

    /* ST = pattern of A(p,p)' */
    P = (L == NULL) ? NULL : L->Perm ;
    ST = CHOLMOD(ptranspose) (A, 0, P, NULL, 0, cm) ;

    /* S = pattern of A(p,p) */
    S = CHOLMOD(transpose) (ST, 0, cm) ;
    ok = CHOLMOD(free_sparse) (&ST, cm) ;

    if (R != NULL && LT != NULL && posdef && A != NULL && A->stype >= 0
	    && S != NULL && Lcopy != NULL)
    {
	LTp = LT->p ;
	LTi = LT->i ;

	Ri = R->i ;
	Rp = R->p ;

	save = my_seed ( ) ;					/* RAND */
	for (trial = 0 ; trial < 30 ; trial++)
	{
	    /* pick a row at random */
	    i = nrand (n) ;					/* RAND */

	    /* compute R = pattern of L(i,0:i-1), using row subtrees */
	    ok = CHOLMOD(row_subtree) (S, NULL, i, Parent, R, cm) ;
	    if (!ok)
	    {
		break ;
	    }
	    rnz = Rp [1] ;

	    /* sort R */
	    qsort (Ri, rnz, sizeof (Int),
		    (int (*) (const void *, const void *)) icomp) ;

	    /* compare with ith column of L transpose */
	    lnz = LTp [i+1] - LTp [i] ;
	    ok = TRUE ;
	    for (k = 0 ; k < MIN (rnz,lnz) ; k++)
	    {
		/* printf ("%d vs %d\n", Ri [k], LTi [LTp [i] + k]) ; */
		ok = ok && (Ri [k] == LTi [LTp [i] + k]) ;
	    }
	    OK (ok) ;
	    OK (rnz == lnz) ;

	    /* compute R = pattern of L(i,0:i-1), using row lsubtrees */
	    ok = CHOLMOD(row_lsubtree) (S, NULL, 0, i, Lcopy, R, cm) ;
	    if (!ok)
	    {
		break ;
	    }
	    rnz = Rp [1] ;

	    /* sort R */
	    qsort (Ri, rnz, sizeof (Int),
		    (int (*) (const void *, const void *)) icomp) ;

	    /* compare with ith column of L transpose */
	    lnz = LTp [i+1] - LTp [i] ;
	    ok = TRUE ;
	    for (k = 0 ; k < MIN (rnz,lnz) ; k++)
	    {
		/* printf ("%d vs %d\n", Ri [k], LTi [LTp [i] + k]) ; */
		ok = ok && (Ri [k] == LTi [LTp [i] + k]) ;
	    }
	    OK (ok) ;
	    OK (rnz == lnz) ;

	    /* L is symbolic, so cholmod_lsubtree will fail */
	    if (check_errors)
	    {
		ok = CHOLMOD(row_lsubtree) (S, NULL, 0, i, L, R, cm) ;
		NOT (ok) ;
	    }

	}
	my_srand (save) ;					/* RAND */
    }

    ok = CHOLMOD(free_factor) (&L, cm) ;			    OK (ok) ;
    ok = CHOLMOD(free_factor) (&Lcopy, cm) ;			    OK (ok) ;
    ok = CHOLMOD(free_sparse) (&LT, cm) ;			    OK (ok) ;
    ok = CHOLMOD(free_sparse) (&R, cm) ;			    OK (ok) ;
    ok = CHOLMOD(free_sparse) (&S, cm) ;			    OK (ok) ;

    /* ---------------------------------------------------------------------- */
    /* simplicial LDL' or LL' factorization with no analysis */
    /* ---------------------------------------------------------------------- */

    for (trial = 0 ; trial <= check_errors ; trial++)
    {

	/* create a simplicial symbolic factor */
	L = CHOLMOD(allocate_factor) (n, cm) ;
	ok = TRUE ;
	Axtype = (A == NULL) ? CHOLMOD_REAL : A->xtype ;

	if (check_errors)
	{
	    OKP (L) ;
	    if (trial == 0)
	    {
		/* convert to packed LDL' first, then unpacked */
		ok = CHOLMOD(change_factor) (Axtype, FALSE, FALSE, TRUE,
			TRUE, L, cm) ;
		OK (ok);
		ok = CHOLMOD(change_factor) (Axtype, FALSE, FALSE, FALSE,
			TRUE, L, cm) ;
		OK (ok) ;
	    }
	    else if (trial == 1)
	    {
		ok = CHOLMOD(rowfac)(NULL, NULL, beta, 0, 0, L, cm) ; NOT (ok) ;
		ok = CHOLMOD(rowfac)(A, NULL, beta, 0, 0, NULL, cm) ; NOT (ok) ;
		ok = CHOLMOD(rowfac)(AT, NULL, beta, 0, 0, L, cm) ;   NOT (ok) ;
		if (n > 1)
		{
		    A1 = CHOLMOD(allocate_sparse)(1, 1, 1, TRUE, TRUE, 1,
			CHOLMOD_PATTERN, cm) ;			      OKP (A1) ;
		    ok = CHOLMOD(rowfac)(A1, NULL, beta, 0, 0, L, cm); NOT (ok);
		    ok = CHOLMOD(free_sparse)(&A1, cm) ;		OK (ok);
		}
	    }
	    else
	    {
		/* convert to symbolic LL' */
		ok = CHOLMOD(change_factor) (CHOLMOD_PATTERN, TRUE, FALSE, TRUE,
			TRUE, L, cm) ;
		OK (ok) ;
		OK (L->is_ll) ;
	    }
	}

	/* factor */
	CHOLMOD(print_factor) (L, "L for rowfac", cm) ;
	CHOLMOD(print_sparse) (A, "A for rowfac", cm) ;

	cm->dbound = 1e-15 ;
	for (k = 0 ; ok && k < n ; k++)
	{
	    if (!CHOLMOD(rowfac) (A, NULL, beta, k, k+1, L, cm))
	    {
		ok = FALSE ;
	    }
	    if (cm->status == CHOLMOD_NOT_POSDEF)
	    {
		/* LL' factorization failed; subsequent rowfac's should fail */
		k++ ;
		ok = CHOLMOD(rowfac) (A, NULL, beta, k, k+1, L, cm) ;
		NOT (ok) ;
		ok = TRUE ;
	    }
	}
	cm->dbound = 0 ;

	if (check_errors)
	{
	    OK (ok) ;
	    ok = CHOLMOD(rowfac) (A, NULL, beta, n, n+1, L, cm) ;    NOT (ok) ;
	    ok = TRUE ;
	}

	/* solve */
	if (ok)
	{

/*
int saveit = cm->print ;
int saveit2 = cm->precise ;
cm->print = 5 ;
cm->precise = TRUE ;

CHOLMOD (print_sparse) (A, "A here", cm) ;
CHOLMOD (print_factor) (L, "L here", cm) ;
CHOLMOD (print_dense) (B, "B here", cm) ;
*/

	    cm->prefer_zomplex = prefer_zomplex ;
	    X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;

/*
CHOLMOD (print_dense) (X, "X here", cm) ;
*/

	    cm->prefer_zomplex = FALSE ;
	    r = resid (A, X, B) ;
	    MAXERR (maxerr, r, 1) ;
	    CHOLMOD(free_dense) (&X, cm) ;

/*
cm->print = saveit ;
cm->precise = saveit2 ;
fprintf (stderr, "solve %8.2e\n", r) ;
*/
	}

	CHOLMOD(free_factor) (&L, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* factor again with entries in the (ignored) lower part A */
    /* ---------------------------------------------------------------------- */

    if (A->packed)
    {
	L = CHOLMOD(allocate_factor) (n, cm) ;
	C = add_gunk (A) ;

/*
	C = CHOLMOD(copy) (A, 0, 1, cm) ;
	if (C != NULL)
	{
	    C->stype = 1 ;
	}
*/

	CHOLMOD(rowfac) (C, NULL, beta, 0, n, L, cm) ;

	X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;

	r = resid (A, X, B) ;
	MAXERR (maxerr, r, 1) ;
	CHOLMOD(free_sparse) (&C, cm) ;
	CHOLMOD(free_factor) (&L, cm) ;
	CHOLMOD(free_dense) (&X, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* factor again using rowfac_mask (for LPDASA only) */
    /* ---------------------------------------------------------------------- */

    r = raw_factor2 (A, 0., 0) ;
    MAXERR (maxerr, r, 1) ;

    r = raw_factor2 (A, 1e-16, 0) ;
    MAXERR (maxerr, r, 1) ;

    /* ---------------------------------------------------------------------- */
    /* free the problem */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(free_sparse) (&AT, cm) ;
    CHOLMOD(free_dense) (&B, cm) ;
    CHOLMOD(free) (n, sizeof (Int), First, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Level, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Parent, cm) ;
    CHOLMOD(free) (n, sizeof (Int), Post, cm) ;
    progress (0, '.') ;
    return (maxerr) ;
}


/* ========================================================================== */
/* === raw_factor2 ========================================================== */
/* ========================================================================== */

/* A->stype can be 0 (lower), 1 (upper) or 0 (unsymmetric).  In the first two
 * cases, Ax=b is solved.  In the third, A*A'x=b is solved.  No analysis and no
 * fill-reducing ordering is used.  Both simplicial LL' and LDL' factorizations
 * are used (testing rowfac_mask, for LPDASA only). */

double raw_factor2 (cholmod_sparse *A, double alpha, int domask)
{
    Int n, i, prefer_zomplex, is_ll, xtype, sorted, axtype, stype ;
    Int *mask = NULL, *RLinkUp = NULL, nz = 0 ;
    Int *Cp = NULL, added_gunk ;
    double maxerr = 0, r = 0 ;
    cholmod_sparse *AT = NULL, *C = NULL, *CT = NULL, *CC = NULL, *C2 = NULL ;
    cholmod_factor *L = NULL ;
    cholmod_dense *B = NULL, *X = NULL ;
    double beta [2] ;

/*
int saveit = cm->print ;
int saveit2 = cm->precise ;
cm->print = 5 ;
cm->precise = TRUE ;
*/

    if (A == NULL)
    {
	return (0) ;
    }
    n = A->nrow ;
    if (n > 1000)
    {
	printf ("\nSkipping rowfac, matrix too large\n") ;
	return (0) ;
    }
    axtype = A->xtype ;
    beta [0] = alpha ;
    beta [1] = 0 ;

    prefer_zomplex = (A->xtype == CHOLMOD_ZOMPLEX) ;
    AT = CHOLMOD(transpose) (A, 2, cm) ;

    /* ensure C has stype of 0 or 1.  Do not prune any entries */
    stype = A->stype ;
    if (stype >= 0)
    {
	A->stype = 0 ;
	C = CHOLMOD(copy_sparse) (A, cm) ;
	A->stype = stype ;
	if (C) C->stype = stype ;
	/*
	C = CHOLMOD(copy) (A, 0, 1, cm) ;
	if (C) C->stype = A->stype ;
	*/
	CT = AT ;

    }
    else
    {
	C = AT ;
	/*
	CT = CHOLMOD(copy_sparse) (A, cm) ;
	*/
	/*
	CT = CHOLMOD(copy) (A, 0, 1, cm) ;
	if (CT) CT->stype = A->stype ;
	*/
	A->stype = 0 ;
	CT = CHOLMOD(copy_sparse) (A, cm) ;
	A->stype = stype ;
	if (CT) CT->stype = stype ;

	/* only domask if C is symmetric and upper part stored */
	domask = FALSE ;

    }

    mask = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    RLinkUp = CHOLMOD(malloc) (n, sizeof (Int), cm) ;

    if (C && cm->status == CHOLMOD_OK)
    {
	for (i = 0 ; i < n ; i++)
	{
	    mask [i] = -1 ;
	    RLinkUp [i] = i+1 ;
	}
    }
    else
    {
	domask = FALSE ;
    }

    if (C && !(C->packed) && !(C->sorted))
    {
	/* do not do the unpacked or unsorted cases */
	domask = FALSE ;
    }

    /* make a copy of C and add some gunk if stype > 0 */
    added_gunk = (C && C->stype > 0) ;
    if (added_gunk)
    {
	C2 = add_gunk (C) ;
    }
    else
    {
	C2 = CHOLMOD(copy_sparse) (C, cm) ;
    }

    CC = CHOLMOD(copy_sparse) (C2, cm) ;

    if (CC && domask)
    {
	Int *Cp, *Ci, p ;
	double *Cx, *Cz ;

	/* this implicitly sets the first row/col of C to zero, except diag. */
	mask [0] = 1 ;

	/* CC = C2, and then set the first row/col to zero, except diagonal */
	Cp = CC->p ;
	Ci = CC->i ;
	Cx = CC->x ;
	Cz = CC->z ;
	nz = Cp [n] ;
	switch (C->xtype)
	{
	    case CHOLMOD_REAL:
		for (p = 1 ; p < nz ; p++)
		{
		    if (Ci [p] == 0) Cx [p] = 0 ;
		}
		break ;

	    case CHOLMOD_COMPLEX:
		for (p = 1 ; p < nz ; p++)
		{
		    if (Ci [p] == 0)
		    {
			Cx [2*p  ] = 0 ;
			Cx [2*p+1] = 0 ;
		    }
		}
		break ;

	    case CHOLMOD_ZOMPLEX:
		for (p = 1 ; p < nz ; p++)
		{
		    if (Ci [p] == 0)
		    {
			Cx [p] = 0 ;
			Cz [p] = 0 ;
		    }
		}
		break ;
	}
    }

    B = rhs (CC, 1, n) ;

    for (sorted = 1 ; sorted >= 0 ; sorted--)
    {

	if (!sorted)
	{
	    if (C2 && !added_gunk) C2->sorted = FALSE ;
	    if (C) C->sorted = FALSE ;
	    if (CT) CT->sorted = FALSE ;
	}

	for (is_ll = 0 ; is_ll <= 1 ; is_ll++)
	{
	    for (xtype = 0 ; xtype <= 1 ; xtype++)
	    {

		L = CHOLMOD(allocate_factor) (n, cm) ;
		if (L) L->is_ll = is_ll ;

		if (xtype)
		{
		    CHOLMOD (change_factor) (axtype, is_ll, 0, 0, 1, L, cm) ;
		}

		CHOLMOD(rowfac_mask) (sorted ? C : C2,
		    CT, beta, 0, n, mask, RLinkUp, L, cm) ;

		cm->prefer_zomplex = prefer_zomplex ;
		X = CHOLMOD(solve) (CHOLMOD_A, L, B, cm) ;
		cm->prefer_zomplex = FALSE ;

		r = resid (CC, X, B) ;
		MAXERR (maxerr, r, 1) ;

		printf ("rowfac mask: resid is %g\n", r) ;

		CHOLMOD(free_factor) (&L, cm) ;
		CHOLMOD(free_dense) (&X, cm) ;
	    }
	}
    }

    CHOLMOD(free) (n, sizeof (Int), mask, cm) ;
    CHOLMOD(free) (n, sizeof (Int), RLinkUp, cm) ;

    CHOLMOD(free_sparse) (&C2, cm) ;
    CHOLMOD(free_sparse) (&CC, cm) ;
    CHOLMOD(free_sparse) (&CT, cm) ;
    CHOLMOD(free_sparse) (&C, cm) ;
    CHOLMOD(free_dense) (&B, cm) ;

    return (maxerr) ;
}
