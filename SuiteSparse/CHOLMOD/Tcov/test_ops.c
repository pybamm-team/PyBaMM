/* ========================================================================== */
/* === Tcov/test_ops ======================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Test CHOLMOD matrix operators. */

#include "cm.h"


/* ========================================================================== */
/* === nzdiag =============================================================== */
/* ========================================================================== */

/* Count the entries on the diagonal */

Int nzdiag (cholmod_sparse *A)
{
    Int *Ap, *Ai, *Anz ;
    Int nnzdiag, packed, j, p, i, pend, ncol ;
    if (A == NULL)
    {
	return (EMPTY) ;
    }
    nnzdiag = 0 ;
    ncol = A->ncol ;
    Ap = A->p ;
    Ai = A->i ;
    Anz = A->nz ;
    packed = A->packed ;
    for (j = 0 ; j < ncol ; j++)
    {
	p = Ap [j] ;
	pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
	for ( ; p < pend ; p++)
	{
	    i = Ai [p] ;
	    if (i == j)
	    {
		nnzdiag++ ;
	    }
	}
    }
    return (nnzdiag) ;
}


/* ========================================================================== */
/* === check_partition ====================================================== */
/* ========================================================================== */

/* Check a node separator, and return the # of nodes in the node separator or
 * -1 if the separater is invalid.  A node j is in the left part if
 * Part [j] = 0, in the right part if Part [j] = 1, and in the separator if
 * Part [j] = 2.
 */

Int check_partition (cholmod_sparse *A, Int *Part)
{
    Int *Ap, *Ai, *Anz ;
    Int chek [3], which, i, j, p, n, pend, packed ;

    if (A == NULL || Part == NULL || A->nrow != A->ncol)
    {
        /* printf ("A NULL, no Partition, or rectangular\n") ; */
	return (EMPTY) ;
    }
    n = A->nrow ;
    Ap = A->p ;
    Ai = A->i ;
    Anz = A->nz ;
    packed = A->packed ;

    chek [0] = 0 ;
    chek [1] = 0 ;
    chek [2] = 0 ;

    /* printf ("\ncheck partition:\n") ; */
    for (j = 0 ; j < n ; j++)
    {
	which = Part [j] ;
        /* printf ("node "ID" in Part "ID"\n", j, which) ; */
	p = Ap [j] ;
	pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
	for ( ; p < pend ; p++)
	{
	    i = Ai [p] ;
            /* printf ("   neighbor "ID" in part "ID"\n", i, Part [i]) ; */
	    if (which == 0)
	    {
		if (Part [i] == 1)
		{
                    /* printf ("   broken!\n") ; */
		    return (EMPTY) ;
		}
	    }
	    else if (which == 1)
	    {
		if (Part [i] == 0)
		{
                    /* printf ("   broken!\n") ; */
		    return (EMPTY) ;
		}
	    }
	}
	if (which < 0 || which > 2)
	{
            /* printf ("   node j, invalid partition, broken!\n") ; */
	    return (EMPTY) ;
	}
	chek [which]++ ;
    }
    /*
    printf ("nodes in left: "ID" right: "ID" separator: "ID"\n",
        chek [0], chek [1], chek [2]) ;
    */
    return (chek [2]) ;
}


/* ========================================================================== */
/* === check_equality ======================================================= */
/* ========================================================================== */

/* Ensure two sparse matrices are identical. */

static void check_equality (cholmod_sparse *E, cholmod_sparse *D, Int xtype)
{
    double *Ex, *Ez, *Dx, *Dz ;
    Int *Ep, *Ei, *Dp, *Di ;
    Int j, nz, p, ncol ;

    if (E == NULL || D == NULL || D->xtype != xtype || E->xtype != xtype)
    {
	return ;
    }

    Ep = E->p ;
    Ei = E->i ;
    Ex = E->x ;
    Ez = E->z ;

    Dp = D->p ;
    Di = D->i ;
    Dx = D->x ;
    Dz = D->z ;

    OK (E->ncol == D->ncol) ;
    OK (E->nrow == D->nrow) ;

    ncol = E->ncol ;

    for (j = 0 ; j <= ncol ; j++)
    {
	OK (Ep [j] == Dp [j]) ;
    }
    nz = Ep [ncol] ;

    for (p = 0 ; p < nz ; p++)
    {
	OK (Ei [p] == Di [p]) ;
    }

    if (xtype == CHOLMOD_REAL)
    {
	for (p = 0 ; p < nz ; p++)
	{
	    OK (Ex [p] == Dx [p]) ;
	}
    }
    else if (xtype == CHOLMOD_COMPLEX)
    {
	for (p = 0 ; p < 2*nz ; p++)
	{
	    OK (Ex [p] == Dx [p]) ;
	}
    }
    else if (xtype == CHOLMOD_ZOMPLEX)
    {
	for (p = 0 ; p < nz ; p++)
	{
	    OK (Ex [p] == Dx [p]) ;
	    OK (Ez [p] == Dz [p]) ;
	}
    }
}


/* ========================================================================== */
/* === test_ops ============================================================= */
/* ========================================================================== */

/* Test various matrix operations. */

double test_ops (cholmod_sparse *A)
{
    double maxerr = 0, r, x, anorm, r1, rinf ;
    double *Sx, *Sz ;
    Int *Pinv, *P, *Si, *Sj, *Q, *Qinv, *fset, *Partition ;
    cholmod_triplet *S ;
    cholmod_sparse *C, *D, *E, *F, *G, *H, *AT, *Zs ;
    cholmod_dense *X, *Y ;
    Int n, kk, k, nrow, ncol, len, nz, ok, i, j, stype, nmin, mode, isreal,
	xtype, xtype2, mtype, asym, xmatched, pmatched, nzoffdiag, nz_diag ;
    size_t nz1, nz2 ;
    void (*save) (int, const char *, int, const char *) ;
    double alpha [2], beta [2], *Xx ;
    FILE *f ;
    int option, save3 ;

    if (A == NULL)
    {
	ERROR (CHOLMOD_INVALID, "nothing for test_ops") ;
	return (1) ;
    }

    nrow = A->nrow ;
    ncol = A->ncol ;
    H = NULL ;
    n = MAX (nrow, ncol) ;
    nmin = MIN (nrow, ncol) ;
    xtype = A->xtype ;
    isreal = (A->xtype == CHOLMOD_REAL) ;

    /* ---------------------------------------------------------------------- */
    /* norm */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(print_sparse) (A, "A for testops", cm) ;
    r1 = CHOLMOD(norm_sparse) (A, 1, cm) ;
    rinf = CHOLMOD(norm_sparse) (A, 0, cm) ;

    anorm = r1 ;

    /* E = pattern of A */
    E = CHOLMOD(copy) (A, 0, 0, cm) ;
    CHOLMOD(print_sparse) (E, "E = pattern of A", cm) ;
    r1 = CHOLMOD(norm_sparse) (E, 1, cm) ;
    rinf = CHOLMOD(norm_sparse) (E, 0, cm) ;
    OK (r1 <= nrow) ;
    OK (rinf <= ncol) ;
    CHOLMOD(free_sparse) (&E, cm) ;

    /* E = pattern of A, but exclude the diagonal */
    E = CHOLMOD(copy) (A, 0, -1, cm) ;
    CHOLMOD(print_sparse) (E, "E = spones (A), excl diag", cm) ;
    r1 = CHOLMOD(norm_sparse) (E, 1, cm) ;
    rinf = CHOLMOD(norm_sparse) (E, 0, cm) ;
    if (nrow < ncol)
    {
	OK (r1 <= nrow) ;
	OK (rinf < MAX (1,ncol)) ;
    }
    else
    {
	OK (r1 < MAX (1,nrow)) ;
	OK (rinf <= ncol) ;
    }
    CHOLMOD(free_sparse) (&E, cm) ;

    /* ---------------------------------------------------------------------- */
    /* copy */
    /* ---------------------------------------------------------------------- */

    if (A->stype)
    {
	/* E = tril (A), no diagonal */
	E = CHOLMOD(copy) (A, -1, -1, cm) ;
	CHOLMOD(print_sparse) (E, "E, lower and no diagonal", cm) ;
	CHOLMOD(band_inplace) (0, n, 0, E, cm) ;
	CHOLMOD(print_sparse) (E, "E Empty", cm) ;
	nz = CHOLMOD(nnz) (E, cm) ;
	if (E != NULL)
	{
	    OK (nz == 0) ;
	}
	CHOLMOD(free_sparse) (&E, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* read/write */
    /* ---------------------------------------------------------------------- */

/*
    i = cm->print ;
    cm->print = 4 ;
    CHOLMOD(print_sparse) (A, "A for read/write", cm) ;
    cm->print = i ;
*/

    /* delete the contents of the temp1.mtx and temp2.mtx file */
    f = fopen ("temp1.mtx", "w") ;
    fprintf (f, "temp1\n") ;
    fclose (f) ;

    f = fopen ("temp3.mtx", "w") ;
    fprintf (f, "temp3\n") ;
    fclose (f) ;

    CHOLMOD(free_work) (cm) ;

    f = fopen ("temp1.mtx", "w") ;
    asym = CHOLMOD(write_sparse) (f, A, NULL, "comments.txt", cm) ;
    fclose (f) ;
    printf ("write_sparse, asym: "ID"\n", asym) ;
    OK (IMPLIES (A != NULL, asym > EMPTY)) ;

    f = fopen ("temp1.mtx", "r") ;
    C = CHOLMOD(read_sparse) (f, cm) ;
    fclose (f) ;
    printf ("got_sparse\n") ;
    CHOLMOD(free_sparse) (&C, cm) ;

    save3 = A->xtype ;
    A->xtype = CHOLMOD_PATTERN ;
    f = fopen ("temp3.mtx", "w") ;
    asym = CHOLMOD(write_sparse) (f, A, NULL, "comments.txt", cm) ;
    A->xtype = save3 ;
    fclose (f) ;
    printf ("write_sparse3, asym: "ID"\n", asym) ;

    f = fopen ("temp3.mtx", "r") ;
    C = CHOLMOD(read_sparse) (f, cm) ;
    fclose (f) ;
    printf ("got_sparse3\n") ;
    CHOLMOD(free_sparse) (&C, cm) ;

    for (i = 0 ; i <= 1 ; i++)
    {

	f = fopen ("temp2.mtx", "w") ;
	fprintf (f, "temp2\n") ;
	fclose (f) ;

	X = CHOLMOD(ones) (4, 4, CHOLMOD_REAL, cm) ;
	if (X != NULL)
	{
	    Xx = X->x ;
	    Xx [0] = (i == 0) ? 1.1e308 : -1.1e308 ;
	}
	f = fopen ("temp2.mtx", "w") ;
	ok = CHOLMOD(write_dense) (f, X, "comments.txt", cm) ;
	fclose (f) ;
	printf ("wrote dense\n") ;

	f = fopen ("temp2.mtx", "r") ;
	Y = CHOLMOD(read_dense) (f, cm) ;
	fclose (f) ;
	printf ("got dense\n") ;
	CHOLMOD(free_dense) (&X, cm) ;
	CHOLMOD(free_dense) (&Y, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* symmetry */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(free_work) (cm) ;
    xmatched = 0 ;
    pmatched = 0 ;
    nzoffdiag = 0 ;
    nz_diag = 0 ;
    for (option = 0 ; option <= 2 ; option++)
    {
	asym = CHOLMOD(symmetry) (A, option, &xmatched, &pmatched, &nzoffdiag,
	    &nz_diag, cm);
	printf
        ("symmetry, asym: "ID" matched "ID" "ID" offdiag "ID" diag "ID"\n",
            asym, xmatched, pmatched, nzoffdiag, nz_diag) ;
    }

    /* ---------------------------------------------------------------------- */
    /* transpose */
    /* ---------------------------------------------------------------------- */

    C = CHOLMOD(allocate_sparse) (A->ncol, A->nrow, A->nzmax, TRUE, TRUE,
	    -(A->stype), A->xtype, cm) ;
    D = CHOLMOD(allocate_sparse) (A->nrow, A->ncol, A->nzmax, TRUE, TRUE,
	    A->stype, A->xtype, cm) ;
    CHOLMOD(free_work) (cm) ;
    ok = (C != NULL && D != NULL) ; 

    /* C = A' */
    if (ok)
    {
	if (A->stype)
	{
	    ok = CHOLMOD(transpose_sym) (A, 2, NULL, C, cm) ;
	}
	else
	{
	    ok = CHOLMOD(transpose_unsym) (A, 2, NULL, NULL, 0, C, cm) ;
	}
	OK (ok || cm->status == CHOLMOD_OUT_OF_MEMORY) ;
    }

    /* D = C' */
    if (ok)
    {
	if (A->stype)
	{
	    ok = CHOLMOD(transpose_sym) (C, 2, NULL, D, cm) ;
	}
	else
	{
	    ok = CHOLMOD(transpose_unsym) (C, 2, NULL, NULL, 0, D, cm) ;
	}
	OK (ok || cm->status == CHOLMOD_OUT_OF_MEMORY) ;
    }

    if (ok)
    {
	ok = CHOLMOD(check_sparse) (D, cm) ;
	OK (ok || cm->status == CHOLMOD_OUT_OF_MEMORY) ;
    }

    CHOLMOD(free_sparse) (&C, cm) ;
    CHOLMOD(free_sparse) (&D, cm) ;

    /* ---------------------------------------------------------------------- */
    /* C = A with jumbled triplets */
    /* ---------------------------------------------------------------------- */

    S = CHOLMOD(sparse_to_triplet) (A, cm) ; /* [ */

    if (S != NULL && nmin > 0)
    {

	/* double the number of entries in S */
	nz1 = S->nzmax ;
	nz2 = 2*nz1 ;
	ok = CHOLMOD(reallocate_triplet) (nz2, S, cm) ;

	if (ok)
	{
	    /* add duplicate entries, but keep the matrix the same */
	    OK (S->nzmax == nz2) ;
	    Si = S->i ;
	    Sj = S->j ;
	    Sx = S->x ;
	    Sz = S->z ;

	    for (k = nz1 ; k < ((Int) nz2) ; k++)
	    {
		kk = nrand (k) ;				/* RAND */
		Si [k] = Si [kk] ;
		Sj [k] = Sj [kk] ;

		if (S->xtype == CHOLMOD_REAL)
		{
		    x = Sx [kk] * (xrand (4.) - 2) ;		/* RAND */
		    Sx [k] = x ;
		    Sx [kk] -= x ;
		}
		else if (S->xtype == CHOLMOD_COMPLEX)
		{
		    x = Sx [2*kk] * (xrand (4.) - 2) ;		/* RAND */
		    Sx [2*k] = x ;
		    Sx [2*kk] -= x ;

		    x = Sx [2*kk+1] * (xrand (4.) - 2) ;	/* RAND */
		    Sx [2*k+1] = x ;
		    Sx [2*kk+1] -= x ;
		}
		else
		{
		    x = Sx [kk] * (xrand (4.) - 2) ;		/* RAND */
		    Sx [k] = x ;
		    Sx [kk] -= x ;

		    x = Sz [kk] * (xrand (4.) - 2) ;		/* RAND */
		    Sz [k] = x ;
		    Sz [kk] -= x ;
		}
	    }

	    /* randomly jumble the entries */
	    for (k = 0 ; k < ((Int) (nz2-1)) ; k++)
	    {
		kk = k + nrand (nz2-k) ;			/* RAND */
		i = Si [k] ;
		Si [k] = Si [kk] ;
		Si [kk] = i ;
		j = Sj [k] ;
		Sj [k] = Sj [kk] ;
		Sj [kk] = j ;

		if (S->xtype == CHOLMOD_REAL)
		{
		    x = Sx [k] ;
		    Sx [k] = Sx [kk] ;
		    Sx [kk] = x ;
		}
		else if (S->xtype == CHOLMOD_COMPLEX)
		{
		    x = Sx [2*k] ;
		    Sx [2*k] = Sx [2*kk] ;
		    Sx [2*kk] = x ;
		    x = Sx [2*k+1] ;
		    Sx [2*k+1] = Sx [2*kk+1] ;
		    Sx [2*kk+1] = x ;
		}
		else
		{
		    x = Sx [k] ;
		    Sx [k] = Sx [kk] ;
		    Sx [kk] = x ;
		    x = Sz [k] ;
		    Sz [k] = Sz [kk] ;
		    Sz [kk] = x ;
		}
	    }
	    S->nnz = nz2 ;
	}
	else
	{
	    OK (S->nzmax == nz1) ;
	    OK (S->nnz == nz1) ;
	}
    }

    CHOLMOD(print_triplet) (S, "S jumbled", cm) ;

    C = CHOLMOD(triplet_to_sparse) (S, 0, cm) ;	/* [ */
    CHOLMOD(print_sparse) (A, "A", cm) ;
    CHOLMOD(print_sparse) (C, "C", cm) ;

    Zs = CHOLMOD(spzeros) (nrow, ncol, 1, xtype, cm) ;	/* [ */

    G = NULL ;
    F = NULL ;

    if (isreal)
    {

	/* G=A+0 */
	G = CHOLMOD(add) (A, Zs, one, one, TRUE, TRUE, cm) ; /* [ */

	/* F = G-C */
	F = CHOLMOD(add) (G, C, one, minusone, TRUE, TRUE, cm) ;	/* [ */

	CHOLMOD(print_sparse) (F, "F", cm) ;
	r = CHOLMOD(norm_sparse) (F, 1, cm) ;
	MAXERR (maxerr, r, anorm) ;
	r = CHOLMOD(norm_sparse) (F, 0, cm) ;
	CHOLMOD(drop) (0, F, cm) ;
	rinf = CHOLMOD(norm_sparse) (F, 0, cm) ;
	if (F != NULL)
	{
	    OK (r == rinf) ;
	}
	MAXERR (maxerr, r, anorm) ;
	MAXERR (maxerr, rinf, anorm) ;

	/* E = F, with change of type and dropping small entries */
	for (stype = -1 ; stype <= 1 ; stype++)
	{
	    if (stype != 0 && (F != NULL && F->nrow != F->ncol))
	    {
		continue ;
	    }
	    for (mode = 0 ; mode <= 1 ; mode++)
	    {
		E = CHOLMOD(copy) (F, stype, mode, cm) ;	/* [ */
		CHOLMOD(drop) (1e-16, E, cm) ;
		r1 = CHOLMOD(norm_sparse) (E, 1, cm) ;
		rinf = CHOLMOD(norm_sparse) (E, 0, cm) ;
		if (E != NULL)
		{
		    if (mode == 0)
		    {
			/* pattern only */
			OK (r1 <= nrow) ;
			OK (rinf <= ncol) ;
		    }
		    else
		    {
			MAXERR (maxerr, r1, anorm) ;
			MAXERR (maxerr, rinf, anorm) ;
		    }
		}
		CHOLMOD(free_sparse) (&E, cm) ;	/* ] */
	    }
	}

	CHOLMOD(free_sparse) (&F, cm) ;	/* ] */
	CHOLMOD(free_sparse) (&G, cm) ;	/* ] */
    }

    Y = CHOLMOD(ones) (nrow, 1, xtype, cm) ;   /* [ */
    X = CHOLMOD(ones) (ncol, 1, xtype, cm) ;   /* [ */
    alpha [0] = 0 ;
    alpha [1] = 0 ;
    beta [0] = 2 ;
    beta [1] = 0 ;
    /* Y = 0*A*X + 2*Y */
    CHOLMOD(sdmult) (A, FALSE, alpha, beta, X, Y, cm) ;
    r = CHOLMOD(norm_dense) (Y, 0, cm) ;
    if (Y != NULL && X != NULL && A != NULL)
    {
	OK ((nrow == 0) ? (r == 0) : (r == 2)) ;
    }

    alpha [0] = 1 ;
    /* Y = 1*(0)*X + 2*Y */
    CHOLMOD(sdmult) (Zs, FALSE, alpha, beta, X, Y, cm) ;
    r = CHOLMOD(norm_dense) (Y, 0, cm) ;
    if (Y != NULL && X != NULL && A != NULL)
    {
	OK ((nrow == 0) ? (r == 0) : (r == 4)) ;
    }

    CHOLMOD(free_dense) (&X, cm) ;   /* ] */
    CHOLMOD(free_dense) (&Y, cm) ;   /* ] */

    CHOLMOD(free_sparse) (&Zs, cm) ; /* ] */
    CHOLMOD(free_sparse) (&C, cm) ;  /* ] */
    CHOLMOD(free_triplet) (&S, cm) ; /* ] */

    /* ---------------------------------------------------------------------- */
    /* C = P*A*Q in triplet form */
    /* ---------------------------------------------------------------------- */

    S = CHOLMOD(sparse_to_triplet) (A, cm) ;
    P = prand (nrow) ;						/* RAND */
    if (A->stype == 0)
    {
	Q = prand (ncol) ;					/* RAND */
    }
    else
    {
	/* if A is symmetric, and stored in either upper or lower form, then
	 * the following code only works if P = Q */
	Q = P ;
    }
    Pinv = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
    Qinv = CHOLMOD(malloc) (ncol, sizeof (Int), cm) ;
    Partition = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;

    if (Pinv != NULL && Qinv != NULL && P != NULL && Q != NULL && S != NULL)
    {
	Si = S->i ;
	Sj = S->j ;
	nz = S->nnz ;
	for (k = 0 ; k < nrow ; k++)
	{
	    Pinv [P [k]] = k ;
	}
	for (k = 0 ; k < ncol ; k++)
	{
	    Qinv [Q [k]] = k ;
	}
	for (k = 0 ; k < nz ; k++)
	{
	    Si [k] = Pinv [Si [k]] ;
	}
	for (k = 0 ; k < nz ; k++)
	{
	    Sj [k] = Qinv [Sj [k]] ;
	}
    }

    C = CHOLMOD(triplet_to_sparse) (S, 0, cm) ;
    D = NULL ;
    E = NULL ;
    F = NULL ;
    G = NULL ;

    if (isreal)
    {

	/* ------------------------------------------------------------------ */
	/* E = P*A*Q in sparse form */
	/* ------------------------------------------------------------------ */

	D = CHOLMOD(copy) (A, 0, 1, cm) ;
	E = CHOLMOD(submatrix) (D, P, nrow, Q, ncol, TRUE, FALSE, cm) ;
	CHOLMOD(sort) (E, cm) ;

	/* ------------------------------------------------------------------ */
	/* F = E-G */
	/* ------------------------------------------------------------------ */

	G = CHOLMOD(copy) (C, 0, 1, cm) ;
	F = CHOLMOD(add) (E, G, one, minusone, TRUE, TRUE, cm) ;
	CHOLMOD(drop) (0, F, cm) ;
	nz = CHOLMOD(nnz) (F, cm) ;
	if (F != NULL)
	{
	    OK (nz == 0) ;
	}
    }

    CHOLMOD(free_sparse) (&F, cm) ;
    CHOLMOD(free_sparse) (&G, cm) ;
    CHOLMOD(free_sparse) (&D, cm) ;
    CHOLMOD(free_sparse) (&E, cm) ;
    CHOLMOD(free_sparse) (&H, cm) ;
    CHOLMOD(free_sparse) (&C, cm) ;
    CHOLMOD(free_triplet) (&S, cm) ;

    /* ---------------------------------------------------------------------- */
    /* submatrix */
    /* ---------------------------------------------------------------------- */

    if (A->stype == 0 && isreal)
    {
	/* E = A(:,:) */
	E = CHOLMOD(submatrix) (A, NULL, -1, NULL, -1, TRUE, TRUE, cm) ;
	/* C = A-E */
	C = CHOLMOD(add) (A, E, one, minusone, TRUE, TRUE, cm) ;
	ok = CHOLMOD(drop) (0., C, cm) ;
	nz = CHOLMOD(nnz) (C, cm) ;
	if (C != NULL)
	{
	    OK (nz == 0) ;
	}
	CHOLMOD(free_sparse) (&C, cm) ;
	CHOLMOD(free_sparse) (&E, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* test band and add, unsymmetric */
    /* ---------------------------------------------------------------------- */

    if (isreal)
    {
	CHOLMOD(print_sparse) (A, "A for do triplet", cm) ;

	/* E = A */
	E = CHOLMOD(copy) (A, 0, 1, cm) ;
	CHOLMOD(print_sparse) (E, "E=triu(A)", cm) ;

	/* E = triu (E) */
	CHOLMOD(band_inplace) (0, ncol, 1, E, cm) ;

	/* F = A */
	F = CHOLMOD(copy) (A, 0, 1, cm) ;
	CHOLMOD(print_sparse) (F, "F=tril(A)", cm) ;

	/* F = tril(F,-1) */
	CHOLMOD(band_inplace) (-nrow, -1, 1, F, cm) ;
	CHOLMOD(print_sparse) (F, "Ftril", cm) ;

	/* G = E+F */
	G = CHOLMOD(add) (E, F, one, one, TRUE, TRUE, cm) ;
	CHOLMOD(print_sparse) (G, "G=E+F", cm) ;

	/* D = A-G, which should be empty */
	D = CHOLMOD(add) (G, A, one, minusone, TRUE, TRUE, cm) ;
	CHOLMOD(print_sparse) (D, "D=A-G", cm) ;

	CHOLMOD(drop) (0, D, cm) ;
	CHOLMOD(print_sparse) (D, "D drop", cm) ;
	nz = CHOLMOD(nnz) (D, cm) ;
	if (D != NULL)
	{
	    OK (nz == 0) ;
	}

	CHOLMOD(free_sparse) (&F, cm) ;
	CHOLMOD(free_sparse) (&D, cm) ;
	CHOLMOD(free_sparse) (&E, cm) ;
	CHOLMOD(free_sparse) (&G, cm) ;

	D = CHOLMOD(band) (A, 1, -1, 0, cm) ;
	nz = CHOLMOD(nnz) (D, cm) ;
	if (D != NULL)
	{
	    OK (nz == 0) ;
	}
	CHOLMOD(free_sparse) (&D, cm) ;
	D = CHOLMOD(band) (A, 0, 0, 0, cm) ;
	nz = CHOLMOD(nnz) (D, cm) ;
	if (D != NULL)
	{
	    OK (nz == nzdiag (D)) ;
	}
	CHOLMOD(free_sparse) (&D, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* test band, add and copy_sparse (symmetric) */
    /* ---------------------------------------------------------------------- */

    if (A->stype && isreal)
    {

	/* E = A, in symmetric/upper form */
	E = CHOLMOD(copy) (A, 1, 1, cm) ;
	CHOLMOD(print_sparse) (E, "E=A in sym/upper form", cm) ;

	/* E = -E */
	CHOLMOD(scale) (M1, CHOLMOD_SCALAR, E, cm) ;

	/* F = A, in symmetric/lower form */
	F = CHOLMOD(copy) (A, -1, 1, cm) ;
	CHOLMOD(print_sparse) (F, "F=A in sym/lower form", cm) ;

	/* C = F (exact copy) */
	C = CHOLMOD(copy_sparse) (F, cm) ;

	/* G = E+C */
	G = CHOLMOD(add) (E, C, one, one, TRUE, FALSE, cm) ;
	CHOLMOD(print_sparse) (G, "G=E+F", cm) ;
	CHOLMOD(sort) (G, cm) ;

	CHOLMOD(drop) (0, G, cm) ;
	CHOLMOD(print_sparse) (G, "G drop", cm) ;
	nz = CHOLMOD(nnz) (G, cm) ;
	if (G != NULL)
	{
	    OK (nz == 0) ;
	}

	CHOLMOD(free_sparse) (&C, cm) ;
	CHOLMOD(free_sparse) (&F, cm) ;
	CHOLMOD(free_sparse) (&E, cm) ;
	CHOLMOD(free_sparse) (&G, cm) ;

    }

    /* ---------------------------------------------------------------------- */
    /* try a dense identity matrix */
    /* ---------------------------------------------------------------------- */

    X = CHOLMOD(eye) (3, 4, CHOLMOD_REAL, cm) ;
    CHOLMOD(print_dense) (X, "Dense identity", cm) ;
    CHOLMOD(free_dense) (&X, cm) ;

    /* ---------------------------------------------------------------------- */
    /* bisector and nested_dissection */
    /* ---------------------------------------------------------------------- */

#ifndef NPARTITION
    if (A != NULL && A->nrow == A->ncol)
    {
	SuiteSparse_long nc, nc_new ;
	Int cnz, csep, save2 ;
	Int *Cnw, *Cew, *Cmember, *CParent, *Perm ;
	double save1 ;

	/* try CHOLMOD's interface to METIS_ComputeVertexSeparator */
	cm->metis_memory = 2.0 ;
        /* cm->print = 5 ; */
	CHOLMOD(print_sparse) (A, "A for bisect", cm) ;
	csep = CHOLMOD(bisect) (A, NULL, 0, TRUE, Partition, cm) ;
	if (csep != EMPTY)
	{
            Int csep2 ;
            /* printf ("csep %g\n", (double) csep) ; */
            csep2 = check_partition (A, Partition) ;
            /* printf ("csep2 %g\n", (double) csep2) ; */
	    OK (csep == csep2) ;
	}

	/* try the raw interface to METIS_ComputeVertexSeparator */
	CHOLMOD(print_sparse) (A, "A for metis bisect", cm) ;

	/* C = A+A', remove the diagonal */
	AT = CHOLMOD(transpose) (A, 0, cm) ;
	E = CHOLMOD(add) (A, AT, one, one, FALSE, TRUE, cm) ;
	CHOLMOD(free_sparse) (&AT, cm) ;
	C = CHOLMOD(copy) (E, 0, -1, cm) ;
	CHOLMOD(print_sparse) (C, "C for metis bisect", cm) ;

	cnz = (C != NULL) ? (C->nzmax) : 0 ;
	Cew = CHOLMOD(malloc) (cnz, sizeof (Int), cm) ;
	Cnw = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
	if (Cnw != NULL)
	{
	    for (j = 0 ; j < (Int) (A->nrow) ; j++)
	    {
		Cnw [j] = 1 ;
	    }
	}
	if (Cew != NULL)
	{
	    for (j = 0 ; j < cnz ; j++)
	    {
		Cew [j] = 1 ;
	    }
	}
	csep = CHOLMOD(metis_bisector) (C, Cnw, Cew, Partition, cm) ;
	if (csep != EMPTY)
	{
	    OK (csep == check_partition (C, Partition)) ;
	}
	CHOLMOD(free) (nrow, sizeof (Int), Cnw, cm) ;
	CHOLMOD(free) (cnz, sizeof (Int), Cew, cm) ;

	CHOLMOD(free_sparse) (&C, cm) ;
	CHOLMOD(free_sparse) (&E, cm) ;

	Cmember = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
	CParent = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
	Perm = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;

	save1 = cm->method [cm->current].nd_small ;
	save2 = cm->method [cm->current].nd_oksep ;
	cm->method [cm->current].nd_small = 1 ;
	cm->method [cm->current].nd_oksep = 1.0 ;

	nc = CHOLMOD(nested_dissection) (A, NULL, 0, Perm, CParent, Cmember,
		cm);
	if (nc > 0)
	{
	    OK (CHOLMOD(check_perm) (Perm, n, n, cm)) ;
	}

	CHOLMOD(free_work) (cm) ;

	/* collapse the septree */
	if (nc > 0 && n > 0)
	{
	    nc_new = CHOLMOD(collapse_septree) (n, nc, 0.1, 400,
		CParent, Cmember, cm) ;

	    /* error checks */
	    save = cm->error_handler ;
	    cm->error_handler = NULL ;
	    nc_new = CHOLMOD(collapse_septree) (n, nc, 0.1, 400,
		CParent, Cmember, NULL) ;
	    OK (nc_new == EMPTY) ;
	    nc_new = CHOLMOD(collapse_septree) (n, nc, 0.1, 400,
		NULL, Cmember, cm) ;
	    OK (nc_new == EMPTY) ;
	    nc_new = CHOLMOD(collapse_septree) (n, nc, 0.1, 400,
		CParent, NULL, cm) ;
	    OK (nc_new == EMPTY) ;
	    nc_new = CHOLMOD(collapse_septree) (0, 1, 0.1, 400,
		CParent, Cmember, cm) ;
	    OK (nc_new == EMPTY) ;
	    nc_new = CHOLMOD(collapse_septree) (1, 1, 0.1, 400,
		CParent, Cmember, cm) ;
	    OK (nc_new == 1 || nc_new == EMPTY) ;
	    nc_new = CHOLMOD(collapse_septree) (Int_max, Int_max,
		0.1, 400, CParent, Cmember, cm) ;
	    OK (nc_new == EMPTY) ;

	    cm->error_handler = save ;
	}

	CHOLMOD(free) (nrow, sizeof (Int), Cmember, cm) ;
	CHOLMOD(free) (nrow, sizeof (Int), CParent, cm) ;
	CHOLMOD(free) (nrow, sizeof (Int), Perm, cm) ;

	cm->method [cm->current].nd_small = save1 ;
	cm->method [cm->current].nd_oksep = save2 ;

    }
#endif

    /* ---------------------------------------------------------------------- */
    /* dense to/from sparse conversions */
    /* ---------------------------------------------------------------------- */

    /* convert A to real, remove zero entries, and then convert to pattern */
    if (MAX (nrow, ncol) < 1000)
    {

	C = CHOLMOD(copy_sparse) (A, cm) ;
	CHOLMOD(sparse_xtype) (CHOLMOD_REAL, C, cm) ;
	CHOLMOD(drop) (0., C, cm) ;
	D = CHOLMOD(copy) (C, 0, 0, cm) ;
	CHOLMOD(sort) (D, cm) ;

	/* X = dense copy of C */
	CHOLMOD(sparse_xtype) (CHOLMOD_PATTERN, C, cm) ;
	X = CHOLMOD(sparse_to_dense) (C, cm) ;
	CHOLMOD(free_sparse) (&C, cm) ;

	/* change X to sparse pattern and then real/complex/zomplex, it should
	 * equal D */
	for (xtype2 = CHOLMOD_REAL ; xtype2 <= CHOLMOD_ZOMPLEX ; xtype2++)
	{
	    E = CHOLMOD(dense_to_sparse) (X, FALSE, cm) ;
	    ok = CHOLMOD(sparse_xtype) (xtype2, E, cm) ;
	    ok = CHOLMOD(sparse_xtype) (xtype2, D, cm) ;
	    if (xtype2 == CHOLMOD_REAL)
	    {
		F = CHOLMOD(add) (E, D, one, minusone, TRUE, TRUE, cm) ;
		r = CHOLMOD(norm_sparse) (F, 0, cm) ;
		if (F != NULL)
		{
		    OK (r == 0) ;
		}
		CHOLMOD(free_sparse) (&F, cm) ;
	    }
	    else
	    {
		check_equality (E, D, xtype2) ;
	    }
	    CHOLMOD(free_sparse) (&E, cm) ;
	}

	CHOLMOD(free_sparse) (&D, cm) ;
	CHOLMOD(free_dense) (&X, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* unsymmetric transpose */
    /* ---------------------------------------------------------------------- */

    len = ncol/2 ;
    fset = prand (ncol) ;					/* RAND */
    CHOLMOD(print_perm) (P, nrow, nrow, "P", cm) ;
    CHOLMOD(print_subset) (fset, ncol, ncol, "fset", cm) ;

    if (isreal)
    {
	C = CHOLMOD(copy) (A, 0, 1, cm) ;
	D = CHOLMOD(ptranspose) (C, 1, P, fset, len, cm) ;
	E = CHOLMOD(transpose) (D, 1, cm) ;
	F = CHOLMOD(transpose) (E, 1, cm) ;
	G = CHOLMOD(add) (D, F, one, minusone, TRUE, FALSE, cm) ;
	r = CHOLMOD(norm_sparse) (G, 0, cm) ;
	if (G != NULL)
	{
	    OK (r == 0) ;
	}
	CHOLMOD(drop) (0, G, cm) ;
	r = CHOLMOD(norm_sparse) (G, 0, cm) ;
	nz = CHOLMOD(nnz) (G, cm) ;
	if (G != NULL)
	{
	    OK (r == 0) ;
	    OK (nz == 0) ;
	}

	CHOLMOD(free_sparse) (&C, cm) ;
	CHOLMOD(free_sparse) (&D, cm) ;
	CHOLMOD(free_sparse) (&E, cm) ;
	CHOLMOD(free_sparse) (&F, cm) ;
	CHOLMOD(free_sparse) (&G, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* symmetric array transpose */
    /* ---------------------------------------------------------------------- */

    if (A->stype != 0)
    {

	/* C = A(p,p).' */
	C = CHOLMOD(ptranspose) (A, 1, P, NULL, 0, cm) ;

	/* D = C(pinv,pinv).' */
	D = CHOLMOD(ptranspose) (C, 1, Pinv, NULL, 0, cm) ;
	CHOLMOD(sort) (D, cm) ;
	CHOLMOD(free_sparse) (&C, cm) ;

	/* C = A, sorted */
	C = CHOLMOD(copy_sparse) (A, cm) ;
	CHOLMOD(sort) (C, cm) ;

	/* C and D should be equal */
	check_equality (C, D, xtype) ;
	CHOLMOD(free_sparse) (&C, cm) ;
	CHOLMOD(free_sparse) (&D, cm) ;

	/* C = A.' */
	C = CHOLMOD(transpose) (A, 1, cm) ;

	/* D = C.' */
	D = CHOLMOD(transpose) (C, 1, cm) ;
	CHOLMOD(sort) (D, cm) ;
	CHOLMOD(free_sparse) (&C, cm) ;

	/* C = A, sorted */
	C = CHOLMOD(copy_sparse) (A, cm) ;
	CHOLMOD(sort) (C, cm) ;

	/* C and D should be equal */
	check_equality (C, D, xtype) ;
	CHOLMOD(free_sparse) (&C, cm) ;
	CHOLMOD(free_sparse) (&D, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* matrix multiply */
    /* ---------------------------------------------------------------------- */

    if (isreal)
    {
	/* this fails for a large arrowhead matrix, so turn off error hanlder */
	save = cm->error_handler ;
	cm->error_handler = NULL ;
	AT = CHOLMOD(transpose) (A, 1, cm) ;
	D = CHOLMOD(copy) (A, 0, 1, cm) ;
	if (n > NLARGE) progress (1, '.') ;
	C = CHOLMOD(aat) (D, NULL, 0, 1, cm) ;
	if (n > NLARGE) progress (1, '.') ;
	CHOLMOD(print_common) ("After A*A'", cm) ;

	for (stype = -1 ; stype <= 1 ; stype++)
	{
	    if (n > NLARGE) progress (1, '.') ;
	    E = CHOLMOD(ssmult) (A, AT, stype, TRUE, TRUE, cm) ;
	    if (n > NLARGE) progress (1, '.') ;
	    G = CHOLMOD(add) (C, E, one, minusone, TRUE, FALSE, cm) ;
	    if (n > NLARGE) progress (1, '.') ;
	    r = CHOLMOD(norm_sparse) (G, 0, cm) ;
	    if (G != NULL)
	    {
		MAXERR (maxerr, r, anorm) ;
	    }
	    CHOLMOD(drop) (0, G, cm) ;
	    r = CHOLMOD(norm_sparse) (G, 0, cm) ;
	    if (G != NULL)
	    {
		MAXERR (maxerr, r, anorm) ;
	    }
	    CHOLMOD(free_sparse) (&E, cm) ;
	    CHOLMOD(free_sparse) (&G, cm) ;
	}

	if (nrow == ncol)
	{
	    /* E = pattern of A */
	    E = CHOLMOD(copy) (A, 0, 0, cm) ;
	    /* G = E*E */
	    if (n > NLARGE) progress (1, '.') ;
	    G = CHOLMOD(ssmult) (E, E, 0, FALSE, FALSE, cm) ;
	    if (n > NLARGE) progress (1, '.') ;
	    CHOLMOD(free_sparse) (&E, cm) ;
	    CHOLMOD(free_sparse) (&G, cm) ;
	}

	cm->error_handler = save ;

	CHOLMOD(free_sparse) (&D, cm) ;
	CHOLMOD(free_sparse) (&C, cm) ;
	CHOLMOD(free_sparse) (&AT, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free P, Q, and their inverses */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(free) (ncol, sizeof (Int), fset, cm) ;
    CHOLMOD(free) (nrow, sizeof (Int), P, cm) ;
    CHOLMOD(free) (nrow, sizeof (Int), Pinv, cm) ;
    if (A->stype == 0)
    {
	CHOLMOD(free) (ncol, sizeof (Int), Q, cm) ;
    }
    CHOLMOD(free) (ncol, sizeof (Int), Qinv, cm) ;
    CHOLMOD(free) (nrow, sizeof (Int), Partition, cm) ;

    progress (0, '.') ;
    return (maxerr) ;
}
