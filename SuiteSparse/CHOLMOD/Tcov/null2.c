/* ========================================================================== */
/* === Tcov/null2 =========================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Null and error tests, continued. */

#include "cm.h"
#include "amd.h"
#ifndef NCAMD
#include "camd.h"
#endif

#define CSETSIZE 5


/* ========================================================================== */
/* === null2 ================================================================ */
/* ========================================================================== */

void null2 (cholmod_triplet *Tok, int do_nantests)
{
    double nm, gsave0, gsave1, r, anorm, beta [2], maxerr, xnan, rcond,
	ax, az, bx, bz, cx, cz, dx, dz, ex, ez ;
    cholmod_sparse *A, *C, *AT, *E, *F, *G, *Sok, *R0, *R1, *Aboth, *Axbad, *I1,
	*Abad, *R, *Acopy, *R3, *Abad2, *I, *I3, *Abad3, *AA, *Rt, *AF, *AFT,
	*I7, *C2, *R2, *Z ;
    cholmod_dense *Xok, *Bok, *Two, *X, *W, *XX, *YY, *Xbad2, *B, *Scale,
	*Y, *X1, *B1, *B2, *X7, *B7 ; 
    cholmod_factor *L, *L2, *L3, *L4, *L5, *L6, *Lcopy, *Lbad, *L7 ;
    cholmod_triplet *T, *T2, *Tz, *T3 ;
    Int *fsetok, *Pok, *Flag, *Head, *Cp, *Ci, *P2, *Parent, *Lperm,
	*Lp, *Li, *Lnz, *Lprev, *Lnext, *Ls, *Lpi, *Lpx, *Super, *Tj, *Ti,
	*Enz, *Ep, *Post, *Cmember, *CParent, *Partition, *Pinv, *ATi, *ATp,
	*LColCount, *ColCount, *First, *Level, *fsetbad, *Pbad, *Lz,
	*R2p, *R2i ;
    double *Xwork, *Cx, *x, *Lx, *Tx, *Az, *R2x ;
    size_t size, nznew, gsave2 ;
    SuiteSparse_long lr ;
    void *pp, *ii, *jj, *xx ;
    Int p, i, j, d, nrhs, nrow, ncol, stype, fsizeok, nz, ok, n2, trial, anz,
	nzmax, cset [CSETSIZE], Axbad_type, isreal, xtype, enz, Lxtype, Cxtype,
	Xxtype, Txtype, Abad2xtype, k, xtype2, Abad3xtype, save1, save2, save3,
	save4, ok1, fnz ;
    int option, asym ;
    FILE *f ;

    beta [0] = 1e-6 ;
    beta [1] = 0 ;
    xnan = 0 ;

    xtype = Tok->xtype ;
    isreal = (xtype == CHOLMOD_REAL) ;

    /* ---------------------------------------------------------------------- */
    /* hypot and divcomplex */
    /* ---------------------------------------------------------------------- */

    maxerr = 0 ;
    ax = 4.3 ;
    az = 9.2 ;
    for (i = 0 ; i <= 1 ; i++)
    {
	if (i == 0)
	{
	    bx = 3.14159 ;
	    bz = -1.2 ;
	}
	else
	{
	    bx = 0.9 ;
	    bz = -1.2 ;
	}
	/* c = a/b */
	CHOLMOD(divcomplex)(ax, az, bx, bz, &cx, &cz) ;
	/* d = c*b */
	dx = cx * bx - cz * bz ;
	dz = cz * bx + cx * bz ;
	/* e = d-a, which should be zero */
	ex = dx - ax ;
	ez = dz - az ;
	r = CHOLMOD(hypot)(ex, ez) ;
	MAXERR (maxerr, r, 1) ;
	OK (r < 1e-14) ;
    }

    /* ---------------------------------------------------------------------- */
    /* create objects to test */
    /* ---------------------------------------------------------------------- */

    printf ("\n------------------------null2 tests:\n") ;

    cm->error_handler = my_handler ;

    CHOLMOD(check_triplet)(Tok, cm) ;
    nrhs = 5 ;
    nrow = Tok->nrow ;
    ncol = Tok->ncol ;
    d = nrow + 2 ;

    A = CHOLMOD(triplet_to_sparse)(Tok, 0, cm) ;    /* [ */

    anorm = CHOLMOD(norm_sparse)(A, 1, cm) ;
    anz = A->nzmax ;

    AT = CHOLMOD(transpose)(A, 2, cm) ;	/* [ */

    printf ("xtrue:\n") ;
    Xok = xtrue (nrow, nrhs, d, xtype) ;	    /* [ */

    printf ("rhs:\n") ;
    Bok = rhs (A, nrhs, d) ;			    /* [ */
    printf ("fset:\n") ;

    fsetok = prand (ncol) ;	/* [ */			/* RAND */
    fsetbad = prand (ncol) ;	/* [ */			/* RAND */

    if (ncol > 0)
    {
	fsetbad [0] = -1 ;
    }
    Pbad = prand (nrow) ;	/* [ */				/* RAND */

    if (nrow > 0)
    {
	Pbad [0] = -1 ;
    }
    I1 = CHOLMOD(speye)(nrow+1, nrow+1, xtype, cm) ;	/* [ */

    fsizeok = (ncol < 2) ? ncol : (ncol/2) ;
    Pok = prand (nrow) ;    /* [  */				/* RAND */

    R2 = CHOLMOD(allocate_sparse)(nrow, 1, nrow, FALSE, TRUE, 0,    /* [ */
	    CHOLMOD_REAL, cm) ;
    OKP (R2) ;

    R2p = R2->p ;
    R2i = R2->i ;
    R2x = R2->x ;
    for (i = 0 ; i < nrow ; i++)
    {
	R2i [i] = Pok [i] ;
	R2x [i] = 1 ;
    }
    R2p [0] = 0 ;
    R2p [1] = nrow ;

    stype = A->stype ;
    Two = CHOLMOD(zeros)(1, 1, xtype, cm) ;	/* [ */
    *((double *)(Two->x)) = 2 ;

    Pinv = CHOLMOD(malloc)(nrow, sizeof (Int), cm) ;	    /* [ */
    Parent = CHOLMOD(malloc)(nrow, sizeof (Int), cm) ;
    Post = CHOLMOD(malloc)(nrow, sizeof (Int), cm) ;
    Cmember = CHOLMOD(malloc)(nrow, sizeof (Int), cm) ;
    CParent = CHOLMOD(malloc)(nrow, sizeof (Int), cm) ;
    Partition = CHOLMOD(malloc)(nrow, sizeof (Int), cm) ;
    ColCount = CHOLMOD(malloc)(nrow, sizeof (Int), cm) ;
    First = CHOLMOD(malloc)(nrow, sizeof (Int), cm) ;
    Level = CHOLMOD(malloc)(nrow, sizeof (Int), cm) ;

    printf ("etree:\n") ;

    if (AT->stype >= 0)
    {
	/* AT is unsymmetric, or symmetric/upper */
	CHOLMOD(etree)(AT, Parent, cm) ;
    }
    else
    {
	/* A is symmetric/upper */
	CHOLMOD(etree)(A, Parent, cm) ;
    }
    CHOLMOD(check_parent)(Parent, nrow, cm) ;
    for (cm->print = 0 ; cm->print <= ((nrow <= 30) ? 5 : 4) ; cm->print++)
    {
	CHOLMOD(print_parent)(Parent, nrow, "Parent", cm) ;
    }
    cm->print = 1 ;

    /* get row 0 and row 1 of A */
    R0 = NULL ;
    R1 = NULL ;
    Aboth = NULL ;
    Sok = NULL ;

    if (isreal)
    {
	Aboth = CHOLMOD(copy)(A, 0, 1, cm) ;	    /* [ */
	Sok = CHOLMOD(copy)(A, 0, 0, cm) ;
	Aboth->sorted = FALSE ;
    }

    if (isreal)	    /* [ */
    {
	if (nrow > 1)
	{
	    cm->print = 4 ;
	    if (nrow < 10)
	    {
		ok = CHOLMOD(print_sparse)(Aboth, "Aboth", cm) ; OK (ok) ;
	    }
	    i = 0 ;
	    R0 = CHOLMOD(submatrix)(Aboth, &i, 1, NULL, -1, TRUE, TRUE, cm) ;
	    ok = CHOLMOD(print_sparse)(R0, "Row zero", cm) ; OK (ok) ;
	    i = 1 ;
	    R1 = CHOLMOD(submatrix)(Aboth, &i, 1, NULL, -1, TRUE, TRUE, cm) ;
	    ok = CHOLMOD(print_sparse)(R1, "Row one", cm) ; OK (ok) ;
	    Rt = CHOLMOD(transpose)(R1, 1, cm) ;

	    C = CHOLMOD(ssmult)(R0, Rt, 0, TRUE, TRUE, cm) ;	OKP (C) ;
	    ok = CHOLMOD(print_sparse)(C, "(Row zero)*(Row one)'", cm) ;OK (ok);
	    ok = CHOLMOD(free_sparse)(&C, cm) ;  OK (ok) ;
	    ok = CHOLMOD(free_sparse)(&Rt, cm) ; OK (ok) ;
	    cm->print = 1 ;
	}
    }

    /* Abad: symmetric but not square, or null if A is square */
    if (A->nrow != A->ncol)
    {
	Abad = CHOLMOD(copy_sparse)(A, cm) ;	    /* [ */
	Abad->stype = 1 ;
    }
    else
    {
	Abad = NULL ;
    }

    /* Abad2: sparse matrix with invalid xtype */
    printf ("allocate Abad2:\n") ;
    Abad2 = CHOLMOD(copy_sparse)(A, cm) ;	    /* [ */
    cm->print = 4 ;
    CHOLMOD(print_sparse)(Abad2, "Abad2", cm) ;
    cm->print = 1 ;
    Abad2xtype = Abad2->xtype ;
    Abad2->xtype = -999 ;

    /* Xbad2: dense matrix with invalid xtype */
    printf ("allocate Xbad2:\n") ;
    Xbad2 = CHOLMOD(zeros)(2, 2, CHOLMOD_REAL, cm) ;	    /* [ */
    Xbad2->xtype = -911 ;

    /* ---------------------------------------------------------------------- */
    /* expect lots of errors */
    /* ---------------------------------------------------------------------- */

    printf ("\n------------------------null2 tests: ERRORs will occur\n") ;
    cm->error_handler = NULL ;

    /* ---------------------------------------------------------------------- */
    /* transpose */
    /* ---------------------------------------------------------------------- */

    C = CHOLMOD(transpose)(Abad2, 1, cm) ;	    NOP (C) ;
    ok = CHOLMOD(sort)(Abad2, cm) ;				    NOT (ok) ;
    ok = CHOLMOD(sort)(NULL, cm) ;				    NOT (ok) ;

    if (nrow > 0)
    {
	C = CHOLMOD(ptranspose)(A, 1, Pbad, NULL, 0, cm) ;	    NOP (C) ;
    }

    C = CHOLMOD(allocate_sparse)(ncol, nrow, anz, TRUE, TRUE,
	    -(A->stype), xtype, cm) ;			    OKP (C) ;
    ok = CHOLMOD(transpose_unsym)(A, 1, NULL, NULL, 0,
	    C, cm) ;					    OK (ok);
    ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;

    C = CHOLMOD(allocate_sparse)(ncol, nrow, anz, TRUE, FALSE,
	    -(A->stype), xtype, cm) ;			    OKP (C) ;
    ok = CHOLMOD(transpose_unsym)(A, 1, NULL, NULL, 0,
	    C, cm) ;					    OK (ok);
    ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;

    C = CHOLMOD(allocate_sparse)(ncol, nrow, anz, TRUE, FALSE,
	    -(A->stype), xtype, cm) ;			    OKP (C) ;
    ok = CHOLMOD(transpose_unsym)(A, 1, Pok, NULL, 0,
	    C, cm) ;					    OK (ok);
    ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;

    C = CHOLMOD(allocate_sparse)(ncol, nrow, anz, TRUE, FALSE,
	    -(A->stype), xtype, cm) ;			    OKP (C) ;
    ok = CHOLMOD(transpose_unsym)(A, 1, Pok, fsetok, fsizeok,
	    C, cm) ;					    OK (ok);
    ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;

    C = CHOLMOD(allocate_sparse)(ncol, nrow, anz, TRUE, FALSE,
	    -(A->stype), xtype, cm) ;			    OKP (C) ;
    ok = CHOLMOD(transpose_unsym)(A, 1, NULL, fsetok, fsizeok,
	    C, cm) ;					    OK (ok);
    ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;

    C = CHOLMOD(allocate_sparse)(ncol, nrow, anz, TRUE, FALSE,
	    -(A->stype), CHOLMOD_PATTERN, cm) ;		    OKP (C) ;
    ok = CHOLMOD(transpose_unsym)(A, 1, NULL, fsetok, fsizeok,
	    C, cm) ;					    OK (ok);

    E = CHOLMOD(allocate_sparse)(nrow, ncol, anz, TRUE, FALSE,
	    (A->stype), CHOLMOD_PATTERN, cm) ;		    OKP (C) ;
    enz = CHOLMOD(nnz)(E, cm) ;
    OK (enz == 0) ;
    ok = CHOLMOD(transpose_unsym)(C, 1, NULL, Pok, nrow,
	    E, cm) ;					    OK (ok);
    ok = CHOLMOD(free_sparse)(&E, cm) ;			    OK (ok) ;

    if (A->nrow != A->ncol)
    {
	ok = CHOLMOD(transpose_sym)(A, 1, NULL, C, cm) ;	    NOT (ok) ;
    }
    ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;

    /* Abad3: sparse matrix with invalid xtype [ */
    printf ("allocate Abad3:\n") ;
    C = CHOLMOD(copy_sparse)(A, cm) ;
    Abad3 = CHOLMOD(transpose)(A, 1, cm) ;	OKP (Abad3) ;
    E = CHOLMOD(transpose)(A, 1, cm) ;		OKP (E) ;
    Abad3xtype = Abad3->xtype ;
    Abad3->xtype = -999 ;

    ok = CHOLMOD(transpose_sym)(C, 1, NULL, Abad3, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(transpose_sym)(Abad3, 1, NULL, C, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(transpose_unsym)(C, 1, NULL, NULL, 0, Abad3, cm) ;NOT (ok);
    ok = CHOLMOD(transpose_unsym)(Abad3, 1, NULL, NULL, 0, C, cm) ;NOT (ok);

    switch (xtype)
    {
	case CHOLMOD_REAL:
	    CHOLMOD(sparse_xtype)(CHOLMOD_COMPLEX, E, cm) ;
	    break ;
	case CHOLMOD_COMPLEX:
	    CHOLMOD(sparse_xtype)(CHOLMOD_ZOMPLEX, E, cm) ;
	    break ;
	case CHOLMOD_ZOMPLEX:
	    CHOLMOD(sparse_xtype)(CHOLMOD_COMPLEX, E, cm) ;
	    break ;
    }

    printf ("mismatch start [:\n") ;
    ok = CHOLMOD(transpose_sym)(C, 1, NULL, E, cm) ;		NOT (ok) ;
    ok = CHOLMOD(transpose_sym)(E, 1, NULL, C, cm) ;		NOT (ok) ;
    ok = CHOLMOD(transpose_sym)(C, 2, NULL, E, cm) ;		NOT (ok) ;
    ok = CHOLMOD(transpose_sym)(E, 2, NULL, C, cm) ;		NOT (ok) ;
    ok = CHOLMOD(transpose_unsym)(C, 1, NULL, NULL, 0, E, cm) ; NOT (ok);
    ok = CHOLMOD(transpose_unsym)(E, 1, NULL, NULL, 0, C, cm) ; NOT (ok);
    ok = CHOLMOD(transpose_unsym)(C, 2, NULL, NULL, 0, E, cm) ; NOT (ok);
    ok = CHOLMOD(transpose_unsym)(E, 2, NULL, NULL, 0, C, cm) ; NOT (ok);
    printf ("mismatch done ]\n") ;

    printf ("wrong dim [:\n") ;
    ok = CHOLMOD(transpose_sym)(C, 1, NULL, I1, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(transpose_sym)(I1, 1, NULL, C, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(transpose_unsym)(C, 1, NULL, NULL, 0, I1, cm) ;    NOT (ok);
    ok = CHOLMOD(transpose_unsym)(I1, 1, NULL, NULL, 0, C, cm) ;    NOT (ok);
    ok = CHOLMOD(transpose_unsym)(C, 2, NULL, NULL, 0, I1, cm) ;    NOT (ok);
    ok = CHOLMOD(transpose_unsym)(I1, 2, NULL, NULL, 0, C, cm) ;    NOT (ok);
    printf ("wrong dim ]\n") ;

    nz = CHOLMOD(nnz)(C, cm) ;
    if (nz > 10)
    {
	printf ("F too small [:\n") ;
	F = CHOLMOD(allocate_sparse)(C->ncol, C->nrow, C->nzmax-5, TRUE, TRUE,
		-(C->stype), C->xtype, cm) ;			    OKP (F) ;
	ok = CHOLMOD(transpose_sym)(C, 1, NULL, F, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(transpose_unsym)(C, 1, NULL, NULL, 0, F, cm) ; NOT (ok);
	CHOLMOD(free_sparse)(&F, cm) ;
	printf ("F too small ]\n") ;
    }

    ok = CHOLMOD(transpose_unsym)(C, 1, NULL, NULL, 0, NULL, cm) ; NOT (ok);
    ok = CHOLMOD(transpose_unsym)(NULL, 1, NULL, NULL, 0, C, cm) ; NOT (ok);

    ok = CHOLMOD(transpose_sym)(C, 1, NULL, NULL, cm) ;	    NOT (ok);
    ok = CHOLMOD(transpose_sym)(NULL, 1, NULL, C, cm) ;	    NOT (ok);

    CHOLMOD(free_sparse)(&C, cm) ;
    CHOLMOD(free_sparse)(&E, cm) ;

    Abad3->xtype = Abad3xtype ;
    CHOLMOD(free_sparse)(&Abad3, cm) ;		    /* ] */

    cm->status = CHOLMOD_OK ;

    /* ---------------------------------------------------------------------- */
    /* aat */
    /* ---------------------------------------------------------------------- */

    C = CHOLMOD(aat)(NULL, NULL, 0, 0, cm) ;			    NOP (C) ;
    if (stype)
    {
	C = CHOLMOD(aat)(A, fsetok, fsizeok, 0, cm) ;
	NOP (C) ;
    }
    else
    {
	C = CHOLMOD(aat)(A, fsetok, fsizeok, 0, cm) ;		    OKP (C) ;
	CHOLMOD(free_sparse)(&C, cm) ;
	C = CHOLMOD(aat)(Abad2, fsetok, fsizeok, 0, cm) ;	    NOP (C) ;
    }

    /* ---------------------------------------------------------------------- */
    /* add */
    /* ---------------------------------------------------------------------- */

    C = CHOLMOD(add)(A, NULL, one, one, TRUE, TRUE, cm) ;	    NOP (C) ;
    C = CHOLMOD(add)(NULL, AT, one, one, TRUE, TRUE, cm) ;	    NOP (C) ;

    C = CHOLMOD(add)(A, AT, one, one, TRUE, TRUE, cm) ;
    if (A->nrow == A->ncol && isreal)
    {
	OKP (C) ;
	/* C should equal 2*A if A=A' */
	if (stype)
	{
	    double *s ;

	    E = CHOLMOD(copy_sparse)(A, cm) ;
	    CHOLMOD(scale)(Two, CHOLMOD_SCALAR, E, cm) ;
	    F = CHOLMOD(add)(C, E, one, minusone, TRUE, TRUE, cm) ;
	    CHOLMOD(drop)(0., F, cm) ;
	    nz = CHOLMOD(nnz)(F, cm) ;
	    OK (nz == 0) ;
	    CHOLMOD(free_sparse)(&E, cm) ;
	    CHOLMOD(free_sparse)(&F, cm) ;

	    Scale = CHOLMOD(zeros)(nrow, 1, CHOLMOD_REAL, cm) ;

	    s = Scale->x ;
	    for (i = 0 ; i < nrow ; i++)
	    {
		s [i] = 2 ;
	    }
	    E = CHOLMOD(copy_sparse)(A, cm) ;
	    CHOLMOD(scale)(Scale, CHOLMOD_ROW, E, cm) ;
	    F = CHOLMOD(add)(C, E, one, minusone, TRUE, TRUE, cm) ;
	    CHOLMOD(drop)(0., F, cm) ;
	    nz = CHOLMOD(nnz)(F, cm) ;
	    r = CHOLMOD(norm_sparse)(F, 0, cm) ;
	    OK (nz == 0) ;
	    CHOLMOD(free_sparse)(&E, cm) ;
	    CHOLMOD(free_sparse)(&F, cm) ;

	    E = CHOLMOD(copy_sparse)(A, cm) ;
	    CHOLMOD(scale)(Scale, CHOLMOD_COL, E, cm) ;
	    F = CHOLMOD(add)(C, E, one, minusone, TRUE, TRUE, cm) ;
	    CHOLMOD(drop)(0., F, cm) ;
	    nz = CHOLMOD(nnz)(F, cm) ;
	    r = CHOLMOD(norm_sparse)(F, 0, cm) ;
	    OK (nz == 0) ;
	    CHOLMOD(free_sparse)(&E, cm) ;
	    CHOLMOD(free_sparse)(&F, cm) ;

	    for (i = 0 ; i < nrow ; i++)
	    {
		s [i] = sqrt (2) ;
	    }
	    E = CHOLMOD(copy_sparse)(A, cm) ;
	    CHOLMOD(scale)(Scale, CHOLMOD_SYM, E, cm) ;
	    F = CHOLMOD(add)(C, E, one, minusone, TRUE, TRUE, cm) ;
	    CHOLMOD(drop)(0., F, cm) ;
	    nz = CHOLMOD(nnz)(F, cm) ;
	    r = CHOLMOD(norm_sparse)(F, 0, cm) ;
	    OK (r < 1e-12*anorm) ;

	    Scale->x = NULL ;
	    CHOLMOD(scale)(Scale, CHOLMOD_SYM, E, cm) ;
	    Scale->x = s ;

	    OKP (E) ;
	    OKP (cm) ;
	    ok = CHOLMOD(scale)(NULL, CHOLMOD_ROW, E, cm) ;	    NOT (ok) ;
	    ok = CHOLMOD(scale)(Scale, CHOLMOD_SYM, NULL, cm) ;	    NOT (ok) ;
	    ok = CHOLMOD(scale)(NULL, CHOLMOD_SYM, NULL, cm) ;	    NOT (ok) ;
	    ok = CHOLMOD(scale)(Scale, -1, E, cm) ;		    NOT (ok) ;

	    CHOLMOD(free_sparse)(&E, cm) ;
	    CHOLMOD(free_sparse)(&F, cm) ;
	    CHOLMOD(free_dense)(&Scale, cm) ;

	}
	CHOLMOD(free_sparse)(&C, cm) ;
    }
    else
    {
	NOP (C) ;
    }

    Axbad = CHOLMOD(copy_sparse)(A, cm) ;	/* [ */
    Axbad_type = Axbad->xtype ;
    Axbad->xtype = CHOLMOD_COMPLEX ;
    C = CHOLMOD(add)(A, Axbad, one, one, TRUE, TRUE, cm) ;	    NOP (C) ;

    if (nrow > 1 && xtype == CHOLMOD_REAL)
    {
	/* C = A (0,:) + A (1,:) */
	C = CHOLMOD(add)(R0, R1, one, one, TRUE, TRUE, cm) ;	    OKP (C) ;
	OK (CHOLMOD(check_sparse)(C, cm)) ;
	ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;
    }
    ok = CHOLMOD(free_sparse)(&C, cm) ;				    OK (ok) ;
    ok = CHOLMOD(free_sparse)(NULL, cm) ;			    OK (ok) ;

    /* ---------------------------------------------------------------------- */
    /* sparse */
    /* ---------------------------------------------------------------------- */

    cm->print = 4 ;

    ok = CHOLMOD(reallocate_sparse)(10, NULL, cm) ;		    NOT (ok) ;
    C = CHOLMOD(allocate_sparse)(10, 10, 10, TRUE, TRUE, 0, -1, cm) ; NOP (C) ;
    ok = CHOLMOD(reallocate_sparse)(Abad2->nzmax, Abad2, cm) ;	    NOT (ok) ;
    C = CHOLMOD(copy_sparse)(Abad2, cm) ;			    NOP (C) ;
    C = CHOLMOD(allocate_sparse)(2, 3, 6, TRUE, TRUE, 1, 0, cm) ;   NOP (C) ;

    C = CHOLMOD(copy)(A, 0, -1, cm) ;				    OKP (C) ;
    E = unpack (C) ;						    OKP (E) ;
    F = CHOLMOD(copy_sparse)(E, cm) ;				    OKP (F) ;
    ok = CHOLMOD(sparse_xtype)(CHOLMOD_REAL, C, cm) ;		    OK (ok) ;
    ok = CHOLMOD(sparse_xtype)(CHOLMOD_REAL, F, cm) ;		    OK (ok) ;
    /* G = C-F */
    G = CHOLMOD(add)(C, F, one, minusone, TRUE, FALSE, cm)  ;	    OKP (G) ;
    ok = CHOLMOD(drop)(0., G, cm) ;				    OK (ok) ;
    nz = CHOLMOD(nnz)(G, cm) ;
    CHOLMOD(print_sparse)(C, "C", cm) ;
    CHOLMOD(print_sparse)(E, "E", cm) ;
    CHOLMOD(print_sparse)(F, "F", cm) ;
    CHOLMOD(print_sparse)(G, "G", cm) ;

    OK (nz == 0) ;
    CHOLMOD(free_sparse)(&C, cm) ;
    CHOLMOD(free_sparse)(&E, cm) ;
    CHOLMOD(free_sparse)(&F, cm) ;
    CHOLMOD(free_sparse)(&G, cm) ;

    cm->print = 1 ;

    /* ---------------------------------------------------------------------- */
    /* scale */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(scale)(Two, -1, C, cm) ;			    NOT (ok) ;
    if (nrow > 1)
    {
	E = CHOLMOD(copy_sparse)(A, cm) ;			    OKP (E) ;
	CHOLMOD(scale)(Two, CHOLMOD_ROW, E, cm) ;		    NOT (ok) ;
	ok = CHOLMOD(free_sparse)(&E, cm) ;			    OK (ok) ;
    }

    /* ---------------------------------------------------------------------- */
    /* amd */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(amd)(NULL, NULL, 0, NULL, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(amd)(A, NULL, 0, NULL, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(amd)(NULL, NULL, 0, Pok, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(amd)(A, NULL, 0, Pok, cm) ;			    OK (ok) ;
    cm->current = -1 ;
    ok = CHOLMOD(amd)(A, NULL, 0, Pok, cm) ;			    OK (ok) ;
    cm->current = 0 ;
    ok = CHOLMOD(print_perm)(Pok, nrow, nrow, "AMD perm", cm) ;	    OK (ok) ;
    i = cm->print ;
    cm->print = 4 ;
    if (A->nrow < 1000 && isreal)
    {
	CHOLMOD(print_sparse)(Aboth, "Aboth", cm) ;
	ok = CHOLMOD(amd)(Aboth, NULL, 0, Pok, cm) ;		    OK (ok) ;
    }
    cm->print = i ;
    ok = CHOLMOD(amd)(Abad2, NULL, 0, Pok, cm) ;		    NOT (ok) ;

    /* ---------------------------------------------------------------------- */
    /* camd */
    /* ---------------------------------------------------------------------- */

#ifndef NCAMD
    ok = CHOLMOD(camd)(NULL, NULL, 0, NULL, NULL, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(camd)(A, NULL, 0, NULL, NULL, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(camd)(NULL, NULL, 0, NULL, Pok, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(camd)(A, NULL, 0, NULL, Pok, cm) ;		    OK (ok) ;
    cm->current = -1 ;
    ok = CHOLMOD(camd)(A, NULL, 0, NULL, Pok, cm) ;		    OK (ok) ;
    cm->current = 0 ;
    ok = CHOLMOD(print_perm)(Pok, nrow, nrow, "CAMD perm", cm) ;    OK (ok) ;
    i = cm->print ;
    cm->print = 4 ;
    if (A->nrow < 1000 && isreal)
    {
	CHOLMOD(print_sparse)(Aboth, "Aboth", cm) ;
	ok = CHOLMOD(camd)(Aboth, NULL, 0, NULL, Pok, cm) ;	    OK (ok) ;
    }
    cm->print = i ;
    ok = CHOLMOD(camd)(Abad2, NULL, 0, NULL, Pok, cm) ;		    NOT (ok) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* analyze */
    /* ---------------------------------------------------------------------- */

    cm->nmethods = 1 ;
    cm->method [0].ordering = -1 ;
    ok = CHOLMOD(print_common)("Bad cm", cm) ;			    NOT (ok) ;
    ok = CHOLMOD(analyze_ordering)(NULL, 0, NULL, NULL, 0,
	    NULL, NULL, NULL, NULL, NULL, cm) ;			    NOT (ok) ;
    L = CHOLMOD(analyze)(NULL, cm) ;				    NOP (L) ;
    L = CHOLMOD(analyze)(Abad2, cm) ;				    NOP (L) ;
    L = CHOLMOD(analyze)(A, cm) ;				    NOP (L) ;

    /* test AMD backup strategy */
    cm->nmethods = 2 ;
    cm->method [0].ordering = -1 ;
    cm->method [1].ordering = -1 ;
    L = CHOLMOD(analyze)(A, cm) ;				    OKP (L) ;

    cm->nmethods = 0 ;	/* restore defaults */
    cm->method [0].ordering = CHOLMOD_GIVEN ;
    cm->method [1].ordering = CHOLMOD_AMD ;
    cm->print = 4 ;
    ok = CHOLMOD(print_common)("OKcm", cm) ;			    OK (ok) ;
    ok = CHOLMOD(print_factor)(L, "L symbolic", cm) ;		    OK (ok) ;
    cm->print = 1 ;
    ok = CHOLMOD(free_factor)(&L, cm) ;				    OK (ok) ;
    ok = CHOLMOD(free_factor)(&L, cm) ;				    OK (ok) ;
    ok = CHOLMOD(free_factor)(NULL, cm) ;			    OK (ok) ;

    /* ---------------------------------------------------------------------- */
    /* band */
    /* ---------------------------------------------------------------------- */

    C = CHOLMOD(band)(NULL, 0, 0, 0, cm) ;			    NOP (C) ;
    C = CHOLMOD(band)(Abad2, 0, 0, 0, cm) ;			    NOP (C) ;

    /* ---------------------------------------------------------------------- */
    /* ccolamd */
    /* ---------------------------------------------------------------------- */

#ifndef NCAMD
    ok = CHOLMOD(ccolamd)(NULL, fsetok, fsizeok, NULL, Pok, cm) ;   NOT (ok) ;
    ok = CHOLMOD(ccolamd)(A, fsetok, fsizeok, NULL, NULL, cm) ;   NOT (ok) ;

    ok = CHOLMOD(ccolamd)(A, fsetok, fsizeok, NULL, Pok, cm) ;
    if (stype)
    {
	NOT (ok) ;
    }
    else
    {
	OK (ok) ;
    }

    cm->current = -1 ;
    ok = CHOLMOD(ccolamd)(A, fsetok, fsizeok, NULL, Pok, cm) ;
    cm->current = 0 ;
    if (stype)
    {
	NOT (ok) ;
    }
    else
    {
	OK (ok) ;
    }
#endif

    /* ---------------------------------------------------------------------- */
    /* copy */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(print_sparse)(Abad, "Abad", cm) ;

    C = CHOLMOD(copy)(Abad, 0, 1, cm) ;
    CHOLMOD(print_sparse)(C, "copy of Abad", cm) ;
    NOP (C) ;

    C = CHOLMOD(copy_sparse)(Abad, cm) ;
    CHOLMOD(print_sparse)(C, "another copy of Abad", cm) ;
    NOP (C) ;

    C = CHOLMOD(copy)(A, 0, -1, cm) ;				    OKP (C) ;
    OK (nzdiag (C) == 0) ;

    ok = CHOLMOD(free_sparse)(&C, cm) ;				    OK (ok) ;

    /* ---------------------------------------------------------------------- */
    /* submatrix */
    /* ---------------------------------------------------------------------- */

    if (A->nrow == A->ncol)
    {
	/* submatrix cannot operation on symmetric matrices */
	C = CHOLMOD(copy)(A, 1, 0, cm) ;			    OKP (C) ;
	E = CHOLMOD(submatrix)(C, NULL, -1, NULL, -1, TRUE, TRUE, cm); NOP (E) ;
	ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;
    }

    E = CHOLMOD(submatrix)(Abad2, NULL, -1, NULL, -1, TRUE, TRUE, cm) ; NOP(E) ;

    if (A->stype == 0 && isreal)
    {
	/* E = A(:,:) */
	E = CHOLMOD(submatrix)(NULL, NULL,-1, NULL,-1, TRUE, TRUE, cm) ; NOP(E);
	E = CHOLMOD(submatrix)(A, NULL, -1, NULL, -1, TRUE, TRUE, cm) ; OKP(E) ;
	/* C = A-E */
	C = CHOLMOD(add)(A, E, one, minusone, TRUE, TRUE, cm) ;	    OKP (C) ;
	ok = CHOLMOD(drop)(0., C, cm) ;				    OK (ok) ;
	ok = CHOLMOD(drop)(0., Abad2, cm) ;			    NOT(ok) ;
	nz = CHOLMOD(nnz)(C, cm) ;
	OK (nz == 0) ;
	ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;
	ok = CHOLMOD(free_sparse)(&E, cm) ;			    OK (ok) ;

	i = -1 ;
	E = CHOLMOD(submatrix)(A, &i, 1, NULL, -1, TRUE, TRUE, cm) ; NOP(E) ;
	E = CHOLMOD(submatrix)(A, NULL, -1, &i, 1, TRUE, TRUE, cm) ; NOP(E) ;
	E = CHOLMOD(submatrix)(A, &i, 1, &i, 1, TRUE, TRUE, cm) ;    NOP(E) ;
	i = 0 ;
	j = -1 ;
	E = CHOLMOD(submatrix)(A, &i, 1, &j, 1, TRUE, TRUE, cm) ;    NOP(E) ;
    }

    /* ---------------------------------------------------------------------- */
    /* read */
    /* ---------------------------------------------------------------------- */

    C = CHOLMOD(read_sparse)(NULL, cm) ;			    NOP (C) ;
    X = CHOLMOD(read_dense)(NULL, cm) ;				    NOP (X) ;
    pp = CHOLMOD(read_matrix)(NULL, 1, NULL, cm) ;		    NOP (pp) ;
    pp = CHOLMOD(read_matrix)((FILE *) 1, 1, NULL, cm) ;	    NOP (pp) ;
    T3 = CHOLMOD(read_triplet)(NULL, cm) ;			    NOP (T3) ;

    /* ---------------------------------------------------------------------- */
    /* write */
    /* ---------------------------------------------------------------------- */

    asym = CHOLMOD(write_sparse) (NULL, NULL, NULL, NULL, cm) ;	NOT (asym>=0);
    asym = CHOLMOD(write_sparse) ((FILE *) 1, NULL, NULL, NULL, cm) ;
								NOT (asym>=0);

    asym = CHOLMOD(write_dense) (NULL, NULL, NULL, cm) ;	NOT (asym>=0);
    asym = CHOLMOD(write_dense) ((FILE *) 1, NULL, NULL, cm) ;	NOT (asym>=0);

    f = fopen ("temp4.mtx", "w") ;
    asym = CHOLMOD(write_sparse) (f, A, NULL, "garbage.txt", cm) ;
    fclose (f) ;
    printf ("write_sparse, asym: %d\n", asym) ;
    OK (asym == EMPTY) ;

    if (A != NULL)
    {
	save1 = A->xtype ;
	A->xtype = 999 ;
	f = fopen ("temp4.mtx", "w") ;
	asym = CHOLMOD(write_sparse) (f, A, NULL, NULL, cm) ;
	fclose (f) ;
	printf ("write_sparse, asym: %d\n", asym) ;
	OK (asym == EMPTY) ;
	A->xtype = save1 ;
    }

    Z = CHOLMOD(speye) (nrow+1, ncol+1, CHOLMOD_PATTERN, cm) ;
    f = fopen ("temp4.mtx", "w") ;
    asym = CHOLMOD(write_sparse) (f, A, Z, NULL, cm) ;
    fclose (f) ;
    printf ("write_sparse, asym: %d with Z\n", asym) ;
    OK (asym == EMPTY) ;

    Z->xtype = 999 ;
    f = fopen ("temp4.mtx", "w") ;
    asym = CHOLMOD(write_sparse) (f, A, Z, NULL, cm) ;
    fclose (f) ;
    printf ("write_sparse, asym: %d with Z2\n", asym) ;
    OK (asym == EMPTY) ;
    Z->xtype = CHOLMOD_PATTERN ;

    CHOLMOD(free_sparse) (&Z, cm) ;

    Z = CHOLMOD(speye) (0, ncol+1, CHOLMOD_PATTERN, cm) ;
    f = fopen ("temp4.mtx", "w") ;
    asym = CHOLMOD(write_sparse) (f, A, Z, NULL, cm) ;
    fclose (f) ;
    printf ("write_sparse, asym: %d with Z\n", asym) ;
    if (A == NULL)
    {
	OK (asym == EMPTY) ;
    }
    else
    {
	OK (asym > EMPTY) ;
    }
    CHOLMOD(free_sparse) (&Z, cm) ;

    X = CHOLMOD(ones) (4, 4, CHOLMOD_REAL, cm) ;
    f = fopen ("temp6.mtx", "w") ;
    asym = CHOLMOD(write_dense) (f, X, "garbage.txt", cm) ;
    fclose (f) ;
    OK (asym == EMPTY) ;

    X->xtype = 999 ;
    f = fopen ("temp6.mtx", "w") ;
    asym = CHOLMOD(write_dense) (f, X, NULL, cm) ;
    fclose (f) ;
    OK (asym == EMPTY) ;
    X->xtype = CHOLMOD_REAL ;
    CHOLMOD(free_dense) (&X, cm) ;

    /* ---------------------------------------------------------------------- */
    /* print_common */
    /* ---------------------------------------------------------------------- */

    cm->print = 4 ;
    ok = CHOLMOD(print_common)("Null", NULL) ;			    NOT (ok) ;
    for (cm->status = CHOLMOD_INVALID ; cm->status <= CHOLMOD_DSMALL ;
	    cm->status++)
    {
	ok = CHOLMOD(print_common)("status", cm) ;		    OK (ok) ;
    }
    cm->status = 999 ;
    ok = CHOLMOD(print_common)("bad status", cm) ;		    NOT (ok) ;
    cm->status = CHOLMOD_OK ;

    Flag = cm->Flag ;
    cm->Flag = NULL ;
    ok = CHOLMOD(print_common)("bad Flag", cm) ;		    NOT (ok) ;
    cm->Flag = Flag ;
    ok = CHOLMOD(print_common)("ok Flag", cm) ;			    OK (ok) ;

    Flag [0] = Int_max ;
    ok = CHOLMOD(print_common)("bad Flag", cm) ;		    NOT (ok) ;
    Flag [0] = -1 ;
    ok = CHOLMOD(print_common)("ok Flag", cm) ;			    OK (ok) ;

    Head = cm->Head ;
    cm->Head = NULL ;
    ok = CHOLMOD(print_common)("bad Head", cm) ;		    NOT (ok) ;
    cm->Head = Head ;
    ok = CHOLMOD(print_common)("ok Head", cm) ;			    OK (ok) ;

    Head [0] = Int_max ;
    ok = CHOLMOD(print_common)("bad Head", cm) ;		    NOT (ok) ;
    Head [0] = -1 ;
    ok = CHOLMOD(print_common)("ok Head", cm) ;			    OK (ok) ;

    Xwork = cm->Xwork ;
    cm->Xwork = NULL ;
    ok = CHOLMOD(print_common)("bad Xwork", cm) ;		    NOT (ok) ;
    cm->Xwork = Xwork ;
    ok = CHOLMOD(print_common)("ok Xwork", cm) ;		    OK (ok) ;

    Xwork [0] = 1 ;
    ok = CHOLMOD(print_common)("bad Xwork", cm) ;		    NOT (ok) ;
    Xwork [0] = 0 ;
    ok = CHOLMOD(print_common)("ok Xwork", cm) ;		    OK (ok) ;

    p = cm->nmethods ;
    i = cm->method [0].ordering ;
    cm->nmethods = 1 ;
    cm->method [0].ordering = 999 ;
    ok = CHOLMOD(print_common)("bad method", cm) ;		    NOT (ok) ;
    cm->nmethods = p ;
    cm->method [0].ordering = i ;

    /* ---------------------------------------------------------------------- */
    /* print_sparse */
    /* ---------------------------------------------------------------------- */

    C = CHOLMOD(copy_sparse)(A, cm) ;				    OKP (C) ;

    cm->print = 3 ;
    C->itype = EMPTY ;
    ok = CHOLMOD(print_sparse)(C, "CIbad", cm) ;		    NOT (ok) ;
    C->itype = CHOLMOD_INTLONG ;
    ok = CHOLMOD(print_sparse)(C, "Cibad", cm) ;		    NOT (ok) ;
    C->itype = cm->itype ;
    cm->print = 1 ;

    cm->print = 4 ;
#ifdef LONG
    C->itype = CHOLMOD_INT ;
#else
    C->itype = CHOLMOD_LONG ;
#endif
    ok = CHOLMOD(print_sparse)(C, "Cibad2", cm) ;		    NOT (ok) ;
    C->itype = cm->itype ;
    cm->print = 1 ;

    C->dtype = CHOLMOD_SINGLE ;
    ok = CHOLMOD(print_sparse)(C, "Cdbad", cm) ;		    NOT (ok) ;
    C->dtype = EMPTY ;
    ok = CHOLMOD(print_sparse)(C, "CDbad", cm) ;		    NOT (ok) ;
    C->dtype = CHOLMOD_DOUBLE ;

    Cxtype = C->xtype ;
    C->xtype = EMPTY ;
    ok = CHOLMOD(print_sparse)(C, "CXbad", cm) ;		    NOT (ok) ;
    C->xtype = Cxtype ;

    ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;			    OK (ok) ;

    Cp = C->p ;
    Ci = C->i ;
    Cx = C->x ;

    C->p = NULL ;
    ok = CHOLMOD(print_sparse)(C, "Cp bad", cm) ;		    NOT (ok) ;
    C->p = Cp ;
    ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;			    OK (ok) ;

    C->i = NULL ;
    ok = CHOLMOD(print_sparse)(C, "Ci bad", cm) ;		    NOT (ok) ;
    C->i = Ci ;
    ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;			    OK (ok) ;

    C->x = NULL ;
    ok = CHOLMOD(print_sparse)(C, "Cx bad", cm) ;		    NOT (ok) ;
    C->x = Cx ;
    ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;			    OK (ok) ;

    Cp [0] = 42 ;
    ok = CHOLMOD(print_sparse)(C, "Cp [0] bad", cm) ;		    NOT (ok) ;
    Cp [0] = 0 ;
    ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;			    OK (ok) ;

    p = Cp [ncol] ;
    Cp [ncol] = C->nzmax + 10 ;
    ok = CHOLMOD(print_sparse)(C, "Cp [ncol] bad", cm) ;	    NOT (ok) ;
    Cp [ncol] = p ;
    ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;			    OK (ok) ;

    p = Cp [ncol] ;
    Cp [ncol] = -1 ;
    ok = CHOLMOD(print_sparse)(C, "Cp [ncol] neg", cm) ;	    NOT (ok) ;
    Cp [ncol] = p ;
    ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;			    OK (ok) ;

    if (ncol > 0)
    {
	p = Cp [1] ;
	Cp [1] = 2*nrow + 1 ;
	ok = CHOLMOD(print_sparse)(C, "Cp [1] bad", cm) ;	    NOT (ok) ;
	Cp [1] = p ;
	ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;		    OK (ok) ;
    }

    if (ncol > 2)
    {
	p = Cp [2] ;
	Cp [2] = Cp [1] - 1 ;
	ok = CHOLMOD(print_sparse)(C, "Cp [2] bad", cm) ;	    NOT (ok) ;
	Cp [2] = p ;
	ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;		    OK (ok) ;
    }

    if (Cp [ncol] > 0)
    {
	i = Ci [0] ;
	Ci [0] = -1 ;
	ok = CHOLMOD(print_sparse)(C, "Ci [0] neg", cm) ;	    NOT (ok) ;
	Ci [0] = i ;
	ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;		    OK (ok) ;
    }

    if (ncol > 0 && C->sorted && Cp [1] - Cp [0] > 2)
    {
	i = Ci [0] ;
	Ci [0] = nrow-1 ;
	ok = CHOLMOD(print_sparse)(C, "Ci [0] unsorted", cm) ;	    NOT (ok) ;
	Ci [0] = i ;
	ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;		    OK (ok) ;
    }

    if (ncol > 0 && C->sorted && ncol > 2 && Cp [1] - Cp [0] > 2)
    {
	/* swap the first two entries */
	p = Ci [0] ;
	Ci [0] = Ci [1] ;
	Ci [1] = p ;
	ok = CHOLMOD(print_sparse)(C, "Ci [0] unsorted", cm) ;	    NOT (ok) ;
	C->sorted = FALSE ;
	ok = CHOLMOD(print_sparse)(C, "Ci [0] unsorted", cm) ;	    OK (ok) ;
	Ci [1] = Ci [0] ;
	ok = CHOLMOD(print_sparse)(C, "Ci [0] duplicate", cm) ;	    NOT (ok) ;
	Ci [1] = p ;
	ok = CHOLMOD(print_sparse)(C, "Ci [0] unsorted", cm) ;	    OK (ok) ;
	p = Ci [0] ;
	Ci [0] = Ci [1] ;
	Ci [1] = p ;
	ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;		    OK (ok) ;
	C->sorted = TRUE ;
	ok = CHOLMOD(print_sparse)(C, "C ok", cm) ;		    OK (ok) ;
    }

    E = CHOLMOD(copy_sparse)(C, cm) ;				    OKP (E) ;
    Enz = CHOLMOD(malloc)(ncol, sizeof (Int), cm) ;		    OKP (Enz) ;
    E->nz = Enz ;
    Ep = E->p ;
    for (j = 0 ; j < ncol ; j++)
    {
	Enz [j] = Ep [j+1] - Ep [j] ;
    }
    E->packed = FALSE ;
    ok = CHOLMOD(print_sparse)(E, "E unpacked ok", cm) ;	    OK (ok) ;

    ok = CHOLMOD(band_inplace)(0, 0, 0, E, cm) ;		    NOT (ok) ;

    E->nz = NULL ;
    ok = CHOLMOD(print_sparse)(E, "E unpacked bad", cm) ;	    NOT (ok) ;
    E->nz = Enz ;
    ok = CHOLMOD(print_sparse)(E, "E unpacked ok", cm) ;	    OK (ok) ;

    F = CHOLMOD(copy)(E, 0, 0, cm) ;
    cm->print = 4 ;
    ok = CHOLMOD(print_sparse)(F, "F pattern ok", cm) ;		    OK (ok) ;
    cm->print = 1 ;

    CHOLMOD(free_sparse)(&F, cm) ;
    CHOLMOD(free_sparse)(&E, cm) ;
    CHOLMOD(free_sparse)(&C, cm) ;

    /* ---------------------------------------------------------------------- */
    /* print_dense */
    /* ---------------------------------------------------------------------- */

    X = CHOLMOD(sparse_to_dense)(NULL, cm) ;			    NOP (X) ;
    X = CHOLMOD(sparse_to_dense)(Abad2, cm) ;			    NOP (X) ;
    C = CHOLMOD(dense_to_sparse)(NULL, TRUE, cm) ;		    NOP (C) ;

    X = CHOLMOD(copy_dense)(Xok, cm) ;

    ok = CHOLMOD(print_dense)(NULL, "null", cm) ;		    NOT (ok) ;

    x = X->x ;
    X->x = NULL ;
    ok = CHOLMOD(print_dense)(X, "Xnull", cm) ;			    NOT (ok) ;
    X->x = x ;
    ok = CHOLMOD(print_dense)(X, "X OK", cm) ;			    OK (ok) ;

    X->nzmax = 1 ;
    ok = CHOLMOD(print_dense)(X, "X nzmax too small", cm) ;	    NOT (ok) ;
    X->nzmax = Xok->nzmax ;
    ok = CHOLMOD(print_dense)(X, "X OK", cm) ;			    OK (ok) ;

    X->d = -1 ;
    ok = CHOLMOD(print_dense)(X, "X d too small", cm) ;		    NOT (ok) ;
    X->d = Xok->d ;
    ok = CHOLMOD(print_dense)(X, "X OK", cm) ;			    OK (ok) ;

    Xxtype = X->xtype ;
    X->xtype = CHOLMOD_PATTERN ;
    ok = CHOLMOD(print_dense)(X, "X pattern", cm) ;		    NOT (ok) ;

    X->xtype = -1 ;
    ok = CHOLMOD(print_dense)(X, "X unknown", cm) ;		    NOT (ok) ;
    X->xtype = Xxtype ;
    ok = CHOLMOD(print_dense)(X, "X OK", cm) ;			    OK (ok) ;

    X->dtype = CHOLMOD_SINGLE ;
    ok = CHOLMOD(print_dense)(X, "X float", cm) ;		    NOT (ok) ;
    X->dtype = -1 ;
    ok = CHOLMOD(print_dense)(X, "X unknown", cm) ;		    NOT (ok) ;
    X->dtype = CHOLMOD_DOUBLE ;
    ok = CHOLMOD(print_dense)(X, "X OK", cm) ;			    OK (ok) ;

    CHOLMOD(free_dense)(&X, cm) ;

    /* ---------------------------------------------------------------------- */
    /* print_subset */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(check_subset)(NULL, 0, 0, cm) ;		    OK (ok) ;
    ok = CHOLMOD(print_subset)(NULL, 0, 0, "null", cm) ;	    OK (ok) ;

    for (i = 0 ; i < CSETSIZE ; i++)
    {
	cset [i] = i ;
    }

    for (cm->print = 0 ; cm->print <= 5 ; cm->print++)
    {
	ok = CHOLMOD(print_subset)(NULL, -1, 10, "[0:9]", cm) ;
	OK (ok) ;
	ok = CHOLMOD(print_subset)(cset, CSETSIZE, CSETSIZE, "cset OK", cm) ;
	OK (ok) ;
	cset [0] = -1 ;
	ok = CHOLMOD(print_subset)(cset, CSETSIZE, CSETSIZE, "cset bad", cm) ;
	NOT (ok) ;
	cset [0] = CSETSIZE-1 ;
	ok = CHOLMOD(print_subset)(cset, CSETSIZE, CSETSIZE, "cset OK", cm) ;
	OK (ok) ;
    }

    cm->print = 1 ;

    /* ---------------------------------------------------------------------- */
    /* print_perm */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(check_perm)(NULL, 0, 0, cm) ;		    OK (ok) ;

    for (cm->print = 3 ; cm->print <= 4 ; cm->print++)
    {
	ok = CHOLMOD(print_perm)(Pok, nrow, nrow, "P OK", cm) ;
	OK (ok) ;
	if (nrow > 0)
	{
	    p = Pok [0] ;
	    Pok [0] = 2*ncol + 1 ;
	    ok = CHOLMOD(print_perm)(Pok, nrow, nrow, "P bad", cm) ;
	    NOT (ok) ;
	    Pok [0] = p ;
	    ok = CHOLMOD(print_perm)(Pok, nrow, nrow, "P OK", cm) ;
	}
	OK (ok) ;
    }
    cm->print = 1 ;

    n2 = 2 * cm->nrow ;
    P2 = prand (n2) ;						/* RAND */

    for (cm->print = 3 ; cm->print <= 4 ; cm->print++)
    {
	ok = CHOLMOD(print_perm)(P2, n2, n2, "P2 OK", cm) ;
	OK (ok) ;
	p = P2 [0] ;
	P2 [0] = -1 ;
	ok = CHOLMOD(print_perm)(P2, n2, n2, "P2 bad", cm) ;
	NOT (ok) ;
	P2 [0] = p ;
	ok = CHOLMOD(print_perm)(P2, n2, n2, "P2 OK", cm) ;
	OK (ok) ;
    }
    cm->print = 1 ;

    CHOLMOD(free)(2 * (cm->nrow), sizeof (Int), P2, cm) ;

    /* ---------------------------------------------------------------------- */
    /* print_parent */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(print_parent)(NULL, 0, "null", cm) ;		    NOT (ok) ;
    if (nrow > 0)
    {
	i = Parent [0] ;
	Parent [0] = -2 ;
	ok = CHOLMOD(print_parent)(Parent, nrow, "bad Parent", cm) ;  NOT (ok) ;
	Parent [0] = i ;
	ok = CHOLMOD(print_parent)(Parent, nrow, "OK Parent", cm) ;    OK (ok) ;
    }

    /* ---------------------------------------------------------------------- */
    /* print_factor */
    /* ---------------------------------------------------------------------- */

    if (A->stype == 0)
    {
	L = CHOLMOD(allocate_factor)(nrow, cm) ;		    OKP (L) ;
	ok = CHOLMOD(super_symbolic)(A, NULL, Parent, L, cm) ;	    NOT (ok) ;
	CHOLMOD(free_factor)(&L, cm) ;
    }

    ok = CHOLMOD(print_factor)(NULL, "L null", cm) ;		    NOT (ok) ;

    /* create a valid symbolic supernodal L */
    cm->supernodal = CHOLMOD_SUPERNODAL ;
    cm->final_asis = TRUE ;
    L = CHOLMOD(analyze)(A, cm) ;	/* [ */			    OKP (L) ;
    ok = CHOLMOD(print_factor)(L, "L ok", cm) ;			    OK (ok) ;

    ok = CHOLMOD(change_factor)(CHOLMOD_ZOMPLEX, TRUE, TRUE, TRUE, TRUE, L, cm);
    NOT (ok) ;

    OK (L->xtype == CHOLMOD_PATTERN) ;
    OK (L->is_super) ;

    L->itype = CHOLMOD_INTLONG ;
    ok = CHOLMOD(print_factor)(L, "L int/long", cm) ;	            NOT (ok) ;
    L->itype = -1 ;
    ok = CHOLMOD(print_factor)(L, "L int unknown", cm) ;	    NOT (ok) ;
    L->itype = cm->itype ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    cm->print = 4 ;
#ifdef LONG
    L->itype = CHOLMOD_INT ;
#else
    L->itype = CHOLMOD_LONG ;
#endif
    ok = CHOLMOD(print_factor)(L, "L bad itype", cm) ;		    NOT (ok) ;
    L->itype = cm->itype ;
    cm->print = 1 ;

    cm->print = 4 ;

    i = L->ordering ;
    L->ordering = -1 ;
    ok = CHOLMOD(print_factor)(L, "L bad ordering", cm) ;	    NOT (ok) ;
    L->ordering = CHOLMOD_GIVEN ;
    ok = CHOLMOD(print_factor)(L, "L given ordering", cm) ;	    OK (ok) ;
    L->ordering = i ;

    Lxtype = L->xtype ;
    L->xtype = CHOLMOD_REAL ;
    ok = CHOLMOD(print_factor)(L, "L real", cm) ;		    NOT (ok) ;
    L->xtype = CHOLMOD_COMPLEX ;
    ok = CHOLMOD(print_factor)(L, "L complex", cm) ;		    NOT (ok) ;
    L->xtype = CHOLMOD_ZOMPLEX ;
    ok = CHOLMOD(print_factor)(L, "L zomplex", cm) ;		    NOT (ok) ;
    L->xtype = -1 ;
    ok = CHOLMOD(print_factor)(L, "L unknown", cm) ;		    NOT (ok) ;
    L->xtype = CHOLMOD_PATTERN ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;
    L->xtype = Lxtype ;

    /* ---------------------------------------------------------------------- */
    /* supernodal factor */
    /* ---------------------------------------------------------------------- */

    /* create a valid supernodal numeric L (simplicial if Supernodal
     * module not installed) */
    ok = CHOLMOD(factorize)(A, L, cm) ;
    OK (ok || cm->status == CHOLMOD_NOT_POSDEF) ;

    if (L->is_super)
    {
	/* there is no supernodal zomplex L */
	ok = CHOLMOD(factor_xtype)(CHOLMOD_ZOMPLEX, L, cm) ;	    NOT (ok) ;
    }

    /* pack the simplicial factor, or return silently if supernodal */
    ok = CHOLMOD(pack_factor)(L, cm) ;				    OK (ok) ;

    Lbad = CHOLMOD(copy_factor)(L, cm) ;	/* [ */
    Lxtype = L->xtype ;
    Lbad->xtype = -1 ;

    OK (L->is_super && L->xtype != CHOLMOD_PATTERN && L->is_ll) ;

    if (A->stype == 0)
    {
	ok = CHOLMOD(super_symbolic)(A, NULL, Parent, L, cm) ;	    NOT (ok) ;
    }
    ok = CHOLMOD(super_symbolic)(A, Abad2, Parent, L, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(super_symbolic)(Abad2, A, Parent, L, cm) ;	    NOT (ok) ;

    W = CHOLMOD(zeros)(nrow, 1, L->xtype, cm) ;			    OKP (W) ;
    X = CHOLMOD(ones)(nrow, 1, L->xtype, cm) ;			    OKP (X) ;
    ok = CHOLMOD(super_lsolve)(L, X, W, cm) ;			    OK (ok) ;
    ok = CHOLMOD(super_ltsolve)(L, X, W, cm) ;			    OK (ok) ;

    ok = CHOLMOD(super_lsolve)(Lbad, X, W, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(super_ltsolve)(Lbad, X, W, cm) ;		    NOT (ok) ;

    XX = CHOLMOD(zeros)(nrow, 1,
	    L->xtype == CHOLMOD_REAL ? CHOLMOD_COMPLEX : CHOLMOD_REAL, cm) ;
    ok = CHOLMOD(super_lsolve)(L, X, XX, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(super_ltsolve)(L, X, XX, cm) ;			    NOT (ok) ;
    CHOLMOD(free_dense)(&XX, cm) ;

    ok = CHOLMOD(super_lsolve)(L, X, W, cm) ;			    OK (ok) ;
    ok = CHOLMOD(super_ltsolve)(L, X, W, cm) ;			    OK (ok) ;

    x = X->x ;
    X->x = NULL  ;
    ok = CHOLMOD(super_lsolve)(L, X, W, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(super_ltsolve)(L, X, W, cm) ;			    NOT (ok) ;
    X->x = x  ;

    x = W->x ;
    W->x = NULL  ;
    ok = CHOLMOD(super_lsolve)(L, X, W, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(super_ltsolve)(L, X, W, cm) ;			    NOT (ok) ;
    W->x = x  ;
    CHOLMOD(free_dense)(&X, cm) ;
    CHOLMOD(free_dense)(&W, cm) ;

    cm->precise = TRUE ;
    ok = CHOLMOD(print_factor)(L, "L supernodal (precise)", cm) ;   OK (ok) ;
    cm->precise = FALSE ;
    ok = CHOLMOD(print_factor)(L, "L supernodal", cm) ;		    OK (ok) ;
    cm->print = 1 ;

    /* cannot realloc a supernodal L */
    ok = CHOLMOD(reallocate_factor)(10000, L, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(reallocate_factor)(10000, NULL, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(pack_factor)(NULL, cm) ;			    NOT (ok) ;

    /* ---------------------------------------------------------------------- */
    /* print factor */
    /* ---------------------------------------------------------------------- */

    Lxtype = L->xtype ;

    i = cm->print ;
    cm->print = 4 ;
    L->xtype = CHOLMOD_PATTERN ;
    ok = CHOLMOD(print_factor)(L, "L pattern", cm) ;		    OK (ok) ;
    C = CHOLMOD(factor_to_sparse)(L, cm) ;			    NOP (C) ;
    L->xtype = Lxtype ;
    cm->print = i ;

    /* check with bad L factor */
    ok = CHOLMOD(print_factor)(Lbad, "L unknown", cm) ;		    NOT (ok) ;
    ok = CHOLMOD(reallocate_factor)(999, Lbad, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(pack_factor)(Lbad, cm) ;			    NOT (ok) ;
    C = CHOLMOD(factor_to_sparse)(Lbad, cm) ;			    NOP (C) ;
    L2 = CHOLMOD(copy_factor)(Lbad, cm) ;			    NOP (L2) ;
    ok = CHOLMOD(factorize)(A, Lbad, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(resymbol)(A, NULL, 0, TRUE, Lbad, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(resymbol_noperm)(A, NULL, 0, TRUE, Lbad, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(rowadd)(nrow-2, A, Lbad, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(rowdel)(nrow-2, NULL, Lbad, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(rowfac)(A, AT, beta, 1, 2, Lbad, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(updown)(+1, A, Lbad, cm) ;			    NOT (ok) ;

    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    L->dtype = CHOLMOD_SINGLE ;
    ok = CHOLMOD(print_factor)(L, "L float", cm) ;		    NOT (ok) ;
    L->dtype = -1 ;
    ok = CHOLMOD(print_factor)(L, "L unknown", cm) ;		    NOT (ok) ;
    L->dtype = CHOLMOD_DOUBLE ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    if (nrow > 0)
    {
	Lperm = L->Perm ;
	p = Lperm [0] ;
	Lperm [0] = -1 ;
	ok = CHOLMOD(print_factor)(L, "L perm invalid", cm) ;	    NOT (ok) ;
	Lperm [0] = p ;
	ok = CHOLMOD(print_factor)(L, "L OK", cm) ;		    OK (ok) ;
    }

    LColCount = L->ColCount ;
    L->ColCount = NULL ;
    ok = CHOLMOD(print_factor)(L, "L no colcount", cm) ;	    NOT (ok) ;
    L->ColCount = LColCount ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    if (nrow > 0)
    {
	LColCount = L->ColCount ;
	p = LColCount [0] ;
	LColCount [0] = -1 ;
	ok = CHOLMOD(print_factor)(L, "L colcount vad", cm) ;	    NOT (ok) ;
	LColCount [0] = p ;
	ok = CHOLMOD(print_factor)(L, "L OK", cm) ;		    OK (ok) ;
    }

    /* ---------------------------------------------------------------------- */
    /* print simplicial factor */
    /* ---------------------------------------------------------------------- */

    /* check LDL' unpacked */
    ok = CHOLMOD(print_factor)(L, "L OK for L2 copy", cm) ;	    OK (ok) ;
    L2 = CHOLMOD(copy_factor)(L, cm) ; /* [ */			    OKP (L2) ;
    ok = CHOLMOD(change_factor)(L->xtype, FALSE, FALSE, FALSE, 
	    TRUE, L2, cm) ;

    /* check LDL' packed */
    L3 = CHOLMOD(copy_factor)(L, cm) ;				    OKP (L3) ;
    ok = CHOLMOD(change_factor)(L->xtype, FALSE, FALSE, TRUE,
	    TRUE, L3, cm) ;
    ok = CHOLMOD(print_factor)(L3, "L3 OK", cm) ;		    OK (ok) ;
    CHOLMOD(free_factor)(&L3, cm) ;				    OK (ok) ;
    ok = CHOLMOD(print_factor)(L2, "L2 OK", cm) ;		    OK (ok) ;
    ok = CHOLMOD(pack_factor)(L2, cm) ;				    OK (ok) ;
    ok = CHOLMOD(print_factor)(L2, "L2 OK packed", cm) ;	    OK (ok) ;

    /* create a simplicial factor from scratch */
    cm->supernodal = CHOLMOD_SIMPLICIAL ;
    cm->final_asis = TRUE ;
    L6 = CHOLMOD(analyze)(A, cm) ;	/* [ */			    OKP (L6) ;
    ok = CHOLMOD(factorize)(A, L6, cm) ;
    OK (cm->status >= CHOLMOD_OK) ;
    cm->supernodal = CHOLMOD_AUTO ;

    ok = CHOLMOD(print_sparse)(A, "A OK", cm) ;			    OK (ok) ;
    ok = CHOLMOD(print_factor)(L6, "L6 OK", cm) ;		    OK (ok) ;

    Lz = L6->z ;
    L6->z = NULL ;
    ok = CHOLMOD(print_factor)(L6, "L6 no z", cm) ;
    if (L6->xtype == CHOLMOD_ZOMPLEX)
    {
	NOT (ok) ;
    }
    else
    {
	OK (ok) ;
    }
    L6->z = Lz ;
    CHOLMOD(free_factor)(&L6, cm) ;	    /* ] */

    Az = A->z ;
    A->z = NULL ;
    ok = CHOLMOD(print_sparse)(A, "A no z", cm) ;
    if (A->xtype == CHOLMOD_ZOMPLEX)
    {
	NOT (ok) ;
    }
    else
    {
	OK (ok) ;
    }
    A->z = Az ;

    Lp = L2->p ;
    Li = L2->i ;
    Lx = L2->x ;
    Lnz = L2->nz ;
    Lnext = L2->next ;
    Lprev = L2->prev ;

    OK (Lp [0] == 0) ;

    L2->p = NULL ;
    ok = CHOLMOD(print_factor)(L2, "L no p", cm) ;		    NOT (ok) ;
    L2->p = Lp ;
    ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

    L2->i = NULL ;
    ok = CHOLMOD(print_factor)(L2, "L no i", cm) ;		    NOT (ok) ;
    L2->i = Li ;
    ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

    L2->x = NULL ;
    ok = CHOLMOD(print_factor)(L2, "L no x", cm) ;		    NOT (ok) ;
    L2->x = Lx ;
    ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

    L2->nz = NULL ;
    ok = CHOLMOD(print_factor)(L2, "L no nz", cm) ;		    NOT (ok) ;
    L2->nz = Lnz ;
    ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

    L2->next = NULL ;
    ok = CHOLMOD(print_factor)(L2, "L no next", cm) ;		    NOT (ok) ;
    L2->next = Lnext ;
    ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

    L2->prev = NULL ;
    ok = CHOLMOD(print_factor)(L2, "L no prev", cm) ;		    NOT (ok) ;
    L2->prev = Lprev ;
    ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

    if (nrow > 0)
    {
	p = Lp [0] ;
	Lp [0] = -1 ;
	ok = CHOLMOD(print_factor)(L2, "Lp bad", cm) ;		    NOT (ok) ;
	Lp [0] = p ;
	ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

	p = Li [0] ;
	Li [0] = -1 ;
	ok = CHOLMOD(print_factor)(L2, "Li bad", cm) ;		    NOT (ok) ;
	Li [0] = p ;
	ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

	p = Lnz [0] ;
	Lnz [0] = -1 ;
	ok = CHOLMOD(print_factor)(L2, "Lnz bad", cm) ;		    NOT (ok) ;
	Lnz [0] = p ;
    }

    ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

    OK (Lnz != NULL) ;

    if (nrow > 0 && Lnz [0] > 3)
    {
	ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;
	p = Li [1] ;
	Li [1] = nrow ;
	ok = CHOLMOD(print_factor)(L2, "Li bad", cm) ;		    NOT (ok) ;
	Li [1] = p ;
	ok = CHOLMOD(print_factor)(L2, "L OK again", cm) ;	    OK (ok) ;

	p = Li [2] ;
	Li [2] = Li [1] ;
	ok = CHOLMOD(print_factor)(L2, "Li bad", cm) ;		    NOT (ok) ;
	Li [2] = p ;
	ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;
    }

    /* check LDL' dynamic link list */
    ok = CHOLMOD(change_factor)(L->xtype, FALSE, FALSE, FALSE,
	    FALSE, L2, cm) ;
								    OK (ok) ;
    ok = CHOLMOD(print_factor)(L2, "L2 OK", cm) ;		    OK (ok) ;
    OK (L2->xtype != CHOLMOD_PATTERN && !(L2->is_ll) && !(L2->is_super)) ;

    /* cannot do a supernodal factorization on a dynamic LDL' factor */
    ok = CHOLMOD(super_numeric)(AT, NULL, Zero, L2, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(super_numeric)(I1, NULL, Zero, L2, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(super_numeric)(I1, I1, Zero, L2, cm) ;		    NOT (ok) ;

    G = CHOLMOD(copy)(I1, 1, 0, cm) ;				    OKP (G) ;
    ok = CHOLMOD(super_numeric)(G, NULL, Zero, L2, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(free_sparse)(&G, cm) ;				    OK (ok) ;

    G = CHOLMOD(copy)(I1, -1, 0, cm) ;				    OKP (G) ;
    ok = CHOLMOD(super_numeric)(G, NULL, Zero, L2, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(free_sparse)(&G, cm) ;				    OK (ok) ;

    ok = CHOLMOD(super_numeric)(AT, I1, Zero, L2, cm) ;		    NOT (ok) ;
    W = CHOLMOD(zeros)(nrow, 1, CHOLMOD_REAL, cm) ;		    OKP (W) ;
    X = CHOLMOD(ones)(nrow, 1, CHOLMOD_REAL, cm) ;		    OKP (X) ;
    ok = CHOLMOD(super_lsolve)(L2, X, W, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(super_ltsolve)(L2, X, W, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(free_dense)(&W, cm) ;				    OK (ok) ;
    ok = CHOLMOD(free_dense)(&X, cm) ;				    OK (ok) ;

    Lnext = L2->next ;
    Lprev = L2->prev ;

    if (nrow > 3)
    {

	p = Lnext [nrow+1] ;
	Lnext [nrow+1] = -1 ;
	ok = CHOLMOD(print_factor)(L2, "Lnext bad", cm) ;	    NOT (ok) ;
	Lnext [nrow+1] = -p ;
	ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

	p = Lnext [2] ;
	Lnext [2] = 2 ;
	ok = CHOLMOD(print_factor)(L2, "Lnext bad", cm) ;	    NOT (ok) ;
	Lnext [2] = p ;
	ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

	p = Lnext [2] ;
	Lnext [2] = -1 ;
	ok = CHOLMOD(print_factor)(L2, "Lnext bad", cm) ;	    NOT (ok) ;
	Lnext [2] = p ;
	ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

	p = Lprev [2] ;
	Lprev [2] = -9 ;
	ok = CHOLMOD(print_factor)(L2, "Lprev bad", cm) ;	    NOT (ok) ;
	Lprev [2] = p ;
	ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

	p = Lnext [nrow] ;
	Lnext [nrow] = 0 ;
	ok = CHOLMOD(print_factor)(L2, "Lnext/prev bad", cm) ;	    NOT (ok) ;
	Lnext [nrow] = p ;
	ok = CHOLMOD(print_factor)(L2, "L OK", cm) ;		    OK (ok) ;

	/* make a non-monotonic copy of L2 and then mangle it */
	L6 = CHOLMOD(copy_factor)(L2, cm) ;
	ok = CHOLMOD(reallocate_column)(0, nrow, L6, cm) ;
	if (ok && !(L6->is_monotonic))
	{
	    ok = CHOLMOD(print_factor)(L6, "L6 monotonic OK ", cm) ; OK (ok) ;
	    L6->is_monotonic = TRUE ;
	    ok = CHOLMOD(print_factor)(L6, "L6 monotonic bad", cm) ; NOT (ok) ;
	}
	CHOLMOD(free_factor)(&L6, cm) ;
    }


    L6 = CHOLMOD(copy_factor)(L, cm) ;				    OKP (L6) ;
    I  = CHOLMOD(speye)(nrow, nrow, L->xtype, cm) ;		    OKP (I) ;
    I3 = CHOLMOD(speye)(nrow, nrow, L->xtype-1, cm) ;		    OKP (I3) ;


    ok = CHOLMOD(super_numeric)(I, I, beta, L6, cm) ;		    OK (ok) ;
    ok = CHOLMOD(super_numeric)(I, I3, beta, L6, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(super_numeric)(I, Abad2, beta, L6, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(super_numeric)(I, I, beta, Lbad, cm) ;		    NOT (ok) ;
    I->stype = -1 ;
    ok = CHOLMOD(super_numeric)(I, I, beta, L6, cm) ;		    OK (ok) ;
    ok = CHOLMOD(super_numeric)(I, NULL, beta, L6, cm) ;	    OK (ok) ;
    I3->stype = -1 ;

    cm->print = 4 ;
    CHOLMOD(print_sparse)(I3, "I3", cm) ;
    CHOLMOD(print_factor)(L6, "L6", cm) ;
    cm->print = 1 ;

    ok = CHOLMOD(super_numeric)(I3, NULL, beta, L6, cm) ;	    NOT (ok) ;
    CHOLMOD(free_sparse)(&I, cm) ;
    I = CHOLMOD(speye)(nrow+1, nrow+1, L->xtype, cm) ;		    OKP (I) ;
    I->stype = -1 ;
    ok = CHOLMOD(super_numeric)(I, I, beta, L6, cm) ;		    NOT (ok) ;


    CHOLMOD(free_sparse)(&I, cm) ;
    CHOLMOD(free_sparse)(&I3, cm) ;
    ok = CHOLMOD(free_factor)(&L6, cm) ;			    OK (ok) ;

    /* check the supernodal L */
    Ls = L->s ;
    Lpi = L->pi ;
    Lpx = L->px ;
    Super = L->super ;
    Lx = L->x ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    L->s = NULL ;
    ok = CHOLMOD(print_factor)(L, "L no s", cm) ;		    NOT (ok) ;
    L->s = Ls ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    L->pi = NULL ;
    ok = CHOLMOD(print_factor)(L, "L no pi", cm) ;		    NOT (ok) ;
    L->pi = Lpi ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    L->px = NULL ;
    ok = CHOLMOD(print_factor)(L, "L no px", cm) ;		    NOT (ok) ;
    L->px = Lpx ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    L->super = NULL ;
    ok = CHOLMOD(print_factor)(L, "L no super", cm) ;		    NOT (ok) ;
    L->super = Super ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;


    L->x = NULL ;
    ok = CHOLMOD(print_factor)(L, "L no x", cm) ;		    NOT (ok) ;
    L->x = Lx ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    p = Ls [0] ;
    Ls [0] = -1 ;
    ok = CHOLMOD(print_factor)(L, "L bad s", cm) ;		    NOT (ok) ;
    Ls [0] = p ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    p = Lpi [0] ;
    Lpi [0] = -1 ;
    ok = CHOLMOD(print_factor)(L, "L bad pi", cm) ;		    NOT (ok) ;
    Lpi [0] = p ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    p = Lpx [0] ;
    Lpx [0] = -1 ;
    ok = CHOLMOD(print_factor)(L, "L bad px", cm) ;		    NOT (ok) ;
    Lpx [0] = p ;
    ok = CHOLMOD(print_factor)(L, "L OK", cm) ;			    OK (ok) ;

    if (nrow > 0)
    {
	p = Super [0] ;
	Super [0] = -1 ;
	ok = CHOLMOD(print_factor)(L, "L bad super", cm) ;	    NOT (ok) ;
	Super [0] = p ;
	ok = CHOLMOD(print_factor)(L, "L OK", cm) ;		    OK (ok) ;

	p = Ls [0] ;
	Ls [0] = 42 ;
	ok = CHOLMOD(print_factor)(L, "L bad s", cm) ;		    NOT (ok) ;
	Ls [0] = p ;
	ok = CHOLMOD(print_factor)(L, "L OK", cm) ;		    OK (ok) ;
    }

    if (nrow > 0 && Lpi [1] - Lpi [0] > 3)
    {
	p = Ls [2] ; 
	Ls [2] = Ls [1] ;
	ok = CHOLMOD(print_factor)(L, "L unsorted s", cm) ;	    NOT (ok) ;
	Ls [2] = p ;
	ok = CHOLMOD(print_factor)(L, "L OK", cm) ;		    OK (ok) ;
    }

    /* ---------------------------------------------------------------------- */
    /* Cholesky */
    /* ---------------------------------------------------------------------- */

    /* test the supernodal symbolic L */
    L3 = CHOLMOD(copy_factor)(L, cm) ;				    OKP (L3) ;
    ok = CHOLMOD(change_factor)(CHOLMOD_PATTERN, TRUE, TRUE, TRUE, TRUE,
	    L3, cm) ;
								    OK (ok) ;

    Ls = L3->s ;
    Lpi = L3->pi ;
    Super = L3->super ;

    if (nrow > 0)
    {
	p = Ls [0] ;
	Ls [0] = 42 ;
	ok = CHOLMOD(print_factor)(L3, "Lsym bad s", cm) ;	    NOT (ok) ;
	Ls [0] = p ;
	ok = CHOLMOD(print_factor)(L3, "Lsym OK", cm) ;		    OK (ok) ;
    }

    if (nrow > 0 && Lpi [1] - Lpi [0] > 3)
    {
	p = Ls [2] ; 
	Ls [2] = Ls [1] ;
	ok = CHOLMOD(print_factor)(L3, "Lsym unsorted s", cm) ;	    NOT (ok) ;
	Ls [2] = p ;
	ok = CHOLMOD(print_factor)(L3, "Lsym OK", cm) ;		    OK (ok) ;
    }

    if (nrow > 0 && L->nsuper > 0)
    {
	Int nscol = Super [1] ;
	Int nsrow = Lpi [1] - Lpi [0] ;
	if (nsrow > nscol + 1)
	{
	    p = Ls [nscol] ; 
	    Ls [nscol] = Ls [nscol+1] ;
	    ok = CHOLMOD(print_factor)(L3, "Lsym unsorted s2", cm) ;  NOT (ok) ;
	    Ls [nscol] = p ;
	    ok = CHOLMOD(print_factor)(L3, "Lsym OK", cm) ;	      OK (ok) ;
	}
    }
    CHOLMOD(free_factor)(&L3, cm) ;

    /* (re)factorize as LL' */
    L5 = CHOLMOD(copy_factor)(L, cm) ;	/* [ */			    OKP (L5) ;

    ok = CHOLMOD(factor_xtype)(-1, L, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(factor_xtype)(CHOLMOD_REAL, NULL, cm) ;	    NOT (ok) ;

    L3 = CHOLMOD(copy_factor)(L, cm) ;				    OKP (L3) ;
    CHOLMOD(print_factor)(L3, "L3 before factorize", cm) ;
    ok = CHOLMOD(change_factor)(L3->xtype, TRUE, FALSE, TRUE, TRUE, L3, cm) ;
    OK (ok) ;

    Acopy = CHOLMOD(copy_sparse)(A, cm) ;   /* [ */
    CHOLMOD(sparse_xtype)(L3->xtype, Acopy, cm) ;

    CHOLMOD(print_sparse)(Acopy, "Acopy for factorize", cm) ;

    ok = CHOLMOD(factorize)(Acopy, L3, cm) ;
    OK (ok || cm->status >= CHOLMOD_OK) ;
    ok = CHOLMOD(free_factor)(&L3, cm) ;			    OK (ok) ;

    CHOLMOD(print_sparse)(A, "A for factorize", cm) ;
    CHOLMOD(print_factor)(L3, "L3 for factorize", cm) ;

    /* refactor, but with wrong-sized A */
    ok = CHOLMOD(print_sparse)(I1, "I1", cm) ;			    OK (ok) ;
    ok = CHOLMOD(factorize)(I1, L, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(factorize)(Abad2, L, cm) ;			    NOT (ok) ;
    C = CHOLMOD(transpose)(I1, 0, cm) ;				    OKP (C) ;
    ok = CHOLMOD(print_sparse)(C, "C = I1'", cm) ;		    OK (ok) ;
    ok = CHOLMOD(free_sparse)(&C, cm) ;				    OK (ok) ;
    ok = CHOLMOD(print_factor)(L, "L OK ", cm) ;		    OK (ok) ;

    /* refactor, with invalid A (NULL, or symmetric but not square) */
    ok = CHOLMOD(print_sparse)(Abad, "Abad", cm) ;		    NOT (ok) ;
    ok = CHOLMOD(factorize)(Abad, L, cm) ;			    NOT (ok) ;

    /* refactorize supernodal LL' */
    printf ("refactorize here\n") ;
    ok = CHOLMOD(print_sparse)(Acopy, "Acopy refactorize", cm) ;    OK (ok) ;
    ok = CHOLMOD(print_factor)(L, "L for refactorize", cm) ;	    OK (ok) ;

    printf ("L->xtype for refactorize %d\n", L->xtype) ;
    ok = CHOLMOD(factorize)(Acopy, L, cm) ;
    OK (ok || cm->status == CHOLMOD_NOT_POSDEF) ;
    ok = CHOLMOD(print_factor)(L, "L ok, here", cm) ;		    OK (ok) ;

    ok = CHOLMOD(factorize)(Acopy, L, cm) ;
    OK (ok || cm->status == CHOLMOD_NOT_POSDEF) ;
    ok = CHOLMOD(print_factor)(L, "L ok, here2", cm) ;		    OK (ok) ;

    /* solve */
    B = CHOLMOD(ones)(nrow, 0, CHOLMOD_REAL, cm) ;		    OKP (B) ;
    X = CHOLMOD(solve)(CHOLMOD_A, L, B, cm) ;			    OKP (X) ;
    ok = CHOLMOD(free_dense)(&X, cm) ;				    OK (ok) ;

    X = CHOLMOD(solve)(-1, L, B, cm) ;				    NOP (X) ;
    ok = CHOLMOD(free_dense)(&B, cm) ;				    OK (ok) ;

    B = CHOLMOD(zeros)(nrow+1, 0, CHOLMOD_REAL, cm) ;		    OKP (B) ;
    X = CHOLMOD(solve)(CHOLMOD_A, L, B, cm) ;			    NOP (X) ;

    B->xtype = 0 ;
    X = CHOLMOD(solve)(CHOLMOD_A, L, B, cm) ;			    NOP (X) ;
    B->xtype = CHOLMOD_REAL ;
    ok = CHOLMOD(free_dense)(&B, cm) ;				    OK (ok) ;

    /* sparse solve */
    if (nrow < 100 && A->stype != 0)
    {
	/* solve A*C=I, so C should equal A inverse */
	I = CHOLMOD(speye)(nrow, nrow, CHOLMOD_REAL, cm) ;	    OKP (I) ;
	C = CHOLMOD(spsolve)(CHOLMOD_A, L, I, cm) ;		    OKP (C) ;
	/* compute norm of A*C-I */
	if (xtype == CHOLMOD_REAL)
	{
	    E = CHOLMOD(ssmult)(A, C, 0, TRUE, FALSE, cm) ;	    OKP (E) ;
	    F = CHOLMOD(add)(E, I, minusone, one, TRUE, FALSE, cm) ;OKP (F) ;
	    cm->print = 4 ;
	    ok = CHOLMOD(print_sparse)(F, "A*inv(A)-I", cm) ;	    OK (ok) ;
	    cm->print = 1 ;
	    r = CHOLMOD(norm_sparse)(F, 1, cm) ;
	    OK (! (r < 0)) ;
	    MAXERR (maxerr, r, 1) ;
	    ok = CHOLMOD(free_sparse)(&E, cm) ;			    OK (ok) ;
	    ok = CHOLMOD(free_sparse)(&F, cm) ;			    OK (ok) ;
	}
	CHOLMOD(free_sparse)(&C, cm) ;

	/* check error cases for sparse solve */
	C = CHOLMOD(spsolve)(CHOLMOD_A, NULL, I, cm) ;		    NOP (C) ;
	C = CHOLMOD(spsolve)(CHOLMOD_A, Lbad, I, cm) ;		    NOP (C) ;
	C = CHOLMOD(spsolve)(CHOLMOD_A, L, NULL, cm) ;		    NOP (C) ;
	I->xtype = 0 ;
	C = CHOLMOD(spsolve)(CHOLMOD_A, L, I, cm) ;		    NOP (C) ;
	I->xtype = CHOLMOD_REAL ;
	I->stype = -1 ;
	C = CHOLMOD(spsolve)(CHOLMOD_A, L, I, cm) ;		    NOP (C) ;
	ok = CHOLMOD(free_sparse)(&I, cm) ;			    OK (ok) ;
	I = CHOLMOD(speye)(nrow+1, nrow+1, CHOLMOD_REAL, cm) ;	    OKP (I) ;
	C = CHOLMOD(spsolve)(CHOLMOD_A, L, I, cm) ;		    NOP (C) ;
	ok = CHOLMOD(free_sparse)(&I, cm) ;			    OK (ok) ;
    }

    /* resymbol */
    ok = CHOLMOD(resymbol)(I1, NULL, 0, TRUE, L, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(resymbol_noperm)(I1, NULL, 0, TRUE, L, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(change_factor)(L->xtype, FALSE, FALSE, FALSE, FALSE, L, cm) ;
								    OK (ok) ;
    ok = CHOLMOD(resymbol)(I1, NULL, 0, TRUE, L, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(resymbol_noperm)(I1, NULL, 0, TRUE, L, cm) ;	    NOT (ok) ;

    ok = CHOLMOD(change_factor)(-1, FALSE, FALSE, FALSE, FALSE, L, cm) ;
    NOT (ok) ;

    ok = CHOLMOD(change_factor)(L->xtype, FALSE, FALSE, FALSE, FALSE, Lbad, cm);
    NOT (ok) ;

    ok = CHOLMOD(resymbol_noperm)(Acopy, NULL, 0, TRUE, L2, cm) ;
    if (Acopy->stype <= 0)
    {
	OK (ok) ;
    }
    else
    {
	NOT (ok) ;
    }

    ok = CHOLMOD(resymbol_noperm)(Abad2, NULL, 0, TRUE, L2, cm) ;   NOT (ok) ;
    ok = CHOLMOD(resymbol)(Abad2, NULL, 0, TRUE, L2, cm) ;	    NOT (ok) ;

    ok = CHOLMOD(resymbol_noperm)(Acopy, NULL, 0, TRUE, NULL, cm) ; NOT (ok) ;
    ok = CHOLMOD(resymbol)(Acopy, NULL, 0, TRUE, L2, cm) ;	    OK (ok) ;

    if (ncol > 0)
    {
	ok = CHOLMOD(print_perm)(fsetbad, ncol, ncol, "bad fset", cm) ;
	NOT (ok) ;
    }

    if (ncol > 1)
    {
	ok = CHOLMOD(resymbol)(Acopy, fsetok, ncol/2, TRUE, L2, cm) ;  OK (ok) ;
	ok = CHOLMOD(resymbol)(Acopy, fsetbad, ncol/2, TRUE, L2, cm) ;
	if (Acopy->stype)
	{
	    /* fset is ignored */
	    OK (ok) ;
	}
	else
	{
	    NOT (ok) ;
	    ok = CHOLMOD(resymbol_noperm)(Acopy, fsetbad, ncol/2, TRUE, L2, cm);
	    NOT (ok) ;
	}
	Acopy->sorted = FALSE ;
	ok = CHOLMOD(resymbol)(Acopy, fsetok, ncol/2, TRUE, L2, cm) ;
	OK (ok) ;
	Acopy->sorted = TRUE ;
    }

    cm->print = 4 ;
    gsave0 = cm->grow0 ;
    gsave1 = cm->grow1 ;
    gsave2 = cm->grow2 ;

    /* reallocate column */
    L4 = NULL ;
    if (nrow > 0)
    {
	ok = CHOLMOD(print_factor)(L, "L ok, for colrealloc", cm) ; OK (ok) ;
	L4 = CHOLMOD(copy_factor)(L, cm) ;
	ok = CHOLMOD(print_factor)(L4, "L4 ok, for colrealloc", cm) ; OK (ok) ;
	OK (nrow == (Int)(L->n)) ;
	ok = CHOLMOD(reallocate_column)(nrow, 1, L4, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(reallocate_column)(nrow-1, 0, L4, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(reallocate_column)(nrow-1, 10, L4, cm) ;	    OK (ok) ;

	cm->grow0 = 2e10 ;
	cm->grow1 = 2 ;

	/* this may or may not fail */
	ok = CHOLMOD(reallocate_column)(0, 10, L4, cm) ;
	CHOLMOD(print_common)("OK or too large", cm) ;
	ok = CHOLMOD(free_factor)(&L4, cm) ;			    OK (ok) ;
    }

    cm->grow0 = gsave0 ;
    cm->grow1 = gsave1 ;
    cm->grow2 = gsave2 ;

    if (ok && nrow > 2)
    {
	L4 = CHOLMOD(copy_factor)(L, cm) ;
	ok = CHOLMOD(resymbol)(A, NULL, 0, TRUE, L4, cm) ;	    OK (ok) ;

	/* make it non-monotonic and then monotonic (LDL' unpacked) */
	ok = CHOLMOD(reallocate_column)(0, nrow-1, L4, cm) ;	    OK (ok) ;

	/* this should be OK for small matrices, but fail for large ones */
	cm->grow0 = nrow ;
	cm->grow1 = nrow ;
	cm->grow2 = nrow ;
	ok = CHOLMOD(change_factor)(CHOLMOD_REAL, FALSE, FALSE, FALSE, TRUE,
		L4, cm) ;

	ok = CHOLMOD(free_factor)(&L4, cm) ;			    OK (ok) ;
	L4 = CHOLMOD(copy_factor)(L, cm) ;
	ok = CHOLMOD(resymbol)(A, NULL, 0, TRUE, L4, cm) ;	    OK (ok) ;
	ok = CHOLMOD(pack_factor)(L4, cm) ;			    OK (ok) ;

	/* now try to make L4 really huge */

	/*
	cm->print = 5 ;
	CHOLMOD(print_sparse) (A, "A for huge", cm) ;
	CHOLMOD(print_factor) (L4, "L4 for huge", cm) ;
	*/

	if (ok && !(L->is_super) && L->xtype != CHOLMOD_PATTERN)
	{

	    cm->grow0 = gsave0 ;
	    cm->grow1 = gsave1 ;
	    cm->grow2 = gsave2 ;

	    ok = CHOLMOD(reallocate_column)(0, nrow-1, L4, cm) ;    OK (ok) ;

	    cm->grow0 = nrow ;
	    cm->grow1 = nrow ;
	    cm->grow2 = nrow ;

	    /*
	    CHOLMOD(print_factor) (L4, "L4 for huge, realloced", cm) ;
	    printf ("L4 for huge is monotonic: %d\n", L4->is_monotonic) ;
	    */

	    if (!(L4->is_monotonic))
	    {
		/* printf ("Make L4 really huge: ") ; */
		ok = CHOLMOD(change_factor)(CHOLMOD_REAL, TRUE, FALSE, FALSE,
		    TRUE, L4, cm) ;
		printf ("L4 huge ok: "ID"\n", ok) ;
	    }
	}
	ok = CHOLMOD(free_factor)(&L4, cm) ;			    OK (ok) ;
    }

    cm->grow0 = gsave0 ;
    cm->grow1 = gsave1 ;
    cm->grow2 = gsave2 ;

    cm->print = 1 ;

    /* ---------------------------------------------------------------------- */
    /* more error tests */
    /* ---------------------------------------------------------------------- */

    cm->error_handler = NULL ;

    /* ---------------------------------------------------------------------- */
    /* modify */
    /* ---------------------------------------------------------------------- */

    X = CHOLMOD(ones)(nrow, 1, CHOLMOD_REAL, cm) ;		    OKP (X) ;
    R = CHOLMOD(dense_to_sparse)(X, TRUE, cm) ;	    /* [ */
    OKP (R) ;

    if (isreal)
    {
	C = CHOLMOD(speye)(nrow, 1, CHOLMOD_REAL, cm) ;		    OKP (C) ;
	ok = CHOLMOD(updown)(+1, C, L, cm) ;			    OK (ok) ;
	X1 = CHOLMOD(ones)(nrow, 1, CHOLMOD_REAL, cm) ;
	B1 = CHOLMOD(eye)(nrow, 1, CHOLMOD_REAL, cm) ;
	ok = CHOLMOD(updown_solve)(+1, C, L, X1, B1, cm) ;	    OK (ok) ;
	B1->xtype = -999 ;
	ok = CHOLMOD(updown_solve)(+1, C, L, X1, B1, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(rowadd_solve)(0, R, beta, L, X1, B1, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(rowdel_solve)(0, R, beta, L, X1, B1, cm) ;	    NOT (ok) ;
	B1->xtype = CHOLMOD_REAL ;
	CHOLMOD(free_dense)(&B1, cm) ;
	B2 = CHOLMOD(ones)(nrow, 2, CHOLMOD_REAL, cm) ;
	ok = CHOLMOD(updown_solve)(+1, C, L, X1, B2, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(rowadd_solve)(0, R, beta, L, X1, B2, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(rowdel_solve)(0, R, beta, L, X1, B2, cm) ;	    NOT (ok) ;

	CHOLMOD(free_dense)(&B2, cm) ;
	CHOLMOD(free_dense)(&X1, cm) ;
	ok = CHOLMOD(updown)(+1, Abad2, L, cm) ;		    NOT (ok) ;

	ok = CHOLMOD(updown)(+1, C, NULL, cm) ;			    NOT (ok) ;

	C->sorted = FALSE ;
	ok = CHOLMOD(updown)(+1, C, L, cm) ;			    NOT (ok) ;
	ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;

	ok = CHOLMOD(updown)(+1, NULL, L, cm) ;			    NOT (ok) ;

	if (nrow > 0)
	{
	    C = CHOLMOD(speye)(nrow-1, 1, CHOLMOD_REAL, cm) ;	    OKP (C) ;
	    ok = CHOLMOD(updown)(+1, C, L, cm) ;		    NOT (ok) ;
	    ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;
	}

	C = CHOLMOD(speye)(nrow, 0, CHOLMOD_REAL, cm) ;		    OKP (C) ;
	ok = CHOLMOD(updown)(+1, C, L, cm) ;			    OK (ok) ;

	ok = CHOLMOD(rowdel)(0, C, L, cm) ;			    NOT (ok) ;
	ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;
    }

    /* ---------------------------------------------------------------------- */
    /* rowfac, rcond */
    /* ---------------------------------------------------------------------- */

    cm->nmethods = 1 ;
    cm->method [0].ordering = CHOLMOD_NATURAL ;
    cm->postorder = FALSE ;

    cm->print = 5 ;
    cm->final_ll = TRUE ;
    for (xtype2 = CHOLMOD_REAL ; xtype2 <= CHOLMOD_ZOMPLEX ; xtype2++)
    {
	cm->supernodal = CHOLMOD_SIMPLICIAL ;

	/* factor a singular matrix (C=LL') */
	printf ("start singular LL'\n") ;
	XX = CHOLMOD(ones)(4, 4, xtype2, cm) ;			    OKP (X) ;
	C = CHOLMOD(dense_to_sparse)(XX, TRUE, cm) ;		    OKP (C) ;
	CHOLMOD(free_dense)(&XX, cm) ;
	C->stype = 1 ;
	CHOLMOD(print_sparse)(C, "C ones", cm) ;
	L6 = CHOLMOD(analyze)(C, cm) ;				    OKP (L6) ;
	ok = CHOLMOD(factorize)(C, L6, cm) ;			    OK (ok) ;
	printf ("status %d\n", cm->status) ;
	ok1 = (cm->status == CHOLMOD_NOT_POSDEF) ;
	ok = CHOLMOD(print_factor)(L6, "L6 singular", cm) ;	    OK (ok) ;
	OK (ok1) ;
	rcond = CHOLMOD(rcond) (L6, cm) ;		    OK (rcond == 0) ;

	/* now make C positive definite */
	CHOLMOD(free_sparse)(&C, cm) ;
	XX = CHOLMOD(ones)(4, 4, xtype2, cm) ;			    OKP (X) ;
	x = XX->x ;
	for (i = 0 ; i < 4 ; i++)
	{
	    if (xtype2 == CHOLMOD_REAL || xtype2 == CHOLMOD_ZOMPLEX)
	    {
		x [i + 4*i] = 42 ;
	    }
	    else /* complex */
	    {
		x [2*(i + 4*i)] = 42 ;
	    }
	}
	C = CHOLMOD(dense_to_sparse)(XX, TRUE, cm) ;		    OKP (C) ;
	CHOLMOD(free_dense)(&XX, cm) ;
	C->stype = 1 ;
	CHOLMOD(print_sparse)(C, "C ok", cm) ;
	ok = CHOLMOD(factorize)(C, L6, cm) ;			    OK (ok) ;
	ok1 = (cm->status == CHOLMOD_OK) ;
	ok = CHOLMOD(print_factor)(L6, "L6 ok", cm) ;		    OK (ok) ;
	OK (ok1) ;
	rcond = CHOLMOD(rcond) (L6, cm) ;		    OK (rcond > 0) ;

/* generate intentional nan's, to test the nan-handling of cholmod_rcond */
if (do_nantests)
{

	xnan = xnan/xnan ;

	/* C(2,2) = nan */
	x = C->x ;
	i = 2 ;
	if (xtype2 == CHOLMOD_REAL || xtype2 == CHOLMOD_ZOMPLEX)
	{
	    x [i + 4*i] = xnan ;
	}
	else /* complex */
	{
	    x [2*(i + 4*i)] = xnan ;
	}
	ok = CHOLMOD(factorize)(C, L6, cm) ;			    OK (ok) ;
	ok1 = (cm->status == CHOLMOD_OK) ;
	ok = CHOLMOD(print_factor)(L6, "L6 nan2", cm) ;		    OK (ok) ;
	printf ("rcond %g\n", rcond) ;
	OK (ok1) ;
	rcond = CHOLMOD(rcond) (L6, cm) ;		    OK (rcond == 0) ;
	CHOLMOD(free_factor)(&L6, cm) ;

	/* C(2,2) = nan, LDL' */
	cm->supernodal = CHOLMOD_SIMPLICIAL ;
	cm->final_ll = TRUE ;
	L6 = CHOLMOD(analyze)(C, cm) ;				    OKP (L6) ;
	ok = CHOLMOD(factorize)(C, L6, cm) ;			    OK (ok) ;
	ok1 = (cm->status == CHOLMOD_OK) ;
	ok = CHOLMOD(print_factor)(L6, "LDL6 nan2", cm) ;	    OK (ok) ;
	OK (ok1) ;
	rcond = CHOLMOD(rcond) (L6, cm) ;		    OK (rcond == 0) ;
	CHOLMOD(free_factor)(&L6, cm) ;

	/* C(2,2) = nan, supernodal */
	cm->supernodal = CHOLMOD_SUPERNODAL ;
	cm->final_ll = FALSE ;
	L6 = CHOLMOD(analyze)(C, cm) ;				    OKP (L6) ;
	ok = CHOLMOD(factorize)(C, L6, cm) ;			    OK (ok) ;
        /* sometimes LAPACK says NaN is not pos.def, sometimes it doesn't...*/
	ok1 = (cm->status == CHOLMOD_OK || cm->status == CHOLMOD_NOT_POSDEF) ;
	ok = CHOLMOD(print_factor)(L6, "L6 supernan2", cm) ;	    OK (ok) ;
	OK (ok1) ;
	rcond = CHOLMOD(rcond) (L6, cm) ;		    OK (rcond == 0) ;
	CHOLMOD(free_factor)(&L6, cm) ;

	/* C(0,0) = nan */
	cm->supernodal = CHOLMOD_SIMPLICIAL ;
	cm->final_ll = FALSE ;
	x [0] = xnan ;
	L6 = CHOLMOD(analyze)(C, cm) ;				    OKP (L6) ;
	ok = CHOLMOD(factorize)(C, L6, cm) ;			    OK (ok) ;
	ok1 = (cm->status == CHOLMOD_OK) ;
	ok = CHOLMOD(print_factor)(L6, "L6 nan0", cm) ;		    OK (ok) ;
	OK (ok1) ;
	rcond = CHOLMOD(rcond) (L6, cm) ;		    OK (rcond == 0) ;
	CHOLMOD(free_factor)(&L6, cm) ;

	/* C(0,0) = nan, LDL' */
	cm->supernodal = CHOLMOD_SIMPLICIAL ;
	cm->final_ll = TRUE ;
	L6 = CHOLMOD(analyze)(C, cm) ;				    OKP (L6) ;
	ok = CHOLMOD(factorize)(C, L6, cm) ;			    OK (ok) ;
	ok1 = (cm->status == CHOLMOD_OK) ;
	ok = CHOLMOD(print_factor)(L6, "LDL6 nan0", cm) ;	    OK (ok) ;
	OK (ok1) ;
	rcond = CHOLMOD(rcond) (L6, cm) ;		    OK (rcond == 0) ;
	CHOLMOD(free_factor)(&L6, cm) ;

	/* C(0,0) = nan, supernodal */
	cm->supernodal = CHOLMOD_SUPERNODAL ;
	cm->final_ll = FALSE ;
	L6 = CHOLMOD(analyze)(C, cm) ;				    OKP (L6) ;
	ok = CHOLMOD(factorize)(C, L6, cm) ;			    OK (ok) ;
        /* sometimes LAPACK says NaN is not pos.def, sometimes it doesn't...*/
	ok1 = (cm->status == CHOLMOD_OK || cm->status == CHOLMOD_NOT_POSDEF) ;
	ok = CHOLMOD(print_factor)(L6, "L6 supernan0", cm) ;	    OK (ok) ;
	OK (ok1) ;
	rcond = CHOLMOD(rcond) (L6, cm) ;		    OK (rcond == 0) ;
}

	CHOLMOD(free_factor)(&L6, cm) ;
	CHOLMOD(free_sparse)(&C, cm) ;
    }
    cm->supernodal = CHOLMOD_AUTO ;
    cm->final_ll = FALSE ;
    cm->print = 1 ;

    /* ---------------------------------------------------------------------- */
    /* refactorize simplicial LDL' */
    /* ---------------------------------------------------------------------- */

    if (nrow < NLARGE)
    {
	L7 = CHOLMOD(analyze) (A, cm) ;				    OKP (L7) ;
	ok = CHOLMOD(factorize) (A, L7, cm) ;			    OK (ok) ;
	ok = CHOLMOD(factorize) (A, L7, cm) ;			    OK (ok) ;
	B7 = CHOLMOD(ones) (nrow, 1, xtype, cm) ;		    OKP (B7) ;
	X7 = CHOLMOD(solve) (CHOLMOD_A, L7, B7, cm) ;		    OKP (X7) ;
	ok = CHOLMOD(free_dense) (&X7, cm) ;			    OK (ok) ;
	ok = CHOLMOD(free_dense) (&B7, cm) ;			    OK (ok) ;
	if (A->stype > 0)
	{
	    ok = CHOLMOD(rowfac) (A, NULL, zero, 0, nrow, L7, cm) ; OK (ok) ;
	    ok = CHOLMOD(rowfac) (A, NULL, zero, 0, nrow, L7, cm) ; OK (ok) ;
	    printf ("I7 :::\n") ;
	    I7 = CHOLMOD(speye) (nrow+1, 1, xtype, cm) ;	    OKP (I7) ;
	    I7->stype = 1 ;
	    ok = CHOLMOD(rowfac) (I7,NULL, zero, 0, nrow, L7, cm) ; NOT(ok) ;
	    printf ("I7 ::: done\n") ;
	    CHOLMOD(free_sparse) (&I7, cm) ;
	}
	ok = CHOLMOD(free_factor) (&L7, cm) ;			    OK (ok) ;
    }

    cm->nmethods = 0 ;	/* restore defaults */
    cm->method [0].ordering = CHOLMOD_GIVEN ;
    cm->postorder = TRUE ;

    /* ---------------------------------------------------------------------- */
    /* row subtree */
    /* ---------------------------------------------------------------------- */

    i = nrow / 2 ;

    C = CHOLMOD(allocate_sparse)(nrow, 1, nrow, TRUE, TRUE, 0,
	    CHOLMOD_REAL, cm) ;					    OKP (C) ;
    C2 = CHOLMOD(allocate_sparse)(nrow, 1, nrow, TRUE, TRUE, 0,
	    CHOLMOD_REAL, cm) ;					    OKP (C) ;
    ok = CHOLMOD(row_subtree)(NULL, NULL, i, Parent, C, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(row_lsubtree)(NULL, NULL, 0, i, L, C2, cm) ;	    NOT (ok) ;

    if (A->stype == 0 && nrow > 0 && AT != NULL)
    {
	ok = CHOLMOD(row_subtree)(A, AT, i, Parent, C, cm) ;	    OK (ok) ;

	ATp = AT->p ;
	ATi = AT->i ;
	fnz = ATp [i+1] - ATp [i] ;
	ok = CHOLMOD(row_lsubtree)(A, ATi, fnz, i, L, C2, cm) ;
        /*
        if (i < ncol) { OK (ok) ; } else { NOT (ok) ; }
        */
        OK (ok) ;

	ok = CHOLMOD(row_lsubtree)(Abad2, ATi, fnz, i, L, C2, cm) ; NOT (ok) ;
	ok = CHOLMOD(row_lsubtree)(A, NULL, fnz, i, L, C2, cm) ;    NOT (ok) ;
	ok = CHOLMOD(row_lsubtree)(A, ATi, fnz, i, L, Abad2, cm) ;  NOT (ok) ;
	ok = CHOLMOD(row_lsubtree)(A, ATi, fnz, i, NULL, C2, cm) ;  NOT (ok) ;
	ok = CHOLMOD(row_lsubtree)(A, ATi, fnz, nrow+1, L, C2, cm) ;NOT (ok) ;

	ok = CHOLMOD(row_subtree)(Abad2, AT, i, Parent, C, cm) ;    NOT (ok) ;
	ok = CHOLMOD(row_subtree)(A, Abad2, i, Parent, C, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(row_subtree)(A, AT, i, Parent, Abad2, cm) ;    NOT (ok) ;
	ok = CHOLMOD(row_subtree)(A, NULL, i, Parent, C, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(row_subtree)(A, AT, nrow+1, Parent, C, cm) ;   NOT (ok) ;
    }
    else
    {
	ok = CHOLMOD(row_subtree)(A, NULL, i, Parent, C, cm) ;
        if (A->stype == 1 && nrow > 0)
        {
            OK (ok) ;
        }
        else
        {
            NOT (ok) ;
        }
        #if 0
	ok = CHOLMOD(row_lsubtree)(A, NULL, 0, i, L, C2, cm) ;
        printf ("%g %g %g\n", (double) A->stype, (double) nrow, (double) ok) ;
        if (A->stype == 1 && nrow > 0 || (A->stype == 0 && nrow == 0 &&
            A->ncol == 1))
        {
            OK (ok) ;
        }
        else
        {
            NOT (ok) ;
        }
        #endif
    }

    ok = CHOLMOD(row_subtree)(A, NULL, i, Parent, NULL, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(row_subtree)(A, NULL, i, NULL, C, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(row_lsubtree)(A, NULL, 0, i, L, NULL, cm) ;	    NOT (ok) ;

    if (A->stype == 1 && nrow > 0)
    {
	/* add extra entries in the (ignored) lower triangular part to AA */
	if (!(A->sorted))
	{
	    ok = CHOLMOD(sort)(A, cm) ;				    OK (ok) ;
	}
	AA = CHOLMOD(copy)(A, 0, 0, cm) ;
	OK (AA->sorted) ;
	AA->stype = 1 ;
	ok = CHOLMOD(row_subtree)(AA, NULL, i, Parent, C, cm) ;	    OK (ok) ;
	ok = CHOLMOD(row_lsubtree)(AA, NULL, 0, i, L, C2, cm) ;	    OK (ok) ;
	ok = CHOLMOD(free_sparse)(&AA, cm) ;			    OK (ok) ;
    }

    ok = CHOLMOD(free_sparse)(&C, cm) ;				    OK (ok) ;
    ok = CHOLMOD(free_sparse)(&C2, cm) ;			    OK (ok) ;

    C = CHOLMOD(speye)(nrow, 0, CHOLMOD_REAL, cm) ;		    OKP (C) ;
    if (A->stype == 0 && AT != NULL && nrow > 0)
    {
	ok = CHOLMOD(row_subtree)(A, AT, i, Parent, C, cm) ;	    NOT (ok) ;

	ATp = AT->p ;
	ATi = AT->i ;
	fnz = ATp [i+1] - ATp [i] ;
	ok = CHOLMOD(row_lsubtree)(A, ATi, fnz, i, L, C, cm) ;	    NOT (ok) ;
    }
    ok = CHOLMOD(free_sparse)(&C, cm) ;				    OK (ok) ;

    L6 = CHOLMOD(allocate_factor)(nrow, cm) ;			    OKP (L6) ;
    if (A->stype == 0 && nrow > 2)
    {
	ok = CHOLMOD(rowfac)(A, AT, beta, 0, 1, L6, cm) ;	    OK (ok) ;
	OK (cm->status == CHOLMOD_OK) ;
	ok = CHOLMOD(rowfac)(A, NULL, beta, 1, 2, L6, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(rowfac)(A, AT, beta, 1, 2, L6, cm) ;	    OK (ok) ;
	ok = CHOLMOD(rowfac)(Abad2, AT, beta, 1, 2, L6, cm) ;	    NOT (ok) ;
	ok = CHOLMOD(rowfac)(A, Abad2, beta, 1, 2, L6, cm) ;	    NOT (ok) ;
    }
    ok = CHOLMOD(free_factor)(&L6, cm) ;			    OK (ok) ;

    /* ---------------------------------------------------------------------- */
    /* horzcat, vertcat */
    /* ---------------------------------------------------------------------- */

    if (A->nrow != A->ncol)
    {
	C = CHOLMOD(horzcat)(A, AT, TRUE, cm) ;			    NOP (C) ;
	C = CHOLMOD(vertcat)(A, AT, TRUE, cm) ;			    NOP (C) ;
    }
    C = CHOLMOD(horzcat)(A, Axbad, TRUE, cm) ;			    NOP (C) ;
    C = CHOLMOD(vertcat)(A, Axbad, TRUE, cm) ;			    NOP (C) ;
    C = CHOLMOD(vertcat)(A, NULL, TRUE, cm) ;			    NOP (C) ;
    C = CHOLMOD(vertcat)(NULL, AT, TRUE, cm) ;			    NOP (C) ;
    C = CHOLMOD(horzcat)(A, NULL, TRUE, cm) ;			    NOP (C) ;
    C = CHOLMOD(horzcat)(NULL, AT, TRUE, cm) ;			    NOP (C) ;

    /* ---------------------------------------------------------------------- */
    /* print_triplet */
    /* ---------------------------------------------------------------------- */

    cm->print = 4 ;
    ok = CHOLMOD(print_triplet)(Tok, "T ok", cm) ;		    OK (ok) ;
    T = CHOLMOD(copy_triplet)(Tok, cm) ;    /* [ */		    OKP (T) ;

    Tz = T->z ;
    T->z = NULL ;
    ok = CHOLMOD(print_triplet)(T, "T no z", cm) ;
    if (T->xtype == CHOLMOD_ZOMPLEX)
    {
	NOT (ok) ;
    }
    else
    {
	OK (ok) ;
    }
    T->z = Tz ;
    cm->print = 1 ;

    ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;

    ok = CHOLMOD(print_triplet)(NULL, "null", cm) ;		    NOT (ok) ;

    p = T->nzmax ;
    T->nzmax = T->nnz - 1 ;
    ok = CHOLMOD(print_triplet)(T, "T nzmax too small", cm) ;	    NOT (ok) ;
    T->nzmax = p ;
    ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;

    T->itype = -1 ;
    ok = CHOLMOD(print_triplet)(T, "T itype bad", cm) ;		    NOT (ok) ;
    T->itype = CHOLMOD_INTLONG ;
    ok = CHOLMOD(print_triplet)(T, "T itype bad", cm) ;		    NOT (ok) ;
    T->itype = cm->itype ;
    ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;

    cm->print = 4 ;
#ifdef LONG
    T->itype = CHOLMOD_INT ;
#else
    T->itype = CHOLMOD_LONG ;
#endif
    ok = CHOLMOD(print_triplet)(T, "T bad itype", cm) ;		    NOT (ok) ;
    T->itype = cm->itype ;
    cm->print = 1 ;

    Txtype = T->xtype ;
    T->xtype = -1 ;
    ok = CHOLMOD(print_triplet)(T, "T xtype bad", cm) ;		    NOT (ok) ;

    T->xtype = Txtype ;
    ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;

    T->dtype = -1 ;
    ok = CHOLMOD(print_triplet)(T, "T dtype bad", cm) ;		    NOT (ok) ;
    T->dtype = CHOLMOD_SINGLE ;
    ok = CHOLMOD(print_triplet)(T, "T dtype bad", cm) ;		    NOT (ok) ;
    T->dtype = CHOLMOD_DOUBLE ;
    ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;

    Tj = T->j ;
    Ti = T->i ;
    Tx = T->x ;

    T->j = NULL  ;
    ok = CHOLMOD(print_triplet)(T, "Tj null", cm) ;		    NOT (ok) ;
    T->j = Tj  ;
    ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;

    T->i = NULL  ;
    ok = CHOLMOD(print_triplet)(T, "Ti null", cm) ;		    NOT (ok) ;
    T->i = Ti  ;
    ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;

    T->x = NULL  ;
    ok = CHOLMOD(print_triplet)(T, "Tx null", cm) ;		    NOT (ok) ;
    T->x = Tx  ;
    ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;

    if (T->nnz > 0)
    {
	p = Ti [0] ;
	Ti [0] = -1 ;
	ok = CHOLMOD(print_triplet)(T, "Ti bad", cm) ;		    NOT (ok) ;
	C = CHOLMOD(triplet_to_sparse)(T, 0, cm) ;		    NOP (C) ;
	Ti [0] = p ;
	ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;

	p = Tj [0] ;
	Tj [0] = -1 ;
	ok = CHOLMOD(print_triplet)(T, "Tj bad", cm) ;		    NOT (ok) ;
	C = CHOLMOD(triplet_to_sparse)(T, 0, cm) ;		    NOP (C) ;
	Tj [0] = p ;
	ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;
    }

    cm->print = 4 ;
    CHOLMOD(triplet_xtype)(CHOLMOD_PATTERN, T, cm) ;
    ok = CHOLMOD(print_triplet)(T, "T pattern ok", cm) ;	    OK (ok) ;
    cm->print = 1 ;

    /* ---------------------------------------------------------------------- */
    /* triplet, realloc_multiple */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;
    OK (cm->status == CHOLMOD_OK) ;

    cm->print = 4 ;
    if (T->nrow != T->ncol)
    {
	OK (T->stype == 0) ;

	CHOLMOD(print_triplet)(T, "T ok", cm) ;

	C = CHOLMOD(triplet_to_sparse)(T, 0, cm) ;
	CHOLMOD(print_sparse)(C, "C ok", cm) ;
	OKP (C) ;
	ok = CHOLMOD(free_sparse)(&C, cm) ;	    		    OK (ok) ;

	Ti = T->i ;
	T->i = NULL ;
	C = CHOLMOD(triplet_to_sparse)(T, 0, cm) ;		    NOP (C) ;
	T->i = Ti ;

	Tj = T->j ;
	T->j = NULL ;
	C = CHOLMOD(triplet_to_sparse)(T, 0, cm) ;		    NOP (C) ;
	T->j = Tj ;

	T->stype = 1 ;
	ok = CHOLMOD(print_triplet)(T, "T bad", cm) ;		    NOT (ok) ;
	C = CHOLMOD(triplet_to_sparse)(T, 0, cm) ;		    NOP (C) ;
	T->stype = 0 ;
	ok = CHOLMOD(print_triplet)(T, "T pattern ok", cm) ;	    OK (ok) ;
    }
    OK (cm->status == CHOLMOD_OK) ;
    cm->print = 1 ;

    ok = CHOLMOD(reallocate_triplet)(1, NULL, cm) ;		    NOT (ok) ;

    CHOLMOD(print_triplet)(T, "T before realloc", cm) ;
    ok = CHOLMOD(reallocate_triplet)(1+(T->nzmax), T, cm) ;	    OK (ok) ;
    CHOLMOD(print_triplet)(T, "T after realloc", cm) ;

    nznew = 10 + T->nzmax ;
    pp = NULL ;

    ok = CHOLMOD(realloc_multiple)(Size_max/2, 2, T->xtype, &(T->i),
	&(T->j), &(T->x), &(T->z), &(T->nzmax), cm) ;		    NOT (ok) ;

    size = 0 ;
    ii = NULL ;
    jj = NULL ;
    xx = NULL ;
    ok = CHOLMOD(realloc_multiple)(Size_max, 2, CHOLMOD_REAL, &ii, &jj, &xx,
	    NULL, &size, cm) ;					    NOT (ok) ;

    ok = CHOLMOD(realloc_multiple)(0, 0, CHOLMOD_PATTERN, &ii, &jj, &xx, NULL,
	    &size, cm) ;					    OK (ok) ;

    ok = CHOLMOD(realloc_multiple)(0, 0, -1, &ii, &jj, &xx, NULL,
	    &size, cm) ;					    NOT (ok) ;

    /* change to pattern-only */
    CHOLMOD(triplet_xtype)(CHOLMOD_PATTERN, T, cm) ;

    ok = CHOLMOD(reallocate_triplet)(1+(T->nzmax), T, cm) ;	    OK (ok) ;

    ok = CHOLMOD(free_triplet)(&T, cm) ;    /* ] */		    OK (ok) ;

    T = CHOLMOD(allocate_triplet)(nrow, ncol, Size_max, 0, CHOLMOD_REAL, cm);
								    NOP (T) ;

    T2 = CHOLMOD(allocate_triplet)(4, 4, 8, 0, CHOLMOD_REAL, cm);   OKP (T2) ;
    ok = CHOLMOD(reallocate_triplet)(12, T2, cm) ;		    OK (ok) ;
    T = CHOLMOD(copy_triplet)(T2, cm) ;				    OKP (T) ;
    CHOLMOD(free_triplet)(&T, cm) ;
    T = CHOLMOD(sparse_to_triplet)(A, cm) ;			    OKP (T) ;
    C = CHOLMOD(triplet_to_sparse)(T, 100, cm) ;		    OKP (C) ;
    CHOLMOD(free_sparse)(&C, cm) ;
    CHOLMOD(free_triplet)(&T, cm) ;

    T2->xtype = -1 ;
    ok = CHOLMOD(reallocate_triplet)(16, T2, cm) ;		    NOT (ok) ;
    T = CHOLMOD(copy_triplet)(T2, cm) ;				    NOP (T) ;
    C = CHOLMOD(triplet_to_sparse)(T2, 100, cm) ;		    NOP (C) ;
    T2->xtype = CHOLMOD_REAL ; 
    CHOLMOD(free_triplet)(&T2, cm) ;

    T = CHOLMOD(allocate_triplet)(4, 4, 16, 0, -1, cm);		    NOP (T) ;

    T = CHOLMOD(sparse_to_triplet)(Abad2, cm) ;			    NOP (T) ;

    for (stype = -1 ; stype <= 1 ; stype++)
    {
	T = CHOLMOD(allocate_triplet)(4, 4, 16, stype, CHOLMOD_PATTERN, cm) ;
	OKP (T) ;
	Ti = T->i ;
	Tj = T->j ;
	k = 0 ;
	for (i = 0 ; i < 4 ; i++)
	{
	    for (j = 0 ; j < 4 ; j++)
	    {
		Ti [k] = i ;
		Tj [k] = j ;
		k++ ;
	    }
	}
	T->nnz = k ;
	C = CHOLMOD(triplet_to_sparse)(T, 0, cm) ;
	cm->print = 4 ;
	printf ("stype "ID"\n", stype) ;
	CHOLMOD(print_triplet)(T, "T from triplet", cm) ;
	CHOLMOD(print_sparse)(C, "C from triplet", cm) ;
	cm->print = 1 ;
	OKP (C) ;
	CHOLMOD(free_sparse)(&C, cm) ;
	CHOLMOD(free_triplet)(&T, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* sparse_to_triplet */
    /* ---------------------------------------------------------------------- */

    if (A->nrow != A->ncol)
    {
	OK (A->stype == 0) ;
	T = CHOLMOD(sparse_to_triplet)(A, cm) ;			    OKP (T) ;
	ok = CHOLMOD(print_triplet)(T, "T ok", cm) ;		    OK (ok) ;

	T2 = CHOLMOD(copy_triplet)(NULL, cm) ;			    NOP (T2) ;

	Ti = T->i ;
	T->i = NULL ;
	T2 = CHOLMOD(copy_triplet)(T, cm) ;			    NOP (T2) ;
	T->i = Ti ;

	Tj = T->j ;
	T->j = NULL ;
	T2 = CHOLMOD(copy_triplet)(T, cm) ;			    NOP (T2) ;
	T->j = Tj ;

	ok = CHOLMOD(free_triplet)(&T, cm) ;			    OK (ok) ;
	A->stype = 1 ;
	T = CHOLMOD(sparse_to_triplet)(A, cm) ;			    NOP (T) ;
	A->stype = 0 ;
	T = CHOLMOD(sparse_to_triplet)(NULL, cm) ;		    NOP (T) ;
    }

    /* ---------------------------------------------------------------------- */
    /* colamd */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(colamd)(A, fsetok, fsizeok, TRUE, NULL, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(colamd)(NULL, fsetok, fsizeok, TRUE, Pok, cm) ;    NOT (ok) ;

    cm->current = 0 ;

    save1 = cm->method [0].prune_dense2 ;
    save2 = cm->method [0].ordering ;
    save4 = cm->nmethods ;

    cm->method [0].prune_dense2 = 0.5 ;
    cm->method [0].ordering = CHOLMOD_COLAMD ;
    cm->nmethods = 1 ;

    ok = CHOLMOD(colamd)(A, fsetok, fsizeok, TRUE, Pok, cm) ;
    if (A->stype == 0)
    {
	save3 = cm->print ;
	cm->print = 5 ;
	ok = CHOLMOD(print_common) ("colamd dense2", cm) ;	    OK (ok) ;
	cm->print = save3 ;
	OK (ok) ;
    }
    else
    {
	NOT (ok) ;
    }

    cm->method [0].prune_dense2 = save1 ;
    cm->method [0].ordering = save2 ;
    cm->nmethods = save4 ;

    cm->current = -1 ;
    ok = CHOLMOD(colamd)(A, fsetok, fsizeok, TRUE, Pok, cm) ;
    if (A->stype == 0)
    {
	OK (ok) ;
    }
    else
    {
	NOT (ok) ;
    }
    cm->current = 0 ;

    ok = CHOLMOD(colamd)(Abad2, NULL, 0, TRUE, Pok, cm) ;	    NOT (ok) ;

    if (ncol > 0)
    {
	ok = CHOLMOD(colamd)(A, fsetbad, ncol, TRUE, Pok, cm) ;	    NOT (ok) ;
    }

    /* mangle the matrix to test integer overflow in colamd */
    if (A->stype == 0)
    {
	nzmax = A->nzmax ;
	A->nzmax = Size_max/2 ;
	ok = CHOLMOD(colamd)(A, fsetok, fsizeok, TRUE, Pok, cm) ;    NOT (ok) ;
	A->nzmax = nzmax ;
    }

    /* ---------------------------------------------------------------------- */
    /* ccolamd/csymamd */
    /* ---------------------------------------------------------------------- */

#ifndef NCAMD
    ok = CHOLMOD(ccolamd)(A, fsetok, fsizeok, NULL, NULL, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(ccolamd)(A, fsetok, fsizeok, NULL, Pok, cm) ;
    if (A->stype == 0)
    {
	OK (ok) ;
    }
    else
    {
	NOT (ok) ;
    }
    ok = CHOLMOD(ccolamd)(Abad2, NULL, 0, NULL, Pok, cm) ;	    NOT (ok) ;

    ok = CHOLMOD(csymamd)(A, NULL, Pok, cm) ;
    if (A->nrow == A->ncol)
    {
	OK (ok) ;
    }
    else
    {
	NOT (ok) ;
    }
    ok = CHOLMOD(csymamd)(Abad2, NULL, Pok, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(csymamd)(NULL, NULL, Pok, cm) ;		    NOT (ok) ;
    ok = CHOLMOD(csymamd)(A, NULL, NULL, cm) ;			    NOT (ok) ;

    /* mangle the matrix to test integer overflow in colamd */
    if (A->stype == 0)
    {
	nzmax = A->nzmax ;
	A->nzmax = Size_max/2 ;
	ok = CHOLMOD(ccolamd)(A, fsetok, fsizeok, NULL, Pok, cm) ;    NOT (ok) ;
	A->nzmax = nzmax ;
    }
#endif

    /* ---------------------------------------------------------------------- */
    /* amd */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(amd)(A, NULL, 0, Pok, cm) ;			    OK (ok) ;

    /* ---------------------------------------------------------------------- */
    /* metis */
    /* ---------------------------------------------------------------------- */

#ifndef NPARTITION
    /* no METIS memory guard */
    cm->metis_memory = 0 ;
    if (A->stype)
    {
	E = CHOLMOD(copy)(A, 0, -1, cm) ;
    }
    else
    {
	E = CHOLMOD(aat)(A, NULL, 0, -1, cm) ;
    }
    enz = CHOLMOD(nnz)(E, cm) ;

    CHOLMOD(print_sparse)(A, "A for metis", cm) ;

    if (A != NULL && Pok != NULL)
    {
	ok = CHOLMOD(metis)(A, NULL, 0, TRUE, Pok, cm) ;

	/* memory guard triggered */
	if (nrow > 0)
	{
	    double density ;

	    cm->metis_memory = Size_max ;
	    ok = CHOLMOD(metis)(A, NULL, 0, FALSE, Pok, cm) ;
	    OK (ok) ;
	    /* Pok should be identity */
	    for (j = 0 ; j < nrow ; j++)
	    {
		OK (Pok [j] == j) ;
	    }

	    /* memory guard triggered */
	    cm->metis_memory = 2 ;
	    cm->metis_nswitch = 10 ;


	    ok = CHOLMOD(metis)(A, NULL, 0, FALSE, Pok, cm) ;	  OK (ok) ;
	    /* Pok should be identity if the matrix is dense */
	    density = ((double) enz) / (((double) nrow) * ((double) nrow)) ;
	    if (nrow > 10 && density > cm->metis_dswitch)
	    {
		for (j = 0 ; j < nrow ; j++)
		{
		    OK (Pok [j] == j) ;
		}
	    }
	}
    }

    /* restore METIS default memory guard */
    cm->metis_memory = 2 ;
    cm->metis_nswitch = 3000 ;

    /* check metis bisector error handling */
    if (E != NULL && enz > 0)
    {
	Int *Anw, *Aew ;
	Anw = CHOLMOD(malloc)(nrow, sizeof (Int), cm) ;
	Aew = CHOLMOD(malloc)(MAX (anz,enz), sizeof (Int), cm) ;
	for (j = 0 ; j < nrow ; j++)
	{
	    Anw [j] = 1 ;
	}
	for (j = 0 ; j < enz ; j++)
	{
	    Aew [j] = 1 ;
	}
	lr = CHOLMOD(metis_bisector)(E, Anw, Aew, Pok, cm) ;
	if (E->stype || E->nrow != E->ncol)
	{
	    NOT (lr >= 0) ;
	}
	else
	{
	    OK (lr >= 0) ;
	}
	lr = CHOLMOD(metis_bisector)(Abad2, Anw, Aew, Pok, cm) ;NOT (lr >= 0);
	lr = CHOLMOD(metis_bisector)(NULL, Anw, Aew, Pok, cm) ;	NOT (lr >= 0);
	lr = CHOLMOD(metis_bisector)(A, Anw, Aew, NULL, cm) ;	NOT (lr >= 0);

	if (A->stype)
	{
	    lr = CHOLMOD(metis_bisector)(A, Anw, Aew, Pok, cm) ; NOT (lr>=0) ;
	}

	CHOLMOD(free)(nrow, sizeof (Int), Anw, cm) ;
	CHOLMOD(free)(MAX (anz,enz), sizeof (Int), Aew, cm) ;
    }

    CHOLMOD(free_sparse)(&E, cm) ;

    CHOLMOD(print_sparse)(Abad, "Abad", cm) ;
    lr = CHOLMOD(bisect)(Abad, NULL, 0, TRUE, Partition, cm) ;
    if (Abad != NULL && Abad->nrow == 0)
    {
	OK (lr == 0) ;
    }
    else
    {
	NOT (lr >= 0) ;
    }

    lr = CHOLMOD(bisect)(A, NULL, 0, TRUE, NULL, cm) ;		NOT (lr >= 0);
    lr = CHOLMOD(bisect)(NULL, NULL, 0, TRUE, Partition, cm) ;	NOT (lr >= 0);

    lr = CHOLMOD(nested_dissection)(NULL, NULL, 0, Pok, CParent,
	    Cmember, cm) ;					NOT (lr>=0) ;

    lr = CHOLMOD(nested_dissection)(A, NULL, 0, NULL, CParent,
	    Cmember, cm) ;					NOT (lr>=0) ;

    lr = CHOLMOD(nested_dissection)(A, NULL, 0, Pok, NULL,
	    Cmember, cm) ;					NOT (lr>=0) ;

    lr = CHOLMOD(nested_dissection)(A, NULL, 0, Pok, CParent,
	    NULL, cm) ;						NOT (lr>=0) ;

    ok = CHOLMOD(metis)(NULL, NULL, 0, TRUE, Pok, cm) ;		NOT (ok) ;
    ok = CHOLMOD(metis)(A, NULL, 0, TRUE, NULL, cm) ;		NOT (ok) ;
    ok = CHOLMOD(metis)(Abad2, NULL, 0, FALSE, Pok, cm) ;	NOT (ok) ;
    lr = CHOLMOD(bisect)(Abad2, NULL, 0, TRUE, Partition, cm) ;	NOT (lr >= 0);
#endif

    /* ---------------------------------------------------------------------- */
    /* etree */
    /* ---------------------------------------------------------------------- */

    if (A->stype < 0)
    {
	ok = CHOLMOD(etree)(A, Parent, cm) ;			    NOT (ok) ;
    }
    ok = CHOLMOD(etree)(Abad2, Parent, cm) ;			    NOT (ok) ;

    /* ---------------------------------------------------------------------- */
    /* etree, postorder, rowcolcount */
    /* ---------------------------------------------------------------------- */

    if (A->stype == 0 && ncol > 0)
    {
	AFT = CHOLMOD(ptranspose)(A, 1, NULL, fsetok, fsizeok, cm) ;  OKP(AFT);
	AF  = CHOLMOD(transpose)(AFT, 1, cm) ;			       OKP(AF);

	ok = CHOLMOD(etree)(NULL, Parent, cm) ;			       NOT(ok);
	ok = CHOLMOD(etree)(AFT, NULL, cm) ;			       NOT(ok);
	ok = CHOLMOD(etree)(AFT, Parent, cm) ;			       OK (ok);

	lr = CHOLMOD(postorder)(Parent, nrow, NULL, Post, cm) ;	OK (lr>=0) ;
	lr = CHOLMOD(postorder)(NULL, nrow, NULL, Post, cm) ;	NOT (lr>=0) ;
	lr = CHOLMOD(postorder)(Parent, nrow, NULL, NULL, cm) ;	NOT (lr>=0) ;


	ok = CHOLMOD(rowcolcounts)(A, fsetok, fsizeok, Parent,
		Post, NULL, ColCount, First, Level, cm) ;		OK (ok);

	ok = CHOLMOD(rowcolcounts)(Abad2, fsetok, fsizeok, Parent,
		Post, NULL, ColCount, First, Level, cm) ;		NOT(ok);

	ok = CHOLMOD(rowcolcounts)(NULL, fsetok, fsizeok, Parent,
		Post, NULL, ColCount, First, Level, cm) ;		NOT(ok);
	ok = CHOLMOD(rowcolcounts)(A, fsetok, fsizeok, NULL,
		Post, NULL, ColCount, First, Level, cm) ;		NOT(ok);
	ok = CHOLMOD(rowcolcounts)(A, fsetok, fsizeok, Parent,
		NULL, NULL, ColCount, First, Level, cm) ;		NOT(ok);
	ok = CHOLMOD(rowcolcounts)(A, fsetok, fsizeok, Parent,
		Post, NULL, NULL, First, Level, cm) ;			NOT(ok);
	ok = CHOLMOD(rowcolcounts)(A, fsetok, fsizeok, Parent,
		Post, NULL, ColCount, NULL, Level, cm) ;		NOT(ok);
	ok = CHOLMOD(rowcolcounts)(A, fsetok, fsizeok, Parent,
		Post, NULL, ColCount, First, NULL, cm) ;		NOT(ok);

	ok = CHOLMOD(rowcolcounts)(A, fsetbad, ncol, Parent,
		Post, NULL, ColCount, First, Level, cm) ;		NOT(ok);
	ok = CHOLMOD(rowcolcounts)(A, fsetok, fsizeok, Parent,
		Post, NULL, ColCount, First, NULL, cm) ;		NOT(ok);

	CHOLMOD(free_sparse)(&AF, cm) ;
	CHOLMOD(free_sparse)(&AFT, cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* norm */
    /* ---------------------------------------------------------------------- */

    nm = CHOLMOD(norm_sparse)(A, 2, cm) ;			NOT (nm>=0) ;
    nm = CHOLMOD(norm_sparse)(Abad, 0, cm) ;			NOT (nm>=0) ;
    nm = CHOLMOD(norm_sparse)(Abad2, 2, cm) ;			NOT (nm>=0) ;
    nm = CHOLMOD(norm_dense)(Bok, 3, cm) ;			NOT (nm>=0) ;
    nm = CHOLMOD(norm_dense)(Bok, 2, cm) ;			NOT (nm>=0) ;
    nm = CHOLMOD(norm_dense)(Xbad2, 1, cm) ;			NOT (nm>=0) ;

    /* ---------------------------------------------------------------------- */
    /* copy dense */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(copy_dense2)(NULL, Bok, cm) ;			NOT (ok) ;
    ok = CHOLMOD(copy_dense2)(Bok, NULL, cm) ;			NOT (ok) ;

    ok = CHOLMOD(copy_dense2)(Bok, Xbad2, cm) ;			NOT (ok) ;
    ok = CHOLMOD(copy_dense2)(Xbad2, Xbad2, cm) ;		NOT (ok) ;

    if (nrow > 1)
    {

	/* wrong dimensions */
	ok = CHOLMOD(copy_dense2)(Two, Bok, cm) ;		NOT (ok) ;

	/* mangled matrix */
	Y = CHOLMOD(copy_dense)(Bok, cm) ;			OKP (Y) ;
	Y->d = 0 ;
	ok = CHOLMOD(copy_dense2)(Bok, Y, cm) ;			NOT (ok) ;
	CHOLMOD(free_dense)(&Y, cm) ;

	Y = CHOLMOD(copy_dense)(Xbad2, cm) ;			NOP (Y) ;
	Y = CHOLMOD(copy_dense)(NULL, cm) ;			NOP (Y) ;
    }

    /* ---------------------------------------------------------------------- */
    /* complex */
    /* ---------------------------------------------------------------------- */

    W = CHOLMOD(eye)(4, 4, CHOLMOD_COMPLEX, cm) ;		OKP (W) ;
    ok = CHOLMOD(dense_xtype)(0, W, cm) ;			NOT (ok) ;
    ok = CHOLMOD(dense_xtype)(CHOLMOD_REAL, W, cm) ;		OK (ok) ;
    ok = CHOLMOD(dense_xtype)(CHOLMOD_REAL, NULL, cm) ;		NOT (ok) ;
    k = W->xtype ;
    W->xtype = -1 ;
    ok = CHOLMOD(dense_xtype)(CHOLMOD_REAL, W, cm) ;		NOT (ok) ;
    W->xtype = k ;
    ok = CHOLMOD(free_dense)(&W, cm) ;				OK (ok) ;

    C = CHOLMOD(speye)(4, 4, CHOLMOD_COMPLEX, cm) ;		OKP (C) ;
    ok = CHOLMOD(sparse_xtype)(-1, C, cm) ;			NOT (ok) ;
    ok = CHOLMOD(sparse_xtype)(CHOLMOD_ZOMPLEX, C, cm) ;	OK (ok) ;
    ok = CHOLMOD(sparse_xtype)(CHOLMOD_ZOMPLEX, NULL, cm) ;	NOT (ok) ;
    T = CHOLMOD(sparse_to_triplet)(C, cm) ;			OKP (T) ;
    ok = CHOLMOD(triplet_xtype)(-1, T, cm) ;			NOT (ok) ;
    ok = CHOLMOD(triplet_xtype)(CHOLMOD_ZOMPLEX, T, cm) ;	OK (ok) ;
    ok = CHOLMOD(triplet_xtype)(CHOLMOD_ZOMPLEX, NULL, cm) ;	NOT (ok) ;

    k = T->xtype ;
    T->xtype = -1 ;
    ok = CHOLMOD(triplet_xtype)(CHOLMOD_REAL, T, cm) ;		NOT (ok) ;
    T->xtype = k ;

    k = C->xtype ;
    C->xtype = -1 ;
    ok = CHOLMOD(sparse_xtype)(CHOLMOD_REAL, C, cm) ;		NOT (ok) ;
    C->xtype = k ;

    ok = CHOLMOD(free_triplet)(&T, cm) ;			OK (ok) ;
    ok = CHOLMOD(free_sparse)(&C, cm) ;				OK (ok) ;

    ok = CHOLMOD(factor_xtype)(CHOLMOD_REAL, Lbad, cm) ;	NOT (ok) ;

    /* ---------------------------------------------------------------------- */
    /* rowadd */
    /* ---------------------------------------------------------------------- */

    x = X->x ;
    X->x = NULL ;
    C = CHOLMOD(dense_to_sparse)(X, TRUE, cm) ;			NOP (C) ;

    if (nrow > 3 && isreal)
    {
	ok = CHOLMOD(rowadd)(1, I1, L, cm) ;			NOT (ok) ;
	ok = CHOLMOD(rowadd)(nrow+1, R, L, cm) ;		NOT (ok) ;
	ok = CHOLMOD(rowadd)(nrow+1, R, L, cm) ;		NOT (ok) ;
	ok = CHOLMOD(rowdel)(nrow+1, NULL, L5, cm) ;		NOT (ok) ;

	ok = CHOLMOD(rowdel)(nrow-2, NULL, L5, cm) ;		OK (ok) ;
	ok = CHOLMOD(rowdel)(nrow-2, NULL, L5, cm) ;		OK (ok) ;
	ok = CHOLMOD(rowdel)(nrow-2, Abad2, L5, cm) ;		NOT (ok) ;
	ok = CHOLMOD(rowdel)(nrow-1, R, L5, cm) ;		NOT (ok) ;

	ok = CHOLMOD(change_factor)(CHOLMOD_REAL, TRUE, FALSE, TRUE, TRUE, L5,
		cm) ;
								OK (ok) ;

	ok = CHOLMOD(rowadd)(nrow-2, NULL, L5, cm) ;		NOT (ok) ;
	ok = CHOLMOD(rowadd)(nrow-2, R, NULL, cm) ;		NOT (ok) ;

	ok = CHOLMOD(rowadd)(nrow-2, R, L5, cm) ;		OK (ok) ;

	ok = CHOLMOD(rowadd)(nrow-2, Abad2, L5, cm) ;		NOT (ok) ;

/*	ok = CHOLMOD(rowadd)(nrow-2, R, L5, cm) ;		NOT (ok) ; */
	ok = CHOLMOD(rowdel)(nrow-2, NULL, L5, cm) ;		OK (ok) ;
	ok = CHOLMOD(change_factor)(CHOLMOD_PATTERN, TRUE,
		TRUE, TRUE, TRUE, L5, cm) ;			NOT (ok) ;
	ok = CHOLMOD(change_factor)(CHOLMOD_REAL, TRUE,
		TRUE, TRUE, TRUE, L5, cm) ;			NOT (ok) ;

	ok = CHOLMOD(rowadd_solve)(nrow-2, R, beta, L5, X, X, cm) ;  NOT (ok) ;
	ok = CHOLMOD(rowdel_solve)(nrow-2, R, beta, L5, X, X, cm) ;  NOT (ok) ;
	ok = CHOLMOD(updown_solve)(TRUE, R, L5, X, X, cm) ;	NOT (ok) ;

	if (nrow < 200 && L5 != NULL && R2 != NULL)
	{
	    cholmod_factor *L8 ;
	    Int *L8p, *L8i, *L8nz, rnz ;
	    double *L8x ;
	    L8 = CHOLMOD(copy_factor) (L5, cm) ;
	    ok = TRUE ;
	    for (k = nrow-1 ; ok && L8 != NULL
		    && L8->xtype == CHOLMOD_REAL && k >= 0 ; k--)
	    {
		for (rnz = 0 ; rnz < nrow ; rnz++)
		{
		    /* first, ensure row i is zero */
		    for (j = 0 ; j < nrow ; j++)
		    {
			L8p = L8->p ;
			L8i = L8->i ;
			L8nz = L8->nz ;
			L8x = L8->x ;
			for (p = L8p [j] ; p < L8p [j] + L8nz [j] ; p++)
			{
			    i = L8i [p] ;
			    if (i == k) L8x [p] = 0 ;
			}
		    }
		    R2p [1] = rnz ;
		    ok = CHOLMOD(rowadd)(k, R2, L8, cm) ;	OK (ok) ;
		    ok = CHOLMOD(rowdel)(k, NULL, L8, cm) ;	OK (ok) ;
		    ok = CHOLMOD(rowadd)(k, R2, L8, cm) ;	OK (ok) ;
		}
	    }
	    CHOLMOD(free_factor) (&L8, cm) ;
	}
    }

    X->x = x ;
    ok = CHOLMOD(free_dense)(&X, cm) ;				OK (ok) ;

    /* ---------------------------------------------------------------------- */
    /* ssmult */
    /* ---------------------------------------------------------------------- */

    if (nrow < 100)
    {
	C = CHOLMOD(ssmult)(A, A, 0, TRUE, TRUE, cm) ;
	if (A->nrow != A->ncol || !isreal)
	{
	    NOP (C) ;
	}
	else
	{
	    OKP (C) ;
	    ok = CHOLMOD(free_sparse)(&C, cm) ;			    OK (ok) ;
	}
	C = CHOLMOD(ssmult)(NULL, A, 0, TRUE, TRUE, cm) ;	    NOP (C) ;
	C = CHOLMOD(ssmult)(A, NULL, 0, TRUE, TRUE, cm) ;	    NOP (C) ;
	C = CHOLMOD(ssmult)(A, Axbad, 0, TRUE, TRUE, cm) ;	    NOP (C) ;
    }

    /* ---------------------------------------------------------------------- */
    /* sdmult */
    /* ---------------------------------------------------------------------- */

    if (nrow > 1)
    {
	ok = CHOLMOD(sdmult)(A, FALSE, one, one, Two, Two, cm) ;    NOT (ok) ;
    }

    YY = CHOLMOD(ones)(A->nrow, 1, xtype, cm) ;			    OKP (YY) ;
    XX = CHOLMOD(ones)(A->ncol, 1, xtype, cm) ;			    OKP (XX) ;
    cm->print = 4 ;
    ok = CHOLMOD(print_dense)(XX, "XX", cm) ;			    OK (ok) ;
    cm->print = 1 ;
    ok = CHOLMOD(sdmult)(A, FALSE, one, one, XX, YY, cm) ;	    OK (ok) ;
    ok = CHOLMOD(sdmult)(NULL, FALSE, one, one, XX, YY, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(sdmult)(A, FALSE, one, one, NULL, YY, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(sdmult)(A, FALSE, one, one, XX, NULL, cm) ;	    NOT (ok) ;

    ok = CHOLMOD(sdmult)(Abad2, FALSE, one, one, XX, YY, cm) ;	    NOT (ok) ;

    XX->xtype++ ;
    ok = CHOLMOD(sdmult)(A, FALSE, one, one, XX, YY, cm) ;	    NOT (ok) ;
    XX->xtype-- ;

    YY->xtype++ ;
    ok = CHOLMOD(sdmult)(A, FALSE, one, one, XX, YY, cm) ;	    NOT (ok) ;
    YY->xtype-- ;

    CHOLMOD(free_dense)(&YY, cm) ;
    CHOLMOD(free_dense)(&XX, cm) ;

    /* ---------------------------------------------------------------------- */
    /* symmetry */
    /* ---------------------------------------------------------------------- */

    for (option = 0 ; option <= 2 ; option++)
    {
	Int xmatched = 0, pmatched = 0, nzoffdiag = 0, nz_diag = 0 ;
	int asym ;
	printf ("test symmetry: option %d\n", option) ;
	save1 = cm->print ;
	cm->print = 5 ;
	CHOLMOD(print_sparse) (A, "A", cm) ;
	cm->print = save1 ;
	asym = CHOLMOD(symmetry) (A, option, &xmatched, &pmatched,
	    &nzoffdiag, &nz_diag, cm) ;
	printf ("asym: %d\n", asym) ;
	OK (A->stype != 0 || asym >= 0) ;
	save1 = A->xtype ;
	A->xtype = CHOLMOD_PATTERN ;
	asym = CHOLMOD(symmetry) (A, option, &xmatched, &pmatched,
	    &nzoffdiag, &nz_diag, cm) ;
	printf ("asym: %d pattern\n", asym) ;
	OK (A->stype != 0 || asym >= 0) ;
	A->xtype = save1 ;
	C = CHOLMOD(copy_sparse) (A, cm) ;
	OKP (C) ;
	ok = CHOLMOD(sparse_xtype) (CHOLMOD_ZOMPLEX, C, cm) ;
	OK (ok) ;
	asym = CHOLMOD(symmetry) (C, option, &xmatched, &pmatched,
	    &nzoffdiag, &nz_diag, cm) ;
	OK (A->stype != 0 || asym >= 0) ;
	printf ("asym: %d zomplex\n", asym) ;

	asym = CHOLMOD(symmetry) (NULL, option, &xmatched, &pmatched,
	    &nzoffdiag, &nz_diag, cm) ;
	NOT (asym >= 0) ;

	C->xtype = 999 ;
	asym = CHOLMOD(symmetry) (C, option, &xmatched, &pmatched,
	    &nzoffdiag, &nz_diag, cm) ;
	NOT (asym >= 0) ;
	C->xtype = CHOLMOD_ZOMPLEX ;

	ok = CHOLMOD(free_sparse)(&C, cm) ;  OK (ok) ;

	C = CHOLMOD(copy) (A, 0, (A->xtype == CHOLMOD_REAL), cm) ;
	OKP (C) ;
	asym = CHOLMOD(symmetry) (C, option, &xmatched, &pmatched,
	    &nzoffdiag, &nz_diag, cm) ;
	OK (asym >= 0) ;
	ok = CHOLMOD(free_sparse)(&C, cm) ;  OK (ok) ;
    }

    /* ---------------------------------------------------------------------- */
    /* memory tests */
    /* ---------------------------------------------------------------------- */

    R3 = CHOLMOD(speye)(nrow, 1, CHOLMOD_PATTERN, cm) ;	/* [ */
    OKP (R3) ;

    test_memory_handler ( ) ;

    ok = CHOLMOD(amd)(A, NULL, 0, Pok, cm) ;
    if (A->nrow == 0)
    {
	OK (ok) ;
    }
    else
    {
	NOT (ok) ;
    }

#ifndef NCAMD
    ok = CHOLMOD(camd)(A, NULL, 0, NULL, Pok, cm) ;
    if (A->nrow == 0)
    {
	OK (ok) ;
    }
    else
    {
	NOT (ok) ;
    }
#endif

    C = CHOLMOD(aat)(A, NULL, 0, 0, cm) ;			    NOP (C) ;
    A->sorted = FALSE ;
    ok = CHOLMOD(check_sparse)(A, cm) ;				    NOT (ok) ;
    A->sorted = TRUE ;

    CHOLMOD(free_work)(cm) ;
    if (A->stype == 0)
    {
	for (trial = 0 ; !ok && trial < 20 ; trial++)
	{
	    my_tries = trial ;
	    printf ("--------------------- trial "ID"\n", my_tries) ;
	    ok = CHOLMOD(colamd)(A, NULL, 0, TRUE, Pok, cm) ;
	}
	OK (ok) ;
    }


#ifndef NCAMD
    test_memory_handler ( ) ;
    ok = CHOLMOD(ccolamd)(A, fsetok, fsizeok, NULL, Pok, cm) ;	    NOT (ok) ;
    ok = CHOLMOD(csymamd)(A, NULL, Pok, cm) ;			    NOT (ok) ;
    for (trial = 0 ; trial < 7 ; trial++)
    {
	test_memory_handler ( ) ;
	my_tries = trial ;
	ok = CHOLMOD(csymamd)(A, NULL, Pok, cm) ;		    NOT (ok) ;
    }

    if (A->nrow == A->ncol && A->packed)
    {
	test_memory_handler ( ) ;
	my_tries = 8 ;
	ok = CHOLMOD(csymamd)(A, NULL, Pok, cm) ;		    OK (ok) ;
	test_memory_handler ( ) ;
	ok = CHOLMOD(csymamd)(A, NULL, Pok, cm) ;		    NOT (ok) ;
	OK (cm->status == CHOLMOD_OUT_OF_MEMORY) ;
    }

    for (trial = 0 ; trial < 5 ; trial++)
    {
	test_memory_handler ( ) ;
	my_tries = trial ;
	ok = CHOLMOD(camd)(A, NULL, 0, NULL, Pok, cm) ;
	if (A->nrow == 0)
	{
	    OK (ok) ;
	}
	else
	{
	    NOT (ok) ;
	}
    }
#endif

    test_memory_handler ( ) ;

    ok = CHOLMOD(etree)(A, Parent, cm) ;			    NOT (ok) ;
    ok = CHOLMOD(factorize)(A, L, cm) ;				    NOT (ok) ;

    pp = CHOLMOD(malloc)(4, 0, cm) ;				    NOP (pp) ;
    pp = CHOLMOD(calloc)(4, 0, cm) ;				    NOP (pp) ;
    pp = CHOLMOD(calloc)(Size_max, 1, cm) ;			    NOP (pp) ;
    pp = NULL ;
    size = 0 ;
    pp = CHOLMOD(realloc)(4, 0, pp, &size, cm) ;		    NOP (pp) ;
    pp = CHOLMOD(realloc)(Size_max, 1, pp, &size, cm) ;		    NOP (pp) ;

    normal_memory_handler ( ) ;
    OK (CHOLMOD(print_sparse)(A, "A ok", cm)) ;
    OK (CHOLMOD(print_factor)(L, "L ok", cm)) ;

    /* test no_workspace_reallocate flag */
    CHOLMOD (free_work) (cm) ;
    CHOLMOD (allocate_work) (1, 1, 1, cm) ;
    OK (cm->status == CHOLMOD_OK) ;
    cm->no_workspace_reallocate = TRUE ;
    ok = CHOLMOD (allocate_work) (2, 1, 1, cm) ;
    NOT (ok) ;
    ok = CHOLMOD (allocate_work) (1, 2, 1, cm) ;
    NOT (ok) ;
    ok = CHOLMOD (allocate_work) (1, 1, 2, cm) ;
    NOT (ok) ;
    cm->no_workspace_reallocate = FALSE ;
    ok = CHOLMOD (allocate_work) (1, 1, 2, cm) ;
    OK (ok) ;

    cm->print = 4 ;
    ok = CHOLMOD(print_factor)(L, "L for copy", cm) ;
    OK (ok) ;
    ok = FALSE ;
    test_memory_handler ( ) ;
    for (trial = 0 ; !ok && trial < 100 ; trial++)
    {
	my_tries = trial ;
	Lcopy = CHOLMOD(copy_factor)(L, cm) ;
	ok = (Lcopy != NULL) ;
    }
    normal_memory_handler ( ) ;
    ok = CHOLMOD(print_factor)(Lcopy, "Lcopy", cm) ;
    OK (ok) ;
    CHOLMOD(free_factor)(&Lcopy, cm) ;
    cm->print = 1 ;

    test_memory_handler ( ) ;
    ok = CHOLMOD(resymbol)(A, NULL, 0, TRUE, L, cm) ;		NOT (ok) ;
    ok = CHOLMOD(resymbol_noperm)(A, NULL, 0, TRUE, L, cm) ;	NOT (ok) ;

    lr = CHOLMOD(postorder)(Parent, nrow, NULL, Post, cm) ;	NOT (lr>=0) ;

    T = CHOLMOD(copy_triplet)(Tok, cm) ;			NOT (ok) ;

#ifndef NPARTITION
    lr = CHOLMOD(nested_dissection)(A, NULL, 0, Pok, CParent,
	    Cmember, cm) ;
    if (nrow == 0)
    {
	OK (lr >= 0) ;
    }
    else
    {
	NOT (lr >= 0) ;
    }

    lr = CHOLMOD(nested_dissection)(Abad2, NULL, 0, Pok, CParent,
	    Cmember, cm) ;
    NOT (lr >= 0) ;

    ok = CHOLMOD(metis)(A, NULL, 0, TRUE, Pok, cm) ;

    if (nrow == 0)
    {
	OK (ok) ;
    }
    else
    {
	NOT (ok) ;
    }
    lr = CHOLMOD(bisect)(A, NULL, 0, TRUE, Partition, cm) ;

    if (nrow == 0)
    {
	OK (lr == 0) ;
    }
    else
    {
	NOT (lr >= 0) ;
    }

    lr = CHOLMOD(bisect)(Abad2, NULL, 0, TRUE, Partition, cm) ; NOT (lr >= 0) ;

#endif

    if (nrow > 3)
    {
	ok = CHOLMOD(rowdel)(nrow-2, NULL, L5, cm) ;		NOT (ok) ;
	ok = CHOLMOD(rowadd)(nrow-2, R, L5, cm) ;		NOT (ok) ;
	ok = CHOLMOD(updown)(+1, A, L, cm) ;			NOT (ok) ;
    }

    C = CHOLMOD(add)(A, A, one, one, TRUE, TRUE, cm) ;		NOP (C) ;
    C = CHOLMOD(ssmult)(A, A, 0, TRUE, TRUE, cm) ;		NOP (C) ;

    ok = CHOLMOD(rowcolcounts)(A, NULL, 0, Parent, Post,
	NULL, ColCount, First, Level, cm) ;			NOT (ok) ;

    ok = CHOLMOD(rowfac)(A, NULL, beta, 0, 0, L, cm) ;		NOT (ok) ;
    ok = CHOLMOD(transpose_unsym)(A, 1, Pok, NULL, 0, R, cm) ;  NOT (ok) ;
    ok = CHOLMOD(transpose_sym)(A, 1, Pok, R, cm) ;		NOT (ok) ;
    if (nrow > 1)
    {
	ok = CHOLMOD(sort)(A, cm) ;				    NOT (ok) ;
    }

    ok = CHOLMOD(row_subtree)(A, AT, 0, Parent, R3, cm) ;	    NOT (ok) ;
    ATi = (AT == NULL) ? NULL : AT->i ;
    ok = CHOLMOD(row_lsubtree)(A, ATi, 0, 0, L, R3, cm) ;	    NOT (ok) ;

    normal_memory_handler ( ) ;

    /* ---------------------------------------------------------------------- */
    /* free the valid objects */
    /* ---------------------------------------------------------------------- */

    cm->status = CHOLMOD_OK ;

    CHOLMOD(free_triplet)(NULL, cm) ;

    CHOLMOD(free_sparse)(&R3, cm) ;	/* ] */
    CHOLMOD(free_sparse)(&R, cm) ;	/* ] */
    CHOLMOD(free_sparse)(&Acopy, cm) ;	/* ] */
    CHOLMOD(free_factor)(&L5, cm) ;	/* ] */
    CHOLMOD(free_factor)(&L2, cm) ;	/* ] */

    Lbad->xtype = Lxtype ;
    CHOLMOD(free_factor)(&Lbad, cm) ;	/* ] */

    CHOLMOD(free_factor)(&L, cm) ;	/* ] */

    CHOLMOD(free_triplet)(&T, cm) ;

    Axbad->xtype = Axbad_type ;
    CHOLMOD(free_sparse)(&Axbad, cm) ;	    /* ] */

    cm->error_handler = my_handler ;

    Xbad2->xtype = CHOLMOD_REAL ;
    CHOLMOD(free_dense)(&Xbad2, cm) ;	    /* ] */

    Abad2->xtype = Abad2xtype ;
    CHOLMOD(free_sparse)(&Abad2, cm) ;	    /* ] */

    CHOLMOD(free_sparse)(&Abad, cm) ;	    /* ] */

    CHOLMOD(free_sparse)(&R0, cm) ;
    CHOLMOD(free_sparse)(&R1, cm) ; /* ] */

    CHOLMOD(free_sparse)(&Aboth, cm) ;	    /* ] */
    CHOLMOD(free_sparse)(&Sok, cm) ;


    CHOLMOD(free)(nrow, sizeof (Int), Pinv, cm) ;
    CHOLMOD(free)(nrow, sizeof (Int), Parent, cm) ;
    CHOLMOD(free)(nrow, sizeof (Int), Post, cm) ;
    CHOLMOD(free)(nrow, sizeof (Int), Cmember, cm) ;
    CHOLMOD(free)(nrow, sizeof (Int), CParent, cm) ;
    CHOLMOD(free)(nrow, sizeof (Int), Partition, cm) ;
    CHOLMOD(free)(nrow, sizeof (Int), ColCount, cm) ;
    CHOLMOD(free)(nrow, sizeof (Int), First, cm) ;
    CHOLMOD(free)(nrow, sizeof (Int), Level, cm) ;  /* ] */

    CHOLMOD(free_dense)(&Two, cm) ;	/* ] */

    CHOLMOD(free_sparse)(&R2, cm) ;	    /* ] */
    CHOLMOD(free)(nrow, sizeof (Int), Pok, cm) ;    /* ] */

    CHOLMOD(free_sparse)(&I1, cm) ;	    /* ] */

    CHOLMOD(free)(nrow, sizeof (Int), Pbad, cm) ;   /* ] */

    CHOLMOD(free)(ncol, sizeof (Int), fsetbad, cm) ;	/* ] */
    CHOLMOD(free)(ncol, sizeof (Int), fsetok, cm) ; /* ] */

    CHOLMOD(free_dense)(&Bok, cm) ; /* ] */

    CHOLMOD(free_dense)(&Xok, cm) ; /* ] */

    CHOLMOD(free_sparse)(&AT, cm) ; /* ] */

    CHOLMOD(free_sparse)(&A, cm) ;  /* ] */

    OK (cm->status == CHOLMOD_OK) ;
    printf ("\n------------------------null2 tests: All OK\n") ;
}
