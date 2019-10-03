/* ========================================================================== */
/* === Tcov/lpdemo ========================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* A rectangular matrix is being tested (# nrows < # cols).  This is a
 * linear programming problem.  Process the system using the same kind of
 * operations that occur in an LP solver (the LP Dual Active Set Algorithm).
 * This routine does not actually solve the LP.  It simply mimics the kind
 * of matrix operations that occur in LPDASA.
 *
 * The active set f is held in fset [0..fsize-1].  It is a subset of the columns
 * of A.  Columns not in the fset are in the list fnot [0..ncol-fsize-1].
 *
 * Rows can be added and deleted from A as well.  A "dead" row is one that has
 * been (temporarily) set to zero in A.  If row i is dead, rflag [i] is 0,
 * and 1 otherwise.
 *
 * The list r of "live" rows is kept in rset [0..rsize-1].  The list of "dead"
 * rows is kept in rnot [0..nrow-rsize-1].
 *
 * The system to solve as r and/or f change is (beta*I + A(r,f)*A(r,f)') x = b.
 * If a row i is deleted from A, it is set to zero.  Row i of L and D are set
 * to the ith row of the identity matrix.
 */

#include "cm.h"
#define MAXCOLS 8


/* ========================================================================== */
/* === Lcheck =============================================================== */
/* ========================================================================== */

/* Testing only: make sure there are no dead rows in L (excluding diagonal) */

static void Lcheck (cholmod_factor *L, Int *rflag)
{
    Int *Lp, *Li, *Lnz ;
    Int i, n, j, p, pend ;
    double *Lx ;

    if (L == NULL)
    {
	return ;
    }

    Lp = L->p ;
    Li = L->i ;
    Lx = L->x ;
    Lnz = L->nz ;
    n = L->n ;

    for (j = 0 ; j < n ; j++)
    {
	p = Lp [j] ;
	pend = p + Lnz [j] ;
	for (p++ ; p < pend ; p++)
	{
	    i = Li [p] ;
	    OK (IMPLIES (!rflag [i], Lx [p] == 0)) ;
	}
    }
}


/* ========================================================================== */
/* === lp_prune ============================================================= */
/* ========================================================================== */

/* C = A (r,f), except that C and A have the same row dimension.  Row i of C
 * and A(:,f) are equal if row i is in the rset.  Row i of C is zero
 * otherwise.  C has as many columns as the size of f. */

cholmod_sparse *lp_prune
(
    cholmod_sparse *A,
    Int *rflag,
    Int *fset,
    Int fsize
)
{
    cholmod_sparse *C ;
    double *Ax, *Cx ;
    Int *Ai, *Ap, *Ci, *Cp ;
    Int i, kk, j, p, nz, nf, ncol ;

    if (A == NULL)
    {
	ERROR (CHOLMOD_INVALID, "nothing to prune") ;
	return (NULL) ;
    }

    Ap = A->p ;
    Ai = A->i ;
    Ax = A->x ;
    ncol = A->ncol ;
    nf = (fset == NULL) ? ncol : fsize ;

    OK (fsize >= 0) ;

    C = CHOLMOD(allocate_sparse) (A->nrow, nf, A->nzmax, A->sorted,
	    TRUE, 0, CHOLMOD_REAL, cm) ;

    if (C == NULL)
    {
	ERROR (CHOLMOD_INVALID, "cannot create pruned C") ;
	return (NULL) ;
    }

    Cp = C->p ;
    Ci = C->i ;
    Cx = C->x ;

    nz = 0 ;

    for (kk = 0 ; kk < nf ; kk++)
    {
	j = (fset == NULL) ? (kk) : (fset [kk]) ;
	Cp [kk] = nz ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;
	    if (rflag [i])
	    {
		Ci [nz] = i ;
		Cx [nz] = Ax [p] ;
		nz++ ;
	    }
	}
    }
    Cp [nf] = nz ;
    return (C) ;
}


/* ========================================================================== */
/* === lp_resid ============================================================= */
/* ========================================================================== */

/* Compute the 2-norm of the residual.
 * norm ((beta*I + C*C')y(r) - b(r)), where C = A (r,f).
 */

double lp_resid
(
    cholmod_sparse *A, 
    Int *rflag,
    Int *fset,
    Int fsize,
    double beta [2],
    cholmod_dense *Y,
    cholmod_dense *B
)
{
    cholmod_dense *R ;
    double *Rx, *Yx ;
    double rnorm, bnorm, ynorm, norm ;
    cholmod_sparse *C ;
    cholmod_dense *W ;
    Int i, nrow ;

    if (A == NULL)
    {
	ERROR (CHOLMOD_INVALID, "cannot compute LP resid") ;
	return (1) ;
    }

    nrow = A->nrow ;
    R = CHOLMOD(zeros) (nrow, 1, CHOLMOD_REAL, cm) ;

    /* C = A(r,f).  In LPDASA, we do this in place, without making a copy. */
    C = lp_prune (A, rflag, fset, fsize) ;

    /* W = C'*Y */
    OK (fsize >= 0) ;
    W = CHOLMOD(zeros) (fsize, 1, CHOLMOD_REAL, cm) ;
    CHOLMOD(sdmult) (C, TRUE, one, zero, Y, W, cm) ;

    /* R = B */
    CHOLMOD(copy_dense2) (B, R, cm) ;

    /* R = C*W - R */
    CHOLMOD(sdmult) (C, FALSE, one, minusone, W, R, cm) ;

    /* R = R + beta*Y, (beta = 1 for dropped rows) */
    if (R != NULL && Y != NULL)
    {
	Rx = R->x ;
	Yx = Y->x ;
	for (i = 0 ; i < nrow ; i++)
	{
	    if (rflag [i])
	    {
		Rx [i] += beta [0] * Yx [i] ;
	    }
	    else
	    {
		Rx [i] += Yx [i] ;
	    }
	}
    }

    /* rnorm = norm (R) */
    rnorm = CHOLMOD(norm_dense) (R, 2, cm) ;
    bnorm = CHOLMOD(norm_dense) (B, 2, cm) ;
    ynorm = CHOLMOD(norm_dense) (Y, 2, cm) ;
    norm = MAX (bnorm, ynorm) ;
    if (norm > 0)
    {
	rnorm /= norm ;
    }

    CHOLMOD(print_dense) (R, "R, resid", cm) ;

    CHOLMOD(free_sparse) (&C, cm) ;
    CHOLMOD(free_dense) (&W, cm) ;
    CHOLMOD(free_dense) (&R, cm) ;

    return (rnorm) ;
}


/* ========================================================================== */
/* === get_row ============================================================== */
/* ========================================================================== */

/* S = column i of beta*I + A(r,f)*A(r,f)' */

cholmod_sparse *get_row 
(
    cholmod_sparse *A,
    Int i,
    Int *rflag,
    Int *fset,
    Int fsize,
    double beta [2]
)
{
    cholmod_sparse *Ri, *R, *C, *S ;
    double *Sx ;
    Int *Sp, *Si ;
    Int p, ii, found ;

    if (rflag [i] == 0)
    {
	S = CHOLMOD(speye) (A->nrow, A->nrow, CHOLMOD_REAL, cm) ;
	CHOLMOD(print_sparse) (S, "S identity", cm) ;
	return (S) ;
    } 
    OK (fsize >= 0) ;

    /* Getting row i of A is expensive.  In LPDASA, we maintain
     * a copy of A(r,f)', and extact row i as column i of that
     * matrix.  We compute S = A(r,f)*A(i,f)' and S(i) += beta
     * in a single pass.  This is a simpler but slower method. */

    /* R = A (i,f)' */
    Ri = CHOLMOD(submatrix) (A, &i, 1, fset, fsize, TRUE, FALSE, cm) ;
    R = CHOLMOD(transpose) (Ri, 1, cm) ;
    CHOLMOD(free_sparse) (&Ri, cm) ;

    /* C = A (r,f) */
    C = lp_prune (A, rflag, fset, fsize) ;

    /* S = C*R */
    S = CHOLMOD(ssmult) (C, R, 0, TRUE, TRUE, cm) ;
    CHOLMOD(free_sparse) (&C, cm) ;
    CHOLMOD(free_sparse) (&R, cm) ;

    if (S == NULL)
    {
	return (NULL) ;
    }

    /* S (i) += beta */
    found = FALSE ;
    Sp = S->p ;
    Si = S->i ;
    Sx = S->x ;
    for (p = Sp [0] ; p < Sp [1] ; p++)
    {
	ii = Si [p] ;
	if (ii == i)
	{
	    found = TRUE ;
	    Sx [p] += beta [0] ;
	    break ;
	}
    }
    if (!found)
    {
	/* oops, row index i is not present in S.  Add it. */
	CHOLMOD(reallocate_sparse) (S->nzmax+1, S, cm) ;
	OK (Sp [1] < (Int) (S->nzmax)) ;
	Si = S->i ;
	Sx = S->x ;
	Si [Sp [1]] = i ;
	Sx [Sp [1]] = beta [0] ;
	Sp [1]++ ;
	S->sorted = FALSE ;
    }

    CHOLMOD(print_sparse) (S, "S", cm) ;

    return (S) ;
}


/* ========================================================================== */
/* === lpdemo =============================================================== */
/* ========================================================================== */

double lpdemo (cholmod_triplet *T)
{
    double r, maxerr = 0, anorm, bnorm, norm, xnorm, ynorm ;
    double *b = NULL, *Yx = NULL, *Xx = NULL, *Sx ;
    cholmod_sparse *A, *AT, *Apermuted, *C, *S, *Row ;
    cholmod_dense *X, *B, *Y, *DeltaB, *R ;
    cholmod_factor *L ;
    Int *init, *rset, *rnot, *fset, *fnot, *rflag, *P, *Pinv, *Lperm, *fflag,
	*Sp, *Si, *StaticParent ;
    Int i, j, k, nrow, ncol, fsize, cols [MAXCOLS+1], trial, rank, kk, rsize,
	p, op, ok ;
    double beta [2], bk [2], yk [2] ;

    /* ---------------------------------------------------------------------- */
    /* convert T into a sparse matrix A */
    /* ---------------------------------------------------------------------- */

    if (T == NULL || T->ncol == 0)
    {
	/* nothing to do */
	return (0) ;
    }

    if (T->xtype != CHOLMOD_REAL)
    {
	return (0) ;
    }

    A = CHOLMOD(triplet_to_sparse) (T, 0, cm) ;

    if (A == NULL)
    {
	ERROR (CHOLMOD_INVALID, "cannot continue LP demo") ;
	return (1) ;
    }

    nrow = A->nrow ;
    ncol = A->ncol ;

    anorm = CHOLMOD(norm_sparse) (A, 1, cm) ;

    /* switch for afiro, but not galenet */
    cm->supernodal_switch = 5 ;

    /* ---------------------------------------------------------------------- */
    /* select a random initial row and column basis */
    /* ---------------------------------------------------------------------- */

    /* select an initial fset of size nrow */
    init = prand (ncol) ;					/* RAND */
    fset = CHOLMOD(malloc) (ncol, sizeof (Int), cm) ;
    fnot = CHOLMOD(malloc) (ncol, sizeof (Int), cm) ;
    fflag = CHOLMOD(malloc) (ncol, sizeof (Int), cm) ;
    fsize = MIN (nrow, ncol) ;

    if (init != NULL && fset != NULL && fflag != NULL)
    {
	for (k = 0 ; k < fsize ; k++)
	{
	    j = init [k] ;
	    fset [k] = j ;
	    fflag [j] = 1 ;
	}
	for ( ; k < ncol ; k++)
	{
	    j = init [k] ;
	    fnot [k-fsize] = j ;
	    fflag [j] = 0 ;
	}
    }

    CHOLMOD(free) (ncol, sizeof (Int), init, cm) ;

    /* all rows are live */
    rsize = nrow ;
    rflag = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
    rset = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
    rnot = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;

    if (rset != NULL && rflag != NULL)
    {
	for (i = 0 ; i < nrow ; i++)
	{
	    rflag [i] = 1 ;
	    rset [i] = i ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* factorize the first matrix, beta*I + A(p,f)*A(p,f)' */
    /* ---------------------------------------------------------------------- */

    beta [0] = 1e-6 ;
    beta [1] = 0 ;

    /* Need to prune entries due to relaxed amalgamation, or else
     * cholmod_row_subtree will not be able to find all the entries in row
     * k of L. */
    cm->final_resymbol = TRUE ;

    cm->final_asis = FALSE ;
    cm->final_super = FALSE ;
    cm->final_ll = FALSE ;
    cm->final_pack = FALSE ;
    cm->final_monotonic = FALSE ;

    L = CHOLMOD(analyze_p) (A, NULL, fset, fsize, cm) ;
    CHOLMOD(factorize_p) (A, beta, fset, fsize, L, cm) ;

    /* get a copy of the fill-reducing permutation P and compute its inverse */
    Lperm = (L != NULL) ? (L->Perm) : NULL ;
    P = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
    Pinv = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;

    if (P != NULL && Pinv != NULL && Lperm != NULL)
    {
	for (k = 0 ; k < nrow ; k++)
	{
	    P [k] = Lperm [k] ;
	    Pinv [P [k]] = k ;
	}
    }
    else
    {
	P = CHOLMOD(free) (nrow, sizeof (Int), P, cm) ;
	Pinv = CHOLMOD(free) (nrow, sizeof (Int), Pinv, cm) ;
    }

    if (cm->print > 1)
    {
	k = cm->print ;
	cm->print = 5 ;
	CHOLMOD(print_common) ("cm for lpdemo", cm) ;
	cm->print = k ;
    }

    /* ---------------------------------------------------------------------- */
    /* A=P*A: permute the rows of A according to P */
    /* ---------------------------------------------------------------------- */

    /* This is done just once, since the system will be solved and modified
     * many times.  It's faster, and easier, to work in the permuted ordering
     * rather than the original ordering. */

    /* A will become unsorted later on; don't bother to sort it here */
    Apermuted = CHOLMOD(submatrix) (A, P, nrow, NULL, -1, TRUE, TRUE, cm) ;
    CHOLMOD(free_sparse) (&A, cm) ;
    A = Apermuted ;

    /* ---------------------------------------------------------------------- */
    /* find the etree of A*A' */
    /* ---------------------------------------------------------------------- */

    /* Since the fset is a subset of 0:ncol-1, and rset is a subset of 0:nrow-1,
     * the nonzero pattern of the Cholesky factorization of A(r,f)*A(r,f)' is a
     * subset of the Cholesky factorization of A*A'.  After many updates/
     * downdates/rowadds/rowdels, any given row i of L may have entries that
     * are not in the factorization of A (r,f)*A(r,f)'.  To drop a row using
     * cholmod_rowdel, we either need to know the pattern of the ith row of L,
     * we can pass NULL and have cholmod_rowdel look at each column 0 to i-1.
     * The StaticParent array is the etree of A*A', and it suffices to compute
     * the pattern of the ith row of L based on that etree, and A and A'
     * (ignoring the fset and rset).  This gives us an upper bound on the
     * nonzero pattern of the ith row of the current L (the factorization
     * of A(r,f)*A(r,f)'.
     */

    /* AT = nonzero pattern of A', used for row-subtree computations */
    AT = CHOLMOD(transpose) (A, 0, cm) ;

    /* Row = cholmod_row_subtree workspace (unsorted, packed, unsym, pattern) */
    Row = CHOLMOD(allocate_sparse) (nrow, 1, nrow, FALSE, TRUE, 0,
	    CHOLMOD_PATTERN, cm) ;

    /* Compute the "static" etree; the etree of A*A' */
    StaticParent = CHOLMOD(malloc) (nrow, sizeof (Int), cm) ;
    CHOLMOD(etree) (AT, StaticParent, cm) ;

    /* ---------------------------------------------------------------------- */
    /* compute initial right-hand-side */
    /* ---------------------------------------------------------------------- */

    /* If row i of the original A and B is row k of the permuted P*A and P*B,
     * then P [k] = i and Pinv [i] = k.  Row indices of A now refer to the
     * permuted form of A, not the original A.  Likewise, row k of B will
     * refer to the permuted row k = Pinv [i], not the original row i.  In a
     * real program, this would affect how B is computed.  This program just
     * creates a random B anyway, so the order of B does not matter.  It does
     * use Pinv [i], just to show you how you would do it.
     */

    B = CHOLMOD(zeros) (nrow, 1, CHOLMOD_REAL, cm) ;

    if (B != NULL && Pinv != NULL)
    {
	b = B->x ;
	for (i = 0 ; i < nrow ; i++)
	{
	    /* row i of the original B is row k of the permuted B */
	    k = Pinv [i] ;
	    b [k] = xrand (1.) ;				/* RAND */
	}
    }

    /* ---------------------------------------------------------------------- */
    /* solve the system */
    /* ---------------------------------------------------------------------- */

    /* Solve the system (beta*I + A(:,f)*A(:,f)')y=b without using L->Perm,
     * since A and B have already been permuted according to L->Perm. */

    DeltaB = CHOLMOD(zeros) (nrow, 1, CHOLMOD_REAL, cm) ;

    /* solve Lx=b */
    X = CHOLMOD(solve) (CHOLMOD_L, L, B, cm) ;

    /* solve DL'y=x */
    Y = CHOLMOD(solve) (CHOLMOD_DLt, L, X, cm) ;

    r = lp_resid (A, rflag, fset, fsize, beta, Y, B) ;
    MAXERR (maxerr, r, 1) ;

    bk [0] = 0 ;
    bk [1] = 0 ;

    yk [0] = 0 ;
    yk [1] = 0 ;

    bnorm = CHOLMOD(norm_dense) (B, 1, cm) ;

    /* ---------------------------------------------------------------------- */
    /* modify the system */
    /* ---------------------------------------------------------------------- */

    ok = (fset != NULL && fnot != NULL && fflag != NULL &&
	  rset != NULL && rnot != NULL && rflag != NULL &&
	  B != NULL && Y != NULL && X != NULL && Row != NULL && A != NULL &&
	  AT != NULL && StaticParent != NULL && DeltaB != NULL && L != NULL &&
	  L->xtype != CHOLMOD_PATTERN && !(L->is_ll) && !(L->is_super)) ;

    for (trial = 1 ; ok && trial < MAX (64, 2*ncol) ; trial++)
    {
	/* select an operation at random */
	op = nrand (6) ;					/* RAND */

	Xx = X->x ;
	Yx = Y->x ;

	switch (op)
	{

	    /* -------------------------------------------------------------- */
	    case 0:	/* update */
	    /* -------------------------------------------------------------- */

		/* pick some columns at random, but not all columns */
		rank = 1 + nrand (MAXCOLS+4) ;			/* RAND */
		rank = MIN (rank, MAXCOLS) ;

		rank = MIN (rank, ncol-fsize-1) ;
		if (rank <= 0)
		{
		    continue ;
		}

		/* remove the columns from fnot and add them to fset */
		for (k = 0 ; k < rank ; k++)
		{
		    kk = nrand (ncol-fsize) ;			/* RAND */
		    j = fnot [kk] ;
		    fnot [kk] = fnot [ncol-fsize-1] ;
		    fset [fsize++] = j ;
		    OK (fsize < ncol) ;
		    cols [k] = j ;
		    fflag [j] = 1 ;
		}

		/* update L, and the solution to Lx=b+deltaB */
		C = lp_prune (A, rflag, cols, rank) ;
		ok = CHOLMOD(updown_solve) (TRUE, C, L, X, DeltaB, cm) ;
		CHOLMOD(free_sparse) (&C, cm) ;
		break ;

	    /* -------------------------------------------------------------- */
	    case 1:	/* downdate */
	    /* -------------------------------------------------------------- */

		/* pick some columns at random, but not all columns */
		rank = 1 + nrand (MAXCOLS+4) ;			/* RAND */
		rank = MIN (rank, MAXCOLS) ;

		rank = MIN (rank, fsize-1) ;
		if (rank <= 0)
		{
		    continue ;
		}

		/* remove the columns from fset and add them to fnot */
		for (k = 0 ; k < rank ; k++)
		{
		    kk = nrand (fsize) ;			/* RAND */
		    j = fset [kk] ;
		    fset [kk] = fset [fsize-1] ;
		    fnot [ncol-fsize] = j ;
		    fsize-- ;
		    OK (fsize > 0) ;
		    cols [k] = j ;
		    fflag [j] = 0 ;
		}

		/* downdate L, and the solution to Lx=b+deltaB */
		C = lp_prune (A, rflag, cols, rank) ;
		ok = CHOLMOD(updown_solve) (FALSE, C, L, X, DeltaB, cm) ;
		CHOLMOD(free_sparse) (&C, cm) ;
		break ;

	    /* -------------------------------------------------------------- */
	    case 2:	/* resymbol (no change to numerical values) */
	    /* -------------------------------------------------------------- */

		/* let resymbol handle the fset */
		C = lp_prune (A, rflag, NULL, 0) ;
		ok = CHOLMOD(resymbol_noperm) (C, fset, fsize, TRUE, L, cm) ;
		CHOLMOD(free_sparse) (&C, cm) ;
		break;

	    /* -------------------------------------------------------------- */
	    case 3:	/* add row */
	    /* -------------------------------------------------------------- */

		/* remove a row from rnot and add to rset */
		if (nrow == rsize)
		{
		    continue ;
		}
		kk = nrand (nrow-rsize) ;			/* RAND */
		i = rnot [kk] ;

		OK (rflag [i] == 0) ;

		rnot [kk] = rnot [nrow-rsize-1] ;
		rset [rsize++] = i ;
		rflag [i] = 1 ;

		/* S = column i of beta*I + A(r,f)*A(r,f)' */
		S = get_row (A, i, rflag, fset, fsize, beta) ;
		ok = (S != NULL) ;

		if (ok)
		{
		    /* pick a random right-hand-side for this new row */
		    b [i] = 1 ; /* xrand (1) */			/* was RAND */
		    bk [0] = b [i] ;
		    bk [1] = 0 ;
		    ok = CHOLMOD(rowadd_solve) (i, S, bk, L, X, DeltaB, cm) ;
		}

		CHOLMOD(free_sparse) (&S, cm) ;
		break ;

	    /* -------------------------------------------------------------- */
	    case 4:	/* delete row */
	    /* -------------------------------------------------------------- */

		/* remove a row from rset and add to rnot */
		if (rsize == 0)
		{
		    continue ;
		}
		kk = nrand (rsize) ;				/* RAND */
		i = rset [kk] ;

		OK (rflag [i] == 1) ;
		rset [kk] = rset [rsize-1] ;
		rnot [nrow-rsize] = i ;
		rsize-- ;

		/* S = column i of beta*I + A(r,f)*A(r,f)' */
		S = get_row (A, i, rflag, fset, fsize, beta) ;
		ok = (S != NULL) ;

		if (ok)
		{
		    /* B = B - S * y(i) */
		    Sp = S->p ;
		    Si = S->i ;
		    Sx = S->x ;
		    for (p = 0 ; p < Sp [1] ; p++)
		    {
			b [Si [p]] -= Sx [p] * Yx [i] ;
		    }
		    /* B(i) = y(i) */
		    b [i] = Yx [i] ;

		    yk [0] = Yx [i] ;
		    yk [1] = 0 ;

		    /* pick a method arbitrarily */
		    if (trial % 2)
		    {
			/* get upper bound nonzero pattern of L(i,0:i-1) */
			CHOLMOD(row_subtree) (A, AT, i, StaticParent, Row, cm) ;
			ok = CHOLMOD(rowdel_solve) (i, Row, yk, L, X, DeltaB,
				cm) ;
		    }
		    else
		    {
			/* Look in all cols 0 to i-1 for entries in L(i,0:i-1).
			 * This is more costly, but requires no knowledge of
			 * an upper bound on the pattern of L. */
			ok = CHOLMOD(rowdel_solve) (i, NULL, yk, L, X, DeltaB,
				cm) ;
		    }

		    /* for testing only, to ensure cholmod_row_subtree worked */
		    if (ok)
		    {
			rflag [i] = 0 ;
			Lcheck (L, rflag) ;
		    }
		}

		if (ok)
		{
		    /* let resymbol handle the fset */
		    C = lp_prune (A, rflag, NULL, 0) ;
		    ok = CHOLMOD(resymbol_noperm) (C, fset, fsize, TRUE, L, cm) ;
		    CHOLMOD(free_sparse) (&C, cm) ;
		}

		CHOLMOD(free_sparse) (&S, cm) ;
		break ;

	    /* -------------------------------------------------------------- */
	    case 5:	/* convert, just for testing */
	    /* -------------------------------------------------------------- */

		/* convert to LDL', optionally packed */
		if (trial % 2)
		{
		    ok = CHOLMOD(change_factor) (CHOLMOD_REAL, FALSE, FALSE,
			    TRUE, TRUE, L, cm) ;
		}
		else
		{ 
		    ok = CHOLMOD(change_factor) (CHOLMOD_REAL, FALSE, FALSE,
			    FALSE, TRUE, L, cm) ;
		}
		break ;

	}

	if (ok)
	{

	    /* scale B and X if their norm is getting large */
	    ynorm = CHOLMOD(norm_dense) (Y, 1, cm) ;
	    bnorm = CHOLMOD(norm_dense) (B, 1, cm) ;
	    xnorm = CHOLMOD(norm_dense) (X, 1, cm) ;
	    norm = MAX (bnorm, xnorm) ;
	    norm = MAX (norm, ynorm) ;
	    if (norm > 1e10)
	    {
		for (i = 0 ; i < nrow ; i++)
		{
		    Xx [i] /= norm ;
		    b  [i] /= norm ;
		}
	    }

	    CHOLMOD(free_dense) (&Y, cm) ;
	    Y = CHOLMOD(solve) (CHOLMOD_DLt, L, X, cm) ;

	    r = lp_resid (A, rflag, fset, fsize, beta, Y, B) ;
	    OK (!ISNAN (r)) ;
	    MAXERR (maxerr, r, 1) ;
	    if (r > 1e-6 && cm->print > 1)
	    {
		printf ("lp err %.1g operation: "ID" ok "ID"\n", r, op, ok) ;
	    }
	    ok = (Y != NULL) ;
	}
    }

    CHOLMOD(free_dense) (&Y, cm) ;
    OK (CHOLMOD(print_common) ("cm in lpdemo", cm)) ;

    /* ---------------------------------------------------------------------- */
    /* convert to LDL packed, LDL unpacked or LL packed and solve again */
    /* ---------------------------------------------------------------------- */

    /* solve the new system and check the residual */

    CHOLMOD(print_factor) (L, "L final, for convert", cm) ;
    if (ok)
    {
	switch (nrand (3))					/* RAND */
	{
	    /* pick one at random */
	    case 0:
	    {
		ok = CHOLMOD(change_factor) (CHOLMOD_REAL, FALSE, FALSE, TRUE,
			TRUE, L, cm) ;
		Y = CHOLMOD(solve) (CHOLMOD_DLt, L, X, cm) ;
		break ;
	    }
	    case 1:
	    {
		ok = CHOLMOD(change_factor) (CHOLMOD_REAL, FALSE, FALSE, FALSE,
			TRUE, L, cm) ;
		Y = CHOLMOD(solve) (CHOLMOD_DLt, L, X, cm) ;
		break ;
	    }
	    case 2:
	    {
		ok = CHOLMOD(change_factor) (CHOLMOD_REAL, TRUE, FALSE, TRUE,
			TRUE, L, cm) ;
		Y = CHOLMOD(solve) (CHOLMOD_LDLt, L, B, cm) ;
		break ;
	    }
	}
	r = lp_resid (A, rflag, fset, fsize, beta, Y, B) ;
	OK (!ISNAN (r)) ;
	MAXERR (maxerr, r, 1) ;
	CHOLMOD(print_factor) (L, "L after convert", cm) ;
    }

    /* ---------------------------------------------------------------------- */
    /* rank-1 update, but only partial Lx=b update */
    /* ---------------------------------------------------------------------- */

    if (ok && fsize < ncol && nrow > 3)
    {
	Int colmark [1] ;

	j = fnot [0] ;
	fnot [0] = fnot [ncol-fsize-1] ;
	fset [fsize++] = j ;
	OK (fsize <= ncol) ;
	cols [0] = j ;
	fflag [j] = 1 ;

	for (colmark [0] = 0 ; colmark [0] <= nrow ; colmark [0]++)
	{
	    cholmod_factor *L2 ;
	    cholmod_dense *X2 ;
	    double *X2x ;
	    L2 = CHOLMOD(copy_factor) (L, cm) ;
	    X2 = CHOLMOD(copy_dense) (X, cm) ;
	    X2x = (X2 == NULL) ? NULL : X2->x ;

	    /* fprintf (stderr, "check colmark "ID"\n", colmark [0]) ; */
	    printf ("check cholmark "ID"\n", colmark [0]) ;
	    /* colmark [0] = 3 ; */

	    /* update L, and the solution to Lx=b+deltaB,
	     * but only update solution in rows 0 to colmark[0] */
	    C = lp_prune (A, rflag, cols, 1) ;
	    ok = CHOLMOD(updown_mark) (TRUE, C, colmark, L2, X2, DeltaB, cm) ;
	    CHOLMOD(free_sparse) (&C, cm) ;

	    /* compare with Lr=b+deltaB */
	    R = CHOLMOD(solve) (CHOLMOD_L, L2, B, cm) ;
	    r = -1 ;
	    if (ok && R != NULL)
	    {
		double *Rx ;
		Rx = R->x ;
		r = 0 ;
		for (i = 0 ; i < colmark [0] ; i++)
		{
		    r = MAX (r, fabs (X2x [i] - Rx [i])) ;
		}
		MAXERR (maxerr, r, 1) ;
	    }
	    printf ("check cholmark resid %6.2e\n", r) ;
	    CHOLMOD(free_dense) (&R, cm) ;
	    CHOLMOD(free_dense) (&X2, cm) ;
	    CHOLMOD(free_factor) (&L2, cm) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* free everything */
    /* ---------------------------------------------------------------------- */

    /* restore defaults */
    cm->final_resymbol = FALSE ;
    cm->final_asis = TRUE ;
    cm->supernodal_switch = 40 ;

    CHOLMOD(free) (nrow, sizeof (Int), StaticParent, cm) ;
    CHOLMOD(free) (nrow, sizeof (Int), Pinv, cm) ;
    CHOLMOD(free) (nrow, sizeof (Int), P, cm) ;
    CHOLMOD(free) (nrow, sizeof (Int), rflag, cm) ;
    CHOLMOD(free) (nrow, sizeof (Int), rset, cm) ;
    CHOLMOD(free) (nrow, sizeof (Int), rnot, cm) ;
    CHOLMOD(free) (ncol, sizeof (Int), fset, cm) ;
    CHOLMOD(free) (ncol, sizeof (Int), fnot, cm) ;
    CHOLMOD(free) (ncol, sizeof (Int), fflag, cm) ;
    CHOLMOD(free_factor) (&L, cm) ;
    CHOLMOD(free_sparse) (&Row, cm) ;
    CHOLMOD(free_sparse) (&AT, cm) ;
    CHOLMOD(free_sparse) (&A, cm) ;
    CHOLMOD(free_dense) (&B, cm) ;
    CHOLMOD(free_dense) (&X, cm) ;
    CHOLMOD(free_dense) (&Y, cm) ;
    CHOLMOD(free_dense) (&DeltaB, cm) ;

    progress (0, '.') ;
    return (maxerr) ;
}
