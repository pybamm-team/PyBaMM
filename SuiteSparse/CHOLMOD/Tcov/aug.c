/* ========================================================================== */
/* === Tcov/aug ============================================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Create the augmented system S = [-I A' ; A alpha*I], solve Sx=b, and return
 * the residual.  The system solved is (alpha*I + A*A')*x=b.
 *
 * r1 = norm (Sx-b)
 * r2 = norm ((alpha*I+AA')x-b)
 * alpha = norm(A)
 */

#include "cm.h"


/* ========================================================================== */
/* === aug ================================================================== */
/* ========================================================================== */

double aug (cholmod_sparse *A)
{
    double r, maxerr = 0, bnorm, anorm ;
    cholmod_sparse *S, *Im, *In, *At, *A1, *A2, *Sup ;
    cholmod_dense *Alpha, *B, *Baug, *X, *W1, *W2, *R, *X2, X2mat ;
    cholmod_factor *L ;
    double *b, *baug, *rx, *w, *x ;
    Int nrow, ncol, nrhs, i, j, d, d2, save, save2, save3 ;

    if (A == NULL)
    {
	ERROR (CHOLMOD_INVALID, "cm: no A for aug") ;
	return (1) ;
    }

    if (A->xtype != CHOLMOD_REAL)
    {
	return (0) ;
    }

    /* ---------------------------------------------------------------------- */
    /* A is m-by-n, B must be m-by-nrhs */
    /* ---------------------------------------------------------------------- */

    nrow = A->nrow ;
    ncol = A->ncol ;
    B = rhs (A, 5, A->nrow + 7) ;

    /* ---------------------------------------------------------------------- */
    /* create scalars */
    /* ---------------------------------------------------------------------- */

    bnorm = CHOLMOD(norm_dense) (B, 0, cm) ;
    anorm = CHOLMOD(norm_sparse) (A, 1, cm) ;

    Alpha = CHOLMOD(eye) (1, 1, CHOLMOD_REAL, cm) ;
    if (Alpha != NULL)
    {
	((double *) (Alpha->x)) [0] = anorm ;
    }

    CHOLMOD(print_dense) (M1, "MinusOne", cm) ;
    CHOLMOD(print_dense) (Alpha, "Alpha = norm(A)", cm) ;

    /* ---------------------------------------------------------------------- */
    /* create augmented system, S = [-I A' ; A anorm*I] */
    /* ---------------------------------------------------------------------- */

    Im = CHOLMOD(speye) (nrow, nrow, CHOLMOD_REAL, cm) ;
    In = CHOLMOD(speye) (ncol, ncol, CHOLMOD_REAL, cm) ;
    CHOLMOD(scale) (Alpha, CHOLMOD_SCALAR, Im, cm) ;
    CHOLMOD(scale) (M1, CHOLMOD_SCALAR, In, cm) ;
    At = CHOLMOD(transpose) (A, 2, cm) ;

    /* use one of two equivalent methods */
    if (nrow % 2)
    {
	/* S = [[-In A'] ; [A alpha*Im]] */
	A1 = CHOLMOD(horzcat) (In, At, TRUE, cm) ;
	A2 = CHOLMOD(horzcat) (A,  Im, TRUE, cm) ;
	S = CHOLMOD(vertcat) (A1, A2, TRUE, cm) ;
    }
    else
    {
	/* S = [[-In ; A] [A' ; alpha*Im]] */
	A1 = CHOLMOD(vertcat) (In, A, TRUE, cm) ;
	A2 = CHOLMOD(vertcat) (At, Im, TRUE, cm) ;
	S = CHOLMOD(horzcat) (A1, A2, TRUE, cm) ;
    }

    CHOLMOD(free_sparse) (&Im, cm) ;
    CHOLMOD(free_sparse) (&In, cm) ;

    CHOLMOD(print_sparse) (S, "S, augmented system", cm) ;

    /* make a symmetric (upper) copy of S */
    Sup = CHOLMOD(copy) (S, 1, 1, cm) ;

    CHOLMOD(print_sparse) (S, "S, augmented system (upper)", cm) ;
    CHOLMOD(print_sparse) (Sup, "Sup", cm) ;

    /* ---------------------------------------------------------------------- */
    /* create augmented right-hand-side, Baug = [ zeros(ncol,nrhs) ; B ] */
    /* ---------------------------------------------------------------------- */

    b = NULL ;
    d = 0 ;
    nrhs = 0 ;
    d2 = 0 ;
    if (B != NULL)
    {
	nrhs = B->ncol ;
	d = B->d ;
	b = B->x ;
	Baug = CHOLMOD(zeros) (nrow+ncol, nrhs, CHOLMOD_REAL, cm) ;
	if (Baug != NULL)
	{
	    d2 = Baug->d ;
	    baug = Baug->x ;
	    for (j = 0 ; j < nrhs ; j++)
	    {
		for (i = 0 ; i < nrow ; i++)
		{
		    baug [(i+ncol)+j*d2] = b [i+j*d] ;
		}
	    }
	}
    }
    else
    {
	Baug = NULL ;
    }

    /* ---------------------------------------------------------------------- */
    /* solve Sx=baug */
    /* ---------------------------------------------------------------------- */

    /* S is symmetric indefinite, so do not use a supernodal LL' */
    save = cm->supernodal ;
    save2 = cm->final_asis ;
    cm->supernodal = CHOLMOD_SIMPLICIAL ;
    cm->final_asis = TRUE ;
    save3 = cm->metis_memory ;
    cm->metis_memory = 2.0 ;
    L = CHOLMOD(analyze) (Sup, cm) ;
    CHOLMOD(factorize) (Sup, L, cm) ;
    X = CHOLMOD(solve) (CHOLMOD_A, L, Baug, cm) ;
    cm->supernodal = save ;
    cm->final_asis = save2 ;
    cm->metis_memory = save3 ;

    /* ---------------------------------------------------------------------- */
    /* compute the residual */
    /* ---------------------------------------------------------------------- */

    r = resid (Sup, X, Baug) ;
    MAXERR (maxerr, r, 1) ;

    /* ---------------------------------------------------------------------- */
    /* create a shallow submatrix of X, X2 = X (ncol:end, :)  */
    /* ---------------------------------------------------------------------- */

    if (X == NULL)
    {
	X2 = NULL ;
    }
    else
    {
	X2 = &X2mat ;
	X2->nrow = nrow ; 
	X2->ncol = nrhs ; 
	X2->nzmax = X->nzmax ;
	X2->d = X->d ;
	X2->x = ((double *) X->x) + ncol ;
	X2->z = NULL ;
	X2->xtype = X->xtype ;
	X2->dtype = X->dtype ;
    }

    CHOLMOD(print_dense) (X, "X", cm) ;
    CHOLMOD(print_dense) (X2, "X2 = X (ncol:end,:)", cm) ;

    /* ---------------------------------------------------------------------- */
    /* compute norm ((alpha*I + A*A')*x-b) */
    /* ---------------------------------------------------------------------- */

    /* W1 = A'*X2 */
    W1 = CHOLMOD(zeros) (ncol, nrhs, CHOLMOD_REAL, cm) ;
    CHOLMOD(sdmult) (A, TRUE, one, zero, X2, W1, cm) ;

    /* W2 = A*W1 */
    W2 = CHOLMOD(zeros) (nrow, nrhs, CHOLMOD_REAL, cm) ;
    CHOLMOD(sdmult) (A, FALSE, one, zero, W1, W2, cm) ;

    /* R = alpha*x + w2 - b */
    R = CHOLMOD(zeros) (nrow, nrhs, CHOLMOD_REAL, cm) ;

    if (R != NULL && W2 != NULL && X != NULL)
    {
	w = W2->x ;
	rx = R->x ;
	x = X2->x ;
	for (j = 0 ; j < nrhs ; j++)
	{
	    for (i = 0 ; i < nrow ; i++)
	    {
		rx [i+j*nrow] = anorm * x [i+j*d2] + w [i+j*nrow] - b [i+j*d] ;
	    }
	}
    }

    r = CHOLMOD(norm_dense) (R, 1, cm) ;
    MAXERR (maxerr, r, bnorm) ;

    /* ---------------------------------------------------------------------- */
    /* free everything */
    /* ---------------------------------------------------------------------- */

    CHOLMOD(free_sparse) (&At, cm) ;
    CHOLMOD(free_sparse) (&A1, cm) ;
    CHOLMOD(free_sparse) (&A2, cm) ;
    CHOLMOD(free_sparse) (&S, cm) ;
    CHOLMOD(free_sparse) (&Sup, cm) ;
    CHOLMOD(free_factor) (&L, cm) ;
    CHOLMOD(free_dense) (&R, cm) ;
    CHOLMOD(free_dense) (&W1, cm) ;
    CHOLMOD(free_dense) (&W2, cm) ;
    CHOLMOD(free_dense) (&B, cm) ;
    CHOLMOD(free_dense) (&Baug, cm) ;
    CHOLMOD(free_dense) (&X, cm) ;
    CHOLMOD(free_dense) (&Alpha, cm) ;

    progress (0, '.') ;
    return (maxerr) ;
}
