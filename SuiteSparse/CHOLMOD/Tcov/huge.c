/* ========================================================================== */
/* === Tcov/huge ============================================================ */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Tests on huge matrices */

#include "cm.h"
#include "amd.h"
#ifndef NCAMD
#include "camd.h"
#endif


/* ========================================================================== */
/* === huge ================================================================= */
/* ========================================================================== */

void huge ( )
{
    cholmod_sparse *A, *C ;
    cholmod_triplet *T ;
    cholmod_factor *L ;
    cholmod_dense *X ;
    size_t n, nbig ;
    int ok = TRUE, save ;
    Int junk ;
    FILE *f ;
    double beta [2] ;

    n = Size_max ;
    CHOLMOD (free_work) (cm) ;
    CHOLMOD (allocate_work) (n, 0, 0, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    n = CHOLMOD(add_size_t) (n, 1, &ok) ;
    NOT (ok) ;

    /* create a fake zero sparse matrix, with huge dimensions */
    A = CHOLMOD (spzeros) (1, 1, 0, CHOLMOD_REAL, cm) ;
    A->nrow = Size_max ;
    A->ncol = Size_max ;
    A->stype = 0 ;

    /* create a fake factor, with huge dimensions.  */
    L = CHOLMOD (allocate_factor)  (1, cm) ;
    OKP (L) ;
    L->n = Size_max ;
    CHOLMOD (factorize) (A, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    /* free the fake factor */
    L->n = 1 ;
    CHOLMOD (free_factor) (&L, cm) ;

    /* create a valid factor to test resymbol */
    C = CHOLMOD (speye) (1, 1, CHOLMOD_REAL, cm) ;
    C->stype = 1 ;
    L = CHOLMOD (analyze) (C, cm) ;
    OKP (L) ;
    CHOLMOD (factorize) (C, L, cm) ;
    ok = CHOLMOD (resymbol) (C, NULL, 0, 0, L, cm) ;
    OK (ok) ;
    C->nrow = Size_max ;
    C->ncol = Size_max ;
    L->n = Size_max ;

    ok = CHOLMOD (resymbol) (C, NULL, 0, 0, L, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    printf ("rowfac:\n") ;
    beta [0] = 1 ;
    beta [1] = 0 ;
    C->xtype = CHOLMOD_COMPLEX ;
    L->xtype = CHOLMOD_COMPLEX ;
    ok = CHOLMOD (rowfac) (C, NULL, beta, 0, 0, L, cm) ;
    printf ("rowfac %d\n", cm->status) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;
    C->xtype = CHOLMOD_REAL ;
    L->xtype = CHOLMOD_REAL ;
    printf ("rowfac done:\n") ;

    C->stype = -1 ;
    ok = CHOLMOD (resymbol_noperm) (C, NULL, 0, 0, L, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    C->ncol = 1 ;
    CHOLMOD (rowadd) (0, C, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    CHOLMOD (rowdel) (0, C, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    C->ncol = 4 ;
    CHOLMOD (updown) (1, C, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    C->nrow = 1 ;
    C->ncol = 1 ;
    L->n = 1 ;
    CHOLMOD (free_sparse) (&C, cm) ;
    CHOLMOD (free_factor) (&L, cm) ;

    C = CHOLMOD (allocate_sparse) (Size_max, Size_max, Size_max, 0, 0, 0, 0, cm);
    NOP (C) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    CHOLMOD (rowcolcounts) (A, NULL, 0,
	&junk, &junk, &junk, &junk, &junk, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    C = CHOLMOD (submatrix) (A, &junk, Size_max/2, &junk, Size_max/2, 0, 0, cm) ;
    NOP (C) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    ok = CHOLMOD (transpose_unsym) (A, 0, &junk, &junk, Size_max, A, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    A->stype = 1 ;
    ok = CHOLMOD (transpose_sym) (A, 0, &junk, A, cm) ;
    NOT (ok) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    C = CHOLMOD (ptranspose) (A, 0, &junk, NULL, 0, cm) ;
    NOP (C) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;
    A->stype = 0 ;

    CHOLMOD (amd) (A, NULL, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    L = CHOLMOD (analyze) (A, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;
    NOP (L) ;

#ifndef NCAMD
    CHOLMOD (camd) (A, NULL, 0, &junk, NULL, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;
#endif

    printf ("calling colamd\n") ;
    CHOLMOD (colamd) (A, NULL, 0, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

#ifndef NCAMD
    printf ("calling ccolamd\n") ;
    CHOLMOD (ccolamd) (A, NULL, 0, NULL, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;
#endif

    CHOLMOD (etree) (A, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    L = CHOLMOD (allocate_factor) (Size_max, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;
    NOP (L) ;

#ifndef NPARTITION
    CHOLMOD (metis) (A, NULL, 0, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    CHOLMOD (bisect) (A, NULL, 0, 0, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    CHOLMOD (nested_dissection) (A, NULL, 0, &junk, &junk, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;
#endif

    CHOLMOD (postorder) (&junk, Size_max, &junk, &junk, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    /* causes overflow in 32-bit version, but not 64-bit */
    f = fopen ("../Tcov/Matrix/mega.tri", "r") ;
    T = CHOLMOD (read_triplet) (f, cm) ;
    if (sizeof (Int) == sizeof (int))
    {
	NOP (T) ;
	OK (cm->status != CHOLMOD_OK) ;
    }
    CHOLMOD (free_triplet) (&T, cm) ;
    fclose (f) ;

    n = Size_max ;
    X = CHOLMOD (allocate_dense) (n, 1, n, CHOLMOD_REAL, cm) ;
    NOP (X) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;

    /* supernodal symbolic test */
    C = CHOLMOD (speye) (1, 1, CHOLMOD_REAL, cm) ;
    C->stype = 1 ;
    save = cm->supernodal ;
    cm->supernodal = CHOLMOD_SIMPLICIAL ;
    L = CHOLMOD (analyze) (C, cm) ;
    OKP (L) ;
    junk = 0 ;
    C->nrow = Size_max ;
    C->ncol = Size_max ;
    L->n = Size_max ;
    CHOLMOD (super_symbolic) (C, C, &junk, L, cm) ;
    OK (cm->status == CHOLMOD_TOO_LARGE) ;
    cm->supernodal = save ;
    C->nrow = 1 ;
    C->ncol = 1 ;
    L->n = 1 ;
    CHOLMOD (free_sparse) (&C, cm) ;
    CHOLMOD (free_factor) (&L, cm) ;

    /* supernodal numeric test */
    C = CHOLMOD (speye) (1, 1, CHOLMOD_REAL, cm) ;
    C->stype = -1 ;
    save = cm->supernodal ;
    cm->supernodal = CHOLMOD_SUPERNODAL ;
    L = CHOLMOD (analyze) (C, cm) ;
    OKP (L) ;
    OK (cm->status == CHOLMOD_OK) ;
    C->nrow = Size_max ;
    C->ncol = Size_max ;
    L->n = Size_max ;
    CHOLMOD (super_numeric) (C, C, beta, L, cm) ;
    cm->supernodal = save ;
    C->nrow = 1 ;
    C->ncol = 1 ;
    L->n = 1 ;
    CHOLMOD (free_sparse) (&C, cm) ;
    CHOLMOD (free_factor) (&L, cm) ;

    /* free the fake matrix */
    A->nrow = 1 ;
    A->ncol = 1 ;
    CHOLMOD (free_sparse) (&A, cm) ;

    fprintf (stderr, "\n") ;
}
