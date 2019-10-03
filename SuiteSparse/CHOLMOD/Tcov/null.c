/* ========================================================================== */
/* === Tcov/null ============================================================ */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Test CHOLMOD with NULL pointers, and other error cases. */

#include "cm.h"


/* ========================================================================== */
/* === my_hander2 =========================================================== */
/* ========================================================================== */

void my_handler2 (int status, const char *file, int line, const char *msg)
{
    printf ("This ERROR is expected: file %s line %d\n%d: %s\n",
	    file, line, status, msg) ;
}


/* ========================================================================== */
/* === null_test ============================================================ */
/* ========================================================================== */

/* This routine is not called during memory testing */

void null_test (cholmod_common *cn)
{
    cholmod_sparse *A = NULL, *F = NULL, *C = NULL, *R = NULL, *B = NULL ;
    cholmod_factor *L = NULL ;
    cholmod_triplet *T = NULL ;
    cholmod_dense *X = NULL, *DeltaB = NULL, *S = NULL, *Y = NULL, *E = NULL ;
    void *p = NULL, *ii = NULL, *jj = NULL, *xx = NULL, *zz = NULL ;
    Int *Perm = NULL, *fset = NULL, *Parent = NULL, *Post = NULL,
	*RowCount = NULL, *ColCount = NULL, *First = NULL, *Level = NULL,
	*UserPerm = NULL, *colmark = NULL, *Constraints = NULL,
	*r = NULL, *c = NULL, *Set = NULL ;
    char *name = NULL ;
    double alpha [2], beta [2], bk [2], yk [2], rcond ;
    double dj = 1, nm = 0, tol = 0 ;
    int ok, stype = 0, xtype = 0, sorted = 0, packed = 0, nint = 0, update = 0,
	postorder = 0, pack = 0, values = 0, mode = 0, sys = 0, norm = 0,
	to_xtype = 0, to_ll = 0, to_super = 0, to_packed = 0, to_monotonic = 0,
	scale = 0, transpose = 0, option = 0, ordering = 0, prefer = 0,
	mtype = 0, asym = 0 ;
    SuiteSparse_long lr = 0, k1 = 0, k2 = 0 ;
    size_t j = 0, need = 0, n = 0, mr = 0, nrow = 0, ncol = 0, iworksize = 0,
	newsize = 0, fsize = 0, d = 0, nzmax = 0, nnew = 0, size = 0,
	nold = 0, xwork = 0, kstart = 0, kend = 0, nr = 0, nc = 0, len = 0,
	krow = 0, k = 0 ;

#ifndef NPARTITION
    Int *Anw = NULL, *Aew = NULL, *Partition = NULL, *CParent = NULL,
        *Cmember = NULL ;
    Int compress = 0 ;
#endif

#ifndef NCAMD
    Int *Cmem2 = NULL ;
#endif

    /* ---------------------------------------------------------------------- */
    /* Core */
    /* ---------------------------------------------------------------------- */

    if (cn == NULL)
    {
	ok = CHOLMOD(start)(cn) ;				NOT (ok) ;
    }
    ok = CHOLMOD(finish)(cn) ;					NOT (ok) ;
    ok = CHOLMOD(defaults)(cn) ;				NOT (ok) ;
    mr = CHOLMOD(maxrank)(n, cn) ;				NOT (mr>0) ;
    ok = CHOLMOD(allocate_work)(nrow, iworksize, xwork, cn) ;	NOT (ok) ;
    ok = CHOLMOD(free_work)(cn) ;				NOT (ok) ;
    lr = CHOLMOD(clear_flag)(cn) ;				NOT (lr>=0) ;

    dj = CHOLMOD(dbound)(dj, cn) ;				OK (dj==0) ;				
    ok = CHOLMOD(error)(CHOLMOD_INVALID, __FILE__, __LINE__, "oops", cn) ;
								NOT (ok) ;
    A = CHOLMOD(allocate_sparse)(nrow, ncol, nzmax, sorted,
	packed, stype, xtype, cn) ;				NOP (A) ;
    ok = CHOLMOD(free_sparse)(&A, cn) ;				NOT (ok) ;
    ok = CHOLMOD(reallocate_sparse)(newsize, A, cn) ;		NOT (ok) ;
    lr = CHOLMOD(nnz)(A, cn) ;					NOT (lr>=0) ;
    A  = CHOLMOD(speye)(nrow, ncol, xtype, cn) ;		NOP (A) ;
    A  = CHOLMOD(spzeros)(nrow, ncol, 0, xtype, cn) ;		NOP (A) ;
    A  = CHOLMOD(ptranspose)(A, values, Perm, fset, fsize, cn);	NOP (A) ;
    A  = CHOLMOD(transpose)(A, values, cn) ;			NOP (A) ;
    ok = CHOLMOD(transpose_unsym)(A, values, Perm, fset, fsize, F, cn) ;
    NOT (ok) ;
    ok = CHOLMOD(transpose_sym)(A, values, Perm, F, cn) ;	NOT (ok) ;
    ok = CHOLMOD(sort)(A, cn) ;					NOT (ok) ;
    A  = CHOLMOD(copy_sparse)(A, cn) ;				NOP (A) ;
    C  = CHOLMOD(aat)(A, fset, fsize, mode, cn) ;		NOP (C) ;

    L  = CHOLMOD(allocate_factor)(n, cn) ;			NOP (L) ;
    ok = CHOLMOD(free_factor)(&L, cn) ;				NOT (ok) ;
    ok = CHOLMOD(reallocate_factor)(newsize, L, cn) ;		NOT (ok) ;
    ok = CHOLMOD(change_factor)(0, 0, 0, 0, 0, L, cn) ;		NOT (ok) ;
    ok = CHOLMOD(pack_factor)(L, cn) ;				NOT (ok) ;
    ok = CHOLMOD(change_factor)(to_xtype, to_ll, to_super,
	to_packed, to_monotonic, L, cn) ;			NOT (ok) ;
    ok = CHOLMOD(reallocate_column)(j, need, L, cn) ;		NOT (ok) ;
    A  = CHOLMOD(factor_to_sparse)(L, cn) ;			NOP (A) ;
    L  = CHOLMOD(copy_factor)(L, cn) ;				NOP (L) ;

    X  = CHOLMOD(allocate_dense)(nrow, ncol, d, xtype, cn) ;	NOP (X) ;
    X  = CHOLMOD(zeros)(nrow, ncol, xtype, cn) ;		NOP (X) ;
    X  = CHOLMOD(ones)(nrow, ncol, xtype, cn) ;			NOP (X) ;
    X  = CHOLMOD(eye)(nrow, ncol, xtype, cn) ;			NOP (X) ;
    ok = CHOLMOD(free_dense)(&X, cn) ;				NOT (ok) ;
    X  = CHOLMOD(sparse_to_dense)(A, cn) ;			NOP (X) ;
    A  = CHOLMOD(dense_to_sparse)(X, values, cn) ;		NOP (A) ;
    Y  = CHOLMOD(copy_dense)(X, cn) ;				NOP (X) ;
    ok = CHOLMOD(copy_dense2)(X, Y, cn) ;			NOT (ok) ;

    T  = CHOLMOD(allocate_triplet)(nrow, ncol, nzmax,
	    stype, xtype, cn) ;					NOP (T) ;
    ok = CHOLMOD(free_triplet)(&T, cn) ;			NOT (ok) ;
    T  = CHOLMOD(sparse_to_triplet)(A, cn) ;			NOP (T) ;
    A  = CHOLMOD(triplet_to_sparse)(T, 0, cn) ;			NOP (A) ;
    T  = CHOLMOD(copy_triplet)(T, cn) ;				NOP (T) ;
    ok = CHOLMOD(reallocate_triplet)(nzmax, T, cn) ;		NOT (ok) ;

    lr = CHOLMOD(postorder)(Parent, nrow, NULL, Post, cn) ;	NOT (lr>=0) ;
    p  = CHOLMOD(malloc)(n, size, cn) ;				NOP (p) ;
    p  = CHOLMOD(calloc)(n, size, cn) ;				NOP (p) ;
    p  = CHOLMOD(free)(n, size, p, cn) ;			NOP (p) ;
    p  = CHOLMOD(realloc)(nnew, size, p, &n, cn) ;		NOP (p) ;
    ok = CHOLMOD(realloc_multiple)(nnew, nint, xtype,
	    &ii, &jj, &xx, &zz, &nold, cn) ;			NOT (ok) ;

    C = CHOLMOD(band)(A, k1, k2, mode, cn) ;			NOP (C) ;
    ok = CHOLMOD(band_inplace)(k1, k2, mode, A, cn) ;		NOT (ok) ;

    ok = CHOLMOD(factor_xtype)(CHOLMOD_REAL, L, cn) ;		NOT (ok) ;
    ok = CHOLMOD(sparse_xtype)(CHOLMOD_REAL, A, cn) ;		NOT (ok) ;
    ok = CHOLMOD(dense_xtype)(CHOLMOD_REAL, X, cn) ;		NOT (ok) ;
    ok = CHOLMOD(triplet_xtype)(CHOLMOD_REAL, T, cn) ;		NOT (ok) ;

    /* ---------------------------------------------------------------------- */
    /* Cholesky */
    /* ---------------------------------------------------------------------- */

    L  = CHOLMOD(analyze)(A, cn) ;				NOP (L) ;    
    L  = CHOLMOD(analyze_p)(A, UserPerm, fset, fsize, cn) ;	NOP (L) ;    
    ok = CHOLMOD(factorize)(A, L, cn) ;				NOT (ok) ;
    ok = CHOLMOD(factorize_p)(A, beta, fset, fsize, L, cn) ;	NOT (ok) ;
    rcond = CHOLMOD(rcond)(L, cn) ;				NOT (rcond>=0) ;
    X = CHOLMOD(solve)(sys, L, Y, cn) ;				NOP (X) ;
    C = CHOLMOD(spsolve)(sys, L, B, cn) ;			NOP (C) ;
    ok = CHOLMOD(etree)(A, Parent, cn) ;			NOT (ok) ;
    ok = CHOLMOD(rowcolcounts)(A, fset, fsize, Parent, Post,
	RowCount, ColCount, First, Level, cn) ;			NOT (ok) ;
    ok = CHOLMOD(amd)(A, fset, fsize, Perm, cn) ;		NOT (ok) ;
    ok = CHOLMOD(camd)(A, fset, fsize, Constraints, Perm, cn) ;	NOT (ok) ;
    ok = CHOLMOD(colamd)(A, fset, fsize, postorder, Perm, cn) ;	NOT (ok) ;
    ok = CHOLMOD(rowfac)(A, F, beta, kstart, kend, L, cn) ;	NOT (ok) ;
    ok = CHOLMOD(row_subtree)(A, F, krow, Parent, R, cn) ;	NOT (ok) ;
    ok = CHOLMOD(row_lsubtree)(A, c, 0, krow, L, R, cn) ;	NOT (ok) ;
    ok = CHOLMOD(resymbol)(A, fset, fsize, pack, L, cn) ;	NOT (ok) ;
    ok = CHOLMOD(resymbol_noperm)(A, fset, fsize, pack, L, cn) ;NOT (ok) ;
    ok = CHOLMOD(analyze_ordering)(A, ordering, Perm, fset,
	fsize, Parent, Post, ColCount, First, Level, cn) ;	NOT (ok) ;

    /* ---------------------------------------------------------------------- */
    /* Modify */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(updown)(update, C, L, cn) ;			NOT (ok) ;
    ok = CHOLMOD(updown_solve)(update, C, L, X, DeltaB, cn) ;	NOT (ok) ;
    ok = CHOLMOD(updown_mark)(update, C, colmark, L, X, DeltaB,
	    cn) ;					NOT (ok) ;
    ok = CHOLMOD(rowadd)(k, R, L, cn) ;				NOT (ok) ;
    ok = CHOLMOD(rowadd_solve)(k, R, bk, L, X, DeltaB, cn) ;	NOT (ok) ;
    ok = CHOLMOD(rowadd_mark)(k, R, bk, colmark, L, X, DeltaB,
	    cn) ;					NOT (ok) ;
    ok = CHOLMOD(rowdel)(k, R, L, cn) ;				NOT (ok) ;
    ok = CHOLMOD(rowdel_solve)(k, R, yk, L, X, DeltaB, cn) ;	NOT (ok) ;
    ok = CHOLMOD(rowdel_mark)(k, R, yk, colmark, L, X, DeltaB,
	    cn) ;					NOT (ok) ;

    /* ---------------------------------------------------------------------- */
    /* MatrixOps */
    /* ---------------------------------------------------------------------- */

    C = CHOLMOD(add)(A, B, alpha, beta, values, sorted, cn) ;	NOP (C) ;

    C = CHOLMOD(copy)(A, stype, mode, cn) ;			NOP (C) ;
    ok = CHOLMOD(drop)(tol, A, cn) ;				NOT (ok) ;
    nm = CHOLMOD(norm_dense)(X, norm, cn) ;			NOT (nm>=0) ;
    nm = CHOLMOD(norm_sparse)(A, norm, cn) ;			NOT (nm>=0) ;
    C = CHOLMOD(horzcat)(A, B, values, cn) ;			NOP (C) ;
    ok = CHOLMOD(scale)(S, scale, A, cn) ;			NOT (ok) ;
    ok = CHOLMOD(sdmult)(A, transpose, alpha, beta, X, Y, cn) ;	NOT (ok) ;
    C = CHOLMOD(ssmult)(A, B, stype, values, sorted, cn) ;	NOP (C) ;
    C = CHOLMOD(submatrix)(A, r, nr, c, nc, values, sorted,
	    cn) ;						NOP (C) ;
    C = CHOLMOD(vertcat)(A, B, values, cn) ;			NOP (C) ;
    asym = CHOLMOD(symmetry)(A, option, NULL, NULL, NULL, NULL,
	    cn) ;						NOT(asym>=0) ;

    /* ---------------------------------------------------------------------- */
    /* Supernodal */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(super_symbolic)(A, F, Parent, L, cn) ;	NOT (ok) ;
    ok = CHOLMOD(super_numeric)(A, F, beta, L, cn) ;		NOT (ok) ;
    ok = CHOLMOD(super_lsolve)(L, X, E, cn) ;			NOT (ok) ;
    ok = CHOLMOD(super_ltsolve)(L, X, E, cn) ;			NOT (ok) ;

    /* ---------------------------------------------------------------------- */
    /* Check */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(check_common)(cn) ;				NOT (ok) ;
    ok = CHOLMOD(print_common)(name, cn) ;			NOT (ok) ;

    ok = CHOLMOD(check_sparse)(A, cn) ;				NOT (ok) ;
    ok = CHOLMOD(print_sparse)(A, name, cn) ;			NOT (ok) ;
    ok = CHOLMOD(check_dense)(X, cn) ;				NOT (ok) ;
    ok = CHOLMOD(print_dense)(X, name, cn) ;			NOT (ok) ;
    ok = CHOLMOD(check_factor)(L, cn) ;				NOT (ok) ;
    ok = CHOLMOD(print_factor)(L, name, cn) ;			NOT (ok) ;
    ok = CHOLMOD(check_triplet)(T, cn) ;			NOT (ok) ;
    ok = CHOLMOD(print_triplet)(T, name, cn) ;			NOT (ok) ;
    ok = CHOLMOD(check_subset)(Set, len, n, cn) ;		NOT (ok) ;
    ok = CHOLMOD(print_subset)(Set, len, n, name, cn) ;		NOT (ok) ;
    ok = CHOLMOD(check_perm)(Perm, n, n, cn) ;			NOT (ok) ;
    ok = CHOLMOD(print_perm)(Perm, n, n, name, cn) ;		NOT (ok) ;
    ok = CHOLMOD(check_parent)(Parent, n, cn) ;			NOT (ok) ;
    ok = CHOLMOD(print_parent)(Parent, n, name, cn) ;		NOT (ok) ;

    A = CHOLMOD(read_sparse)(NULL, cn) ;			NOP (A) ;
    p = CHOLMOD(read_matrix)(NULL, prefer, &mtype, cn) ;	NOP (p) ;
    X = CHOLMOD(read_dense)(NULL, cn) ;				NOP (X) ;
    T = CHOLMOD(read_triplet)(NULL, cn) ;			NOP (T) ;

    asym = CHOLMOD(write_dense) (NULL, NULL, NULL, cn) ;	NOT (asym>=0) ;
    asym = CHOLMOD(write_dense) ((FILE *) 1, NULL, NULL, cn) ;	NOT (asym>=0) ;

    asym = CHOLMOD(write_sparse)(NULL, NULL, NULL, NULL, cn) ;	NOT (asym>=0) ;
    asym = CHOLMOD(write_sparse)((FILE *) 1, NULL, NULL, NULL,
	    cn) ;						NOT (asym>=0) ;

    /* ---------------------------------------------------------------------- */
    /* Partition */
    /* ---------------------------------------------------------------------- */

#ifndef NPARTITION
    lr = CHOLMOD(nested_dissection)(A, fset, fsize, Perm,
	    CParent, Cmember, cn) ;				NOT (lr >= 0) ;
    lr = CHOLMOD(collapse_septree) (n, n, 1., 4,
	    CParent, Cmember, cn) ;				NOT (lr >= 0) ;
    ok = CHOLMOD(metis)(A, fset, fsize, postorder, Perm, cn) ;	NOT (ok) ;
    lr = CHOLMOD(bisect)(A, fset, fsize, compress,
	    Partition, cn) ;					NOT (lr >= 0) ;
    lr = CHOLMOD(metis_bisector)(A, Anw, Aew, Partition, cn) ;	NOT (lr >= 0) ;
#endif

#ifndef NCAMD
    ok = CHOLMOD(ccolamd)(A, fset, fsize, Cmem2, Perm, cn) ;	NOT (ok) ;
    ok = CHOLMOD(csymamd)(A, Cmem2, Perm, cn) ;		        NOT (ok) ;
#endif

}

/* ========================================================================== */
/* === null_test2 =========================================================== */
/* ========================================================================== */

void null_test2 (void)
{
    cholmod_dense *X, *Xbad = NULL ;
    cholmod_sparse *Sbad = NULL, *A ;
    int ok ;

    /* ---------------------------------------------------------------------- */
    /* Test Core Common */
    /* ---------------------------------------------------------------------- */

    ok = CHOLMOD(allocate_work)(Size_max, 1, 1, cm) ;		NOT (ok) ;
    ok = CHOLMOD(allocate_work)(1, Size_max, 1, cm) ;		NOT (ok) ;
    ok = CHOLMOD(allocate_work)(1, 1, Size_max, cm) ;		NOT (ok) ;

    /* free a NULL pointer */
    CHOLMOD(free)(42, sizeof (char), NULL, cm) ;
    cm->print = 5 ; CHOLMOD(print_common)("cm", cm) ; cm->print = 3 ;

    cm->maxrank = 3 ;
    cm->maxrank = CHOLMOD(maxrank)(5, cm) ; OK (cm->maxrank == 4) ;
    cm->maxrank = 1 ;
    cm->maxrank = CHOLMOD(maxrank)(5, cm) ; OK (cm->maxrank == 2) ;
    cm->maxrank = 8 ;

    /* test the error handler */
    cm->error_handler = my_handler2 ;
    CHOLMOD(drop)(0., NULL, cm) ;
    cm->error_handler = NULL ;

    /* ---------------------------------------------------------------------- */
    /* dense */
    /* ---------------------------------------------------------------------- */

    X = CHOLMOD(allocate_dense)(5, 4, 1, CHOLMOD_REAL, cm) ;	    NOP (X) ;
    X = CHOLMOD(allocate_dense)(1, Int_max, 1, CHOLMOD_REAL, cm) ;  NOP (X) ;
    X = CHOLMOD(allocate_dense)(1, 1, 1, -1, cm) ;		    NOP (X) ;
    CHOLMOD(free_dense)(&X, cm) ;

    /* free a NULL dense matrix */
    ok = CHOLMOD(free_dense)(&X, cm) ;				    OK (ok) ;
    ok = CHOLMOD(free_dense)(NULL, cm) ;			    OK (ok) ;

    /* make an invalid sparse matrix */
    Sbad = CHOLMOD(speye)(2, 3, CHOLMOD_REAL, cm) ;		    OKP (Sbad) ;
    Sbad->stype = 1 ;
    ok = CHOLMOD(check_sparse)(Sbad, cm) ;			    NOT (ok) ;
    X = CHOLMOD(sparse_to_dense)(Sbad, cm) ;			    NOP (X) ;
    ok = CHOLMOD(free_sparse)(&Sbad, cm) ;			    OK (ok) ;

    /* make an invalid dense matrix */
    Xbad = CHOLMOD(eye)(4, 4, CHOLMOD_REAL, cm) ;		    OKP (Xbad) ;
    Xbad->d = 1 ;
    ok = CHOLMOD(check_dense)(Xbad, cm) ;			    NOT (ok) ;
    A = CHOLMOD(dense_to_sparse)(Xbad, TRUE, cm) ;
    ok = CHOLMOD(free_dense)(&Xbad, cm) ;			    OK (ok) ;
    CHOLMOD(print_common)("cm", cm) ;
    cm->print = 5 ; CHOLMOD(print_sparse)(A, "Bad A", cm) ; cm->print = 3 ;
    NOP (A) ;

    /* ---------------------------------------------------------------------- */
    /* sparse */
    /* ---------------------------------------------------------------------- */

    /* free a NULL sparse matrix */
    ok = CHOLMOD(free_sparse)(&A, cm) ;				    OK (ok) ;
    ok = CHOLMOD(free_sparse)(NULL, cm) ;			    OK (ok) ;
    A = CHOLMOD(copy_sparse)(NULL, cm) ;			    NOP (A) ;

    /* ---------------------------------------------------------------------- */
    /* error tests done */
    /* ---------------------------------------------------------------------- */

    printf ("------------------ null tests done\n") ;
}
