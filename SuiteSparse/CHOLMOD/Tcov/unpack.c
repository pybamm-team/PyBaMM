/* ========================================================================== */
/* === Tcov/unpack ========================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Create an unpacked, unsorted version of a matrix, with random-sized gaps in
 * each column. */

#include "cm.h"


/* ========================================================================== */
/* === unpack =============================================================== */
/* ========================================================================== */

cholmod_sparse *unpack (cholmod_sparse *A)
{
    double x ;
    double *Ax, *Cx, *Az, *Cz ;
    Int *Ap, *Ai, *Anz, *Cp, *Ci, *Cnz ;
    cholmod_sparse *C ;
    Int i, j, p, q, pdest, pend, nrow, ncol, nzmax, sorted, packed, stype,
	extra ;

    if (A == NULL)
    {
	return (NULL) ;
    }

    extra = 10 ;

    nrow = A->nrow ;
    ncol = A->ncol ;
    nzmax = A->nzmax ;
    sorted = A->sorted ;
    packed = A->packed ;
    stype = A->stype ;

    C = CHOLMOD(allocate_sparse) (nrow, ncol, nzmax + extra*ncol, FALSE,
	FALSE, stype, A->xtype, cm) ;

    if (C == NULL)
    {
	return (NULL) ;
    }

    Ap = A->p ;
    Ai = A->i ;
    Ax = A->x ;
    Az = A->z ;
    Anz = A->nz ;

    Cp = C->p ;
    Ci = C->i ;
    Cx = C->x ;
    Cz = C->z ;
    Cnz = C->nz ;
    nzmax = C->nzmax ;
    nzmax = MAX (1, nzmax) ;

    for (p = 0 ; p < nzmax ; p++)
    {
	Ci [p] = 0 ;
    }
    if (A->xtype == CHOLMOD_REAL)
    {
	for (p = 0 ; p < nzmax ; p++)
	{
	    Cx [p] = 0 ;
	}
    }
    else if (A->xtype == CHOLMOD_COMPLEX)
    {
	for (p = 0 ; p < 2*nzmax ; p++)
	{
	    Cx [p] = 0 ;
	}
    }
    else if (A->xtype == CHOLMOD_ZOMPLEX)
    {
	for (p = 0 ; p < nzmax ; p++)
	{
	    Cx [p] = 0 ;
	    Cz [p] = 0 ;
	}
    }

    pdest = 0 ;
    for (j = 0 ; j < ncol ; j++)
    {
	/* copy the column into C */
	p = Ap [j] ;
	Cp [j] = pdest ;
	pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
	Cnz [j] = pend - p ;
	for ( ; p < pend ; p++)
	{
	    Ci [pdest] = Ai [p] ;
	    if (A->xtype == CHOLMOD_REAL)
	    {
		Cx [pdest] = Ax [p] ;
	    }
	    else if (A->xtype == CHOLMOD_COMPLEX)
	    {
		Cx [2*pdest  ] = Ax [2*p] ;
		Cx [2*pdest+1] = Ax [2*p+1] ;
	    }
	    else if (A->xtype == CHOLMOD_ZOMPLEX)
	    {
		Cx [pdest] = Ax [p] ;
		Cz [pdest] = Az [p] ;
	    }
	    pdest++ ;
	}

	/* jumble the column */
	p = Cp [j] ;
	pend = p + Cnz [j] ;
	for ( ; p < pend-1 ; p++)
	{
	    q = p + nrand (pend-p) ;				/* RAND */
	    i = Ci [p] ;
	    Ci [p] = Ci [q] ;
	    Ci [q] = i ;

	    if (A->xtype == CHOLMOD_REAL)
	    {
		x = Cx [p] ;
		Cx [p] = Cx [q] ;
		Cx [q] = x ;
	    }
	    else if (A->xtype == CHOLMOD_COMPLEX)
	    {
		x = Cx [2*p] ;
		Cx [2*p] = Cx [2*q] ;
		Cx [2*q] = x ;

		x = Cx [2*p+1] ;
		Cx [2*p+1] = Cx [2*q+1] ;
		Cx [2*q+1] = x ;
	    }
	    else if (A->xtype == CHOLMOD_ZOMPLEX)
	    {
		x = Cx [p] ;
		Cx [p] = Cx [q] ;
		Cx [q] = x ;

		x = Cz [p] ;
		Cz [p] = Cz [q] ;
		Cz [q] = x ;
	    }
	}

	/* add some random blank space */
	pdest += nrand (extra) ;				/* RAND */
	for (p = pend ; p < pdest ; p++)
	{
	    Ci [p] = 0 ;
	    if (A->xtype == CHOLMOD_REAL)
	    {
		Cx [p] = 0 ;
	    }
	    else if (A->xtype == CHOLMOD_COMPLEX)
	    {
		Cx [2*p] = 0 ;
		Cx [2*p+1] = 0 ;
	    }
	    else if (A->xtype == CHOLMOD_ZOMPLEX)
	    {
		Cx [p] = 0 ;
		Cz [p] = 0 ;
	    }
	}
    }
    Cp [ncol] = pdest ;

    return (C) ;
}
