//==============================================================================
// ssmult_transpose
//==============================================================================

// C = A' or A.' where the input matrix A may have unsorted columns.  The output
// C is always returned with sorted columns.

#include "sfmult.h"

mxArray *ssmult_transpose	// returns C = A' or A.'
(
    const mxArray *A,
    int conj			// compute A' if true, compute A.' if false
)
{
    Int *Cp, *Ci, *Ap, *Ai, *W ;
    double *Cx, *Cz, *Ax, *Az ;	    // (TO DO): do single too
    mxArray *C ;
    Int p, pend, q, i, j, n, m, anz, cnz ;
    int C_is_complex ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    m = mxGetM (A) ;
    n = mxGetN (A) ;
    Ap = mxGetJc (A) ;
    Ai = mxGetIr (A) ;
    Ax = mxGetPr (A) ;
    Az = mxGetPi (A) ;
    anz = Ap [n] ;
    C_is_complex = mxIsComplex (A) ;

    //--------------------------------------------------------------------------
    // allocate C but do not initialize it
    //--------------------------------------------------------------------------

    cnz = MAX (anz, 1) ;
    C = mxCreateSparse (0, 0, 0, C_is_complex ? mxCOMPLEX : mxREAL) ;
    MXFREE (mxGetJc (C)) ;
    MXFREE (mxGetIr (C)) ;
    MXFREE (mxGetPr (C)) ;
    MXFREE (mxGetPi (C)) ;
    Cp = mxMalloc ((m+1) * sizeof (Int)) ;
    Ci = mxMalloc (MAX (cnz,1) * sizeof (Int)) ;
    Cx = mxMalloc (MAX (cnz,1) * sizeof (double)) ;
    Cz = C_is_complex ? mxMalloc (MAX (cnz,1) * sizeof (double)) : NULL ;
    mxSetJc (C, Cp) ;
    mxSetIr (C, Ci) ;
    mxSetPr (C, Cx) ;
    mxSetPi (C, Cz) ;
    mxSetNzmax (C, cnz) ;
    mxSetM (C, n) ;
    mxSetN (C, m) ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    W = mxCalloc (MAX (m,1), sizeof (Int)) ;

    //--------------------------------------------------------------------------
    // compute row counts
    //--------------------------------------------------------------------------

    for (p = 0 ; p < anz ; p++)
    {
	W [Ai [p]]++ ;
    }

    //--------------------------------------------------------------------------
    // compute column pointers of C and copy back into W
    //--------------------------------------------------------------------------

    for (p = 0, i = 0 ; i < m ; i++)
    {
	Cp [i] = p ;
	p += W [i] ;
	W [i] = Cp [i] ;
    }
    Cp [m] = p ;

    //--------------------------------------------------------------------------
    // C = A'
    //--------------------------------------------------------------------------

    p = 0 ;
    if (!C_is_complex)
    {
	// C = A' (real case)
	for (j = 0 ; j < n ; j++)
	{
	    pend = Ap [j+1] ;
	    for ( ; p < pend ; p++)
	    {
		q = W [Ai [p]]++ ;	// find position for C(j,i)
		Ci [q] = j ;		// place A(i,j) as entry C(j,i)
		Cx [q] = Ax [p] ;
	    }
	}
    }
    else if (conj)
    {
	// C = A' (complex conjugate)
	for (j = 0 ; j < n ; j++)
	{
	    pend = Ap [j+1] ;
	    for ( ; p < pend ; p++)
	    {
		q = W [Ai [p]]++ ;	// find position for C(j,i)
		Ci [q] = j ;		// place A(i,j) as entry C(j,i)
		Cx [q] = Ax [p] ;
		Cz [q] = -Az [p] ;
	    }
	}
    }
    else
    {
	// C = A.' (complex case)
	for (j = 0 ; j < n ; j++)
	{
	    pend = Ap [j+1] ;
	    for ( ; p < pend ; p++)
	    {
		q = W [Ai [p]]++ ;	// find position for C(j,i)
		Ci [q] = j ;		// place A(i,j) as entry C(j,i)
		Cx [q] = Ax [p] ;
		Cz [q] = Az [p] ;
	    }
	}
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    MXFREE (W) ;
    return (C) ;
}
