//==============================================================================
// sfmult_xA: Y = X*A, unblocked
//==============================================================================

// This kernel is unique; it operates on all of X and Y, not just a few rows or
// columns.  It is used only when A is very sparse and X is large since it can
// be very slow otherwise.

#include "sfmult.h"

void sfmult_xA		// y = (A'*x')' = x*A,	x is k-by-m, and y is k-by-n
(
    // --- outputs, not initialized on input
    double *Yx,		// k-by-n
    double *Yz,		// k-by-n if Y is complex (TO DO)

    // --- inputs, not modified
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex (TO DO)
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// k-by-m
    const double *Xz,	// k-by-m if X complex (TO DO)
    int ac,		// true: use conj(A), otherwise use A (TO DO)
    int xc,		// true: use conj(X), otherwise use X (TO DO)
    int yc		// true: compute conj(Y), otherwise compute Y (TO DO)
    , Int k
)
{
    double a ;
    const double *xx, *xz ;
    Int p, pend, j, i, k1 ;

    p = 0 ;
    for (j = 0 ; j < n ; j++)
    {
	pend = Ap [j+1] ;
	for (k1 = 0 ; k1 < k ; k1++)
	{
	    Yx [k1] = 0 ;
	}
	for ( ; p < pend ; p++)
	{
	    i = Ai [p] ;
	    a = Ax [p] ;
	    xx = Xx + i*k ;
	    xz = Xz + i*k ;
	    k1 = k % 4 ;
	    switch (k1)
	    {
		case 3: Yx [2] += a * xx [2] ;
		case 2: Yx [1] += a * xx [1] ;
		case 1: Yx [0] += a * xx [0] ;
		case 0: ;
	    }
	    for ( ; k1 < k ; k1 += 4)
	    {
		Yx [k1  ] += a * xx [k1  ] ;
		Yx [k1+1] += a * xx [k1+1] ;
		Yx [k1+2] += a * xx [k1+2] ;
		Yx [k1+3] += a * xx [k1+3] ;
	    }
	}
	Yx += k ;
	Yz += k ;
    }
}

