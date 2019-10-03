//==============================================================================
//=== sfmult_atxtyn_k ==========================================================
//==============================================================================

// y = A'*x'	    where x has 2, 3, or 4 rows

// compare with sfmult_atxtyt

// sfmult_AT_XT_YN_2  y = A'*x'  where x is 2-by-m, and y is n-by-2
// sfmult_AT_XT_YN_3  y = A'*x'  where x is 3-by-m, and y is n-by-3 (ldx = 4)
// sfmult_AT_XT_YN_4  y = A'*x'  where x is 4-by-m, and y is n-by-4

#include "sfmult.h"

//==============================================================================
//=== sfmult_AT_XT_YN_2 ========================================================
//==============================================================================

void sfmult_AT_XT_YN_2	// y = A'*x'	x is 2-by-m, and y is n-by-2
(
    // --- outputs, not initialized on input
    double *Yx,		// n-by-2
    double *Yz,		// n-by-2 if Y is complex (TO DO)

    // --- inputs, not modified
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex (TO DO)
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// 2-by-m
    const double *Xz,	// 2-by-m if X complex (TO DO)
    int ac,		// true: use conj(A), otherwise use A (TO DO)
    int xc,		// true: use conj(X), otherwise use X (TO DO)
    int yc		// true: compute conj(Y), otherwise compute Y (TO DO)
)
{
    double y [2], a [4] ;
    Int p, pend, j, i0, i1, i2 ;

    p = 0 ;
    for (j = 0 ; j < n ; j++)
    {
	pend = Ap [j+1] ;
	y [0] = 0 ;
	y [1] = 0 ;
	switch ((pend - p) % 3)
	{
	    case 2:
		i0 = Ai [p] ;
		a [0] = Ax [p] ;
		y [0] += a [0] * Xx [2*i0  ] ;
		y [1] += a [0] * Xx [2*i0+1] ;
		p++ ;
	    case 1:
		i0 = Ai [p] ;
		a [0] = Ax [p] ;
		y [0] += a [0] * Xx [2*i0  ] ;
		y [1] += a [0] * Xx [2*i0+1] ;
		p++ ;
	    case 0: ;
	}
	for ( ; p < pend ; p += 3)
	{
	    i0 = Ai [p  ] ;
	    i1 = Ai [p+1] ;
	    i2 = Ai [p+2] ;
	    a [0] = Ax [p  ] ;
	    a [1] = Ax [p+1] ;
	    a [2] = Ax [p+2] ;
	    y [0] += a [0] * Xx [2*i0  ] ;
	    y [1] += a [0] * Xx [2*i0+1] ;
	    y [0] += a [1] * Xx [2*i1  ] ;
	    y [1] += a [1] * Xx [2*i1+1] ;
	    y [0] += a [2] * Xx [2*i2  ] ;
	    y [1] += a [2] * Xx [2*i2+1] ;
	}
	Yx [j  ] = y [0] ;
	Yx [j+n] = y [1] ;
    }
}


//==============================================================================
//=== sfmult_AT_XT_YN_3 ========================================================
//==============================================================================

void sfmult_AT_XT_YN_3	// y = A'*x'	x is 3-by-m, and y is n-by-3 (ldx = 4)
(
    // --- outputs, not initialized on input
    double *Yx,		// n-by-3
    double *Yz,		// n-by-3 if Y is complex (TO DO)

    // --- inputs, not modified
    const int *Ap,	// size n+1 column pointers
    const int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex (TO DO)
    int m,		// A is m-by-n
    int n,
    const double *Xx,	// 3-by-m
    const double *Xz,	// 3-by-m if X complex (TO DO)
    int ac,		// true: use conj(A), otherwise use A (TO DO)
    int xc,		// true: use conj(X), otherwise use X (TO DO)
    int yc		// true: compute conj(Y), otherwise compute Y (TO DO)
)
{
    double y [4], a [2] ;
    int p, pend, j, i0, i1 ;

    p = 0 ;
    for (j = 0 ; j < n ; j++)
    {
	pend = Ap [j+1] ;
	y [0] = 0 ;
	y [1] = 0 ;
	y [2] = 0 ;
	if ((pend - p) % 2)
	{
	    i0 = Ai [p] ;
	    a [0] = Ax [p] ;
	    y [0] += a [0] * Xx [4*i0  ] ;
	    y [1] += a [0] * Xx [4*i0+1] ;
	    y [2] += a [0] * Xx [4*i0+2] ;
	    p++ ;
	}
	for ( ; p < pend ; p += 2)
	{
	    i0 = Ai [p  ] ;
	    i1 = Ai [p+1] ;
	    a [0] = Ax [p  ] ;
	    a [1] = Ax [p+1] ;
	    y [0] += a [0] * Xx [4*i0  ] ;
	    y [1] += a [0] * Xx [4*i0+1] ;
	    y [2] += a [0] * Xx [4*i0+2] ;
	    y [0] += a [1] * Xx [4*i1  ] ;
	    y [1] += a [1] * Xx [4*i1+1] ;
	    y [2] += a [1] * Xx [4*i1+2] ;
	}
	Yx [j    ] = y [0] ;
	Yx [j+  n] = y [1] ;
	Yx [j+2*n] = y [2] ;
    }
}


//==============================================================================
//=== sfmult_AT_XT_YN_4 ========================================================
//==============================================================================

void sfmult_AT_XT_YN_4	// y = A'*x'	x is 4-by-m, and y is n-by-4
(
    // --- outputs, not initialized on input
    double *Yx,		// n-by-4
    double *Yz,		// n-by-4 if Y is complex (TO DO)

    // --- inputs, not modified
    const int *Ap,	// size n+1 column pointers
    const int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex (TO DO)
    int m,		// A is m-by-n
    int n,
    const double *Xx,	// 4-by-m
    const double *Xz,	// 4-by-m if X complex (TO DO)
    int ac,		// true: use conj(A), otherwise use A (TO DO)
    int xc,		// true: use conj(X), otherwise use X (TO DO)
    int yc		// true: compute conj(Y), otherwise compute Y (TO DO)
)
{
    double y [4], a ;
    int p, pend, j, i ;

    p = 0 ;
    for (j = 0 ; j < n ; j++)
    {
	pend = Ap [j+1] ;
	y [0] = 0 ;
	y [1] = 0 ;
	y [2] = 0 ;
	y [3] = 0 ;
	for ( ; p < pend ; p++)
	{
	    i = Ai [p] ;
	    a = Ax [p] ;
	    y [0] += a * Xx [4*i  ] ;
	    y [1] += a * Xx [4*i+1] ;
	    y [2] += a * Xx [4*i+2] ;
	    y [3] += a * Xx [4*i+3] ;
	}
	Yx [j    ] = y [0] ;
	Yx [j+  n] = y [1] ;
	Yx [j+2*n] = y [2] ;
	Yx [j+3*n] = y [3] ;
    }
}
