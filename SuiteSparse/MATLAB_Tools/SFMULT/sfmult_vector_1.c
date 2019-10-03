//==============================================================================
//=== sfmult_vector_1 ==========================================================
//==============================================================================

// y = A*x or A'*x where x is a vector

// sfmult_AN_x_1    y = A*x	x is n-by-1, y is m-by-1
// sfmult_AT_x_1    y = A'*x	x is m-by-1, y is n-by-1

#include "sfmult.h"

//==============================================================================
//=== sfmult_AN_x_1 ============================================================
//==============================================================================

void sfmult_AN_x_1	// y = A*x	x is n-by-1 unit stride, y is m-by-1
(
    // --- outputs, not initialized on input
    double *Yx,		// m-by-1
    double *Yz,		// m-by-1 if Y is complex (TO DO)

    // --- inputs, not modified
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex (TO DO)
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// n-by-1
    const double *Xz,	// n-by-1 if X complex
    int ac,		// true: use conj(A), otherwise use A (TO DO)
    int xc,		// true: use conj(X), otherwise use X (TO DO)
    int yc		// true: compute conj(Y), otherwise compute Y (TO DO)
)
{
    double y [4], x ;
    Int p, pend, i, j, i0, i1, i2, i3 ;

    for (i = 0 ; i < m ; i++)
    {
	Yx [i] = 0 ;
    }
    p = 0 ;
    for (j = 0 ; j < n ; j++)
    {
	pend = Ap [j+1] ;
	x = Xx [j] ;
	switch ((pend - p) % 4)
	{
	    case 3: Yx [Ai [p]] += Ax [p] * x ; p++ ;
	    case 2: Yx [Ai [p]] += Ax [p] * x ; p++ ;
	    case 1: Yx [Ai [p]] += Ax [p] * x ; p++ ;
	    case 0: ;
	}
	for ( ; p < pend ; p += 4)
	{
	    i0 = Ai [p  ] ;
	    i1 = Ai [p+1] ;
	    i2 = Ai [p+2] ;
	    i3 = Ai [p+3] ;
	    y [0] = Yx [i0] + Ax [p  ] * x ;
	    y [1] = Yx [i1] + Ax [p+1] * x ;
	    y [2] = Yx [i2] + Ax [p+2] * x ;
	    y [3] = Yx [i3] + Ax [p+3] * x ;
	    Yx [i0] = y [0] ;
	    Yx [i1] = y [1] ;
	    Yx [i2] = y [2] ;
	    Yx [i3] = y [3] ;
	}
    }
}


//==============================================================================
//=== sfmult_AT_x_1 ============================================================
//==============================================================================

void sfmult_AT_x_1	// y = A'*x	x is m-by-1 unit stride, y is n-by-1
(
    // --- outputs, not initialized on input
    double *Yx,		// n-by-1
    double *Yz,		// n-by-1 if Y is complex (TO DO)

    // --- inputs, not modified
    const Int *Ap,	// size n+1 column pointers
    const Int *Ai,	// size nz = Ap[n] row indices
    const double *Ax,	// size nz values
    const double *Az,	// size nz imaginary values if A is complex (TO DO)
    Int m,		// A is m-by-n
    Int n,
    const double *Xx,	// m-by-1
    const double *Xz,	// m-by-1 if X complex
    int ac,		// true: use conj(A), otherwise use A (TO DO)
    int xc,		// true: use conj(X), otherwise use X (TO DO)
    int yc		// true: compute conj(Y), otherwise compute Y (TO DO)
)
{
    double y ;
    Int p, pend, j ;

    p = 0 ;
    for (j = 0 ; j < n ; j++)
    {
	pend = Ap [j+1] ;
	y = 0 ;
	switch ((pend - p) % 4)
	{
	    case 3: y += Ax [p] * Xx [Ai [p]] ; p++ ;
	    case 2: y += Ax [p] * Xx [Ai [p]] ; p++ ;
	    case 1: y += Ax [p] * Xx [Ai [p]] ; p++ ;
	    case 0: ;
	}
	for ( ; p < pend ; p += 4)
	{
	    y += Ax [p  ] * Xx [Ai [p  ]] ;
	    y += Ax [p+1] * Xx [Ai [p+1]] ;
	    y += Ax [p+2] * Xx [Ai [p+2]] ;
	    y += Ax [p+3] * Xx [Ai [p+3]] ;
	}
	Yx [j] = y ;
    }
}
