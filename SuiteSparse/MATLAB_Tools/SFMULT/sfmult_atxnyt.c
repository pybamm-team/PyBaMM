//==============================================================================
//=== sfmult_AT_XN_YT ==========================================================
//==============================================================================

// y = (A'*x)'	    A is m-by-n, x is m-by-k, y is k-by-n

// compare with sfmult_AT_XT_YT for kernel usage
// compare with sfmult_AT_XN_YN for outer loop structure but different kernels

#include "sfmult.h"

mxArray *sfmult_AT_XN_YT    // y = (A'*x)'
(
    const mxArray *A,
    const mxArray *X,
    int ac,		// if true: conj(A)   if false: A. ignored if A real
    int xc,		// if true: conj(x)   if false: x. ignored if x real
    int yc		// if true: conj(y)   if false: y. ignored if y real
)
{
    mxArray *Y ;
    double *Ax, *Az, *Xx, *Xz, *Yx, *Yz, *Wx, *Wz ;
    Int *Ap, *Ai ;
    Int m, n, k, k1, i ;
    int Acomplex, Xcomplex, Ycomplex ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    m = mxGetM (A) ;
    n = mxGetN (A) ;
    k = mxGetN (X) ;
    if (m != mxGetM (X)) sfmult_invalid ( ) ;
    Acomplex = mxIsComplex (A) ;
    Xcomplex = mxIsComplex (X) ;
    Ap = mxGetJc (A) ;
    Ai = mxGetIr (A) ;
    Ax = mxGetPr (A) ;
    Az = mxGetPi (A) ;
    Xx = mxGetPr (X) ;
    Xz = mxGetPi (X) ;

    //--------------------------------------------------------------------------
    // allocate result
    //--------------------------------------------------------------------------

    Ycomplex = Acomplex || Xcomplex ;
    Y = sfmult_yalloc (k, n, Ycomplex) ;
    Yx = mxGetPr (Y) ;
    Yz = mxGetPi (Y) ;

    //--------------------------------------------------------------------------
    // special cases
    //--------------------------------------------------------------------------

    if (k == 0 || m == 0 || n == 0 || Ap [n] == 0)
    {
	// Y = 0
	return (sfmult_yzero (Y)) ;
    }
    if (k == 1)
    {
	// Y = A' * x
	sfmult_AT_x_1 (Yx, Yz, Ap, Ai, Ax, Az, m, n, Xx, Xz, ac, xc, yc) ;
	return (Y) ;
    }

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    sfmult_walloc ((k == 2) ? 2 : 4, m, &Wx, &Wz) ;

    //--------------------------------------------------------------------------
    // Y = (A'*X)', in blocks of up to 4 columns of X, using sfmult_atxtyt
    //--------------------------------------------------------------------------

    k1 = k % 4 ;
    if (k1 == 1)
    {
	// Y (1,:) = (A' * X(:,1))'
	sfmult_AT_xk_1 (Yx, Yz, Ap, Ai, Ax, Az, m, n, Xx, Xz, ac, xc, yc, k) ;
	Yx += 1 ;
	Yz += 1 ;
	Xx += m ;
	Xz += m ;
    }
    else if (k1 == 2)
    {
	// W = X (:,1:2)'
	for (i = 0 ; i < m ; i++)
	{
	    Wx [2*i  ] = Xx [i  ] ;
	    Wx [2*i+1] = Xx [i+m] ;
	}
	// Y (1:2,:) = (A' * W')'
	sfmult_AT_XT_YT_2 (Yx, Yz, Ap, Ai, Ax, Az, m, n, Wx, Wz, ac, xc, yc, k);
	Yx += 2 ;
	Yz += 2 ;
	Xx += 2*m ;
	Xz += 2*m ;
    }
    else if (k1 == 3)
    {
	// W = X (:,1:3)'
	for (i = 0 ; i < m ; i++)
	{
	    Wx [4*i  ] = Xx [i    ] ;
	    Wx [4*i+1] = Xx [i+m  ] ;
	    Wx [4*i+2] = Xx [i+2*m] ;
	}
	// Y (1:3,:) = (A' * W')'
	sfmult_AT_XT_YT_3 (Yx, Yz, Ap, Ai, Ax, Az, m, n, Wx, Wz, ac, xc, yc, k);
	Yx += 3 ;
	Yz += 3 ;
	Xx += 3*m ;
	Xz += 3*m ;
    }
    for ( ; k1 < k ; k1 += 4)
    {
	// W = X (:,1:4)'
	for (i = 0 ; i < m ; i++)
	{
	    Wx [4*i  ] = Xx [i    ] ;
	    Wx [4*i+1] = Xx [i+m  ] ;
	    Wx [4*i+2] = Xx [i+2*m] ;
	    Wx [4*i+3] = Xx [i+3*m] ;
	}
	// Y (k1+(1:4),:) = (A' * W')'
	sfmult_AT_XT_YT_4 (Yx, Yz, Ap, Ai, Ax, Az, m, n, Wx, Wz, ac, xc, yc, k);
	Yx += 4 ;
	Yz += 4 ;
	Xx += 4*m ;
	Xz += 4*m ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    mxFree (Wx) ;
    return (Y) ;
}
