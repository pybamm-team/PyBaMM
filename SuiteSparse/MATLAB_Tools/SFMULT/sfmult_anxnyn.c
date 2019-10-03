//==============================================================================
//=== sfmult_AN_XN_YN ==========================================================
//==============================================================================

// y = A*x	    A is m-by-n, x is n-by-k, y is m-by-k

// compare with sfmult_AN_XN_YT for kernel usage
// compare with sfmult_AN_XT_YN for outer loop structure but different kernels

#include "sfmult.h"

mxArray *sfmult_AN_XN_YN	// y = A*x
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
    if (n != mxGetM (X)) sfmult_invalid ( ) ;
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
    Y = sfmult_yalloc (m, k, Ycomplex) ;
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
	// Y = A*X
	sfmult_AN_x_1 (Yx, Yz, Ap, Ai, Ax, Az, m, n, Xx, Xz, ac, xc, yc) ;
	return (Y) ;
    }

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    sfmult_walloc ((k == 2) ? 2 : 4, m, &Wx, &Wz) ;

    //--------------------------------------------------------------------------
    // Y = A*X, in blocks of up to 4 columns of X, using sfmult_anxnyt
    //--------------------------------------------------------------------------

    k1 = k % 4 ;
    if (k1 == 1)
    {
	// Y (:,1) = A * X(:,1)
	sfmult_AN_x_1 (Yx, Yz, Ap, Ai, Ax, Az, m, n, Xx, Xz, ac, xc, yc) ;
	Yx += m ;
	Yz += m ;
	Xx += n ;
	Xz += n ;
    }
    else if (k1 == 2)
    {
	// W = (A * X(:,1:2))'
	sfmult_AN_XN_YT_2 (Wx, Wz, Ap, Ai, Ax, Az, m, n, Xx, Xz, ac, xc, yc) ;
	// Y (:,1:2) = W'

#if 0
	for (i = 0 ; i < m ; i++) Yx [i] = Wx [2*i  ] ; Yx += m ; Yz += m ;
	for (i = 0 ; i < m ; i++) Yx [i] = Wx [2*i+1] ; Yx += m ; Yz += m ;
#else
	for (i = 0 ; i < m ; i++)
	{
	    Yx [i  ] = Wx [2*i  ] ;
	    Yx [i+m] = Wx [2*i+1] ;
	}
	Yx += 2*m ;
	Yz += 2*m ;
#endif

	Xx += 2*n ;
	Xz += 2*n ;
    }
    else if (k1 == 3)
    {
	// W = (A * X(:,1:3))'
	sfmult_AN_XN_YT_3 (Wx, Wz, Ap, Ai, Ax, Az, m, n, Xx, Xz, ac, xc, yc) ;
	// Y (:,1:3) = W'

#if 0
	for (i = 0 ; i < m ; i++) Yx [i] = Wx [4*i  ] ; Yx += m ; Yz += m ;
	for (i = 0 ; i < m ; i++) Yx [i] = Wx [4*i+1] ; Yx += m ; Yz += m ;
	for (i = 0 ; i < m ; i++) Yx [i] = Wx [4*i+2] ; Yx += m ; Yz += m ;
#else
	for (i = 0 ; i < m ; i++)
	{
	    Yx [i    ] = Wx [4*i  ] ;
	    Yx [i+  m] = Wx [4*i+1] ;
	    Yx [i+2*m] = Wx [4*i+2] ;
	}
	Yx += 3*m ;
	Yz += 3*m ;
#endif

	Xx += 3*n ;
	Xz += 3*n ;
    }
    for ( ; k1 < k ; k1 += 4)
    {
	// W = (A * X(:,1:4))'
	sfmult_AN_XN_YT_4 (Wx, Wz, Ap, Ai, Ax, Az, m, n, Xx, Xz, ac, xc, yc) ;
	// Y (:,k1+(1:4),:) = W'

#if 0
	for (i = 0 ; i < m ; i++) Yx [i] = Wx [4*i  ] ; Yx += m ; Yz += m ;
	for (i = 0 ; i < m ; i++) Yx [i] = Wx [4*i+1] ; Yx += m ; Yz += m ;
	for (i = 0 ; i < m ; i++) Yx [i] = Wx [4*i+2] ; Yx += m ; Yz += m ;
	for (i = 0 ; i < m ; i++) Yx [i] = Wx [4*i+3] ; Yx += m ; Yz += m ;

#else
	for (i = 0 ; i < m ; i++)
	{
	    Yx [i    ] = Wx [4*i  ] ;
	    Yx [i+  m] = Wx [4*i+1] ;
	    Yx [i+2*m] = Wx [4*i+2] ;
	    Yx [i+3*m] = Wx [4*i+3] ;
	}
	Yx += 4*m ;
	Yz += 4*m ;
#endif

	Xx += 4*n ;
	Xz += 4*n ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    mxFree (Wx) ;
    return (Y) ;
}
