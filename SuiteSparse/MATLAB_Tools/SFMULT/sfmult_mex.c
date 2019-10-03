// -----------------------------------------------------------------------------
// sfmult mexFunction
// -----------------------------------------------------------------------------

// y = A*x and variants.  Either A or x can be sparse, the other must be full

#include "sfmult.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    const mxArray *A, *X ;
    int at, ac, xt, xc, yt, yc, do_sparse_times_full ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    if (nargin < 2 || nargin > 8 || nargout > 1)
    {
	mexErrMsgTxt ("Usage: y = sfmult (A,x, at,ac, xt,xc, yt,yc)") ;
    }

    if (mxIsSparse (pargin [0]))
    {
	// sfmult (A,x) will do A*x where A is sparse and x is full
	do_sparse_times_full = 1 ;
	A = pargin [0] ;
	X = pargin [1] ;
    }
    else
    {
	// sfmult (x,A) will do x*A where A is sparse and x is full
	do_sparse_times_full = 0 ;
	A = pargin [1] ;
	X = pargin [0] ;
    }

    if (!mxIsSparse (A) || mxIsSparse (X))
    {
	mexErrMsgTxt ("one matrix must be sparse and the other full") ; 
    }

    at = (nargin > 2) ? mxGetScalar (pargin [2]) : 0 ;
    ac = (nargin > 3) ? mxGetScalar (pargin [3]) : 0 ;
    xt = (nargin > 4) ? mxGetScalar (pargin [4]) : 0 ;
    xc = (nargin > 5) ? mxGetScalar (pargin [5]) : 0 ;
    yt = (nargin > 6) ? mxGetScalar (pargin [6]) : 0 ;
    yc = (nargin > 7) ? mxGetScalar (pargin [7]) : 0 ;

    // -------------------------------------------------------------------------
    // y = A*x or x*A or variants
    // -------------------------------------------------------------------------

    pargout [0] = do_sparse_times_full ?
	sfmult (A, X, at, ac, xt, xc, yt, yc) :
	fsmult (A, X, at, ac, xt, xc, yt, yc) ;

    // (TO DO) convert y to real if imag(y) is all zero
}
