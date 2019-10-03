/* -------------------------------------------------------------------------- */
/* ssmult mexFunction */
/* -------------------------------------------------------------------------- */

#include "ssmult.h"

/* C = A*B and variants.  Both A and B must be sparse */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    const mxArray *A, *B ;
    int at, ac, bt, bc, ct, cc ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 2 || nargin > 8 || nargout > 1)
    {
        mexErrMsgTxt ("Usage: C = ssmult (A,B, at,ac, bt,bc, ct,cc)") ;
    }

    A = pargin [0] ;
    B = pargin [1] ;

    at = (nargin > 2) ? mxGetScalar (pargin [2]) : 0 ;
    ac = (nargin > 3) ? mxGetScalar (pargin [3]) : 0 ;
    bt = (nargin > 4) ? mxGetScalar (pargin [4]) : 0 ;
    bc = (nargin > 5) ? mxGetScalar (pargin [5]) : 0 ;
    ct = (nargin > 6) ? mxGetScalar (pargin [6]) : 0 ;
    cc = (nargin > 7) ? mxGetScalar (pargin [7]) : 0 ;

    /* ---------------------------------------------------------------------- */
    /* C = A*B or variants */
    /* ---------------------------------------------------------------------- */

    pargout [0] = ssmult (A, B, at, ac, bt, bc, ct, cc) ;
}
