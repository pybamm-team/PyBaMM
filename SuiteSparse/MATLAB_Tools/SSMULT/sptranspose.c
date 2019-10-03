/* -------------------------------------------------------------------------- */
/* sptranspose mexFunction */
/* -------------------------------------------------------------------------- */

/* C = A' or A.' */

#include "ssmult.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    const mxArray *A ;
    int conj ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 1 || nargin > 2 || nargout > 1)
    {
        mexErrMsgTxt ("Usage: C = sptranspose (A,conj)") ;
    }

    A = pargin [0] ;
    if (!mxIsSparse (A))
    {
        mexErrMsgTxt ("A must be sparse") ; 
    }

    conj = (nargin > 1) ? mxGetScalar (pargin [1]) : 0 ;

    /* ---------------------------------------------------------------------- */
    /* C = A' or A.' */
    /* ---------------------------------------------------------------------- */

    pargout [0] = ssmult_transpose (A, conj) ;
}
