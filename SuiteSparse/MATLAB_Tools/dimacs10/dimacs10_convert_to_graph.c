#include "mex.h"
/* convert_to_graph mexFunction

    function S = convert_to_graph (A, binary)
    if (nargin < 2)
        binary = 0 ;
    end
    S = tril (A,-1) + triu (A,1) ;
    if (binary)
        S = spones (S) ;
    end

Ignores the imaginary part of a complex matrix A.
 */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    mwSignedIndex *Ap, *Ai, *Sp, *Si, m, n, nz, j, p, i ;
    double *Sx, *Ax ;
    int binary ;
    if (nargin < 1 || nargin > 2 || nargout > 1)
    {
        mexErrMsgTxt ("usage: S = convert_to_graph (A, binary)") ;
    }

    /* get the input A matrix */
    Ap = (mwSignedIndex *) mxGetJc (pargin [0]) ;
    Ai = (mwSignedIndex *) mxGetIr (pargin [0]) ;
    Ax = mxGetPr (pargin [0]) ;
    m = mxGetM (pargin [0]) ;
    n = mxGetN (pargin [0]) ;
    nz = Ap [n] ;
    if (nz == 0) nz = 1 ;

    /* get the 2nd input argument */
    binary = (nargin < 2) ? 0 : ((int) mxGetScalar (pargin [1])) ;

    /* allocate the output matrix */
    pargout [0] = mxCreateSparse (m, n, nz, mxREAL) ;
    Sp = (mwSignedIndex *) mxGetJc (pargout [0]) ;
    Si = (mwSignedIndex *) mxGetIr (pargout [0]) ;
    Sx = mxGetPr (pargout [0]) ;

    /* strip the diagonal from A */
    nz = 0 ;
    for (j = 0 ; j < n ; j++)
    {
        Sp [j] = nz ;
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;
            if (i != j)
            {
                Si [nz] = i ;
                if (!binary) Sx [nz] = Ax [p] ;
                nz++ ;
            }
        }
    }
    Sp [n] = nz ;

    /* convert the result to binary, if requested */
    if (binary)
    {
        for (p = 0 ; p < nz ; p++)
        {
            Sx [p] = 1 ;
        }
    }
}
