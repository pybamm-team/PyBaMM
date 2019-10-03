#include "cs_mex.h"
/* cs_thumb: convert a sparse matrix to a dense 2D thumbnail matrix of size
 * at most k-by-k.  k defaults to 256.  A helper mexFunction for cspy. */

#define INDEX(i,j,lda) ((i)+(j)*(lda))
#define ISNAN(x) ((x) != (x))
#ifdef DBL_MAX
#define BIG_VALUE DBL_MAX
#else
#define BIG_VALUE 1.7e308
#endif

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    CS_INT m, n, mn, m2, n2, k, s, j, ij, sj, si, p, *Ap, *Ai ;
    double aij, ax, az, *S, *Ax, *Az ;
    if (nargout > 1 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: S = cs_thumb(A,k)") ;
    }
    cs_mex_check (0, -1, -1, 0, 1, 1, pargin [0]) ;
    m = mxGetM (pargin [0]) ;
    n = mxGetN (pargin [0]) ;
    mn = CS_MAX (m,n) ;
    k = (nargin == 1) ? 256 : mxGetScalar (pargin [1]) ;    /* get k */
    /* s = size of each submatrix; A(1:s,1:s) maps to S(1,1) */
    s = (mn < k) ? 1 : (CS_INT) ceil ((double) mn / (double) k) ;
    m2 = (CS_INT) ceil ((double) m / (double) s) ;
    n2 = (CS_INT) ceil ((double) n / (double) s) ;
    /* create S */
    pargout [0] = mxCreateDoubleMatrix (m2, n2, mxREAL) ;
    S = mxGetPr (pargout [0]) ;
    Ap = (CS_INT *) mxGetJc (pargin [0]) ;
    Ai = (CS_INT *) mxGetIr (pargin [0]) ;
    Ax = mxGetPr (pargin [0]) ;
    Az = (mxIsComplex (pargin [0])) ? mxGetPi (pargin [0]) : NULL ;
    for (j = 0 ; j < n ; j++)
    {
        sj = j/s ;
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            si = Ai [p] / s ;
            ij = INDEX (si,sj,m2) ;
            ax = Ax [p] ;
            az = Az ? Az [p] : 0 ;
            if (az == 0)
            {
                aij = fabs (ax) ;
            }
            else
            {
                aij = sqrt (ax*ax + az*az) ;
            }
            if (ISNAN (aij)) aij = BIG_VALUE ;
            aij = CS_MIN (BIG_VALUE, aij) ;
            S [ij] = CS_MAX (S [ij], aij) ;
        }
    }
}
