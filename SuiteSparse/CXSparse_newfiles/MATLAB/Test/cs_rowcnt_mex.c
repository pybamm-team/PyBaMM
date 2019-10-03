/* Compute the row counts of the Cholesky factor L of the matrix A.  Uses
 * the lower triangular part of A. */

#include "cs_mex.h"

static
void firstdesc (CS_INT n, CS_INT *parent, CS_INT *post, CS_INT *first,
    CS_INT *level)
{
    CS_INT len, i, k, r, s ;
    for (i = 0 ; i < n ; i++) first [i] = -1 ;
    for (k = 0 ; k < n ; k++)
    {
        i = post [k] ;      /* node i of etree is kth postordered node */
        len = 0 ;           /* traverse from i towards the root */
        for (r = i ; r != -1 && first [r] == -1 ; r = parent [r], len++)
            first [r] = k ;
        len += (r == -1) ? (-1) : level [r] ;   /* root node or end of path */
        for (s = i ; s != r ; s = parent [s]) level [s] = len-- ;
    }
}

/* return rowcount [0..n-1] */
static
CS_INT *rowcnt (cs_dl *A, CS_INT *parent, CS_INT *post)
{
    CS_INT i, j, k, len, s, p, jprev, q, n, sparent, jleaf, *Ap, *Ai, *maxfirst,
        *ancestor, *prevleaf, *w, *first, *level, *rowcount ;
    n = A->n ; Ap = A->p ; Ai = A->i ;                  /* get A */
    w = cs_dl_malloc (5*n, sizeof (CS_INT)) ;           /* get workspace */
    ancestor = w ; maxfirst = w+n ; prevleaf = w+2*n ; first = w+3*n ;
    level = w+4*n ;
    rowcount = cs_dl_malloc (n, sizeof (CS_INT)) ;      /* allocate result */
    firstdesc (n, parent, post, first, level) ; /* find first and level */
    for (i = 0 ; i < n ; i++)
    {
        rowcount [i] = 1 ;      /* count the diagonal of L */
        prevleaf [i] = -1 ;     /* no previous leaf of the ith row subtree */
        maxfirst [i] = -1 ;     /* max first[j] for node j in ith subtree */
        ancestor [i] = i ;      /* every node is in its own set, by itself */
    }
    for (k = 0 ; k < n ; k++)
    {
        j = post [k] ;          /* j is the kth node in the postordered etree */
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;
            q = cs_dl_leaf (i, j, first, maxfirst, prevleaf, ancestor, &jleaf) ;
            if (jleaf) rowcount [i] += (level [j] - level [q]) ;
        }
        if (parent [j] != -1) ancestor [j] = parent [j] ;
    }
    cs_dl_free (w) ;
    return (rowcount) ;
}

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs_dl *A, Amatrix ;
    double *x ;
    CS_INT i, m, n, *parent, *post, *rowcount ;

    if (nargout > 1 || nargin != 3)
    {
        mexErrMsgTxt ("Usage: r = cs_rowcnt(A,parent,post)") ;
    }

    /* get inputs */
    A = cs_dl_mex_get_sparse (&Amatrix, 1, 0, pargin [0]) ;
    n = A->n ;

    parent = cs_dl_mex_get_int (n, pargin [1], &i, 0) ;
    post = cs_dl_mex_get_int (n, pargin [2], &i, 1) ;

    rowcount = rowcnt (A, parent, post) ;

    pargout [0] = mxCreateDoubleMatrix (1, n, mxREAL) ;
    x = mxGetPr (pargout [0]) ;
    for (i = 0 ; i < n ; i++) x [i] = rowcount [i] ;

    cs_dl_free (rowcount) ;
}
