#include "cs_mex.h"
/* cs_etree: elimination tree of A or A'*A */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs_dl Amatrix, *A ;
    CS_INT n, *parent, *post ;
    int ata ;
    char mode [20] ;
    if (nargout > 2 || nargin < 1 || nargin > 2)
    {
        mexErrMsgTxt ("Usage: [parent,post] = cs_etree(A,mode)") ;
    }
    ata = 0 ;                                           /* get mode */
    if (nargin > 1 && mxIsChar (pargin [1]))
    {
        mxGetString (pargin [1], mode, 8) ;
        ata = (mode [0] == 'c') ;
    }
    A = cs_dl_mex_get_sparse (&Amatrix, !ata, 0, pargin [0]) ;  /* get A */
    n = A->n ;
    parent = cs_dl_etree (A, ata) ;                     /* compute etree */
    if (nargout > 1)
    {
        post = cs_dl_post (parent, n) ;                 /* postorder the etree*/
        pargout [1] = cs_dl_mex_put_int (post, n, 1, 1) ;       /* return post */
    }
    pargout [0] = cs_dl_mex_put_int (parent, n, 1, 1) ; /* return parent */
}
