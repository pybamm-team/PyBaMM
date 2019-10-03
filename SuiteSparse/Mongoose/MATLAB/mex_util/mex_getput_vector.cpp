#include "mongoose_mex.hpp"

namespace Mongoose
{

/* check MATLAB input argument */
void cs_mex_check (csi nel, csi m, csi n, csi square, csi sparse, csi values,
    const mxArray *A)
{
    csi nnel, mm = mxGetM (A), nn = mxGetN (A) ;
    if (values)
    {
        if (mxIsComplex (A))
        {
            mexErrMsgTxt ("matrix must be real; try CXSparse instead") ;
        }
    }
    if (sparse && !mxIsSparse (A)) mexErrMsgTxt ("matrix must be sparse") ;
    if (!sparse)
    {
        if (mxIsSparse (A)) mexErrMsgTxt ("matrix must be full") ;
        if (values && !mxIsDouble (A)) mexErrMsgTxt ("matrix must be double") ;
    }
    if (nel)
    {
        /* check number of elements */
        nnel = mxGetNumberOfElements (A) ;
        if (m >= 0 && n >= 0 && m*n != nnel) mexErrMsgTxt ("wrong length") ;
    }
    else
    {
        /* check row and/or column dimensions */
        if (m >= 0 && m != mm) mexErrMsgTxt ("wrong dimension") ;
        if (n >= 0 && n != nn) mexErrMsgTxt ("wrong dimension") ;
    }
    if (square && mm != nn) mexErrMsgTxt ("matrix must be square") ;
}

/* get a MATLAB sparse matrix and convert to cs */
cs *cs_mex_get_sparse (cs *A, csi square, csi values, const mxArray *Amatlab)
{
    cs_mex_check (0, -1, -1, square, 1, values, Amatlab) ;
    A->m = mxGetM (Amatlab) ;
    A->n = mxGetN (Amatlab) ;
    A->p = (csi *) mxGetJc (Amatlab) ;
    A->i = (csi *) mxGetIr (Amatlab) ;
    A->x = values ? mxGetPr (Amatlab) : NULL ;
    A->nzmax = mxGetNzmax (Amatlab) ;
    A->nz = -1 ;    /* denotes a compressed-col matrix, instead of triplet */
    return (A) ;
}

/* return a sparse matrix to MATLAB */
mxArray *cs_mex_put_sparse (cs **Ahandle)
{
    cs *A ;
    mxArray *Amatlab ;
    if (!Ahandle || !CS_CSC ((*Ahandle))) mexErrMsgTxt ("invalid sparse matrix") ;
    A = *Ahandle ;
    Amatlab = mxCreateSparse (0, 0, 0, mxREAL) ;
    mxSetM (Amatlab, A->m) ;
    mxSetN (Amatlab, A->n) ;
    mxSetNzmax (Amatlab, A->nzmax) ;
    SuiteSparse_free (mxGetJc (Amatlab)) ;
    SuiteSparse_free (mxGetIr (Amatlab)) ;
    mxSetJc (Amatlab, (mwIndex *) A->p) ;  /* assign A->p pointer to MATLAB A */
    mxSetIr (Amatlab, (mwIndex *) A->i) ;
    SuiteSparse_free (mxGetPr (Amatlab)) ;
    if (A->x == NULL)
    {
        /* A is a pattern only matrix; return all 1's to MATLAB */
        csi i, nz ;
        nz = A->p [A->n] ;
        A->x = (double*) SuiteSparse_malloc ( (nz > 1) ? nz : 1, sizeof (double)) ;
        for (i = 0 ; i < nz ; i++)
        {
            A->x [i] = 1 ;
        }
    }
    mxSetPr (Amatlab, A->x) ;
    mexMakeMemoryPersistent (A->p) ;    /* ensure MATLAB does not free A->p */
    mexMakeMemoryPersistent (A->i) ;
    mexMakeMemoryPersistent (A->x) ;
    SuiteSparse_free (A) ;              /* frees A struct only, not A->p, etc */
    *Ahandle = NULL ;
    return (Amatlab) ;
}

/* get a MATLAB dense column vector */
double *gp_mex_get_double (Int n, const mxArray *X)
{
    return (mxGetPr (X)) ;
}

/* return a double vector to MATLAB */
double *gp_mex_put_double (Int n, const double *b, mxArray **X)
{
    double *x ;
    Int k ;
    *X = mxCreateDoubleMatrix (n, 1, mxREAL) ;      /* create x */
    x = mxGetPr (*X) ;
    for (k = 0 ; k < n ; k++) x [k] = b [k] ;       /* copy x = b */
    return (x) ;
}

/* get a MATLAB flint array and convert to Int */
Int *gp_mex_get_int
(
    Int n,
    const mxArray *Imatlab,
    Int *imax,
    Int lo
)
{
    double *p ;
    Int i, k, *C = (Int*) SuiteSparse_malloc(n, sizeof (Int));

    p = mxGetPr (Imatlab) ;
    *imax = 0 ;
    for (k = 0 ; k < n ; k++)
    {
        i = (Int) p[k];
        C [k] = i - 1 ;
        if (i < lo) mexErrMsgTxt ("index out of bounds") ;
        *imax = std::max(*imax, i) ;
    }
    return (C) ;
}

/* return an Int array to MATLAB as a flint row vector */
mxArray *gp_mex_put_int(Int *p, Int n, Int offset, Int do_free)
{
    mxArray *X = mxCreateDoubleMatrix (1, n, mxREAL) ;
    double *x = mxGetPr (X) ;
    Int k ;
    for (k = 0 ; k < n ; k++) x [k] = (p ? p [k] : k) + offset ;
    if (do_free) SuiteSparse_free(p);
    return (X) ;
}

/* return an Int array to MATLAB as a flint row vector */
mxArray *gp_mex_put_logical(bool *p, Int n)
{
    mxArray *X = mxCreateDoubleMatrix(1, n, mxREAL);
    double *x = mxGetPr(X);
    for(Int k = 0; k < n; k++) x[k] = p[k] ? 1.0 : 0.0;
    return X;
}

}
