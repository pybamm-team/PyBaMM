#include "cs_mex.h"
/* check MATLAB input argument */
void cs_mex_check (CS_INT nel, CS_INT m, CS_INT n, int square, int sparse,
    int values, const mxArray *A)
{
    CS_INT nnel, mm = mxGetM (A), nn = mxGetN (A) ;
#ifdef NCOMPLEX
    if (values)
    {
        if (mxIsComplex (A)) mexErrMsgTxt ("complex matrices not supported") ;
    }
#endif
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

/* get a real (or pattern) MATLAB sparse matrix and convert to cs_dl */
cs_dl *cs_dl_mex_get_sparse (cs_dl *A, int square, int values,
    const mxArray *Amatlab)
{
    cs_mex_check (0, -1, -1, square, 1, values, Amatlab) ;
    A->m = mxGetM (Amatlab) ;
    A->n = mxGetN (Amatlab) ;
    A->p = (CS_INT *) mxGetJc (Amatlab) ;
    A->i = (CS_INT *) mxGetIr (Amatlab) ;
    A->x = values ? mxGetPr (Amatlab) : NULL ;
    A->nzmax = mxGetNzmax (Amatlab) ;
    A->nz = -1 ;    /* denotes a compressed-col matrix, instead of triplet */
    return (A) ;
}


/* return a real sparse matrix to MATLAB */
mxArray *cs_dl_mex_put_sparse (cs_dl **Ahandle)
{
    cs_dl *A ;
    mxArray *Amatlab ;
    if (!Ahandle || !CS_CSC ((*Ahandle))) mexErrMsgTxt ("invalid sparse matrix") ;
    A = *Ahandle ;
    Amatlab = mxCreateSparse (0, 0, 0, mxREAL) ;
    mxSetM (Amatlab, A->m) ;
    mxSetN (Amatlab, A->n) ;
    mxSetNzmax (Amatlab, A->nzmax) ;
    cs_free (mxGetJc (Amatlab)) ;
    cs_free (mxGetIr (Amatlab)) ;
    cs_free (mxGetPr (Amatlab)) ;
    mxSetJc (Amatlab, (void *) (A->p)) ; /* assign A->p pointer to MATLAB A */
    mxSetIr (Amatlab, (void *) (A->i)) ;
    if (A->x == NULL)
    {
        /* A is a pattern only matrix; return all 1's to MATLAB */
        CS_INT i, nz ;
        nz = A->p [A->n] ;
        A->x = cs_dl_malloc (CS_MAX (nz,1), sizeof (double)) ;
        for (i = 0 ; i < nz ; i++)
        {
            A->x [i] = 1 ;
        }
    }
    mxSetPr (Amatlab, A->x) ;
    cs_free (A) ;                       /* frees A struct only, not A->p, etc */
    *Ahandle = NULL ;
    return (Amatlab) ;
}

/* get a real MATLAB dense column vector */
double *cs_dl_mex_get_double (CS_INT n, const mxArray *X)
{
    cs_mex_check (0, n, 1, 0, 0, 1, X) ;
    return (mxGetPr (X)) ;
}

/* return a double vector to MATLAB */
double *cs_dl_mex_put_double (CS_INT n, const double *b, mxArray **X)
{
    double *x ;
    CS_INT k ;
    *X = mxCreateDoubleMatrix (n, 1, mxREAL) ;      /* create x */
    x = mxGetPr (*X) ;
    for (k = 0 ; k < n ; k++) x [k] = b [k] ;       /* copy x = b */
    return (x) ;
}

/* get a MATLAB flint array and convert to CS_INT */
CS_INT *cs_dl_mex_get_int (CS_INT n, const mxArray *Imatlab, CS_INT *imax,
    int lo)
{
    double *p ;
    CS_INT i, k, *C = cs_dl_malloc (n, sizeof (CS_INT)) ;
    cs_mex_check (1, n, 1, 0, 0, 1, Imatlab) ;
    if (mxIsComplex (Imatlab))
    {
        mexErrMsgTxt ("integer input cannot be complex") ;
    }
    p = mxGetPr (Imatlab) ;
    *imax = 0 ;
    for (k = 0 ; k < n ; k++)
    {
        i = p [k] ;
        C [k] = i - 1 ;
        if (i < lo) mexErrMsgTxt ("index out of bounds") ;
        *imax = CS_MAX (*imax, i) ;
    }
    return (C) ;
}

/* return an CS_INT array to MATLAB as a flint row vector */
mxArray *cs_dl_mex_put_int (CS_INT *p, CS_INT n, CS_INT offset, int do_free)
{
    mxArray *X = mxCreateDoubleMatrix (1, n, mxREAL) ;
    double *x = mxGetPr (X) ;
    CS_INT k ;
    for (k = 0 ; k < n ; k++) x [k] = (p ? p [k] : k) + offset ;
    if (do_free) cs_free (p) ;
    return (X) ;
}

#ifndef NCOMPLEX

/* copy a MATLAB real or complex vector into a cs_cl complex vector */
static cs_complex_t *cs_cl_get_vector (CS_INT n, CS_INT size,
    const mxArray *Xmatlab)
{
    CS_INT p ;
    double *X, *Z ;
    cs_complex_t *Y ;
    X = mxGetPr (Xmatlab) ;
    Z = (mxIsComplex (Xmatlab)) ? mxGetPi (Xmatlab) : NULL ;
    Y = cs_dl_malloc (size, sizeof (cs_complex_t)) ;
    for (p = 0 ; p < n ; p++)
    {
        Y [p] = X [p] + I * (Z ? Z [p] : 0) ;
    }
    return (Y) ;
}

/* get a real or complex MATLAB sparse matrix and convert to cs_cl */
cs_cl *cs_cl_mex_get_sparse (cs_cl *A, int square, const mxArray *Amatlab)
{
    cs_mex_check (0, -1, -1, square, 1, 1, Amatlab) ;
    A->m = mxGetM (Amatlab) ;
    A->n = mxGetN (Amatlab) ;
    A->p = (CS_INT *) mxGetJc (Amatlab) ;
    A->i = (CS_INT *) mxGetIr (Amatlab) ;
    A->nzmax = mxGetNzmax (Amatlab) ;
    A->x = cs_cl_get_vector (A->p [A->n], A->nzmax, Amatlab) ;
    A->nz = -1 ;    /* denotes a compressed-col matrix, instead of triplet */
    return (A) ;
}

/* return a complex sparse matrix to MATLAB */
mxArray *cs_cl_mex_put_sparse (cs_cl **Ahandle)
{
    cs_cl *A ;
    double *x, *z ;
    mxArray *Amatlab ;
    CS_INT k ;
    if (!Ahandle || !CS_CSC ((*Ahandle))) mexErrMsgTxt ("invalid sparse matrix") ;
    A = *Ahandle ;
    if (A->x == NULL) mexErrMsgTxt ("invalid complex sparse matrix") ;
    Amatlab = mxCreateSparse (0, 0, 0, mxCOMPLEX) ;
    mxSetM (Amatlab, A->m) ;
    mxSetN (Amatlab, A->n) ;
    mxSetNzmax (Amatlab, A->nzmax) ;
    cs_cl_free (mxGetJc (Amatlab)) ;
    cs_cl_free (mxGetIr (Amatlab)) ;
    cs_cl_free (mxGetPr (Amatlab)) ;
    cs_cl_free (mxGetPi (Amatlab)) ;
    mxSetJc (Amatlab, (void *) (A->p)) ; /* assign A->p pointer to MATLAB A */
    mxSetIr (Amatlab, (void *) (A->i)) ;
    x = cs_dl_malloc (A->nzmax, sizeof (double)) ;
    z = cs_dl_malloc (A->nzmax, sizeof (double)) ;
    for (k = 0 ; k < A->nzmax ; k++)
    {
        x [k] = creal (A->x [k]) ;      /* copy and split numerical values */
        z [k] = cimag (A->x [k]) ;
    }
    cs_cl_free (A->x) ;                 /* free copy of complex values */
    mxSetPr (Amatlab, x) ;              /* x and z will not be NULL, even if nz==0 */
    mxSetPi (Amatlab, z) ;
    cs_cl_free (A) ;                    /* frees A struct only, not A->p, etc */
    *Ahandle = NULL ;
    return (Amatlab) ;
}

/* get a real or complex MATLAB dense column vector, and copy to cs_complex_t */
cs_complex_t *cs_cl_mex_get_double (CS_INT n, const mxArray *X)
{
    cs_mex_check (0, n, 1, 0, 0, 1, X) ;
    return (cs_cl_get_vector (n, n, X)) ;
}

/* copy a complex vector back to MATLAB and free it */
mxArray *cs_cl_mex_put_double (CS_INT n, cs_complex_t *b)
{
    double *x, *z ;
    mxArray *X ;
    CS_INT k ;
    X = mxCreateDoubleMatrix (n, 1, mxCOMPLEX) ;    /* create x */
    x = mxGetPr (X) ;
    z = mxGetPi (X) ;
    for (k = 0 ; k < n ; k++)
    {
        x [k] = creal (b [k]) ;     /* copy x = b */
        z [k] = cimag (b [k]) ;
    }
    cs_cl_free (b) ;
    return (X) ;
}
#endif
