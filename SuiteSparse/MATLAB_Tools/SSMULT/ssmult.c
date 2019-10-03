/* ========================================================================== */
/* == ssmult ================================================================ */
/* ========================================================================== */

/* C = A*B where A and B are sparse, or variants (A'*B, A*B', etc) using either
   the saxpy method (for most matrices) or the dot product method (when C is
   small compared with A and/or B).
 */

#include "ssmult.h"

/* -------------------------------------------------------------------------- */
/* ssmult_invalid */
/* -------------------------------------------------------------------------- */

void ssmult_invalid (int error)
{
    if (error == ERROR_DIMENSIONS)
    {
        mexErrMsgTxt ("Error using ==> ssmult\n"
            "Inner matrix dimensions must agree.") ;
    }
    else if (error == ERROR_TOO_LARGE)
    {
        mexErrMsgTxt ("Error using ==> ssmult\n"
            "Problem too large.") ;
    }
}


/* -------------------------------------------------------------------------- */
/* ssmult_use_dot */
/* -------------------------------------------------------------------------- */

/* Determine workspace required for C = A'*B using the dot product method and
   the saxpy method where A is m-by-n, B is m-by-k, C is n-by-k, and return
   true if the dot product method would use less workspace than the saxpy
   method.
 */

static int ssmult_use_dot (const mxArray *A, const mxArray *B)
{
    Int *Ap ;
    Int m, n, k, anz ;
    size_t dot_workspace, saxpy_workspace ;

    m = mxGetM (A) ;
    n = mxGetN (A) ;
    k = mxGetN (B) ;
    Ap = (Int *) mxGetJc (A) ;
    anz = Ap [n] ;

    /* ssmult_dot requires a full array C of n*k Reals */

    dot_workspace = n * k * sizeof (double) ;   /* (twice if A or B complex) */

    if (((double) n) * ((double) k) * ((double) (sizeof (double))) !=
        (double) dot_workspace)
    {
        /* integer overflow computing dot_workspace; use saxpy method */
        return (0) ;
    }

    /* computing T = A' requires a workspace of size m Int's, and then the
     * space for T which is (m+1) Int's for the column pointers, nnz(A) Int's
     * for the row indices, and nnz(A) Reals for the values.   The m Int
     * workspace is freed, but reallocated by ssmult_saxpy as the Flag array.
     * ssmult_saxpy requires an additional m Reals workspace.
     */

    saxpy_workspace =
        (2*m+1 + anz) * sizeof (Int) +
        + (anz + m) * sizeof (double) ;         /* (twice if A complex) */

    /* use ssmult_dot if it requires less workspace */
    return (dot_workspace < saxpy_workspace) ;
}


/* -------------------------------------------------------------------------- */
/* ssmult */
/* -------------------------------------------------------------------------- */

mxArray *ssmult         /* returns C = A*B or variants */
(
    const mxArray *A,
    const mxArray *B,
    int at,             /* if true: trans(A)  if false: A */
    int ac,             /* if true: conj(A)   if false: A. ignored if A real */
    int bt,             /* if true: trans(B)  if false: B */
    int bc,             /* if true: conj(B)   if false: B. ignored if B real */
    int ct,             /* if true: trans(C)  if false: C */
    int cc              /* if true: conj(C)   if false: C. ignored if C real */
)
{
    mxArray *C, *T = NULL ;

    if (!mxIsSparse (A) || !mxIsSparse (B))
    {
        mexErrMsgTxt ("A and B must be sparse") ; 
    }

    if (at)
    {
        if (bt)
        {
            if (ct)
            {
                /* C = (A'*B')'     A is m-by-n, B is k-by-m, C is k-by-n */
                C = ssmult_saxpy (B, A, bc, ac, cc, 1) ;        /* C = B*A */
            }
            else
            {
                /* C = A'*B'        A is m-by-n, B is k-by-m, C is n-by-k */
                T = ssmult_saxpy (B, A, bc, ac, 0, 0) ;         /* T = B*A */
                C = ssmult_transpose (T, cc) ;                  /* C = T'  */
            }
        }
        else
        {
            if (ct)
            {
                /* C = (A'*B)'      A is m-by-n, B is m-by-k, C is k-by-n */
                if (ssmult_use_dot (B, A))
                {
                    C = ssmult_dot (B, A, bc, ac, cc) ;         /* C = B'*A */
                }
                else
                {
                    T = ssmult_transpose (B, bc) ;              /* T = B' */
                    C = ssmult_saxpy (T, A, 0, ac, cc, 1) ;     /* C = T*A */
                }
            }
            else
            {
                /* C = A'*B         A is m-by-n, B is m-by-k, C is n-by-k */
                if (ssmult_use_dot (A, B))
                {
                    C = ssmult_dot (A, B, ac, bc, cc) ;         /* C = A'*B */
                }
                else
                {
                    T = ssmult_transpose (A, ac) ;              /* T = A' */
                    C = ssmult_saxpy (T, B, 0, bc, cc, 1) ;     /* C = T*B */
                }
            }
        }
    }
    else
    {
        if (bt)
        {
            if (ct)
            {
                /* C = (A*B')'      A is m-by-n, B is k-by-n, C is k-by-m */
                T = ssmult_transpose (A, ac) ;                  /* T = A' */
                C = ssmult_saxpy (B, T, bc, 0, cc, 1) ;         /* C = B*T */
            }
            else
            {
                /* C = A*B'         A is m-by-n, B is k-by-n, C is m-by-k */
                T = ssmult_transpose (B, bc) ;                  /* T = B' */
                C = ssmult_saxpy (A, T, ac, 0, cc, 1) ;         /* C = A*T */
            }
        }
        else
        {
            if (ct)
            {
                /* C = (A*B)'       A is m-by-n, B is n-by-k, C is k-by-m */
                T = ssmult_saxpy (A, B, ac, bc, 0, 0) ;         /* T = A*B */
                C = ssmult_transpose (T, cc) ;                  /* C = T' */
            }
            else
            {
                /* C = A*B          A is m-by-n, B is n-by-k, C is m-by-k */
                C = ssmult_saxpy (A, B, ac, bc, cc, 1) ;        /* C = A*B */
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* free workspace and return result */
    /* ---------------------------------------------------------------------- */

    if (T) mxDestroyArray (T) ;
    return (C) ;
}
