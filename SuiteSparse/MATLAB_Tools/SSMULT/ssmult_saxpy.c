/* ========================================================================== */
/* === ssmult_saxpy ========================================================= */
/* ========================================================================== */

/* C = ssmult_saxpy (A,B) multiplies two sparse matrices A and B, in MATLAB.
 * Either A or B, or both, can be complex.  C is returned as a proper MATLAB
 * sparse matrix, with sorted row indices, or with unsorted row indices.  No
 * explicit zero entries appear in C.  If A or B are complex, but the imaginary
 * part of C is computed to be zero, then C is returned as a real sparse matrix
 * (as in MATLAB).
 *
 * Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com
 */

#include "ssmult.h"

/* -------------------------------------------------------------------------- */
/* merge Left [0..nleft-1] and Right [0..nright-1] into S [0..nleft+nright-1] */
/* -------------------------------------------------------------------------- */

static void merge
(
    Int S [ ],              /* output of length nleft + nright */
    const Int Left [ ],     /* left input of length nleft */
    const Int nleft,
    const Int Right [ ],    /* right input of length nright */
    const Int nright
)
{
    Int p, pleft, pright ;

    /* merge the two inputs, Left and Right, while both inputs exist */
    for (p = 0, pleft = 0, pright = 0 ; pleft < nleft && pright < nright ; p++)
    {
        if (Left [pleft] < Right [pright])
        {
            S [p] = Left [pleft++] ;
        }
        else
        {
            S [p] = Right [pright++] ;
        }
    }

    /* either input is exhausted; copy the remaining list into S */
    for ( ; pleft < nleft ; p++)
    {
        S [p] = Left [pleft++] ;
    }
    for ( ; pright < nright ; p++)
    {
        S [p] = Right [pright++] ;
    }
}


/* -------------------------------------------------------------------------- */
/* SORT(a,b) sorts [a b] in ascending order, so that a < b holds on output */
/* -------------------------------------------------------------------------- */

#define SORT(a,b) { if (a > b) { t = a ; a = b ; b = t ; } }


/* -------------------------------------------------------------------------- */
/* BUBBLE(a,b) sorts [a b] in ascending order, and sets done to 0 if it swaps */
/* -------------------------------------------------------------------------- */

#define BUBBLE(a,b) { if (a > b) { t = a ; a = b ; b = t ; done = 0 ; } }


/* -------------------------------------------------------------------------- */
/* ssmergesort */
/* -------------------------------------------------------------------------- */

/* ssmergesort (A,W,n) sorts an Int array A of length n in ascending order. W is
 * a workspace array of size n.  function is used for sorting the row indices
 * in each column of C.  Small lists (of length SMALL or less) are sorted with
 * a bubble sort.  A value of 10 for SMALL works well on an Intel Core Duo, an
 * Intel Pentium 4M, and a 64-bit AMD Opteron.  SMALL must be in the range 4 to
 * 10. */

#ifndef SMALL
#define SMALL 10
#endif

static void ssmergesort
(
    Int A [ ],      /* array to sort, of size n */
    Int W [ ],      /* workspace of size n */
    Int n
)
{
    if (n <= SMALL)
    {

        /* ------------------------------------------------------------------ */
        /* bubble sort for small lists of length SMALL or less */
        /* ------------------------------------------------------------------ */

        Int t, done ;
        switch (n)
        {

#if SMALL >= 10
            case 10:
                /* 10-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                BUBBLE (A [4], A [5]) ;
                BUBBLE (A [5], A [6]) ;
                BUBBLE (A [6], A [7]) ;
                BUBBLE (A [7], A [8]) ;
                BUBBLE (A [8], A [9]) ;
                if (done) return ;
#endif

#if SMALL >= 9
            case 9:
                /* 9-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                BUBBLE (A [4], A [5]) ;
                BUBBLE (A [5], A [6]) ;
                BUBBLE (A [6], A [7]) ;
                BUBBLE (A [7], A [8]) ;
                if (done) return ;
#endif

#if SMALL >= 8
            case 8:
                /* 7-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                BUBBLE (A [4], A [5]) ;
                BUBBLE (A [5], A [6]) ;
                BUBBLE (A [6], A [7]) ;
                if (done) return ;
#endif

#if SMALL >= 7
            case 7:
                /* 7-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                BUBBLE (A [4], A [5]) ;
                BUBBLE (A [5], A [6]) ;
                if (done) return ;
#endif

#if SMALL >= 6
            case 6:
                /* 6-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                BUBBLE (A [4], A [5]) ;
                if (done) return ;
#endif

#if SMALL >= 5
            case 5:
                /* 5-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                BUBBLE (A [3], A [4]) ;
                if (done) return ;
#endif

            case 4:
                /* 4-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                BUBBLE (A [2], A [3]) ;
                if (done) return ;

            case 3:
                /* 3-element bubble sort */
                done = 1 ;
                BUBBLE (A [0], A [1]) ;
                BUBBLE (A [1], A [2]) ;
                if (done) return ;

            case 2:
                /* 2-element bubble sort */
                SORT (A [0], A [1]) ; 

            case 1:
            case 0:
                /* nothing to do */
                ;
        }

    }
    else
    {

        /* ------------------------------------------------------------------ */
        /* recursive ssmergesort if A has length 5 or more */
        /* ------------------------------------------------------------------ */

        Int n1, n2, n3, n4, n12, n34, n123 ;

        n12 = n / 2 ;           /* split n into n12 and n34 */
        n34 = n - n12 ;

        n1 = n12 / 2 ;          /* split n12 into n1 and n2 */
        n2 = n12 - n1 ;

        n3 = n34 / 2 ;          /* split n34 into n3 and n4 */
        n4 = n34 - n3 ;

        n123 = n12 + n3 ;       /* start of 4th subset = n1 + n2 + n3 */

        ssmergesort (A,        W, n1) ;       /* sort A [0  ... n1-1] */
        ssmergesort (A + n1,   W, n2) ;       /* sort A [n1 ... n12-1] */
        ssmergesort (A + n12,  W, n3) ;       /* sort A [n12 ... n123-1] */
        ssmergesort (A + n123, W, n4) ;       /* sort A [n123 ... n-1]  */

        /* merge A [0 ... n1-1] and A [n1 ... n12-1] into W [0 ... n12-1] */
        merge (W, A, n1, A + n1, n2) ;

        /* merge A [n12 ... n123-1] and A [n123 ... n-1] into W [n12 ... n-1] */
        merge (W + n12, A + n12, n3, A + n123, n4) ;

        /* merge W [0 ... n12-1] and W [n12 ... n-1] into A [0 ... n-1] */
        merge (A, W, n12, W + n12, n34) ;
    }
}


/* -------------------------------------------------------------------------- */
/* ssmult_saxpy */
/* -------------------------------------------------------------------------- */

mxArray *ssmult_saxpy       /* return C = A*B */
(
    const mxArray *A,
    const mxArray *B,
    int ac,                 /* if true use conj(A) */
    int bc,                 /* if true use conj(B) */
    int cc,                 /* if true compute conj(C) */
    int sorted              /* if true, return C with sorted columns */
)
{
    double bkj, bzkj, cij, czij ;
    mxArray *C ;
    double *Ax, *Az, *Bx, *Bz, *Cx, *Cz, *W, *Wz ;
    Int *Ap, *Ai, *Bp, *Bi, *Cp, *Ci, *Flag ;
    Int Anrow, Ancol, Bnrow, Bncol, p, i, j, k, cnz, pb, pa, pbend, paend, mark,
        pc, pcstart, blen, drop, zallzero, pend, needs_sorting, pcmax ;
    int A_is_complex, B_is_complex, C_is_complex, A_is_diagonal,
        A_is_permutation ;

    /* ---------------------------------------------------------------------- */
    /* get inputs and workspace */
    /* ---------------------------------------------------------------------- */

    Ap = (Int *) mxGetJc (A) ;
    Ai = (Int *) mxGetIr (A) ;
    Ax = mxGetPr (A) ;
    Az = mxGetPi (A) ;
    Anrow = mxGetM (A) ;
    Ancol = mxGetN (A) ;
    A_is_complex = mxIsComplex (A) ;

    Bp = (Int *) mxGetJc (B) ;
    Bi = (Int *) mxGetIr (B) ;
    Bx = mxGetPr (B) ;
    Bz = mxGetPi (B) ;
    Bnrow = mxGetM (B) ;
    Bncol = mxGetN (B) ;
    B_is_complex = mxIsComplex (B) ;

    if (Ancol != Bnrow) ssmult_invalid (ERROR_DIMENSIONS) ;

    Flag = mxCalloc (Anrow, sizeof (Int)) ;     /* workspace */
    Cp = mxMalloc ((Bncol+1) * sizeof (Int)) ;  /* allocate C column pointers */

    /* ---------------------------------------------------------------------- */
    /* compute # of nonzeros in result */
    /* ---------------------------------------------------------------------- */

    pb = 0 ;
    cnz = 0 ;
    mark = 0 ;
    for (j = 0 ; j < Bncol ; j++)
    {
        /* Compute nnz (C (:,j)) */
        mark-- ;                        /* Flag [0..n-1] != mark is now true */ 
        pb = Bp [j] ;
        pbend = Bp [j+1] ;
        pcstart = cnz ;
        pcmax = cnz + Anrow ;
        Cp [j] = cnz ;
        /* cnz += nnz (C (:,j)), stopping early if nnz(C(:,j)) == Anrow */
        for ( ; pb < pbend && cnz < pcmax ; pb++)
        {
            k = Bi [pb] ;               /* nonzero entry B(k,j) */
            paend = Ap [k+1] ;
            for (pa = Ap [k] ; pa < paend ; pa++)
            {
                i = Ai [pa] ;           /* nonzero entry A(i,k) */
                if (Flag [i] != mark)
                {
                    /* C(i,j) is a new nonzero */
                    Flag [i] = mark ;   /* mark i as appearing in C(:,j) */
                    cnz++ ;
                }
            }
        }
        if (cnz < pcstart)
        {
            /* integer overflow has occurred */
            ssmult_invalid (ERROR_TOO_LARGE) ;
        }
    }
    Cp [Bncol] = cnz ;

    /* ---------------------------------------------------------------------- */
    /* allocate C */
    /* ---------------------------------------------------------------------- */

    Ci = mxMalloc (MAX (cnz,1) * sizeof (Int)) ;
    Cx = mxMalloc (MAX (cnz,1) * sizeof (double)) ;
    C_is_complex = A_is_complex || B_is_complex ;
    Cz = (C_is_complex) ? mxMalloc (MAX (cnz,1) * sizeof (double)) : NULL ;

    /* ---------------------------------------------------------------------- */
    /* C = A*B */
    /* ---------------------------------------------------------------------- */

    if (sorted)
    {
#undef UNSORTED
        if (A_is_complex)
        {
            if (B_is_complex)
            {
                /* all of C, A, and B are complex */
#define ACOMPLEX
#define BCOMPLEX
#include "ssmult_template.c"
            }
            else
            {
                /* C and A are complex, B is real */
#define ACOMPLEX
#include "ssmult_template.c"
            }
        }
        else
        {
            if (B_is_complex)
            {
                /* C and B are complex, A is real */
#define BCOMPLEX
#include "ssmult_template.c"
            }
            else
            {
                /* all of C, A, and B are real */
#include "ssmult_template.c"
            }
        }
    }
    else
    {
#define UNSORTED
        if (A_is_complex)
        {
            if (B_is_complex)
            {
                /* all of C, A, and B are complex */
#define ACOMPLEX
#define BCOMPLEX
#include "ssmult_template.c"
            }
            else
            {
                /* C and A are complex, B is real */
#define ACOMPLEX
#include "ssmult_template.c"
            }
        }
        else
        {
            if (B_is_complex)
            {
                /* C and B are complex, A is real */
#define BCOMPLEX
#include "ssmult_template.c"
            }
            else
            {
                /* all of C, A, and B are real */
#include "ssmult_template.c"
            }
        }
    }

/*
    for (j = 0 ; j < Bncol ; j++)
    {
        for (p = Cp [j] ; p < Cp [j+1] ; p++)
        {
            printf ("C(%d,%d) = %g", Ci [p], j, Cx [p]) ;
            if (Cz != NULL) printf (" %g ", Cz [p]) ;
            printf ("\n") ;
        }
    }
*/
    /* ---------------------------------------------------------------------- */
    /* free workspace */
    /* ---------------------------------------------------------------------- */

    MXFREE (Flag) ;

    /* ---------------------------------------------------------------------- */
    /* convert C to real if Cz is all zero */
    /* ---------------------------------------------------------------------- */

    if (C_is_complex && zallzero)
    {
        C_is_complex = 0 ;
        MXFREE (Cz) ;
        Cz = NULL ;
    }

    /* ---------------------------------------------------------------------- */
    /* drop zeros from C and reduce its size, if any zeros appear */
    /* ---------------------------------------------------------------------- */

/*
    for (j = 0 ; j < Bncol ; j++)
    {
        for (p = Cp [j] ; p < Cp [j+1] ; p++)
        {
            printf ("C(%d,%d) = %g", Ci [p], j, Cx [p]) ;
            if (Cz != NULL) printf (" %g ", Cz [p]) ;
            printf ("\n") ;
        }
    }
*/

    if (drop)
    {
        if (C_is_complex)
        {
            for (cnz = 0, p = 0, j = 0 ; j < Bncol ; j++)
            {
                Cp [j] = cnz ;
                pend = Cp [j+1] ;
                for ( ; p < pend ; p++)
                {
                    if (Cx [p] != 0 || Cz [p] != 0)
                    {
                        Ci [cnz] = Ci [p] ;         /* keep this entry */
                        Cx [cnz] = Cx [p] ;
                        Cz [cnz] = (cc ? (-Cz [p]) : (Cz [p])) ;
                        cnz++ ;
                    }
                }
            }
            Cz = mxRealloc (Cz, MAX (cnz,1) * sizeof (double)) ;
        }
        else
        {
            for (cnz = 0, p = 0, j = 0 ; j < Bncol ; j++)
            {
                Cp [j] = cnz ;
                pend = Cp [j+1] ;
                for ( ; p < pend ; p++)
                {
                    if (Cx [p] != 0)
                    {
                        Ci [cnz] = Ci [p] ;         /* keep this entry */
                        Cx [cnz] = Cx [p] ;
                        cnz++ ;
                    }
                }
            }
        }
        Cp [Bncol] = cnz ;
        Ci = mxRealloc (Ci, MAX (cnz,1) * sizeof (Int)) ;
        Cx = mxRealloc (Cx, MAX (cnz,1) * sizeof (double)) ;
    }
    else if (cc && C_is_complex)
    {
        /* compute conj(C), but with no dropping of entries */
        cnz = Cp [Bncol] ;
        for (p = 0 ; p < cnz ; p++)
        {
            Cz [p] = -Cz [p] ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* return C */
    /* ---------------------------------------------------------------------- */

/*
    for (j = 0 ; j < Bncol ; j++)
    {
        for (p = Cp [j] ; p < Cp [j+1] ; p++)
        {
            printf ("C(%d,%d) = %g", Ci [p], j, Cx [p]) ;
            if (Cz != NULL) printf (" %g ", Cz [p]) ;
            printf ("\n") ;
        }
    }
*/

    C = mxCreateSparse (0, 0, 0, C_is_complex ? mxCOMPLEX : mxREAL) ;
    MXFREE (mxGetJc (C)) ;
    MXFREE (mxGetIr (C)) ;
    MXFREE (mxGetPr (C)) ;
    MXFREE (mxGetPi (C)) ;
    mxSetJc (C, (mwIndex *) Cp) ;
    mxSetIr (C, (mwIndex *) Ci) ;
    mxSetPr (C, Cx) ;
    mxSetPi (C, Cz) ;
    mxSetNzmax (C, MAX (cnz,1)) ;
    mxSetM (C, Anrow) ;
    mxSetN (C, Bncol) ;

/*
    for (j = 0 ; j < Bncol ; j++)
    {
        for (p = Cp [j] ; p < Cp [j+1] ; p++)
        {
            printf ("C(%d,%d) = %g", Ci [p], j, Cx [p]) ;
            if (Cz != NULL) printf (" %g ", Cz [p]) ;
            printf ("\n") ;
        }
    }
*/
    return (C) ;
}
