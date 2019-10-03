/* ========================================================================== */
/* == ssmult_dot ============================================================ */
/* ========================================================================== */

/*
   C = A'*B using the sparse-dot-product method.  Computes C as a full matrix
   first and then converts the result to sparse format.  It is thus useful only
   if C is small compared to A and/or B.  It is very fast if A and B are long
   column vectors, because in that case, computing A' takes a long time.

   A is m-by-n, B is m-by-k, and thus C is n-by-k.

   The time take by this function is at least proportional to n*k + flops(A*B)),
   but it can be higher because a sparse dot product of x'*y where x and y are
   column vectors can take up to O (nnz(x) + nnz(y)).  The sparse dot product
   can terminate early.  In particular, the time is precisely
   O (nnz (x (1:t)) + nnz (y (1:t))) where t = min (max (find (x), find (y))).
   This sparse dot is used for each pair of columns of A and B.  The workspace
   required by this method is n*k*sizeof(double) or twice that if A or B are
   complex.

   By comparison, the saxpy method to compute C=A*B takes O (m+n+k+flops(A*B))
   time and uses only O(m) workspace.  However, C=A'*B using that method must
   transpose A first, taking another O(m+n+nnz(A)) time and adding O(m+nnz(A))
   workspace.

   Note that m does NOT appear in the time or memory complexity of ssmult_dot
   when C=A'*B is computed.  Thus, if m is huge compared to n, k, nnz(A), and
   so on, then it can be far faster and use far less memory.  For exampe, if
   A and B are very long and very sparse column vectors, the dot product method
   is much faster than the saxpy method.

   Comparing flop counts of the two methods is not trivial.  Thus, when
   computing C=A'*B, ssmult uses whichever method requires the least workspace. 

   Sparse dot product based matrix multiplication algorithm in MATLAB notation:
  
        function C = ssmult_dot (A,B)
        % C = A'*B                  A is m-by-n, B is m-by-k, C is n-by-k
        C = zeros (n,k) ;
        for i = 1:n
            for j = 1:k
                C(i,j) = A (:,i)'*B(:,j) ;
            end
        end
        C = sparse (C) ;
 */

#include "ssmult.h"

/* -------------------------------------------------------------------------- */
/* ssmult_dot */
/* -------------------------------------------------------------------------- */

mxArray *ssmult_dot     /* returns C = A'*B */
(
    const mxArray *A,
    const mxArray *B,
    int ac,             /* if true: conj(A)   if false: A. ignored if A real */
    int bc,             /* if true: conj(B)   if false: B. ignored if B real */
    int cc              /* if true: conj(C)   if false: C. ignored if C real */
)
{
    double cx, cz, ax, az, bx, bz ;
    mxArray *C ;
    double *Ax, *Az, *Bx, *Bz, *Cx, *Cz ;
    Int *Ap, *Ai, *Bp, *Bi, *Cp, *Ci ;
    Int m, n, k, cnzmax, i, j, p, paend, pbend, ai, bi, cnz, pa, pb, zallzero,
        A_is_complex, B_is_complex, C_is_complex ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    m = mxGetM (A) ;
    n = mxGetN (A) ;
    k = mxGetN (B) ;

    if (m != mxGetM (B)) ssmult_invalid (ERROR_DIMENSIONS) ;

    Ap = (Int *) mxGetJc (A) ;
    Ai = (Int *) mxGetIr (A) ;
    Ax = mxGetPr (A) ;
    Az = mxGetPi (A) ;
    A_is_complex = mxIsComplex (A) ;

    Bp = (Int *) mxGetJc (B) ;
    Bi = (Int *) mxGetIr (B) ;
    Bx = mxGetPr (B) ;
    Bz = mxGetPi (B) ;
    B_is_complex = mxIsComplex (B) ;

    /* ---------------------------------------------------------------------- */
    /* allocate C as an n-by-k full matrix but do not initialize it */
    /* ---------------------------------------------------------------------- */

    /* NOTE: integer overflow cannot occur here, because this function is not
       called unless O(n*k) is less than O(m+nnz(A)).  The test is done
       in the caller, not here.
     */

    cnzmax = n*k ;
    cnzmax = MAX (cnzmax, 1) ;
    Cx = mxMalloc (cnzmax * sizeof (double)) ;
    C_is_complex = A_is_complex || B_is_complex ;
    Cz = C_is_complex ?  mxMalloc (cnzmax * sizeof (double)) : NULL ;

    /* ---------------------------------------------------------------------- */
    /* C = A'*B using sparse dot products */
    /* ---------------------------------------------------------------------- */

    /*
       NOTE:  this method REQUIRES the columns of A and B to be sorted on input.
       That is, the row indices in each column must appear in ascending order.
       This is the standard in all versions of MATLAB to date, and likely will
       be for some time.  However, if MATLAB were to use unsorted sparse
       matrices in the future (a lazy sort) then a test should be included in
       ssmult to not use ssmult_dot if A or B are unsorted, or they should be
       sorted on input.
     */

    cnz = 0 ;
    for (j = 0 ; j < k ; j++)
    {
        for (i = 0 ; i < n ; i++)
        {
            /* compute C (i,j) = A (:,i)' * B (:,j) */
            pa = Ap [i] ;
            paend = Ap [i+1] ;
            pb = Bp [j] ;
            pbend = Bp [j+1] ;

            if (pa == paend            /* nnz (A (:,i)) == 0 */
            || pb == pbend             /* nnz (B (:,j)) == 0 */
            || Ai [paend-1] < Bi [pb]  /* max(find(A(:,i)))<min(find(B(:,j))) */
            || Ai [pa] > Bi [pbend-1]) /* min(find(A(:,i)))>max(find(B(:,j))) */
            {
                Cx [i+j*n] = 0 ;        /* no work to do */
                if (C_is_complex) 
                {
                    Cz [i+j*n] = 0 ;
                }
                continue ;
            }
            cx = 0 ;
            cz = 0 ;
            while (pa < paend && pb < pbend)
            {
                /* The dot product looks like the merge in ssmergesort, except*/
                /* no "clean-up" phase is need when one list is exhausted. */
                ai = Ai [pa] ;
                bi = Bi [pb] ;
                if (ai == bi)
                {
                    /* c += A (ai,i) * B (ai,j), and "consume" both entries */
                    if (!C_is_complex)
                    {
                        cx += Ax [pa] * Bx [pb] ;
                    }
                    else
                    {
                        /* complex case */
                        ax = Ax [pa] ;
                        bx = Bx [pb] ;
                        az = Az ? (ac ? (-Az [pa]) : Az [pa]) : 0.0 ;
                        bz = Bz ? (bc ? (-Bz [pb]) : Bz [pb]) : 0.0 ;
                        cx += ax * bx - az * bz ;
                        cz += az * bx + ax * bz ;
                    }
                    pa++ ;
                    pb++ ;
                }
                else if (ai < bi)
                {
                    /* consume A(ai,i) and discard it, since B(ai,j) is zero */
                    pa++ ;
                }
                else
                {
                    /* consume B(bi,j) and discard it, since A(ai,i) is zero */
                    pb++ ;
                }
            }
            Cx [i+j*n] = cx ;
            if (C_is_complex)
            {
                Cz [i+j*n] = cz ;
            }
        }

        /* count the number of nonzeros in C(:,j) */
        for (i = 0 ; i < n ; i++)
        {
            /* This could be done above, except for the gcc compiler bug when
               cx is an 80-bit nonzero in register above, but becomes zero here
               when stored into memory.  We need the latter, to correctly handle
               the case when cx underflows to zero in 64-bit floating-point.
               Do not attempt to "optimize" this code by doing this test above,
               unless the gcc compiler bug is fixed (as of gcc version 4.1.0).
             */
            if (Cx [i+j*n] != 0 || (C_is_complex && Cz [i+j*n] != 0))
            {
                cnz++ ;
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* convert C to real if the imaginary part is all zero */
    /* ---------------------------------------------------------------------- */

    if (C_is_complex)
    {
        zallzero = 1 ;
        for (p = 0 ; zallzero && p < cnzmax ; p++)
        {
            if (Cz [p] != 0)
            {
                zallzero = 0 ;
            }
        }
        if (zallzero)
        {
            /* the imaginary part of C is all zero */
            C_is_complex = 0 ;
            mxFree (Cz) ;
            Cz = NULL ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* allocate integer part of C but do not initialize it */
    /* ---------------------------------------------------------------------- */

    cnz = MAX (cnz, 1) ;
    C = mxCreateSparse (0, 0, 0, C_is_complex ? mxCOMPLEX : mxREAL) ;
    MXFREE (mxGetJc (C)) ;
    MXFREE (mxGetIr (C)) ;
    MXFREE (mxGetPr (C)) ;
    MXFREE (mxGetPi (C)) ;
    Cp = mxMalloc ((k + 1) * sizeof (Int)) ;
    Ci = mxMalloc (MAX (cnz,1) * sizeof (Int)) ;
    mxSetJc (C, (mwIndex *) Cp) ;
    mxSetIr (C, (mwIndex *) Ci) ;
    mxSetM (C, n) ;
    mxSetN (C, k) ;

    /* ---------------------------------------------------------------------- */
    /* C = sparse (C).  Note that this is done in-place. */
    /* ---------------------------------------------------------------------- */

    p = 0 ;
    for (j = 0 ; j < k ; j++)
    {
        Cp [j] = p ;
        for (i = 0 ; i < n ; i++)
        {
            cx = Cx [i+j*n] ;
            cz = (C_is_complex ? Cz [i+j*n] : 0) ;
            if (cx != 0 || cz != 0)
            {
                Ci [p] = i ;
                Cx [p] = cx ;
                if (C_is_complex) Cz [p] = (cc ? (-cz) : cz) ;
                p++ ;
            }
        }
    }
    Cp [k] = p ;

    /* ---------------------------------------------------------------------- */
    /* reduce the size of Cx and Cz and return result */
    /* ---------------------------------------------------------------------- */

    if (cnz < cnzmax)
    {
        Cx = mxRealloc (Cx, cnz * sizeof (double)) ;
        if (C_is_complex)
        {
            Cz = mxRealloc (Cz, cnz * sizeof (double)) ;
        }
    }

    mxSetNzmax (C, cnz) ;
    mxSetPr (C, Cx) ;
    if (C_is_complex)
    {
        mxSetPi (C, Cz) ;
    }
    return (C) ;
}
