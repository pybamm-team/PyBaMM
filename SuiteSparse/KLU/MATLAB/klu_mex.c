/* ========================================================================== */
/* === klu mexFunction ====================================================== */
/* ========================================================================== */

/* KLU: a MATLAB interface to a "Clark Kent" sparse LU factorization algorithm.

    3 or 4 input arguments: factorize and solve, returning the solution:

        x = klu (A, '\', b)
        x = klu (A, '\', b, opts)
        x = klu (b, '/', A)
        x = klu (b, '/', A, opts)

    A can be the LU struct, instead:

        x = klu (LU, '\', b)
        x = klu (LU, '\', b, opts)
        x = klu (b, '/', LU)
        x = klu (b, '/', LU, opts)

    where LU is a struct containing members: L, U, p, q, R, F, and r.  Only L
    and U are required.  The factorization is L*U+F = R\A(p,q), where r defines
    the block boundaries of the BTF form, and F contains the entries in the
    upper block triangular part.

    with 1 or 2 input arguments: factorize, returning the LU struct:

        LU = klu (A)
        LU = klu (A, opts)

    2nd optional output: info, which is only meaningful if A was factorized.

    A must be square.  b can be a matrix, but it cannot be sparse.

    Obscure options, mainly for testing:

        opts.memgrow    1.2     when L and U need to grow, inc. by this ratio.
                                valid range: 1 or more.
        opts.imemamd    1.2     initial size of L and U with AMD or other
                                symmetric ordering is 1.2*nnz(L)+n;
                                valid range 1 or more.
        opts.imem       10      initial size of L and U is 10*nnz(A)+n if a
                                symmetric ordering not used; valid range 1 or
                                more
*/

/* ========================================================================== */

#include "klu.h"
#include <string.h>
#define Long SuiteSparse_long

#ifndef NCHOLMOD
#include "klu_cholmod.h"
#endif

#include "mex.h"
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define ABS(x) (((x) < 0) ? -(x) : (x))
#define STRING_MATCH(s1,s2) (strcmp ((s1), (s2)) == 0)

/* Complex division.  This uses ACM Algo 116, by R. L. Smith, 1962. */
/* Note that c cannot be the same variable as a or b */
#define DIV(cx,cz,ax,az,bx,bz) \
{ \
    double r, den ; \
    if (ABS (bx) >= ABS (bz)) \
    { \
        r = bz / bx ; \
        den = bx + r * bz ; \
        cx = (ax + az * r) / den ; \
        cz = (az - ax * r) / den ; \
    } \
    else \
    { \
        r = bx / bz ; \
        den = r * bx + bz ; \
        cx = (ax * r + az) / den ; \
        cz = (az * r - ax) / den ; \
    } \
}

/* complex multiply/subtract, c -= a*b */
/* Note that c cannot be the same variable as a or b */
#define MULT_SUB(cx,cz,ax,az,bx,bz) \
{ \
    cx -= ax * bx - az * bz ; \
    cz -= az * bx + ax * bz ; \
}

/* complex multiply/subtract, c -= a*conj(b) */
/* Note that c cannot be the same variable as a or b */
#define MULT_SUB_CONJ(cx,cz,ax,az,bx,bz) \
{ \
    cx -= ax * bx + az * bz ; \
    cz -= az * bx - ax * bz ; \
}

/* ========================================================================== */
/* === klu mexFunction ====================================================== */
/* ========================================================================== */

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double ukk, lkk, rs, s, lik, uik, x [4], offik, z, ukkz, lkkz, sz, wx, wz ;
    double *X, *B, *Xz, *Xx, *Bx, *Bz, *A, *Ax, *Az, *Lx, *Ux, *Rs, *Offx, *Wx,
        *Uz, *Lz, *Offz, *Wz, *W, *Xi, *Bi ;
    Long *Ap, *Ai, *Lp, *Li, *Up, *Ui, *P, *Q, *R, *Rp, *Ri, *Offp, *Offi ;
    char *operator ;
    mxArray *L_matlab, *U_matlab, *p_matlab, *q_matlab, *R_matlab, *F_matlab,
        *r_matlab, *field ;
    const mxArray *A_matlab = NULL, *LU_matlab, *B_matlab = NULL, *opts_matlab ;
    klu_l_symbolic *Symbolic ;
    klu_l_numeric *Numeric ;
    klu_l_common Common ;
    Long n = 0, k, nrhs = 0, do_solve, do_factorize, symmetric,
        A_complex = 0, B_complex, nz, do_transpose = 0, p, pend, nblocks,
        R1 [2], chunk, nr, i, j, block, k1, k2, nk, bn = 0, ordering ;
    int mx_int ;
    static const char *fnames [ ] = {
        "noffdiag",     /* # of off-diagonal pivots */
        "nrealloc",     /* # of memory reallocations */
        "rcond",        /* cheap reciprocal number estimate */
        "rgrowth",      /* reciprocal pivot growth */
        "flops",        /* flop count */
        "nblocks",      /* # of blocks in BTF form (1 if not computed) */
        "ordering",     /* AMD, COLAMD, natural, cholmod(AA'), cholmod(A+A') */
        "scale",        /* scaling (<=0: none, 1: sum, 2: max */
        "lnz",          /* nnz(L), including diagonal */
        "unz",          /* nnz(U), including diagonal */
        "offnz",        /* nnz(F), including diagonal */
        "tol",          /* pivot tolerance used */
        "memory"        /* peak memory usage */
        },
        *LUnames [ ] = { "L", "U", "p", "q", "R", "F", "r" } ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    if (nargin < 1 || nargin > 4 || nargout > 3)
    {
        mexErrMsgTxt (
            "Usage: x = klu(A,'\',b), x = klu(A,'/',b) or LU = klu(A)") ;
    }

    /* return the solution x, or just do LU factorization */
    do_solve = (nargin > 2) ;

    /* determine size of the MATLAB integer */
    if (sizeof (Long) == sizeof (INT32_T))
    {
        mx_int = mxINT32_CLASS ;
    }
    else
    {
        mx_int = mxINT64_CLASS ;
    }

    if (do_solve)
    {

        /* ------------------------------------------------------------------ */
        /* slash or backslash */
        /* ------------------------------------------------------------------ */

        /* usage, where opts is the optional 4th input argument:
            x = klu (A,  '\', b)
            x = klu (LU, '\', b)
            x = klu (b,  '/', A)
            x = klu (b,  '/', LU)
         */

        /* determine the operator, slash (/) or backslash (\) */
        if (!mxIsChar (pargin [1]))
        {
            mexErrMsgTxt ("invalid operator") ;
        }
        operator = mxArrayToString (pargin [1]) ;
        if (STRING_MATCH (operator, "\\"))
        {
            do_transpose = 0 ;
            A_matlab = pargin [0] ;
            B_matlab = pargin [2] ;
            nrhs = mxGetN (B_matlab) ;
            bn = mxGetM (B_matlab) ;
        }
        else if (STRING_MATCH (operator, "/"))
        {
            do_transpose = 1 ;
            A_matlab = pargin [2] ;
            B_matlab = pargin [0] ;
            nrhs = mxGetM (B_matlab) ;
            bn = mxGetN (B_matlab) ;
        }
        else
        {
            mexErrMsgTxt ("invalid operator") ;
        }

        if (mxIsSparse (B_matlab))
        {
            mexErrMsgTxt ("B cannot be sparse") ;
        }

        opts_matlab = (nargin > 3) ? pargin [3] : NULL ;

        /* determine if the factorization needs to be performed */
        do_factorize = !mxIsStruct (A_matlab) ;
        if (do_factorize)
        {
            LU_matlab = NULL ;
        }
        else
        {
            LU_matlab = A_matlab ;
            A_matlab = NULL ;
        }

    }
    else
    {

        /* ------------------------------------------------------------------ */
        /* factorize A and return LU factorization */
        /* ------------------------------------------------------------------ */

        /* usage, where opts in the optional 2nd input argument:
            LU = klu (A)
         */

        LU_matlab = NULL ;
        A_matlab = pargin [0] ;
        B_matlab = NULL ;
        opts_matlab = (nargin > 1) ? pargin [1] : NULL ;
        do_factorize = 1 ;
        if (mxIsStruct (A_matlab))
        {
            mexErrMsgTxt ("invalid input, A must be a sparse matrix") ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* get options and set Common defaults */
    /* ---------------------------------------------------------------------- */

    klu_l_defaults (&Common) ;

    /* factorization options */
    if (opts_matlab != NULL && mxIsStruct (opts_matlab))
    {
        if ((field = mxGetField (opts_matlab, 0, "tol")) != NULL)
        {
            Common.tol = mxGetScalar (field) ;
        }
        if ((field = mxGetField (opts_matlab, 0, "memgrow")) != NULL)
        {
            Common.memgrow = mxGetScalar (field) ;
        }
        if ((field = mxGetField (opts_matlab, 0, "imemamd")) != NULL)
        {
            Common.initmem_amd = mxGetScalar (field) ;
        }
        if ((field = mxGetField (opts_matlab, 0, "imem")) != NULL)
        {
            Common.initmem = mxGetScalar (field) ;
        }
        if ((field = mxGetField (opts_matlab, 0, "btf")) != NULL)
        {
            Common.btf = mxGetScalar (field) ;
        }
        if ((field = mxGetField (opts_matlab, 0, "ordering")) != NULL)
        {
            Common.ordering = mxGetScalar (field) ;
        }
        if ((field = mxGetField (opts_matlab, 0, "scale")) != NULL)
        {
            Common.scale = mxGetScalar (field) ;
        }
        if ((field = mxGetField (opts_matlab, 0, "maxwork")) != NULL)
        {
            Common.maxwork = mxGetScalar (field) ;
        }
    }

    if (Common.ordering < 0 || Common.ordering > 4)
    {
        mexErrMsgTxt ("invalid ordering option") ;
    }
    ordering = Common.ordering ;

#ifndef NCHOLMOD
    /* ordering option 3,4 becomes KLU option 3, with symmetric 0 or 1 */
    symmetric = (Common.ordering == 4) ;
    if (symmetric) Common.ordering = 3 ;
    Common.user_order = klu_l_cholmod ;
    Common.user_data = &symmetric ;
#else
    /* CHOLMOD, METIS, CAMD, CCOLAMD, not available */
    if (Common.ordering > 2)
    {
        mexErrMsgTxt ("invalid ordering option") ;
    }
#endif

    if (Common.scale < 1 || Common.scale > 2)
    {
        Common.scale = -1 ; /* no scaling, and no error checking either */
    }

    /* ---------------------------------------------------------------------- */
    /* factorize, if needed */
    /* ---------------------------------------------------------------------- */

    if (do_factorize)
    {

        /* get input matrix A to factorize */
        n = mxGetN (A_matlab) ;
        if (!mxIsSparse (A_matlab) || n != mxGetM (A_matlab) || n == 0)
        {
            mexErrMsgTxt ("A must be sparse, square, and non-empty") ;
        }

        Ap = (Long *) mxGetJc (A_matlab) ;
        Ai = (Long *) mxGetIr (A_matlab) ;
        Ax = mxGetPr (A_matlab) ;
        Az = mxGetPi (A_matlab) ;
        nz = Ap [n] ;
        A_complex = mxIsComplex (A_matlab) ;

        if (do_solve && (n != bn || nrhs == 0))
        {
            mexErrMsgTxt ("B must be non-empty with same number of rows as A") ;
        }

        /* ------------------------------------------------------------------ */
        /* analyze */
        /* ------------------------------------------------------------------ */

        Symbolic = klu_l_analyze (n, Ap, Ai, &Common) ;
        if (Symbolic == (klu_l_symbolic *) NULL)
        {
            mexErrMsgTxt ("klu symbolic analysis failed") ;
        }

        /* ------------------------------------------------------------------ */
        /* factorize */
        /* ------------------------------------------------------------------ */

        if (A_complex)
        {
            /* A is complex */
            A = mxMalloc (nz * 2 * sizeof (double)) ;
            for (k = 0 ; k < nz ; k++)
            {
                A [2*k  ] = Ax [k] ;        /* real part */
                A [2*k+1] = Az [k] ;        /* imaginary part */
            }
            Numeric = klu_zl_factor (Ap, Ai, A, Symbolic, &Common) ;
            if (nargout > 1)
            {
                /* flops and rgrowth, if requested */
                klu_zl_flops (Symbolic, Numeric, &Common) ;
                klu_zl_rgrowth (Ap, Ai, A, Symbolic, Numeric, &Common) ;
            }
            mxFree (A) ;
        }
        else
        {
            /* A is real */
            Numeric = klu_l_factor (Ap, Ai, Ax, Symbolic, &Common) ;
            if (nargout > 1)
            {
                /* flops, if requested */
                klu_l_flops (Symbolic, Numeric, &Common) ;
                klu_l_rgrowth (Ap, Ai, Ax, Symbolic, Numeric, &Common) ;
            }
        }
        if (Common.status != KLU_OK)
        {
            mexErrMsgTxt ("klu numeric factorization failed") ;
        }

        /* ------------------------------------------------------------------ */
        /* compute cheap condition number estimate */
        /* ------------------------------------------------------------------ */

        if (A_complex)
        {
            klu_zl_rcond (Symbolic, Numeric, &Common) ;
        }
        else
        {
            klu_l_rcond (Symbolic, Numeric, &Common) ;
        }

        /* ------------------------------------------------------------------ */
        /* return info, if requested */
        /* ------------------------------------------------------------------ */

#define INFO(i,x) \
        mxSetFieldByNumber (pargout [1], 0, i, mxCreateDoubleScalar (x))

        if (nargout > 1)
        {
            pargout [1] = mxCreateStructMatrix (1, 1, 13, fnames) ;
            INFO (0, Common.noffdiag) ;
            INFO (1, Common.nrealloc) ;
            INFO (2, Common.rcond) ;
            INFO (3, Common.rgrowth) ;
            INFO (4, Common.flops) ;
            INFO (5, Symbolic->nblocks) ;
            INFO (6, ordering) ;
            INFO (7, Common.scale) ;
            INFO (8, Numeric->lnz) ;
            INFO (9, Numeric->unz) ;
            INFO (10, Numeric->nzoff) ;
            INFO (11, Common.tol) ;
            INFO (12, Common.mempeak) ;
        }
        if (nargout > 2)
        {
            /* this is done separately, since it's costly */
            klu_l_condest (Ap, Ax, Symbolic, Numeric, &Common) ;
            pargout [2] = mxCreateDoubleMatrix (1, 1, mxREAL) ;
            Wx = mxGetPr (pargout [2]) ;
            Wx [0] = Common.condest ;
        }

    }
    else
    {
        /* create an empty "info" and "condest" output */
        if (nargout > 1)
        {
            pargout [1] = mxCreateDoubleMatrix (0, 0, mxREAL) ;
        }
        if (nargout > 2)
        {
            pargout [2] = mxCreateDoubleMatrix (0, 0, mxREAL) ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* solve, or return LU factorization */
    /* ---------------------------------------------------------------------- */

    if (do_solve)
    {

        /* ------------------------------------------------------------------ */
        /* solve, x = klu ( ... ) usage */
        /* ------------------------------------------------------------------ */

        B_complex = mxIsComplex (B_matlab) ;

        if (do_factorize)
        {

            /* -------------------------------------------------------------- */
            /* solve using KLU factors computed above */
            /* -------------------------------------------------------------- */

            /* klu (A,'\',b) or klu (b,'/',A) usage */

            /* create X */
            if (do_transpose)
            {
                pargout [0] = mxCreateDoubleMatrix (nrhs, n,
                    (A_complex || B_complex) ?  mxCOMPLEX : mxREAL) ;
            }
            else
            {
                pargout [0] = mxCreateDoubleMatrix (n, nrhs,
                    (A_complex || B_complex) ?  mxCOMPLEX : mxREAL) ;
            }

            if (A_complex)
            {

                /* ---------------------------------------------------------- */
                /* A is complex, but B might be real */
                /* ---------------------------------------------------------- */

                X = mxMalloc (n * nrhs * 2 * sizeof (double)) ;
                Bx = mxGetPr (B_matlab) ;
                Bz = mxGetPi (B_matlab) ;

                if (do_transpose)
                {

                    /* X = B', merge and transpose B */
                    for (j = 0 ; j < nrhs ; j++)
                    {
                        for (i = 0 ; i < n ; i++)
                        {
                            X [2*(i+j*n)  ] = Bx [j+i*nrhs] ;   /* real */
                            X [2*(i+j*n)+1] = Bz ? (-Bz [j+i*nrhs]) : 0 ;
                        }
                    }

                    /* solve A'x=b (complex conjugate) */
                    klu_zl_tsolve (Symbolic, Numeric, n, nrhs, X, 1, &Common) ;

                    /* split and transpose the solution */
                    Xx = mxGetPr (pargout [0]) ;
                    Xz = mxGetPi (pargout [0]) ;
                    for (j = 0 ; j < nrhs ; j++)
                    {
                        for (i = 0 ; i < n ; i++)
                        {
                            Xx [j+i*nrhs] = X [2*(i+j*n)  ] ;  /* real part */
                            Xz [j+i*nrhs] = -X [2*(i+j*n)+1] ; /* imag part */
                        }
                    }

                }
                else
                {

                    /* X = B, but create merged X from a split B */
                    for (k = 0 ; k < n*nrhs ; k++)
                    {
                        X [2*k  ] = Bx [k] ;                /* real part */
                        X [2*k+1] = Bz ? (Bz [k]) : 0 ;     /* imaginary part */
                    }

                    /* solve Ax=b */
                    klu_zl_solve (Symbolic, Numeric, n, nrhs, X, &Common) ;

                    /* split the solution into real and imaginary parts */
                    Xx = mxGetPr (pargout [0]) ;
                    Xz = mxGetPi (pargout [0]) ;
                    for (k = 0 ; k < n*nrhs ; k++)
                    {
                        Xx [k] = X [2*k  ] ;        /* real part */
                        Xz [k] = X [2*k+1] ;        /* imaginary part */
                    }
                }

                mxFree (X) ;
            }
            else
            {

                if (do_transpose)
                {

                    /* solve in chunks of 4 columns at a time */
                    W = mxMalloc (n * MAX (nrhs,4) * sizeof (double)) ;
                    X = mxGetPr (pargout [0]) ;
                    B = mxGetPr (B_matlab) ;
                    Xi = mxGetPi (pargout [0]) ;
                    Bi = mxGetPi (B_matlab) ;

                    for (chunk = 0 ; chunk < nrhs ; chunk += 4)
                    {

                        /* A is real: real(X) = real(b) / real(A) */
                        Long chunksize = MIN (nrhs - chunk, 4) ;
                        for (j = 0 ; j < chunksize ; j++)
                        {
                            for (i = 0 ; i < n ; i++)
                            {
                                W [i+j*n] = B [i*nrhs+j] ;
                            }
                        }
                        klu_l_tsolve (Symbolic, Numeric, n, chunksize, W,
                                &Common) ;
                        for (j = 0 ; j < chunksize ; j++)
                        {
                            for (i = 0 ; i < n ; i++)
                            {
                                X [i*nrhs+j] = W [i+j*n] ;
                            }
                        }
                        X += 4 ;
                        B += 4 ;

                        if (B_complex)
                        {
                            /* B is complex: imag(X) = imag(B) / real(A) */

                            for (j = 0 ; j < chunksize ; j++)
                            {
                                for (i = 0 ; i < n ; i++)
                                {
                                    W [i+j*n] = Bi [i*nrhs+j] ;
                                }
                            }
                            klu_l_tsolve (Symbolic, Numeric, n, chunksize, W,
                                &Common) ;
                            for (j = 0 ; j < chunksize ; j++)
                            {
                                for (i = 0 ; i < n ; i++)
                                {
                                    Xi [i*nrhs+j] = W [i+j*n] ;
                                }
                            }
                            Xi += 4 ;
                            Bi += 4 ;
                        }

                    }
                    mxFree (W) ;

                }
                else
                {

                    /* A is real: real(X) = real(A) \ real(b) */
                    X = mxGetPr (pargout [0]) ;
                    B = mxGetPr (B_matlab) ;
                    for (k = 0 ; k < n*nrhs ; k++)
                    {
                        X [k] = B [k] ;     
                    }
                    klu_l_solve (Symbolic, Numeric, n, nrhs, X, &Common) ;
                    if (B_complex)
                    {
                        /* B is complex: imag(X) = real(A) \ imag(B) */
                        X = mxGetPi (pargout [0]) ;
                        B = mxGetPi (B_matlab) ;
                        for (k = 0 ; k < n*nrhs ; k++)
                        {
                            X [k] = B [k] ;         
                        }
                        klu_l_solve (Symbolic, Numeric, n, nrhs, X, &Common) ;
                    }
                }
            }

            /* -------------------------------------------------------------- */
            /* free Symbolic and Numeric objects */
            /* -------------------------------------------------------------- */

            klu_l_free_symbolic (&Symbolic, &Common) ;
            if (A_complex)
            {
                klu_zl_free_numeric (&Numeric, &Common) ;
            }
            else
            {
                klu_l_free_numeric (&Numeric, &Common) ;
            }

        }
        else
        {

            /* -------------------------------------------------------------- */
            /* solve using LU struct given on input */
            /* -------------------------------------------------------------- */

            /* the factorization is L*U+F = R\A(p,q), where L*U is block
               diagonal, and F contains the entries in the upper block
               triangular part */

            L_matlab = mxGetField (LU_matlab, 0, "L") ;
            U_matlab = mxGetField (LU_matlab, 0, "U") ;
            p_matlab = mxGetField (LU_matlab, 0, "p") ;
            q_matlab = mxGetField (LU_matlab, 0, "q") ;
            R_matlab = mxGetField (LU_matlab, 0, "R") ;
            F_matlab = mxGetField (LU_matlab, 0, "F") ;
            r_matlab = mxGetField (LU_matlab, 0, "r") ;

            if (!L_matlab || !U_matlab || !mxIsSparse (L_matlab) ||
                !mxIsSparse (U_matlab))
            {
                mexErrMsgTxt ("invalid LU struct") ;
            }

            n = mxGetM (L_matlab) ;
            if (n != mxGetN (L_matlab) ||
                n != mxGetM (U_matlab) || n != mxGetN (U_matlab)
                /* ... */
                )
            {
                mexErrMsgTxt ("invalid LU struct") ;
            }

            if (n != bn || nrhs == 0)
            {
                mexErrMsgTxt (
                    "B must be non-empty with same number of rows as L and U") ;
            }

            /* get L */
            if (!mxIsSparse (L_matlab) ||
                n != mxGetM (L_matlab) || n != mxGetN (L_matlab))
            {
                mexErrMsgTxt ("LU.L must be sparse and same size as A") ;
            }

            Lp = (Long *) mxGetJc (L_matlab) ;
            Li = (Long *) mxGetIr (L_matlab) ;
            Lx = mxGetPr (L_matlab) ;
            Lz = mxGetPi (L_matlab) ;

            /* get U */
            if (!mxIsSparse (U_matlab) ||
                n != mxGetM (U_matlab) || n != mxGetN (U_matlab))
            {
                mexErrMsgTxt ("LU.U must be sparse and same size as A") ;
            }
            Up = (Long *) mxGetJc (U_matlab) ;
            Ui = (Long *) mxGetIr (U_matlab) ;
            Ux = mxGetPr (U_matlab) ;
            Uz = mxGetPi (U_matlab) ;

            /* get p */
            if (p_matlab)
            {
                if (mxGetNumberOfElements (p_matlab) != n
                    || mxIsSparse (p_matlab)
                    || mxGetClassID (p_matlab) != mx_int)
                {
                    mexErrMsgTxt ("P invalid") ;
                }
                P = (Long *) mxGetData (p_matlab) ;
                for (k = 0 ; k < n ; k++)
                {
                    if (P [k] < 1 || P [k] > n) mexErrMsgTxt ("P invalid") ;
                }
            }
            else
            {
                /* no P, use identity instead */
                P = NULL ;
            }

            /* get q */
            if (q_matlab)
            {
                if (mxGetNumberOfElements (q_matlab) != n
                    || mxIsSparse (q_matlab)
                    || mxGetClassID (q_matlab) != mx_int)
                {
                    mexErrMsgTxt ("Q invalid") ;
                }
                Q = (Long *) mxGetData (q_matlab) ;
                for (k = 0 ; k < n ; k++)
                {
                    if (Q [k] < 1 || Q [k] > n) mexErrMsgTxt ("Q invalid.") ;
                }
            }
            else
            {
                /* no Q, use identity instead */
                Q = NULL ;
            }

            /* get r */
            R1 [0] = 1 ;
            R1 [1] = n+1 ;
            if (r_matlab)
            {
                nblocks = mxGetNumberOfElements (r_matlab) - 1 ;
                if (nblocks < 1 || nblocks > n || mxIsSparse (r_matlab)
                    || mxGetClassID (r_matlab) != mx_int)
                {
                    mexErrMsgTxt ("r invalid") ;
                }
                R = (Long *) mxGetData (r_matlab) ;
                if (R [0] != 1) mexErrMsgTxt ("r invalid") ;
                for (k = 1 ; k <= nblocks ; k++)
                {
                    if (R [k] <= R [k-1] || R [k] > n+1)
                    {
                        mexErrMsgTxt ("rinvalid") ;
                    }
                }
                if (R [nblocks] != n+1) mexErrMsgTxt ("r invalid") ;
            }
            else
            {
                /* no r */
                nblocks = 1 ;
                R = R1 ;
            }

            /* get R, scale factors */
            if (R_matlab)
            {
                /* ensure R is sparse, real, and has the right size */
                if (!mxIsSparse (R_matlab) ||
                    n != mxGetM (R_matlab) || n != mxGetN (R_matlab))
                {
                    mexErrMsgTxt ("LU.R must be sparse and same size as A") ;
                }
                Rp = (Long *) mxGetJc (R_matlab) ;
                Rs = mxGetPr (R_matlab) ;
                if (Rp [n] != n)
                {
                    mexErrMsgTxt ("LU.R invalid, must be diagonal") ;
                }
            }
            else
            {
                /* no scale factors */
                Rs = NULL ;
            }

            /* get F, off diagonal entries */
            if (F_matlab)
            {
                if (!mxIsSparse (F_matlab) ||
                    n != mxGetM (F_matlab) || n != mxGetN (F_matlab))
                {
                    mexErrMsgTxt ("LU.F must be sparse and same size as A") ;
                }
                Offp = (Long *) mxGetJc (F_matlab) ;
                Offi = (Long *) mxGetIr (F_matlab) ;
                Offx = mxGetPr (F_matlab) ;
                Offz = mxGetPi (F_matlab) ;
            }
            else
            {
                /* no off-diagonal entries */
                Offp = NULL ;
                Offi = NULL ;
                Offx = NULL ;
                Offz = NULL ;
            }

            /* -------------------------------------------------------------- */
            /* solve */
            /* -------------------------------------------------------------- */

            if (mxIsComplex (L_matlab) || mxIsComplex (U_matlab) ||
                (F_matlab && mxIsComplex (F_matlab)) || B_complex)
            {

                /* ========================================================== */
                /* === complex case ========================================= */
                /* ========================================================== */

                /* create X */
                if (do_transpose)
                {
                    pargout [0] = mxCreateDoubleMatrix (nrhs, n, mxCOMPLEX) ;
                }
                else
                {
                    pargout [0] = mxCreateDoubleMatrix (n, nrhs, mxCOMPLEX) ;
                }
                Xx = mxGetPr (pargout [0]) ;
                Xz = mxGetPi (pargout [0]) ;

                Bx = mxGetPr (B_matlab) ;
                Bz = mxGetPi (B_matlab) ;

                /* get workspace */
                Wx = mxMalloc (n * sizeof (double)) ;
                Wz = mxMalloc (n * sizeof (double)) ;

                /* ---------------------------------------------------------- */
                /* do just one row/column of the right-hand-side at a time */
                /* ---------------------------------------------------------- */

                if (do_transpose)
                {

                    for (chunk = 0 ; chunk < nrhs ; chunk++)
                    {

                        /* -------------------------------------------------- */
                        /* transpose and permute right hand side, W = Q'*B' */
                        /* -------------------------------------------------- */

                        for (k = 0 ; k < n ; k++)
                        {
                            i = Q ? (Q [k] - 1) : k ;
                            Wx [k] = Bx [i*nrhs] ;
                            Wz [k] = Bz ? (-Bz [i*nrhs]) : 0 ;
                        }

                        /* -------------------------------------------------- */
                        /* solve W = (L*U + Off)'\W */
                        /* -------------------------------------------------- */

                        for (block = 0 ; block < nblocks ; block++)
                        {

                            /* ---------------------------------------------- */
                            /* block of size nk, rows/columns k1 to k2-1 */
                            /* ---------------------------------------------- */

                            k1 = R [block] - 1 ;        /* R is 1-based */
                            k2 = R [block+1] - 1 ;
                            nk = k2 - k1 ;

                            /* ---------------------------------------------- */
                            /* block back-substitution for off-diagonal-block */
                            /* ---------------------------------------------- */

                            if (block > 0 && Offp != NULL)
                            {
                                for (k = k1 ; k < k2 ; k++)
                                {
                                    pend = Offp [k+1] ;
                                    for (p = Offp [k] ; p < pend ; p++)
                                    {
                                        i = Offi [p] ;
                                        /* W [k] -= W [i] * conj(Off [p]) ; */
                                        z = Offz ? Offz [p] : 0 ;
                                        MULT_SUB_CONJ (Wx [k], Wz [k],
                                            Wx [i], Wz [i], Offx [p], z) ;
                                    }
                                }
                            }


                            /* solve the block system */
                            if (nk == 1)
                            {

                                /* W [k1] /= conj (L(k1,k1)) ; */
                                p = Lp [k1] ;
                                s = Lx [p] ;
                                sz = Lz ? (-Lz [p]) : 0 ;
                                DIV (wx, wz, Wx [k1], Wz [k1], s, sz) ;
                                Wx [k1] = wx ;
                                Wz [k1] = wz ;

                                /* W [k1] /= conj (U(k1,k1)) ; */
                                p = Up [k1] ;
                                s = Ux [p] ;
                                sz = Uz ? (-Uz [p]) : 0 ;
                                DIV (wx, wz, Wx [k1], Wz [k1], s, sz) ;
                                Wx [k1] = wx ;
                                Wz [k1] = wz ;

                            }
                            else
                            {

                                /* ------------------------------------------ */
                                /* W = U'\W and then W=L'\W */
                                /* ------------------------------------------ */

                                /* W = U'\W */
                                for (k = k1 ; k < k2 ; k++)
                                {
                                    pend = Up [k+1] - 1 ;
                                    /* w = W [k] */
                                    wx = Wx [k] ;
                                    wz = Wz [k] ;
                                    for (p = Up [k] ; p < pend ; p++)
                                    {
                                        i = Ui [p] ;
                                        /* w -= W [i] * conj(U [p]) */
                                        z = Uz ? Uz [p] : 0 ;
                                        MULT_SUB_CONJ (wx, wz,
                                            Wx [i], Wz [i], Ux [p], z) ;
                                    }
                                    /* W [k] = w / conj(ukk) ; */
                                    ukk = Ux [pend] ;
                                    ukkz = Uz ? (-Uz [pend]) : 0 ;
                                    DIV (Wx [k], Wz [k], wx, wz, ukk, ukkz) ;
                                }

                                /* W = L'\W */
                                for (k = k2-1 ; k >= k1 ; k--)
                                {
                                    p = Lp [k] ;
                                    pend = Lp [k+1] ;
                                    /* w = W [k] */
                                    wx = Wx [k] ;
                                    wz = Wz [k] ;
                                    lkk = Lx [p] ;
                                    lkkz = Lz ? (-Lz [p]) : 0 ;
                                    for (p++ ; p < pend ; p++)
                                    {
                                        i = Li [p] ;
                                        /* w -= W [i] * conj (Lx [p]) ; */
                                        z = Lz ? Lz [p] : 0 ;
                                        MULT_SUB_CONJ (wx, wz,
                                            Wx [i], Wz [i], Lx [p], z) ;
                                    }
                                    /* W [k] = w / conj(lkk) ; */
                                    DIV (Wx [k], Wz [k], wx, wz, lkk, lkkz) ;
                                }
                            }
                        }

                        /* -------------------------------------------------- */
                        /* scale, permute, and tranpose: X = (P*(R\W))' */
                        /* -------------------------------------------------- */

                        if (Rs == NULL)
                        {
                            /* no scaling */
                            for (k = 0 ; k < n ; k++)
                            {
                                i = P ? (P [k] - 1) : k ;
                                Xx [i*nrhs] = Wx [k] ;
                                Xz [i*nrhs] = Wz ? (-Wz [k]) : 0 ;
                            }
                        }
                        else
                        {
                            /* with scaling */
                            for (k = 0 ; k < n ; k++)
                            {
                                i = P ? (P [k] - 1) : k ;
                                rs = Rs [k] ;
                                Xx [i*nrhs] = Wx [k] / rs ;
                                Xz [i*nrhs] = Wz ? (-Wz [k] / rs) : 0 ;
                            }
                        }

                        /* -------------------------------------------------- */
                        /* go to the next row of B and X */
                        /* -------------------------------------------------- */

                        Xx++ ;
                        Xz++ ;
                        Bx++ ;
                        if (Bz) Bz++ ;
                    }

                }
                else
                {

                    for (chunk = 0 ; chunk < nrhs ; chunk++)
                    {

                        /* -------------------------------------------------- */
                        /* scale and permute the right hand side, W = P*(R\B) */
                        /* -------------------------------------------------- */

                        if (Rs == NULL)
                        {
                            /* no scaling */
                            for (k = 0 ; k < n ; k++)
                            {
                                i = P ? (P [k] - 1) : k ;
                                Wx [k] = Bx [i] ;
                                Wz [k] = Bz ? Bz [i] : 0 ;
                            }
                        }
                        else
                        {
                            /* with scaling */
                            for (k = 0 ; k < n ; k++)
                            {
                                i = P ? (P [k] - 1) : k ;
                                rs = Rs [k] ;
                                Wx [k] = Bx [i] / rs ;
                                Wz [k] = Bz ? (Bz [i] / rs) : 0 ;
                            }
                        }

                        /* -------------------------------------------------- */
                        /* solve W = (L*U + Off)\W */
                        /* -------------------------------------------------- */

                        for (block = nblocks-1 ; block >= 0 ; block--)
                        {

                            /* ---------------------------------------------- */
                            /* block of size nk, rows/columns k1 to k2-1 */
                            /* ---------------------------------------------- */

                            k1 = R [block] - 1 ;        /* R is 1-based */
                            k2 = R [block+1] - 1 ;
                            nk = k2 - k1 ;

                            /* solve the block system */
                            if (nk == 1)
                            {

                                /* W [k1] /= L(k1,k1) ; */
                                p = Lp [k1] ;
                                s = Lx [p] ;
                                sz = Lz ? Lz [p] : 0 ;
                                DIV (wx, wz, Wx [k1], Wz [k1], s, sz) ;
                                Wx [k1] = wx ;
                                Wz [k1] = wz ;

                                /* W [k1] /= U(k1,k1) ; */
                                p = Up [k1] ;
                                s = Ux [p] ;
                                sz = Uz ? Uz [p] : 0 ;
                                DIV (wx, wz, Wx [k1], Wz [k1], s, sz) ;
                                Wx [k1] = wx ;
                                Wz [k1] = wz ;

                            }
                            else
                            {

                                /* ------------------------------------------ */
                                /* W = L\W and then W=U\W */
                                /* ------------------------------------------ */

                                /* W = L\W */
                                for (k = k1 ; k < k2 ; k++)
                                {
                                    p = Lp [k] ;
                                    pend = Lp [k+1] ;
                                    lkk = Lx [p] ;
                                    lkkz = Lz ? Lz [p] : 0 ;
                                    /* w = W [k] / lkk ; */
                                    DIV (wx, wz, Wx [k], Wz [k], lkk, lkkz) ;
                                    Wx [k] = wx ;
                                    Wz [k] = wz ;
                                    for (p++ ; p < pend ; p++)
                                    {
                                        i = Li [p] ;
                                        /* W [i] -= Lx [p] * w ; */
                                        z = Lz ? Lz [p] : 0 ;
                                        MULT_SUB (Wx [i], Wz [i], Lx [p], z,
                                            wx, wz) ;
                                    }
                                }

                                /* W = U\W */
                                for (k = k2-1 ; k >= k1 ; k--)
                                {
                                    pend = Up [k+1] - 1 ;
                                    ukk = Ux [pend] ;
                                    ukkz = Uz ? Uz [pend] : 0 ;
                                    /* w = W [k] / ukk ; */
                                    DIV (wx, wz, Wx [k], Wz [k], ukk, ukkz) ;
                                    Wx [k] = wx ;
                                    Wz [k] = wz ;
                                    for (p = Up [k] ; p < pend ; p++)
                                    {
                                        i = Ui [p] ;
                                        /* W [i] -= U [p] * w ; */
                                        z = Uz ? Uz [p] : 0 ;
                                        MULT_SUB (Wx [i], Wz [i], Ux [p], z,
                                            wx, wz) ;
                                    }
                                }
                            }

                            /* ---------------------------------------------- */
                            /* block back-substitution for off-diagonal-block */
                            /* ---------------------------------------------- */

                            if (block > 0 && Offp != NULL)
                            {
                                for (k = k1 ; k < k2 ; k++)
                                {
                                    pend = Offp [k+1] ;
                                    wx = Wx [k] ;
                                    wz = Wz [k] ;
                                    for (p = Offp [k] ; p < pend ; p++)
                                    {
                                        i = Offi [p] ;
                                        /* W [Offi [p]] -= Offx [p] * w ; */
                                        z = Offz ? Offz [p] : 0 ;
                                        MULT_SUB (Wx [i], Wz [i], Offx [p], z,
                                            wx, wz) ;
                                    }
                                }
                            }
                        }

                        /* -------------------------------------------------- */
                        /* permute the result, X = Q*W */
                        /* -------------------------------------------------- */

                        for (k = 0 ; k < n ; k++)
                        {
                            i = Q ? (Q [k] - 1) : k ;
                            Xx [i] = Wx [k] ;
                            Xz [i] = Wz [k] ;
                        }

                        /* -------------------------------------------------- */
                        /* go to the next column of B and X */
                        /* -------------------------------------------------- */

                        Xx += n ;
                        Xz += n ;
                        Bx += n ;
                        if (Bz) Bz += n ;
                    }
                }

                /* free workspace */
                mxFree (Wx) ;
                mxFree (Wz) ;

            }
            else
            {

                /* ========================================================== */
                /* === real case ============================================ */
                /* ========================================================== */

                /* create X */
                if (do_transpose)
                {
                    pargout [0] = mxCreateDoubleMatrix (nrhs, n, mxREAL) ;
                }
                else
                {
                    pargout [0] = mxCreateDoubleMatrix (n, nrhs, mxREAL) ;
                }

                Xx = mxGetPr (pargout [0]) ;
                Bx = mxGetPr (B_matlab) ;

                if (do_transpose)
                {

                    /* ------------------------------------------------------ */
                    /* solve in chunks of one row at a time */
                    /* ------------------------------------------------------ */

                    /* get workspace */
                    Wx = mxMalloc (n * sizeof (double)) ;

                    for (chunk = 0 ; chunk < nrhs ; chunk++)
                    {

                        /* -------------------------------------------------- */
                        /* transpose and permute right hand side, W = Q'*B' */
                        /* -------------------------------------------------- */

                        for (k = 0 ; k < n ; k++)
                        {
                            i = Q ? (Q [k] - 1) : k ;
                            Wx [k] = Bx [i*nrhs] ;
                        }

                        /* -------------------------------------------------- */
                        /* solve W = (L*U + Off)'\W */
                        /* -------------------------------------------------- */

                        for (block = 0 ; block < nblocks ; block++)
                        {
                    
                            /* ---------------------------------------------- */
                            /* block of size nk, rows/columns k1 to k2-1 */
                            /* ---------------------------------------------- */

                            k1 = R [block] - 1 ;        /* R is 1-based */
                            k2 = R [block+1] - 1 ;
                            nk = k2 - k1 ;

                            /* ---------------------------------------------- */
                            /* block back-substitution for off-diagonal-block */
                            /* ---------------------------------------------- */

                            if (block > 0 && Offp != NULL)
                            {
                                for (k = k1 ; k < k2 ; k++)
                                {
                                    pend = Offp [k+1] ;
                                    for (p = Offp [k] ; p < pend ; p++)
                                    {
                                        Wx [k] -= Wx [Offi [p]] * Offx [p] ;
                                    }
                                }
                            }

                            /* solve the block system */
                            if (nk == 1)
                            {
                                Wx [k1] /= Lx [Lp [k1]] ;
                                Wx [k1] /= Ux [Up [k1]] ;
                            }
                            else
                            {

                                /* ------------------------------------------ */
                                /* W = U'\W and then W=L'\W */
                                /* ------------------------------------------ */

                                /* W = U'\W */
                                for (k = k1 ; k < k2 ; k++)
                                {
                                    pend = Up [k+1] - 1 ;
                                    for (p = Up [k] ; p < pend ; p++)
                                    {
                                        Wx [k] -= Wx [Ui [p]] * Ux [p] ;
                                    }
                                    Wx [k] /= Ux [pend] ;
                                }

                                /* W = L'\W */
                                for (k = k2-1 ; k >= k1 ; k--)
                                {
                                    p = Lp [k] ;
                                    pend = Lp [k+1] ;
                                    lkk = Lx [p] ;
                                    for (p++ ; p < pend ; p++)
                                    {
                                        Wx [k] -= Wx [Li [p]] * Lx [p] ;
                                    }
                                    Wx [k] /= lkk ;
                                }
                            }
                        }

                        /* -------------------------------------------------- */
                        /* scale, permute, and tranpose: X = (P*(R\W))' */
                        /* -------------------------------------------------- */

                        if (Rs == NULL)
                        {
                            /* no scaling */
                            for (k = 0 ; k < n ; k++)
                            {
                                i = P ? (P [k] - 1) : k ;
                                Xx [i*nrhs] = Wx [k] ;
                            }
                        }
                        else
                        {
                            /* with scaling */
                            for (k = 0 ; k < n ; k++)
                            {
                                i = P ? (P [k] - 1) : k ;
                                rs = Rs [k] ;
                                Xx [i*nrhs] = Wx [k] / rs ;
                            }
                        }

                        /* -------------------------------------------------- */
                        /* go to the next row of B and X */
                        /* -------------------------------------------------- */

                        Xx++ ;
                        Bx++ ;
                    }

                }
                else
                {

                    /* ------------------------------------------------------ */
                    /* solve in chunks of 4 columns at a time */
                    /* ------------------------------------------------------ */

                    /* get workspace */
                    Wx = mxMalloc (n * MAX (4, nrhs) * sizeof (double)) ;

                    for (chunk = 0 ; chunk < nrhs ; chunk += 4)
                    {
                        /* -------------------------------------------------- */
                        /* get the size of the current chunk */
                        /* -------------------------------------------------- */

                        nr = MIN (nrhs - chunk, 4) ;

                        /* -------------------------------------------------- */
                        /* scale and permute the right hand side, W = P*(R\B) */
                        /* -------------------------------------------------- */

                        if (Rs == NULL)
                        {

                            /* no scaling */
                            switch (nr)
                            {

                                case 1:

                                    for (k = 0 ; k < n ; k++)
                                    {
                                        i = P ? (P [k] - 1) : k ;
                                        Wx [k] = Bx [i] ;
                                    }
                                    break ;

                                case 2:

                                    for (k = 0 ; k < n ; k++)
                                    {
                                        i = P ? (P [k] - 1) : k ;
                                        Wx [2*k    ] = Bx [i      ] ;
                                        Wx [2*k + 1] = Bx [i + n  ] ;
                                    }
                                    break ;

                                case 3:

                                    for (k = 0 ; k < n ; k++)
                                    {
                                        i = P ? (P [k] - 1) : k ;
                                        Wx [3*k    ] = Bx [i      ] ;
                                        Wx [3*k + 1] = Bx [i + n  ] ;
                                        Wx [3*k + 2] = Bx [i + n*2] ;
                                    }
                                    break ;

                                case 4:

                                    for (k = 0 ; k < n ; k++)
                                    {
                                        i = P ? (P [k] - 1) : k ;
                                        Wx [4*k    ] = Bx [i      ] ;
                                        Wx [4*k + 1] = Bx [i + n  ] ;
                                        Wx [4*k + 2] = Bx [i + n*2] ;
                                        Wx [4*k + 3] = Bx [i + n*3] ;
                                    }
                                    break ;
                            }

                        }
                        else
                        {

                            switch (nr)
                            {

                                case 1:

                                    for (k = 0 ; k < n ; k++)
                                    {
                                        i = P ? (P [k] - 1) : k ;
                                        rs = Rs [k] ;
                                        Wx [k] = Bx [i] / rs ;
                                    }
                                    break ;

                                case 2:

                                    for (k = 0 ; k < n ; k++)
                                    {
                                        i = P ? (P [k] - 1) : k ;
                                        rs = Rs [k] ;
                                        Wx [2*k    ] = Bx [i      ] / rs ;
                                        Wx [2*k + 1] = Bx [i + n  ] / rs ;
                                    }
                                    break ;

                                case 3:

                                    for (k = 0 ; k < n ; k++)
                                    {
                                        i = P ? (P [k] - 1) : k ;
                                        rs = Rs [k] ;
                                        Wx [3*k    ] = Bx [i      ] / rs ;
                                        Wx [3*k + 1] = Bx [i + n  ] / rs ;
                                        Wx [3*k + 2] = Bx [i + n*2] / rs ;
                                    }
                                    break ;

                                case 4:

                                    for (k = 0 ; k < n ; k++)
                                    {
                                        i = P ? (P [k] - 1) : k ;
                                        rs = Rs [k] ;
                                        Wx [4*k    ] = Bx [i      ] / rs ;
                                        Wx [4*k + 1] = Bx [i + n  ] / rs ;
                                        Wx [4*k + 2] = Bx [i + n*2] / rs ;
                                        Wx [4*k + 3] = Bx [i + n*3] / rs ;
                                    }
                                    break ;
                            }
                        }

                        /* -------------------------------------------------- */
                        /* solve W = (L*U + Off)\W */
                        /* -------------------------------------------------- */

                        for (block = nblocks-1 ; block >= 0 ; block--)
                        {

                            /* ---------------------------------------------- */
                            /* block of size nk is rows/columns k1 to k2-1 */
                            /* ---------------------------------------------- */

                            k1 = R [block] - 1 ;            /* R is 1-based */
                            k2 = R [block+1] - 1 ;
                            nk = k2 - k1 ;

                            /* solve the block system */
                            if (nk == 1)
                            {

                                /* this is not done if L comes from KLU, since
                                   in that case, L is unit lower triangular */
                                s = Lx [Lp [k1]] ;
                                if (s != 1.0) switch (nr)
                                {
                                    case 1:
                                        Wx [k1] /= s ;
                                        break ;
                                    case 2:
                                        Wx [2*k1] /= s ;
                                        Wx [2*k1 + 1] /= s ;
                                        break ;
                                    case 3:
                                        Wx [3*k1] /= s ;
                                        Wx [3*k1 + 1] /= s ;
                                        Wx [3*k1 + 2] /= s ;
                                        break ;
                                    case 4:
                                        Wx [4*k1] /= s ;
                                        Wx [4*k1 + 1] /= s ;
                                        Wx [4*k1 + 2] /= s ;
                                        Wx [4*k1 + 3] /= s ;
                                        break ;
                                }

                                s = Ux [Up [k1]] ;
                                if (s != 1.0) switch (nr)
                                {
                                    case 1:
                                        Wx [k1] /= s ;
                                        break ;
                                    case 2:
                                        Wx [2*k1] /= s ;
                                        Wx [2*k1 + 1] /= s ;
                                        break ;
                                    case 3:
                                        Wx [3*k1] /= s ;
                                        Wx [3*k1 + 1] /= s ;
                                        Wx [3*k1 + 2] /= s ;
                                        break ;
                                    case 4:
                                        Wx [4*k1] /= s ;
                                        Wx [4*k1 + 1] /= s ;
                                        Wx [4*k1 + 2] /= s ;
                                        Wx [4*k1 + 3] /= s ;
                                        break ;
                                }

                            }
                            else
                            {

                                /* ------------------------------------------ */
                                /* W = L\W and then W=U\W */
                                /* ------------------------------------------ */

                                switch (nr)
                                {

                                    case 1:
                                        /* W = L\W */
                                        for (k = k1 ; k < k2 ; k++)
                                        {
                                            p = Lp [k] ;
                                            pend = Lp [k+1] ;
                                            lkk = Lx [p++] ;
                                            x [0] = Wx [k] / lkk ;
                                            Wx [k] = x [0] ;
                                            for ( ; p < pend ; p++)
                                            {
                                                Wx [Li [p]] -= Lx [p] * x [0] ;
                                            }
                                        }

                                        /* W = U\W */
                                        for (k = k2-1 ; k >= k1 ; k--)
                                        {
                                            pend = Up [k+1] - 1 ;
                                            ukk = Ux [pend] ;
                                            x [0] = Wx [k] / ukk ;
                                            Wx [k] = x [0] ;
                                            for (p = Up [k] ; p < pend ; p++)
                                            {
                                                Wx [Ui [p]] -= Ux [p] * x [0] ;
                                            }
                                        }
                                        break ;

                                    case 2:

                                        /* W = L\W */
                                        for (k = k1 ; k < k2 ; k++)
                                        {
                                            p = Lp [k] ;
                                            pend = Lp [k+1] ;
                                            lkk = Lx [p++] ;
                                            x [0] = Wx [2*k    ] / lkk ;
                                            x [1] = Wx [2*k + 1] / lkk ;
                                            Wx [2*k    ] = x [0] ;
                                            Wx [2*k + 1] = x [1] ;
                                            for ( ; p < pend ; p++)
                                            {
                                                i = Li [p] ;
                                                lik = Lx [p] ;
                                                Wx [2*i    ] -= lik * x [0] ;
                                                Wx [2*i + 1] -= lik * x [1] ;
                                            }
                                        }

                                        /* W = U\W */
                                        for (k = k2-1 ; k >= k1 ; k--)
                                        {
                                            pend = Up [k+1] - 1 ;
                                            ukk = Ux [pend] ;
                                            x [0] = Wx [2*k    ] / ukk ;
                                            x [1] = Wx [2*k + 1] / ukk ;
                                            Wx [2*k    ] = x [0] ;
                                            Wx [2*k + 1] = x [1] ;
                                            for (p = Up [k] ; p < pend ; p++)
                                            {
                                                i = Ui [p] ;
                                                uik = Ux [p] ;
                                                Wx [2*i    ] -= uik * x [0] ;
                                                Wx [2*i + 1] -= uik * x [1] ;
                                            }
                                        }
                                        break ;

                                    case 3:

                                        /* W = L\W */
                                        for (k = k1 ; k < k2 ; k++)
                                        {
                                            p = Lp [k] ;
                                            pend = Lp [k+1] ;
                                            lkk = Lx [p++] ;
                                            x [0] = Wx [3*k    ] / lkk ;
                                            x [1] = Wx [3*k + 1] / lkk ;
                                            x [2] = Wx [3*k + 2] / lkk ;
                                            Wx [3*k    ] = x [0] ;
                                            Wx [3*k + 1] = x [1] ;
                                            Wx [3*k + 2] = x [2] ;
                                            for ( ; p < pend ; p++)
                                            {
                                                i = Li [p] ;
                                                lik = Lx [p] ;
                                                Wx [3*i    ] -= lik * x [0] ;
                                                Wx [3*i + 1] -= lik * x [1] ;
                                                Wx [3*i + 2] -= lik * x [2] ;
                                            }
                                        }

                                        /* W = U\W */
                                        for (k = k2-1 ; k >= k1 ; k--)
                                        {
                                            pend = Up [k+1] - 1 ;
                                            ukk = Ux [pend] ;
                                            x [0] = Wx [3*k    ] / ukk ;
                                            x [1] = Wx [3*k + 1] / ukk ;
                                            x [2] = Wx [3*k + 2] / ukk ;
                                            Wx [3*k    ] = x [0] ;
                                            Wx [3*k + 1] = x [1] ;
                                            Wx [3*k + 2] = x [2] ;
                                            for (p = Up [k] ; p < pend ; p++)
                                            {
                                                i = Ui [p] ;
                                                uik = Ux [p] ;
                                                Wx [3*i    ] -= uik * x [0] ;
                                                Wx [3*i + 1] -= uik * x [1] ;
                                                Wx [3*i + 2] -= uik * x [2] ;
                                            }
                                        }
                                        break ;

                                    case 4:

                                        /* W = L\W */
                                        for (k = k1 ; k < k2 ; k++)
                                        {
                                            p = Lp [k] ;
                                            pend = Lp [k+1] ;
                                            lkk = Lx [p++] ;
                                            x [0] = Wx [4*k    ] / lkk ;
                                            x [1] = Wx [4*k + 1] / lkk ;
                                            x [2] = Wx [4*k + 2] / lkk ;
                                            x [3] = Wx [4*k + 3] / lkk ;
                                            Wx [4*k    ] = x [0] ;
                                            Wx [4*k + 1] = x [1] ;
                                            Wx [4*k + 2] = x [2] ;
                                            Wx [4*k + 3] = x [3] ;
                                            for ( ; p < pend ; p++)
                                            {
                                                i = Li [p] ;
                                                lik = Lx [p] ;
                                                Wx [4*i    ] -= lik * x [0] ;
                                                Wx [4*i + 1] -= lik * x [1] ;
                                                Wx [4*i + 2] -= lik * x [2] ;
                                                Wx [4*i + 3] -= lik * x [3] ;
                                            }
                                        }

                                        /* Wx = U\Wx */
                                        for (k = k2-1 ; k >= k1 ; k--)
                                        {
                                            pend = Up [k+1] - 1 ;
                                            ukk = Ux [pend] ;
                                            x [0] = Wx [4*k    ] / ukk ;
                                            x [1] = Wx [4*k + 1] / ukk ;
                                            x [2] = Wx [4*k + 2] / ukk ;
                                            x [3] = Wx [4*k + 3] / ukk ;
                                            Wx [4*k    ] = x [0] ;
                                            Wx [4*k + 1] = x [1] ;
                                            Wx [4*k + 2] = x [2] ;
                                            Wx [4*k + 3] = x [3] ;
                                            for (p = Up [k] ; p < pend ; p++)
                                            {
                                                i = Ui [p] ;
                                                uik = Ux [p] ;
                                                Wx [4*i    ] -= uik * x [0] ;
                                                Wx [4*i + 1] -= uik * x [1] ;
                                                Wx [4*i + 2] -= uik * x [2] ;
                                                Wx [4*i + 3] -= uik * x [3] ;
                                            }
                                        }
                                        break ;
                                }
                            }

                            /* ---------------------------------------------- */
                            /* block back-substitution for off-diagonal-block */
                            /* ---------------------------------------------- */

                            if (block > 0 && Offp != NULL)
                            {
                                switch (nr)
                                {

                                    case 1:

                                        for (k = k1 ; k < k2 ; k++)
                                        {
                                            pend = Offp [k+1] ;
                                            x [0] = Wx [k] ;
                                            for (p = Offp [k] ; p < pend ; p++)
                                            {
                                                Wx [Offi [p]] -= Offx[p] * x[0];
                                            }
                                        }
                                        break ;

                                    case 2:

                                        for (k = k1 ; k < k2 ; k++)
                                        {
                                            pend = Offp [k+1] ;
                                            x [0] = Wx [2*k    ] ;
                                            x [1] = Wx [2*k + 1] ;
                                            for (p = Offp [k] ; p < pend ; p++)
                                            {
                                                i = Offi [p] ;
                                                offik = Offx [p] ;
                                                Wx [2*i    ] -= offik * x [0] ;
                                                Wx [2*i + 1] -= offik * x [1] ;
                                            }
                                        }
                                        break ;

                                    case 3:

                                        for (k = k1 ; k < k2 ; k++)
                                        {
                                            pend = Offp [k+1] ;
                                            x [0] = Wx [3*k    ] ;
                                            x [1] = Wx [3*k + 1] ;
                                            x [2] = Wx [3*k + 2] ;
                                            for (p = Offp [k] ; p < pend ; p++)
                                            {
                                                i = Offi [p] ;
                                                offik = Offx [p] ;
                                                Wx [3*i    ] -= offik * x [0] ;
                                                Wx [3*i + 1] -= offik * x [1] ;
                                                Wx [3*i + 2] -= offik * x [2] ;
                                            }
                                        }
                                        break ;

                                    case 4:

                                        for (k = k1 ; k < k2 ; k++)
                                        {
                                            pend = Offp [k+1] ;
                                            x [0] = Wx [4*k    ] ;
                                            x [1] = Wx [4*k + 1] ;
                                            x [2] = Wx [4*k + 2] ;
                                            x [3] = Wx [4*k + 3] ;
                                            for (p = Offp [k] ; p < pend ; p++)
                                            {
                                                i = Offi [p] ;
                                                offik = Offx [p] ;
                                                Wx [4*i    ] -= offik * x [0] ;
                                                Wx [4*i + 1] -= offik * x [1] ;
                                                Wx [4*i + 2] -= offik * x [2] ;
                                                Wx [4*i + 3] -= offik * x [3] ;
                                            }
                                        }
                                        break ;
                                }
                            }
                        }

                        /* -------------------------------------------------- */
                        /* permute the result, X = Q*W */
                        /* -------------------------------------------------- */

                        switch (nr)
                        {

                            case 1:

                                for (k = 0 ; k < n ; k++)
                                {
                                    i = Q ? (Q [k] - 1) : k ;
                                    Xx [i] = Wx [k] ;
                                }
                                break ;

                            case 2:

                                for (k = 0 ; k < n ; k++)
                                {
                                    i = Q ? (Q [k] - 1) : k ;
                                    Xx [i      ] = Wx [2*k    ] ;
                                    Xx [i + n  ] = Wx [2*k + 1] ;
                                }
                                break ;

                            case 3:

                                for (k = 0 ; k < n ; k++)
                                {
                                    i = Q ? (Q [k] - 1) : k ;
                                    Xx [i      ] = Wx [3*k    ] ;
                                    Xx [i + n  ] = Wx [3*k + 1] ;
                                    Xx [i + n*2] = Wx [3*k + 2] ;
                                }
                                break ;

                            case 4:

                                for (k = 0 ; k < n ; k++)
                                {
                                    i = Q ? (Q [k] - 1) : k ;
                                    Xx [i      ] = Wx [4*k    ] ;
                                    Xx [i + n  ] = Wx [4*k + 1] ;
                                    Xx [i + n*2] = Wx [4*k + 2] ;
                                    Xx [i + n*3] = Wx [4*k + 3] ;
                                }
                                break ;
                        }

                        /* -------------------------------------------------- */
                        /* go to the next chunk of B and X */
                        /* -------------------------------------------------- */

                        Xx += n*4 ;
                        Bx += n*4 ;
                    }
                }

                /* free workspace */
                mxFree (Wx) ;
            }

        }

    }
    else
    {

        /* ------------------------------------------------------------------ */
        /* LU = klu (A) usage; extract factorization */
        /* ------------------------------------------------------------------ */

        /* sort the row indices in each column of L and U */
        if (A_complex)
        {
            klu_zl_sort (Symbolic, Numeric, &Common) ;
        }
        else
        {
            klu_l_sort (Symbolic, Numeric, &Common) ;
        }

        /* L */
        L_matlab = mxCreateSparse (n, n, Numeric->lnz,
            A_complex ? mxCOMPLEX: mxREAL) ;
        Lp = (Long *) mxGetJc (L_matlab) ;
        Li = (Long *) mxGetIr (L_matlab) ;
        Lx = mxGetPr (L_matlab) ;
        Lz = mxGetPi (L_matlab) ;

        /* U */
        U_matlab = mxCreateSparse (n, n, Numeric->unz,
            A_complex ? mxCOMPLEX: mxREAL) ;
        Up = (Long *) mxGetJc (U_matlab) ;
        Ui = (Long *) mxGetIr (U_matlab) ;
        Ux = mxGetPr (U_matlab) ;
        Uz = mxGetPi (U_matlab) ;

        /* p */
        p_matlab = mxCreateNumericMatrix (1, n, mx_int, mxREAL) ;
        P = (Long *) mxGetData (p_matlab) ;

        /* q */
        q_matlab = mxCreateNumericMatrix (1, n, mx_int, mxREAL) ;
        Q = (Long *) mxGetData (q_matlab) ;

        /* R, as a sparse diagonal matrix */
        R_matlab = mxCreateSparse (n, n, n+1, mxREAL) ;
        Rp = (Long *) mxGetJc (R_matlab) ;
        Ri = (Long *) mxGetIr (R_matlab) ;
        Rs = mxGetPr (R_matlab) ;
        for (k = 0 ; k <= n ; k++)
        {
            Rp [k] = k ;
            Ri [k] = k ;
        }

        /* F, off diagonal blocks */
        F_matlab = mxCreateSparse (n, n, Numeric->nzoff,
            A_complex ? mxCOMPLEX: mxREAL) ;
        Offp = (Long *) mxGetJc (F_matlab) ;
        Offi = (Long *) mxGetIr (F_matlab) ;
        Offx = mxGetPr (F_matlab) ;
        Offz = mxGetPi (F_matlab) ;

        /* r, block boundaries */
        nblocks = Symbolic->nblocks ;
        r_matlab = mxCreateNumericMatrix (1, nblocks+1, mx_int, mxREAL) ;
        R = (Long *) mxGetData (r_matlab) ;

        /* extract the LU factorization from KLU Numeric and Symbolic objects */
        if (A_complex)
        {
            klu_zl_extract (Numeric, Symbolic, Lp, Li, Lx, Lz, Up, Ui, Ux, Uz,
                Offp, Offi, Offx, Offz, P, Q, Rs, R, &Common) ;
        }
        else
        {
            klu_l_extract (Numeric, Symbolic, Lp, Li, Lx, Up, Ui, Ux,
                Offp, Offi, Offx, P, Q, Rs, R, &Common) ;
        }

        /* fix p and q for 1-based indexing */
        for (k = 0 ; k < n ; k++)
        {
            P [k]++ ;
            Q [k]++ ;
        }

        /* fix r for 1-based indexing */
        for (k = 0 ; k <= nblocks ; k++)
        {
            R [k]++ ;
        }

        /* create output LU struct */
        pargout [0] = mxCreateStructMatrix (1, 1, 7, LUnames) ;
        mxSetFieldByNumber (pargout [0], 0, 0, L_matlab) ;
        mxSetFieldByNumber (pargout [0], 0, 1, U_matlab) ;
        mxSetFieldByNumber (pargout [0], 0, 2, p_matlab) ;
        mxSetFieldByNumber (pargout [0], 0, 3, q_matlab) ;
        mxSetFieldByNumber (pargout [0], 0, 4, R_matlab) ;
        mxSetFieldByNumber (pargout [0], 0, 5, F_matlab) ;
        mxSetFieldByNumber (pargout [0], 0, 6, r_matlab) ;

        /* ------------------------------------------------------------------ */
        /* free Symbolic and Numeric objects */
        /* ------------------------------------------------------------------ */

        klu_l_free_symbolic (&Symbolic, &Common) ;
        klu_l_free_numeric (&Numeric, &Common) ;
    }
}
