/* ========================================================================== */
/* === KLU DEMO (long integer version) ====================================== */
/* ========================================================================== */

/* Read in a Matrix Market matrix (using CHOLMOD) and solve a linear system.
 * SuiteSparse_long is normally a "long", but it becomes "_int64" on Windows 64. */

#include <math.h>
#include <stdio.h>
#include "klu.h"
#define Long SuiteSparse_long

/* for handling complex matrices */
#define REAL(X,i) (X [2*(i)])
#define IMAG(X,i) (X [2*(i)+1])
#define CABS(X,i) (sqrt (REAL (X,i) * REAL (X,i) + IMAG (X,i) * IMAG (X,i)))

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

/* ========================================================================== */
/* === klu_l_backslash ====================================================== */
/* ========================================================================== */

static Long klu_l_backslash    /* return 1 if successful, 0 otherwise */
(
    /* --- input ---- */
    Long n,             /* A is n-by-n */
    Long *Ap,           /* size n+1, column pointers */
    Long *Ai,           /* size nz = Ap [n], row indices */
    double *Ax,         /* size nz, numerical values */
    Long isreal,        /* nonzero if A is real, 0 otherwise */
    double *B,          /* size n, right-hand-side */

    /* --- output ---- */
    double *X,          /* size n, solution to Ax=b */
    double *R,          /* size n, residual r = b-A*x */

    /* --- scalar output --- */
    Long *lunz,         /* nnz (L+U+F) */
    double *rnorm,      /* norm (b-A*x,1) / norm (A,1) */

    /* --- workspace - */

    klu_l_common *Common    /* default parameters and statistics */
)
{
    double anorm = 0, asum ;
    klu_l_symbolic *Symbolic ;
    klu_l_numeric *Numeric ;
    Long i, j, p ;

    if (!Ap || !Ai || !Ax || !B || !X || !B) return (0) ;

    /* ---------------------------------------------------------------------- */
    /* symbolic ordering and analysis */
    /* ---------------------------------------------------------------------- */

    Symbolic = klu_l_analyze (n, Ap, Ai, Common) ;
    if (!Symbolic) return (0) ;

    if (isreal)
    {

        /* ------------------------------------------------------------------ */
        /* factorization */
        /* ------------------------------------------------------------------ */

        Numeric = klu_l_factor (Ap, Ai, Ax, Symbolic, Common) ;
        if (!Numeric)
        {
            klu_l_free_symbolic (&Symbolic, Common) ;
            return (0) ;
        }

        /* ------------------------------------------------------------------ */
        /* statistics (not required to solve Ax=b) */
        /* ------------------------------------------------------------------ */

        klu_l_rgrowth (Ap, Ai, Ax, Symbolic, Numeric, Common) ;
        klu_l_condest (Ap, Ax, Symbolic, Numeric, Common) ;
        klu_l_rcond (Symbolic, Numeric, Common) ;
        klu_l_flops (Symbolic, Numeric, Common) ;
        *lunz = Numeric->lnz + Numeric->unz - n + 
            ((Numeric->Offp) ? (Numeric->Offp [n]) : 0) ;

        /* ------------------------------------------------------------------ */
        /* solve Ax=b */
        /* ------------------------------------------------------------------ */

        for (i = 0 ; i < n ; i++)
        {
            X [i] = B [i] ;
        }
        klu_l_solve (Symbolic, Numeric, n, 1, X, Common) ;

        /* ------------------------------------------------------------------ */
        /* compute residual, rnorm = norm(b-Ax,1) / norm(A,1) */
        /* ------------------------------------------------------------------ */

        for (i = 0 ; i < n ; i++)
        {
            R [i] = B [i] ;
        }
        for (j = 0 ; j < n ; j++)
        {
            asum = 0 ;
            for (p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                /* R (i) -= A (i,j) * X (j) */
                R [Ai [p]] -= Ax [p] * X [j] ;
                asum += fabs (Ax [p]) ;
            }
            anorm = MAX (anorm, asum) ;
        }
        *rnorm = 0 ;
        for (i = 0 ; i < n ; i++)
        {
            *rnorm = MAX (*rnorm, fabs (R [i])) ;
        }

        /* ------------------------------------------------------------------ */
        /* free numeric factorization */
        /* ------------------------------------------------------------------ */

        klu_l_free_numeric (&Numeric, Common) ;

    }
    else
    {

        /* ------------------------------------------------------------------ */
        /* statistics (not required to solve Ax=b) */
        /* ------------------------------------------------------------------ */

        Numeric = klu_zl_factor (Ap, Ai, Ax, Symbolic, Common) ;
        if (!Numeric)
        {
            klu_l_free_symbolic (&Symbolic, Common) ;
            return (0) ;
        }

        /* ------------------------------------------------------------------ */
        /* statistics */
        /* ------------------------------------------------------------------ */

        klu_zl_rgrowth (Ap, Ai, Ax, Symbolic, Numeric, Common) ;
        klu_zl_condest (Ap, Ax, Symbolic, Numeric, Common) ;
        klu_zl_rcond (Symbolic, Numeric, Common) ;
        klu_zl_flops (Symbolic, Numeric, Common) ;
        *lunz = Numeric->lnz + Numeric->unz - n + 
            ((Numeric->Offp) ? (Numeric->Offp [n]) : 0) ;

        /* ------------------------------------------------------------------ */
        /* solve Ax=b */
        /* ------------------------------------------------------------------ */

        for (i = 0 ; i < 2*n ; i++)
        {
            X [i] = B [i] ;
        }
        klu_zl_solve (Symbolic, Numeric, n, 1, X, Common) ;

        /* ------------------------------------------------------------------ */
        /* compute residual, rnorm = norm(b-Ax,1) / norm(A,1) */
        /* ------------------------------------------------------------------ */

        for (i = 0 ; i < 2*n ; i++)
        {
            R [i] = B [i] ;
        }
        for (j = 0 ; j < n ; j++)
        {
            asum = 0 ;
            for (p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                /* R (i) -= A (i,j) * X (j) */
                i = Ai [p] ;
                REAL (R,i) -= REAL(Ax,p) * REAL(X,j) - IMAG(Ax,p) * IMAG(X,j) ;
                IMAG (R,i) -= IMAG(Ax,p) * REAL(X,j) + REAL(Ax,p) * IMAG(X,j) ;
                asum += CABS (Ax, p) ;
            }
            anorm = MAX (anorm, asum) ;
        }
        *rnorm = 0 ;
        for (i = 0 ; i < n ; i++)
        {
            *rnorm = MAX (*rnorm, CABS (R, i)) ;
        }

        /* ------------------------------------------------------------------ */
        /* free numeric factorization */
        /* ------------------------------------------------------------------ */

        klu_zl_free_numeric (&Numeric, Common) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free symbolic analysis, and residual */
    /* ---------------------------------------------------------------------- */

    klu_l_free_symbolic (&Symbolic, Common) ;
    return (1) ;
}


/* ========================================================================== */
/* === klu_l_demo =========================================================== */
/* ========================================================================== */

/* Given a sparse matrix A, set up a right-hand-side and solve X = A\b */

static void klu_l_demo (Long n, Long *Ap, Long *Ai, double *Ax, Long isreal)
{
    double rnorm ;
    klu_l_common Common ;
    double *B, *X, *R ;
    Long i, lunz ;

    printf ("KLU: %s, version: %d.%d.%d\n", KLU_DATE, KLU_MAIN_VERSION,
        KLU_SUB_VERSION, KLU_SUBSUB_VERSION) ;

    /* ---------------------------------------------------------------------- */
    /* set defaults */
    /* ---------------------------------------------------------------------- */

    klu_l_defaults (&Common) ;

    /* ---------------------------------------------------------------------- */
    /* create a right-hand-side */
    /* ---------------------------------------------------------------------- */

    if (isreal)
    {
        /* B = 1 + (1:n)/n */
        B = klu_l_malloc (n, sizeof (double), &Common) ;
        X = klu_l_malloc (n, sizeof (double), &Common) ;
        R = klu_l_malloc (n, sizeof (double), &Common) ;
        if (B)
        {
            for (i = 0 ; i < n ; i++)
            {
                B [i] = 1 + ((double) i+1) / ((double) n) ;
            }
        }
    }
    else
    {
        /* real (B) = 1 + (1:n)/n, imag(B) = (n:-1:1)/n */
        B = klu_l_malloc (n, 2 * sizeof (double), &Common) ;
        X = klu_l_malloc (n, 2 * sizeof (double), &Common) ;
        R = klu_l_malloc (n, 2 * sizeof (double), &Common) ;
        if (B)
        {
            for (i = 0 ; i < n ; i++)
            {
                REAL (B, i) = 1 + ((double) i+1) / ((double) n) ;
                IMAG (B, i) = ((double) n-i) / ((double) n) ;
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* X = A\b using KLU and print statistics */
    /* ---------------------------------------------------------------------- */

    if (!klu_l_backslash (n, Ap, Ai, Ax, isreal, B, X, R, &lunz, &rnorm,
        &Common))
    {
        printf ("KLU failed\n") ;
    }
    else
    {
        printf ("n %ld nnz(A) %ld nnz(L+U+F) %ld resid %g\n"
            "recip growth %g condest %g rcond %g flops %g\n",
            n, Ap [n], lunz, rnorm, Common.rgrowth, Common.condest,
            Common.rcond, Common.flops) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free the problem */
    /* ---------------------------------------------------------------------- */

    if (isreal)
    {
        klu_l_free (B, n, sizeof (double), &Common) ;
        klu_l_free (X, n, sizeof (double), &Common) ;
        klu_l_free (R, n, sizeof (double), &Common) ;
    }
    else
    {
        klu_l_free (B, 2*n, sizeof (double), &Common) ;
        klu_l_free (X, 2*n, sizeof (double), &Common) ;
        klu_l_free (R, 2*n, sizeof (double), &Common) ;
    }
    printf ("peak memory usage: %g bytes\n\n", (double) (Common.mempeak)) ;
}


/* ========================================================================== */
/* === main ================================================================= */
/* ========================================================================== */

/* Read in a sparse matrix in Matrix Market format using CHOLMOD, and then
 * solve Ax=b with KLU.  Note that CHOLMOD is only used to read the matrix. */

#include "cholmod.h"

int main (void)
{
    cholmod_sparse *A ;
    cholmod_common ch ;
    cholmod_l_start (&ch) ;
    A = cholmod_l_read_sparse (stdin, &ch) ;
    if (A)
    {
        if (A->nrow != A->ncol || A->stype != 0
            || (!(A->xtype == CHOLMOD_REAL || A->xtype == CHOLMOD_COMPLEX)))
        {
            printf ("invalid matrix\n") ;
        }
        else
        {
            klu_l_demo (A->nrow, A->p, A->i, A->x, A->xtype == CHOLMOD_REAL) ;
        }
        cholmod_l_free_sparse (&A, &ch) ;
    }
    cholmod_l_finish (&ch) ;
    return (0) ;
}
