/* ========================================================================== */
/* === KLU test ============================================================= */
/* ========================================================================== */

/* Exhaustive test for KLU and BTF (int, long, real, and complex versions) */

#include <string.h>
#include "cholmod.h"
#include "klu_cholmod.h"
#include "klu_internal.h"

#define ID Int_id

#define NRHS 6

#define HALT { fprintf (stderr, "Test failure: %d\n", __LINE__) ; abort () ; }
#define OK(a) { if (!(a)) HALT ; }
#define FAIL(a) { if (a) HALT ; }

#define MAX(a,b) (((a) > (b)) ?  (a) : (b))


#ifdef DLONG

#define klu_z_scale klu_zl_scale
#define klu_z_solve klu_zl_solve
#define klu_z_tsolve klu_zl_tsolve
#define klu_z_free_numeric klu_zl_free_numeric
#define klu_z_factor klu_zl_factor
#define klu_z_refactor klu_zl_refactor
#define klu_z_lsolve klu_zl_lsolve
#define klu_z_ltsolve klu_zl_ltsolve
#define klu_z_usolve klu_zl_usolve
#define klu_z_utsolve klu_zl_utsolve
#define klu_z_defaults klu_zl_defaults
#define klu_z_rgrowth klu_zl_rgrowth
#define klu_z_rcond klu_zl_rcond
#define klu_z_extract klu_zl_extract
#define klu_z_condest klu_zl_condest
#define klu_z_flops klu_zl_flops

#define klu_scale klu_l_scale
#define klu_solve klu_l_solve
#define klu_tsolve klu_l_tsolve
#define klu_free_numeric klu_l_free_numeric
#define klu_factor klu_l_factor
#define klu_refactor klu_l_refactor
#define klu_lsolve klu_l_lsolve
#define klu_ltsolve klu_l_ltsolve
#define klu_usolve klu_l_usolve
#define klu_utsolve klu_l_utsolve
#define klu_defaults klu_l_defaults
#define klu_rgrowth klu_l_rgrowth
#define klu_rcond klu_l_rcond
#define klu_extract klu_l_extract
#define klu_condest klu_l_condest
#define klu_flops klu_l_flops

#define klu_analyze klu_l_analyze
#define klu_analyze_given klu_l_analyze_given
#define klu_malloc klu_l_malloc
#define klu_free klu_l_free
#define klu_realloc klu_l_realloc
#define klu_free_symbolic klu_l_free_symbolic
#define klu_free_numeric klu_l_free_numeric
#define klu_defaults klu_l_defaults

#define klu_cholmod klu_l_cholmod

#endif


#ifdef DLONG

#define CHOLMOD_print_sparse cholmod_l_print_sparse
#define CHOLMOD_print_dense cholmod_l_print_dense
#define CHOLMOD_copy_sparse cholmod_l_copy_sparse
#define CHOLMOD_copy_dense cholmod_l_copy_dense
#define CHOLMOD_transpose cholmod_l_transpose
#define CHOLMOD_sdmult cholmod_l_sdmult
#define CHOLMOD_norm_dense cholmod_l_norm_dense
#define CHOLMOD_norm_sparse cholmod_l_norm_sparse
#define CHOLMOD_free_sparse cholmod_l_free_sparse
#define CHOLMOD_free_dense cholmod_l_free_dense
#define CHOLMOD_start cholmod_l_start
#define CHOLMOD_read_sparse cholmod_l_read_sparse
#define CHOLMOD_allocate_dense cholmod_l_allocate_dense
#define CHOLMOD_finish cholmod_l_finish

#else

#define CHOLMOD_print_sparse cholmod_print_sparse
#define CHOLMOD_print_dense cholmod_print_dense
#define CHOLMOD_copy_sparse cholmod_copy_sparse
#define CHOLMOD_copy_dense cholmod_copy_dense
#define CHOLMOD_transpose cholmod_transpose
#define CHOLMOD_sdmult cholmod_sdmult
#define CHOLMOD_norm_dense cholmod_norm_dense
#define CHOLMOD_norm_sparse cholmod_norm_sparse
#define CHOLMOD_free_sparse cholmod_free_sparse
#define CHOLMOD_free_dense cholmod_free_dense
#define CHOLMOD_start cholmod_start
#define CHOLMOD_read_sparse cholmod_read_sparse
#define CHOLMOD_allocate_dense cholmod_allocate_dense
#define CHOLMOD_finish cholmod_finish

#endif

/* ========================================================================== */
/* === random numbers ======================================================= */
/* ========================================================================== */

#define MY_RAND_MAX 32767

static unsigned long next = 1 ;

static Int my_rand (void)
{
   next = next * 1103515245 + 12345 ;
   return ((unsigned)(next/65536) % (MY_RAND_MAX+1)) ;
}

static void my_srand (unsigned seed)
{
   next = seed ;
}

/* ========================================================================== */
/* === memory management ==================================================== */
/* ========================================================================== */

void *my_malloc (size_t size) ;
void *my_calloc (size_t n, size_t size) ;
void *my_realloc (void *p, size_t size) ;
void my_free (void *p) ;

Int my_tries = -1 ;

void *my_malloc (size_t size)
{
    if (my_tries == 0) return (NULL) ;      /* pretend to fail */
    if (my_tries > 0) my_tries-- ;
    return (malloc (size)) ;
}

void *my_calloc (size_t n, size_t size)
{
    if (my_tries == 0) return (NULL) ;      /* pretend to fail */
    if (my_tries > 0) my_tries-- ;
    return (calloc (n, size)) ;
}

void *my_realloc (void *p, size_t size)
{
    if (my_tries == 0) return (NULL) ;      /* pretend to fail */
    if (my_tries > 0) my_tries-- ;
    return (realloc (p, size)) ;
}

void my_free (void *p)
{
    if (p) free (p) ;
}

static void normal_memory_handler ( void )
{
    SuiteSparse_config.malloc_func = malloc ;
    SuiteSparse_config.calloc_func = calloc ;
    SuiteSparse_config.realloc_func = realloc ;
    SuiteSparse_config.free_func = free ;

    my_tries = -1 ;
}

static void test_memory_handler ( void )
{
    SuiteSparse_config.malloc_func = my_malloc ;
    SuiteSparse_config.calloc_func = my_calloc ;
    SuiteSparse_config.realloc_func = my_realloc ;
    SuiteSparse_config.free_func = my_free ;
    my_tries = -1 ;
}


/* ========================================================================== */
/* === print_sparse ========================================================= */
/* ========================================================================== */

/* print a sparse matrix */

static void print_sparse (Int n, Int isreal, Int *Ap, Int *Ai, double *Ax,
    double *Az)
{
    double ax, az ;
    Int i, j, p ;
    for (j = 0 ; j < n ; j++)
    {
        printf ("column "ID":\n", j) ;
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            i = Ai [p] ;
            if (isreal)
            {
                ax = Ax [p] ;
                az = 0 ;
            }
            else if (Az)
            {
                /* split complex */
                ax = Ax [p] ;
                az = Az [p] ;
            }
            else
            {
                /* merged complex */
                ax = Ax [2*p  ] ;
                az = Ax [2*p+1] ;
            }
            printf ("  row "ID" : %g", i, ax) ;
            if (!isreal)
            {
                printf (" + (%g)i", az) ;
            }
            printf ("\n") ;
        }
    }
    fflush (stdout) ;
}


/* ========================================================================== */
/* === print_int ============================================================ */
/* ========================================================================== */

/* print an Int vector */

static void print_int (Int n, Int *P)
{
    Int j ;
    for (j = 0 ; j < n ; j++)
    {
        printf (" "ID" : "ID"\n", j, P [j]) ;
    }
    fflush (stdout) ;
}


/* ========================================================================== */
/* === print_double ========================================================= */
/* ========================================================================== */

/* print a double vector */

static void print_double (Int n, double *X)
{
    Int j ;
    for (j = 0 ; j < n ; j++)
    {
        printf (" "ID" : %g\n", j, X [j]) ;
    }
    fflush (stdout) ;
}


/* ========================================================================== */
/* === ludump =============================================================== */
/* ========================================================================== */

/* extract and print the LU factors */

static void ludump (KLU_symbolic *Symbolic, KLU_numeric *Numeric, Int isreal,
    cholmod_common *ch, KLU_common *Common)
{
    Int *Lp, *Li, *Up, *Ui, *Fp, *Fi, *P, *Q, *R ;
    double *Lx, *Ux, *Fx, *Lz, *Uz, *Fz, *Rs ;
    Int n, lnz, unz, fnz, nb, result ;

    if (Symbolic == NULL || Numeric == NULL)
    {
        return ;
    }

    n = Symbolic->n ;
    lnz = Numeric->lnz ;
    unz = Numeric->unz ;
    fnz = Numeric->Offp [n] ;
    nb = Symbolic->nblocks ;

    printf ("n "ID" lnz "ID" unz "ID" fnz "ID" nblocks "ID" isreal "ID"\n",
        n, lnz, unz, fnz, nb, isreal) ;
    fflush (stdout) ;

    Lp = malloc ((n+1) * sizeof (Int)) ;
    Li = malloc (lnz * sizeof (Int)) ;
    Lx = malloc (lnz * sizeof (double)) ;
    Lz = malloc (lnz * sizeof (double)) ;

    Up = malloc ((n+1) * sizeof (Int)) ;
    Ui = malloc (unz * sizeof (Int)) ;
    Ux = malloc (unz * sizeof (double)) ;
    Uz = malloc (unz * sizeof (double)) ;

    Fp = malloc ((n+1) * sizeof (Int)) ;
    Fi = malloc (fnz * sizeof (Int)) ;
    Fx = malloc (fnz * sizeof (double)) ;
    Fz = malloc (fnz * sizeof (double)) ;

    P = malloc (n * sizeof (Int)) ;
    Q = malloc (n * sizeof (Int)) ;
    Rs = malloc (n * sizeof (double)) ;
    R = malloc ((nb+1) * sizeof (double)) ;

    if (isreal)
    {
        result = klu_extract (Numeric, Symbolic, Lp, Li, Lx,
            Up, Ui, Ux, Fp, Fi, Fx, P, Q, Rs, R, Common) ;
    }
    else
    {
        result = klu_z_extract (Numeric, Symbolic, Lp, Li, Lx, Lz,
            Up, Ui, Ux, Uz, Fp, Fi, Fx, Fz, P, Q, Rs, R, Common) ;
    }

    if (my_tries != 0) OK (result) ;

    if (ch->print >= 5)
    {
        printf ("------ L:\n") ; print_sparse (n, isreal, Lp, Li, Lx, Lz) ;
        printf ("------ U:\n") ; print_sparse (n, isreal, Up, Ui, Ux, Uz) ;
        printf ("------ F:\n") ; print_sparse (n, isreal, Fp, Fi, Fx, Fz) ;
        printf ("------ P:\n") ; print_int (n, P) ;
        printf ("------ Q:\n") ; print_int (n, Q) ;
        printf ("------ Rs:\n") ; print_double (n, Rs) ;
        printf ("------ R:\n") ; print_int (nb+1, R) ;
    }

    free (Lp) ;
    free (Li) ;
    free (Lx) ;
    free (Lz) ;

    free (Up) ;
    free (Ui) ;
    free (Ux) ;
    free (Uz) ;

    free (Fp) ;
    free (Fi) ;
    free (Fx) ;
    free (Fz) ;

    free (P) ;
    free (Q) ;
    free (Rs) ;
    free (R) ;
}


/* ========================================================================== */
/* === randperm ============================================================= */
/* ========================================================================== */

/* return a random permutation vector */

static Int *randperm (Int n, Int seed)
{
    Int *p, k, j, t ;
    p = malloc (n * sizeof (Int)) ;
    for (k = 0 ; k < n ; k++)
    {
        p [k] = k ;
    }
    my_srand (seed) ;                   /* get new random number seed */
    for (k = 0 ; k < n ; k++)
    {
        j = k + (my_rand ( ) % (n-k)) ; /* j = my_rand in range k to n-1 */
        t = p [j] ;                     /* swap p[k] and p[j] */
        p [j] = p [k] ;
        p [k] = t ;
    }
    return (p) ;
}


/* ========================================================================== */
/* === do_1_solve =========================================================== */
/* ========================================================================== */

static double do_1_solve (cholmod_sparse *A, cholmod_dense *B,
    cholmod_dense *Xknown, Int *Puser, Int *Quser,
    KLU_common *Common, cholmod_common *ch, Int *isnan)
{
    Int *Ai, *Ap ;
    double *Ax, *Xknownx, *Xx, *Ax2, *Axx ;
    KLU_symbolic *Symbolic = NULL ; 
    KLU_numeric *Numeric = NULL ;
    cholmod_dense *X = NULL, *R = NULL ;
    cholmod_sparse *AT = NULL, *A2 = NULL, *AT2 = NULL ;
    double one [2], minusone [2],
        rnorm, anorm, xnorm, relresid, relerr, err = 0. ;
    Int i, j, nrhs2, isreal, n, nrhs, transpose, step, k, save, tries ;

    printf ("\ndo_1_solve: btf "ID" maxwork %g scale "ID" ordering "ID" user: "
        ID" P,Q: %d halt: "ID"\n",
        Common->btf, Common->maxwork, Common->scale, Common->ordering,
        Common->user_data ?  (*((Int *) Common->user_data)) : -1,
        (Puser != NULL || Quser != NULL), Common->halt_if_singular) ;
    fflush (stdout) ;
    fflush (stderr) ;

    CHOLMOD_print_sparse (A, "A", ch) ;
    CHOLMOD_print_dense (B, "B", ch) ;

    Ap = A->p ;
    Ai = A->i ;
    Ax = A->x ;
    n = A->nrow ;
    isreal = (A->xtype == CHOLMOD_REAL) ;
    /* Bx = B->x ; */
    Xknownx = Xknown->x ;
    nrhs = B->ncol ;

    one [0] = 1 ;
    one [1] = 0 ;

    minusone [0] = -1 ;
    minusone [1] = 0 ;

    /* ---------------------------------------------------------------------- */
    /* symbolic analysis */
    /* ---------------------------------------------------------------------- */

    Symbolic = NULL ;
    my_tries = 0 ;
    for (tries = 0 ; Symbolic == NULL && my_tries == 0 ; tries++)
    {
        my_tries = tries ;
        if (Puser != NULL || Quser != NULL)
        {
            Symbolic = klu_analyze_given (n, Ap, Ai, Puser, Quser, Common) ;
        }
        else
        {
            Symbolic = klu_analyze (n, Ap, Ai, Common) ;
        }
    }
    printf ("sym try "ID" btf "ID" ordering "ID"\n",
        tries, Common->btf, Common->ordering) ;
    if (Symbolic == NULL)
    {
        printf ("Symbolic is null\n") ;
        return (998) ;
    }
    my_tries = -1 ;

    /* create a modified version of A */

    A2 = CHOLMOD_copy_sparse (A, ch) ;
    Ax2 = A2->x ;
    my_srand (42) ;
    for (k = 0 ; k < Ap [n] * (isreal ? 1:2) ; k++)
    {
        Ax2 [k] = Ax [k] * 
            (1 + 1e-4 * ((double) my_rand ( )) / ((double) MY_RAND_MAX)) ;
    }

    AT = isreal ? NULL : CHOLMOD_transpose (A, 1, ch) ;
    AT2 = isreal ? NULL : CHOLMOD_transpose (A2, 1, ch) ;

    /* ---------------------------------------------------------------------- */
    /* factorize then solve */
    /* ---------------------------------------------------------------------- */

    for (step = 1 ; step <= 3 ; step++)
    {
        printf ("step: "ID"\n", step) ;
        fflush (stdout) ;

        /* ------------------------------------------------------------------ */
        /* factorization or refactorization */
        /* ------------------------------------------------------------------ */

        /* step 1: factor
           step 2: refactor with same A
           step 3: refactor with modified A, and scaling forced on
           and solve each time
        */

        if (step == 1)
        {
            /* numeric factorization */

            Numeric = NULL ;
            my_tries = 0 ;
            for (tries = 0 ; Numeric == NULL && my_tries == 0 ; tries++)
            {
                my_tries = tries ;
                if (isreal)
                {
                    Numeric = klu_factor (Ap, Ai, Ax, Symbolic, Common) ;
                }
                else
                {
                    Numeric = klu_z_factor (Ap, Ai, Ax, Symbolic, Common) ;
                }
            }
            printf ("num try "ID" btf "ID"\n", tries, Common->btf) ;
            my_tries = -1 ;

            if (Common->status == KLU_OK ||
               (Common->status == KLU_SINGULAR && !Common->halt_if_singular))
            {
                OK (Numeric) ;
            }
            else
            {
                FAIL (Numeric) ;
            }

            if (Common->status < KLU_OK)
            {
                printf ("factor failed: "ID"\n", Common->status) ;
            }

        }
        else if (step == 2)
        {

            /* numeric refactorization with same values, same scaling */
            if (isreal)
            {
                klu_refactor (Ap, Ai, Ax, Symbolic, Numeric, Common) ;
            }
            else
            {
                klu_z_refactor (Ap, Ai, Ax, Symbolic, Numeric, Common) ;
            }

        }
        else
        {

            /* numeric refactorization with different values */
            save = Common->scale ;
            if (Common->scale == 0)
            {
                Common->scale = 1 ;
            }
            for (tries = 0 ; tries <= 1 ; tries++)
            {
                my_tries = tries ;
                if (isreal)
                {
                    klu_refactor (Ap, Ai, Ax2, Symbolic, Numeric, Common) ;
                }
                else
                {
                    klu_z_refactor (Ap, Ai, Ax2, Symbolic, Numeric, Common) ;
                }
            }
            my_tries = -1 ;
            Common->scale = save ;
        }

        if (Common->status == KLU_SINGULAR)
        {
            printf ("# singular column : "ID"\n", Common->singular_col) ;
        }

        /* ------------------------------------------------------------------ */
        /* diagnostics */
        /* ------------------------------------------------------------------ */

        Axx = (step == 3) ? Ax2 : Ax ;

        if (isreal)
        {
            klu_rgrowth (Ap, Ai, Axx, Symbolic, Numeric, Common) ;
            klu_condest (Ap, Axx, Symbolic, Numeric, Common) ;
            klu_rcond (Symbolic, Numeric, Common) ;
            klu_flops (Symbolic, Numeric, Common) ;
        }
        else
        {
            klu_z_rgrowth (Ap, Ai, Axx, Symbolic, Numeric, Common) ;
            klu_z_condest (Ap, Axx, Symbolic, Numeric, Common) ;
            klu_z_rcond (Symbolic, Numeric, Common) ;
            klu_z_flops (Symbolic, Numeric, Common) ;
        }

        printf ("growth %g condest %g rcond %g flops %g\n",
            Common->rgrowth, Common->condest, Common->rcond, Common->flops) ;

        ludump (Symbolic, Numeric, isreal, ch, Common) ;

        if (Numeric == NULL || Common->status < KLU_OK)
        {
            continue ;
        }

        /* ------------------------------------------------------------------ */
        /* solve */
        /* ------------------------------------------------------------------ */

        /* forward/backsolve to solve A*X=B or A'*X=B */ 
        for (transpose = (isreal ? 0 : -1) ; transpose <= 1 ; transpose++)
        {

            for (nrhs2 = 1 ; nrhs2 <= nrhs ; nrhs2++)
            {
                /* mangle B so that it has only nrhs2 columns */
                B->ncol = nrhs2 ;

                X = CHOLMOD_copy_dense (B, ch) ;
                CHOLMOD_print_dense (X, "X before solve", ch) ;
                Xx = X->x ;

                if (isreal)
                {
                    if (transpose)
                    {
                        /* solve A'x=b */
                        klu_tsolve (Symbolic, Numeric, n, nrhs2, Xx, Common) ;
                    }
                    else
                    {
                        /* solve A*x=b */
                        klu_solve (Symbolic, Numeric, n, nrhs2, Xx, Common) ;
                    }
                }
                else
                {
                    if (transpose)
                    {
                        /* solve A'x=b (if 1) or A.'x=b (if -1) */
                        klu_z_tsolve (Symbolic, Numeric, n, nrhs2, Xx,
                            (transpose == 1), Common) ;
                    }
                    else
                    {
                        /* solve A*x=b */
                        klu_z_solve (Symbolic, Numeric, n, nrhs2, Xx, Common) ;
                    }
                }

                CHOLMOD_print_dense (X, "X", ch) ;

                /* compute the residual, R = B-A*X, B-A'*X, or B-A.'*X */
                R = CHOLMOD_copy_dense (B, ch) ;
                if (transpose == -1)
                {
                    /* R = B-A.'*X (use A.' explicitly) */
                    CHOLMOD_sdmult ((step == 3) ? AT2 : AT,
                        0, minusone, one, X, R, ch) ;
                }
                else
                {
                    /* R = B-A*X or B-A'*X */
                    CHOLMOD_sdmult ((step == 3) ? A2 :A,
                        transpose, minusone, one, X, R, ch) ;
                }

                CHOLMOD_print_dense (R, "R", ch) ;

                /* compute the norms of R, A, X, and B */
                rnorm = CHOLMOD_norm_dense (R, 1, ch) ;
                anorm = CHOLMOD_norm_sparse ((step == 3) ? A2 : A, 1, ch) ;
                xnorm = CHOLMOD_norm_dense (X, 1, ch) ;
                /* bnorm = CHOLMOD_norm_dense (B, 1, ch) ; */

                CHOLMOD_free_dense (&R, ch) ;

                /* relative residual = norm (r) / (norm (A) * norm (x)) */
                relresid = rnorm ;
                if (anorm > 0)
                {
                    relresid /= anorm ;
                }
                if (xnorm > 0)
                {
                    relresid /= xnorm ;
                }

                if (SCALAR_IS_NAN (relresid))
                {
                    *isnan = TRUE ;
                }
                else
                {
                    err = MAX (err, relresid) ;
                }

                /* relative error = norm (x - xknown) / norm (xknown) */
                /* overwrite X with X - Xknown */
                if (transpose || step == 3)
                {
                    /* not computed */
                    relerr = -1 ;
                }
                else
                {
                    for (j = 0 ; j < nrhs2 ; j++)
                    {
                        for (i = 0 ; i < n ; i++)
                        {
                            if (isreal)
                            {
                                Xx [i+j*n] -= Xknownx [i+j*n] ;
                            }
                            else
                            {
                                Xx [2*(i+j*n)  ] -= Xknownx [2*(i+j*n)  ] ;
                                Xx [2*(i+j*n)+1] -= Xknownx [2*(i+j*n)+1] ;
                            }
                        }
                    }
                    relerr = CHOLMOD_norm_dense (X, 1, ch) ;
                    xnorm = CHOLMOD_norm_dense (Xknown, 1, ch) ;
                    if (xnorm > 0)
                    {
                        relerr /= xnorm ;
                    }

                    if (SCALAR_IS_NAN (relerr))
                    {
                        *isnan = TRUE ;
                    }
                    else
                    {
                        err = MAX (relerr, err) ;
                    }

                }

                CHOLMOD_free_dense (&X, ch) ;

                printf (ID" "ID" relresid %10.3g   relerr %10.3g %g\n", 
                    transpose, nrhs2, relresid, relerr, err) ;

                B->ncol = nrhs ;    /* restore B */
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* free factorization and temporary matrices, and return */
    /* ---------------------------------------------------------------------- */

    klu_free_symbolic (&Symbolic, Common) ;
    if (isreal)
    {
        klu_free_numeric (&Numeric, Common) ;
    }
    else
    {
        klu_z_free_numeric (&Numeric, Common) ;
    }
    CHOLMOD_free_sparse (&A2, ch) ;
    CHOLMOD_free_sparse (&AT, ch) ;
    CHOLMOD_free_sparse (&AT2, ch) ;
    fflush (stdout) ;
    fflush (stderr) ;
    return (err) ;
}


/* ========================================================================== */
/* === do_solves ============================================================ */
/* ========================================================================== */

/* test KLU with many options */

static double do_solves (cholmod_sparse *A, cholmod_dense *B, cholmod_dense *X,
    Int *Puser, Int *Quser, KLU_common *Common, cholmod_common *ch, Int *isnan)
{
    double err, maxerr = 0 ;
    Int n = A->nrow, sflag ;
    *isnan = FALSE ;

    /* ---------------------------------------------------------------------- */
    /* test KLU with the system A*X=B and default options */
    /* ---------------------------------------------------------------------- */

    maxerr = do_1_solve (A, B, X, NULL, NULL, Common, ch, isnan) ;

    /* ---------------------------------------------------------------------- */
    /* test with non-default options */
    /* ---------------------------------------------------------------------- */

    Common->user_order = klu_cholmod ;
    for (Common->btf = 0 ; Common->btf <= 2 ; Common->btf++)
    {
        Common->maxwork = (Common->btf == 2) ? 0.001 : 0 ;

        for (Common->halt_if_singular = 0 ; Common->halt_if_singular <= 1 ;
            Common->halt_if_singular++)
        {
            for (Common->scale = 0 ; Common->scale <= 2 ; Common->scale++)

            {
                fprintf (stderr, ".") ;
                fflush (stderr) ;

                /* orderings: 0: AMD, 1: COLAMD, 2: natural, 3: user function */
                for (Common->ordering = 0 ; Common->ordering <= 3 ;
                     Common->ordering++)
                {
                    err = do_1_solve (A, B, X, NULL, NULL, Common, ch, isnan) ;
                    maxerr = MAX (maxerr, err) ;
                }

                /* user-ordering, unsymmetric case */
                Common->ordering = 3 ;
                Common->user_data = &sflag ;
                sflag = 0 ;
                err = do_1_solve (A, B, X, NULL, NULL, Common, ch, isnan) ;
                maxerr = MAX (maxerr, err) ;
                Common->user_data = NULL ;

                /* Puser and Quser, but only for small matrices */
                Common->ordering = 2 ;
                if (n < 200)
                {
                    err = do_1_solve (A, B, X, Puser, Quser, Common, ch, isnan);
                    maxerr = MAX (maxerr, err) ;
                }
            }
        }
    }

    /* restore defaults */
    Common->btf = TRUE ;
    Common->maxwork = 0 ;
    Common->ordering = 0 ;
    Common->scale = -1 ;
    Common->halt_if_singular = TRUE ;
    Common->user_order = NULL ; 

    my_tries = -1 ;
    return (maxerr) ;
}


/* ========================================================================== */
/* === main ================================================================= */
/* ========================================================================== */

int main (void)
{
    KLU_common Common ;
    cholmod_sparse *A, *A2 ;
    cholmod_dense *X, *B ;
    cholmod_common ch ;
    Int *Ap, *Ai, *Puser, *Quser, *Gunk ;
    double *Ax, *Xx, *A2x ;
    double one [2], zero [2], xsave, maxerr ;
    Int n, i, j, nz, save, isreal, k, isnan ;
    KLU_symbolic *Symbolic, *Symbolic2 ;
    KLU_numeric *Numeric ;

    one [0] = 1 ;
    one [1] = 0 ;
    zero [0] = 0 ;
    zero [1] = 0 ;

    printf ("klu test: -------------------------------------------------\n") ;
    OK (klu_defaults (&Common)) ;
    CHOLMOD_start (&ch) ;
    ch.print = 0 ;
    normal_memory_handler ( ) ;

    /* ---------------------------------------------------------------------- */
    /* read in a sparse matrix from stdin */
    /* ---------------------------------------------------------------------- */

    A = CHOLMOD_read_sparse (stdin, &ch) ;

    if (A->nrow != A->ncol || A->stype != 0)
    {
        fprintf (stderr, "error: only square unsymmetric matrices handled\n") ;
        CHOLMOD_free_sparse (&A, &ch) ;
        return (0) ;
    }
    if (!(A->xtype == CHOLMOD_REAL || A->xtype == CHOLMOD_COMPLEX))
    {
        fprintf (stderr, "error: only real or complex matrices hanlded\n") ;
        CHOLMOD_free_sparse (&A, &ch) ;
        return (0) ;
    }

    n = A->nrow ;
    Ap = A->p ;
    Ai = A->i ;
    Ax = A->x ;
    nz = Ap [n] ;
    isreal = (A->xtype == CHOLMOD_REAL) ;

    /* ---------------------------------------------------------------------- */
    /* construct random permutations */
    /* ---------------------------------------------------------------------- */

    Puser = randperm (n, n) ;
    Quser = randperm (n, n) ;

    /* ---------------------------------------------------------------------- */
    /* select known solution to Ax=b */
    /* ---------------------------------------------------------------------- */

    X = CHOLMOD_allocate_dense (n, NRHS, n, A->xtype, &ch) ;
    Xx = X->x ;
    for (j = 0 ; j < NRHS ; j++)
    {
        for (i = 0 ; i < n ; i++)
        {
            if (isreal)
            {
                Xx [i] = 1 + ((double) i) / ((double) n) + j * 100;
            }
            else
            {
                Xx [2*i  ] = 1 + ((double) i) / ((double) n) + j * 100 ;
                Xx [2*i+1] =  - ((double) i+1) / ((double) n + j) ;
                if (j == NRHS-1)
                {
                    Xx [2*i+1] = 0 ;    /* zero imaginary part */
                }
                else if (j == NRHS-2)
                {
                    Xx [2*i] = 0 ;      /* zero real part */
                }
            }
        }
        Xx += isreal ? n : 2*n ;
    }

    /* B = A*X */
    B = CHOLMOD_allocate_dense (n, NRHS, n, A->xtype, &ch) ;
    CHOLMOD_sdmult (A, 0, one, zero, X, B, &ch) ;
    /* Bx = B->x ; */

    /* ---------------------------------------------------------------------- */
    /* test KLU */
    /* ---------------------------------------------------------------------- */

    test_memory_handler ( ) ;
    maxerr = do_solves (A, B, X, Puser, Quser, &Common, &ch, &isnan) ;

    /* ---------------------------------------------------------------------- */
    /* basic error checking */
    /* ---------------------------------------------------------------------- */

    FAIL (klu_defaults (NULL)) ;

    FAIL (klu_extract (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_extract (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_z_extract (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_z_extract (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_analyze (0, NULL, NULL, NULL)) ;
    FAIL (klu_analyze (0, NULL, NULL, &Common)) ;

    FAIL (klu_analyze_given (0, NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_analyze_given (0, NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_cholmod (0, NULL, NULL, NULL, NULL)) ;

    FAIL (klu_factor (NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_factor (NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_z_factor (NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_z_factor (NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_refactor (NULL, NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_refactor (NULL, NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_z_refactor (NULL, NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_z_refactor (NULL, NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_rgrowth (NULL, NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_rgrowth (NULL, NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_z_rgrowth (NULL, NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_z_rgrowth (NULL, NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_condest (NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_condest (NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_z_condest (NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_z_condest (NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_flops (NULL, NULL, NULL)) ;
    FAIL (klu_flops (NULL, NULL, &Common)) ;

    FAIL (klu_z_flops (NULL, NULL, NULL)) ;
    FAIL (klu_z_flops (NULL, NULL, &Common)) ;

    FAIL (klu_rcond (NULL, NULL, NULL)) ;
    FAIL (klu_rcond (NULL, NULL, &Common)) ;

    FAIL (klu_z_rcond (NULL, NULL, NULL)) ;
    FAIL (klu_z_rcond (NULL, NULL, &Common)) ;

    FAIL (klu_free_symbolic (NULL, NULL)) ;
    OK (klu_free_symbolic (NULL, &Common)) ;

    FAIL (klu_free_numeric (NULL, NULL)) ;
    OK (klu_free_numeric (NULL, &Common)) ;

    FAIL (klu_z_free_numeric (NULL, NULL)) ;
    OK (klu_z_free_numeric (NULL, &Common)) ;

    FAIL (klu_scale (0, 0, NULL, NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_scale (0, 0, NULL, NULL, NULL, NULL, NULL, &Common)) ;
    OK (klu_scale (-1, 0, NULL, NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_z_scale (0, 0, NULL, NULL, NULL, NULL, NULL, NULL)) ;
    FAIL (klu_z_scale (0, 0, NULL, NULL, NULL, NULL, NULL, &Common)) ;
    OK (klu_z_scale (-1, 0, NULL, NULL, NULL, NULL, NULL, &Common)) ;

    FAIL (klu_solve (NULL, NULL, 0, 0, NULL, NULL)) ;
    FAIL (klu_solve (NULL, NULL, 0, 0, NULL, &Common)) ;

    FAIL (klu_z_solve (NULL, NULL, 0, 0, NULL, NULL)) ;
    FAIL (klu_z_solve (NULL, NULL, 0, 0, NULL, &Common)) ;

    FAIL (klu_tsolve (NULL, NULL, 0, 0, NULL, NULL)) ;
    FAIL (klu_tsolve (NULL, NULL, 0, 0, NULL, &Common)) ;

    FAIL (klu_z_tsolve (NULL, NULL, 0, 0, NULL, 0, NULL)) ;
    FAIL (klu_z_tsolve (NULL, NULL, 0, 0, NULL, 0, &Common)) ;

    FAIL (klu_malloc (0, 0, NULL)) ;
    FAIL (klu_malloc (0, 0, &Common)) ;
    FAIL (klu_malloc (Int_MAX, 1, &Common)) ;

    FAIL (klu_realloc (0, 0, 0, NULL, NULL)) ;
    FAIL (klu_realloc (0, 0, 0, NULL, &Common)) ;
    FAIL (klu_realloc (Int_MAX, 1, 0, NULL, &Common)) ;
    Gunk = (Int *) klu_realloc (1, 0, sizeof (Int), NULL, &Common) ;
    OK (Gunk) ;
    OK (klu_realloc (Int_MAX, 1, sizeof (Int), Gunk, &Common)) ;
    OK (Common.status == KLU_TOO_LARGE) ;
    klu_free (Gunk, 1, sizeof (Int), &Common) ;

    /* ---------------------------------------------------------------------- */
    /* mangle the matrix, and other error checking */
    /* ---------------------------------------------------------------------- */

    printf ("\nerror handling:\n") ;
    Symbolic = klu_analyze (n, Ap, Ai, &Common) ;
    OK (Symbolic) ;

    Xx = X->x ;
    if (nz > 0)
    {

        /* ------------------------------------------------------------------ */
        /* row index out of bounds */
        /* ------------------------------------------------------------------ */

        save = Ai [0] ;
        Ai [0] = -1 ;
        FAIL (klu_analyze (n, Ap, Ai, &Common)) ;
        if (isreal)
        {
            FAIL (klu_scale (1, n, Ap, Ai, Ax, Xx, Puser, &Common)) ;
        }
        else
        {
            FAIL (klu_z_scale (1, n, Ap, Ai, Ax, Xx, Puser, &Common)) ;
        }
        Ai [0] = save ;

        /* ------------------------------------------------------------------ */
        /* row index out of bounds */
        /* ------------------------------------------------------------------ */

        save = Ai [0] ;
        Ai [0] = Int_MAX ;
        FAIL (klu_analyze (n, Ap, Ai, &Common)) ;
        if (isreal)
        {
            FAIL (klu_scale (1, n, Ap, Ai, Ax, Xx, Puser, &Common)) ;
        }
        else
        {
            FAIL (klu_z_scale (1, n, Ap, Ai, Ax, Xx, Puser, &Common)) ;
        }
        Ai [0] = save ;

        /* ------------------------------------------------------------------ */
        /* column pointers mangled */
        /* ------------------------------------------------------------------ */

        save = Ap [n] ;
        Ap [n] = -1 ;
        FAIL (klu_analyze (n, Ap, Ai, &Common)) ;
        if (isreal)
        {
            FAIL (klu_scale (1, n, Ap, Ai, Ax, Xx, Puser, &Common)) ;
        }
        else
        {
            FAIL (klu_z_scale (1, n, Ap, Ai, Ax, Xx, Puser, &Common)) ;
        }
        Ap [n] = save ;

        /* ------------------------------------------------------------------ */
        /* column pointers mangled */
        /* ------------------------------------------------------------------ */

        save = Ap [n] ;
        Ap [n] = Ap [n-1] - 1 ;
        FAIL (klu_analyze (n, Ap, Ai, &Common)) ;
        if (isreal)
        {
            FAIL (klu_scale (1, n, Ap, Ai, Ax, Xx, Puser, &Common)) ;
        }
        else
        {
            FAIL (klu_z_scale (1, n, Ap, Ai, Ax, Xx, Puser, &Common)) ;
        }
        Ap [n] = save ;

        /* ------------------------------------------------------------------ */
        /* duplicates */
        /* ------------------------------------------------------------------ */

        if (n > 1 && Ap [1] - Ap [0] > 1)
        {
            save = Ai [1] ;
            Ai [1] = Ai [0] ;
            FAIL (klu_analyze (n, Ap, Ai, &Common)) ;
            if (isreal)
            {
                FAIL (klu_scale (1, n, Ap, Ai, Ax, Xx, Puser, &Common)) ;
            }
            else
            {
                FAIL (klu_z_scale (1, n, Ap, Ai, Ax, Xx, Puser, &Common)) ;
            }
            Ai [1] = save ;
        }

        /* ------------------------------------------------------------------ */
        /* invalid ordering */
        /* ------------------------------------------------------------------ */

        save = Common.ordering ;
        Common.ordering = 42 ;
        FAIL (klu_analyze (n, Ap, Ai, &Common)) ;
        Common.ordering = save ;

        /* ------------------------------------------------------------------ */
        /* invalid ordering (klu_cholmod, with NULL user_ordering) */
        /* ------------------------------------------------------------------ */

        save = Common.ordering ;
        Common.user_order = NULL ;
        Common.ordering = 3 ;
        FAIL (klu_analyze (n, Ap, Ai, &Common)) ;
        Common.ordering = save ;
    }

    /* ---------------------------------------------------------------------- */
    /* tests with valid symbolic factorization */
    /* ---------------------------------------------------------------------- */

    Common.halt_if_singular = FALSE ;
    Common.scale = 0 ;
    Numeric = NULL ;

    if (nz > 0)
    {

        /* ------------------------------------------------------------------ */
        /* Int overflow */
        /* ------------------------------------------------------------------ */

        if (n == 100)
        {
            Common.ordering = 2 ;
            Symbolic2 = klu_analyze (n, Ap, Ai, &Common) ;
            OK (Symbolic2) ;
            Common.memgrow = Int_MAX ;
            if (isreal)
            {
                Numeric = klu_factor (Ap, Ai, Ax, Symbolic2, &Common) ;
            }
            else
            {
                Numeric = klu_z_factor (Ap, Ai, Ax, Symbolic2, &Common) ;
            }
            Common.memgrow = 1.2 ;
            Common.ordering = 0 ;
            klu_free_symbolic (&Symbolic2, &Common) ;
            klu_free_numeric (&Numeric, &Common) ;
        }

        /* ------------------------------------------------------------------ */
        /* Int overflow again */
        /* ------------------------------------------------------------------ */

        Common.initmem = Int_MAX ;
        Common.initmem_amd = Int_MAX ;
        if (isreal)
        {
            Numeric = klu_factor (Ap, Ai, Ax, Symbolic, &Common) ;
        }
        else
        {
            Numeric = klu_z_factor (Ap, Ai, Ax, Symbolic, &Common) ;
        }
        Common.initmem = 10 ;
        Common.initmem_amd = 1.2 ;
        klu_free_numeric (&Numeric, &Common) ;

        /* ------------------------------------------------------------------ */
        /* mangle the matrix */
        /* ------------------------------------------------------------------ */

        save = Ai [0] ;
        Ai [0] = -1 ;

        if (isreal)
        {
            Numeric = klu_factor (Ap, Ai, Ax, Symbolic, &Common) ;
        }
        else
        {
            Numeric = klu_z_factor (Ap, Ai, Ax, Symbolic, &Common) ;
        }
        FAIL (Numeric) ;
        Ai [0] = save ;

        /* ------------------------------------------------------------------ */
        /* nan and inf handling */
        /* ------------------------------------------------------------------ */

        xsave = Ax [0] ;
        Ax [0] = one [0] / zero [0] ;
        if (isreal)
        {
            Numeric = klu_factor (Ap, Ai, Ax, Symbolic, &Common) ;
            klu_rcond (Symbolic, Numeric, &Common) ;
            klu_condest (Ap, Ax, Symbolic, Numeric, &Common) ;
        }
        else
        {
            Numeric = klu_z_factor (Ap, Ai, Ax, Symbolic, &Common) ;
            klu_z_rcond (Symbolic, Numeric, &Common) ;
            klu_z_condest (Ap, Ax, Symbolic, Numeric, &Common) ;
        }
        printf ("Nan case: rcond %g condest %g\n",
            Common.rcond, Common.condest) ;
        OK (Numeric) ;
        Ax [0] = xsave ;

        /* ------------------------------------------------------------------ */
        /* mangle the matrix again */
        /* ------------------------------------------------------------------ */

        save = Ai [0] ;
        Ai [0] = -1 ;
        if (isreal)
        {
            FAIL (klu_refactor (Ap, Ai, Ax, Symbolic, Numeric, &Common)) ;
        }
        else
        {
            FAIL (klu_z_refactor (Ap, Ai, Ax, Symbolic, Numeric, &Common)) ;
        }
        Ai [0] = save ;

        /* ------------------------------------------------------------------ */
        /* all zero */
        /* ------------------------------------------------------------------ */

        A2 = CHOLMOD_copy_sparse (A, &ch) ;
        A2x = A2->x ;
        for (k = 0 ; k < nz * (isreal ? 1:2) ; k++)
        {
            A2x [k] = 0 ;
        }
        for (Common.halt_if_singular = 0 ; Common.halt_if_singular <= 1 ;
            Common.halt_if_singular++)
        {
            for (Common.scale = -1 ; Common.scale <= 2 ; Common.scale++)
            {
                if (isreal)
                {
                    klu_refactor (Ap, Ai, A2x, Symbolic, Numeric, &Common) ;
                    klu_condest (Ap, A2x, Symbolic, Numeric, &Common) ;
                }
                else
                {
                    klu_z_refactor (Ap, Ai, A2x, Symbolic, Numeric, &Common) ;
                    klu_z_condest (Ap, A2x, Symbolic, Numeric, &Common) ;
                }
                OK (Common.status = KLU_SINGULAR) ;
            }
        }
        CHOLMOD_free_sparse (&A2, &ch) ;

        /* ------------------------------------------------------------------ */
        /* all one, or all 1i for complex case */
        /* ------------------------------------------------------------------ */

        A2 = CHOLMOD_copy_sparse (A, &ch) ;
        A2x = A2->x ;
        for (k = 0 ; k < nz ; k++)
        {
            if (isreal)
            {
                A2x [k] = 1 ;
            }
            else
            {
                A2x [2*k  ] = 0 ;
                A2x [2*k+1] = 1 ;
            }
        }
        Common.halt_if_singular = 0 ;
        Common.scale = 0 ;
        if (isreal)
        {
            klu_refactor (Ap, Ai, A2x, Symbolic, Numeric, &Common) ;
            klu_condest (Ap, A2x, Symbolic, Numeric, &Common) ;
        }
        else
        {
            klu_z_refactor (Ap, Ai, A2x, Symbolic, Numeric, &Common) ;
            klu_z_condest (Ap, A2x, Symbolic, Numeric, &Common) ;
        }
        OK (Common.status = KLU_SINGULAR) ;
        CHOLMOD_free_sparse (&A2, &ch) ;
    }

    klu_free_symbolic (&Symbolic, &Common) ;
    if (isreal)
    {
        klu_free_numeric (&Numeric, &Common) ;
    }
    else
    {
        klu_z_free_numeric (&Numeric, &Common) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free problem and quit */
    /* ---------------------------------------------------------------------- */

    CHOLMOD_free_dense (&X, &ch) ;
    CHOLMOD_free_dense (&B, &ch) ;
    CHOLMOD_free_sparse (&A, &ch) ;
    free (Puser) ;
    free (Quser) ;
    CHOLMOD_finish (&ch) ;
    fprintf (stderr, " maxerr %10.3e", maxerr) ;
    printf (" maxerr %10.3e", maxerr) ;
    if (maxerr < 1e-8)
    {
        fprintf (stderr, "  test passed") ;
        printf ("  test passed") ;
    }
    else
    {
        fprintf (stderr, "  test FAILED") ;
        printf ("  test FAILED") ;
    }
    if (isnan)
    {
        fprintf (stderr, " *") ;
        printf (" *") ;
    }
    fprintf (stderr, "\n") ;
    printf ("\n-----------------------------------------------------------\n") ;
    return (0) ;
}
