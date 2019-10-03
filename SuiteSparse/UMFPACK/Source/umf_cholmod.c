/* ========================================================================== */
/* === umf_cholmod ========================================================== */
/* ========================================================================== */

/* umfpack_cholmod: user-defined ordering function to interface UMFPACK
 * to CHOLMOD.
 *
 * This routine is an example of a user-provided ordering function for UMFPACK.
 *
 * This function can be passed to umfpack_*_fsymbolic as the
 * user_ordering function pointer.
 */

#include "umf_internal.h"
#include "umf_cholmod.h"

#ifndef NCHOLMOD
#include "cholmod.h"
#endif

#if defined (DINT) || defined (ZINT)
#define CHOLMOD_start       cholmod_start
#define CHOLMOD_transpose   cholmod_transpose
#define CHOLMOD_analyze     cholmod_analyze
#define CHOLMOD_free_sparse cholmod_free_sparse
#define CHOLMOD_free_factor cholmod_free_factor
#define CHOLMOD_finish      cholmod_finish
#define CHOLMOD_print_common      cholmod_print_common
#else
#define CHOLMOD_start       cholmod_l_start
#define CHOLMOD_transpose   cholmod_l_transpose
#define CHOLMOD_analyze     cholmod_l_analyze
#define CHOLMOD_free_sparse cholmod_l_free_sparse
#define CHOLMOD_free_factor cholmod_l_free_factor
#define CHOLMOD_finish      cholmod_l_finish
#define CHOLMOD_print_common      cholmod_l_print_common
#endif

int UMF_cholmod
(
    /* inputs */
    Int nrow,               /* A is nrow-by-ncol */
    Int ncol,               /* A is nrow-by-ncol */
    Int symmetric,          /* if true and nrow=ncol do A+A', else do A'A */
    Int Ap [ ],             /* column pointers, size ncol+1 */
    Int Ai [ ],             /* row indices, size nz = Ap [ncol] */
    /* output */
    Int Perm [ ],           /* fill-reducing permutation, size ncol */
    /* user-defined */
    void *user_params,      /* Int array of size 3 */
    double user_info [3]    /* [0]: max col count for L=chol(P(A+A')P')
                               [1]: nnz (L)
                               [2]: flop count for chol, if A real */
)
{
#ifndef NCHOLMOD
    double dmax, flops, c, lnz ;
    cholmod_sparse Amatrix, *A, *AT, *S ;
    cholmod_factor *L ;
    cholmod_common cm ;
    Int *P, *ColCount ;
    Int k, ordering_option, print_level, *params ;

    params = (Int *) user_params ;
    ordering_option = params [0] ;
    print_level = params [1] - 1 ;
    params [2] = -1 ;

    if (Ap == NULL || Ai == NULL || Perm == NULL || nrow < 0 || ncol < 0)
    {
        /* invalid inputs */
        return (FALSE) ;
    }
    if (nrow != ncol)
    {
        /* force symmetric to be false */
        symmetric = FALSE ;
    }

    /* start CHOLMOD */
    CHOLMOD_start (&cm) ;
    cm.supernodal = CHOLMOD_SIMPLICIAL ;
    cm.print = print_level ;

    /* adjust cm based on ordering_option */
    switch (ordering_option)
    {

        default:
        case UMFPACK_ORDERING_AMD:
            /* AMD on A+A' if symmetric, COLAMD on A otherwise */
            cm.nmethods = 1 ;
            cm.method [0].ordering = symmetric ? CHOLMOD_AMD : CHOLMOD_COLAMD ;
            cm.postorder = TRUE ;
            break ;

        case UMFPACK_ORDERING_METIS:
            /* metis on A+A' if symmetric, A'A otherwise */
            cm.nmethods = 1 ;
            cm.method [0].ordering = CHOLMOD_METIS ;
            cm.postorder = TRUE ;
            break ;

        case UMFPACK_ORDERING_NONE:
        case UMFPACK_ORDERING_GIVEN:
        case UMFPACK_ORDERING_USER:
            /* no ordering.  No input permutation here, and no user
               function, so all these are the same as "none". */
            cm.nmethods = 1 ;
            cm.method [0].ordering = CHOLMOD_NATURAL ;
            cm.postorder = FALSE ;
            break ;

        case UMFPACK_ORDERING_BEST:
            /* try AMD, METIS and NESDIS on A+A', or COLAMD(A), METIS(A'A),
               and NESDIS (A'A) */
            cm.nmethods = 3 ;
            cm.method [0].ordering = symmetric ? CHOLMOD_AMD : CHOLMOD_COLAMD ;
            cm.method [1].ordering = CHOLMOD_METIS ;
            cm.method [2].ordering = CHOLMOD_NESDIS ;
            cm.postorder = TRUE ;
            break ;

        case UMFPACK_ORDERING_CHOLMOD:
            /* no change to CHOLMOD defaults:
            Do not use given permutation, since it's not provided.
            Try AMD.  If fill-in and flop count are low, use AMD.
            Otherwise, try METIS and take the best of AMD and METIS.
            cm.method [0].ordering = CHOLMOD_GIVEN
            cm.method [1].ordering = CHOLMOD_AMD
            cm.method [2].ordering = CHOLMOD_METIS
            cm.nmethods = 2 if METIS installed, 3 otherwise ('given' is skipped)
            */
            break ;
    }

    /* construct a CHOLMOD version of the input matrix A */
    A = &Amatrix ;
    A->nrow = nrow ;                /* A is nrow-by-ncol */
    A->ncol = ncol ;
    A->nzmax = Ap [ncol] ;          /* with nzmax entries */
    A->packed = TRUE ;              /* there is no A->nz array */
    if (symmetric)
    {
        A->stype = 1 ;                  /* A is symmetric */
    }
    else
    {
        A->stype = 0 ;                  /* A is unsymmetric */
    }
    A->itype = CHOLMOD_INT ;
    A->xtype = CHOLMOD_PATTERN ;
    A->dtype = CHOLMOD_DOUBLE ;
    A->nz = NULL ;
    A->p = Ap ;                     /* column pointers */
    A->i = Ai ;                     /* row indices */
    A->x = NULL ;                   /* no numerical values */
    A->z = NULL ;
    A->sorted = FALSE ;             /* columns of A might not be sorted */

    if (symmetric)
    {
        /* CHOLMOD with order the symmetric matrix A */
        AT = NULL ;
        S = A ;
    }
    else
    {
        /* S = A'.  CHOLMOD will order S*S', which is A'*A */
        AT = CHOLMOD_transpose (A, 0, &cm) ;
        S = AT ;
    }

    /* order and analyze S or S*S' */
    L = CHOLMOD_analyze (S, &cm) ;
    CHOLMOD_free_sparse (&AT, &cm) ;
    if (L == NULL)
    {
        return (FALSE) ;
    }

    /* determine the ordering used */
    switch (L->ordering)
    {

        case CHOLMOD_AMD:
        case CHOLMOD_COLAMD:
            params [2] = UMFPACK_ORDERING_AMD ;
            break ;

        case CHOLMOD_METIS:
        case CHOLMOD_NESDIS:
            params [2] = UMFPACK_ORDERING_METIS ;
            break ;

        case CHOLMOD_GIVEN:
        case CHOLMOD_NATURAL:
        default:
            params [2] = UMFPACK_ORDERING_NONE ;
            break ;
    }

    /* copy the permutation from L to the output and compute statistics */
    P = L->Perm ;
    ColCount = L->ColCount ;
    dmax = 1 ;
    lnz = 0 ;
    flops = 0 ;
    for (k = 0 ; k < ncol ; k++)
    {
        Perm [k] = P [k] ;
        c = ColCount [k] ;
        if (c > dmax) dmax = c ;
        lnz += c ;
        flops += c*c ;
    }
    user_info [0] = dmax ;
    user_info [1] = lnz ;
    user_info [2] = flops ;

    CHOLMOD_free_factor (&L, &cm) ;
    if (print_level > 0) 
    {
        CHOLMOD_print_common ("for UMFPACK", &cm) ;
    }
    CHOLMOD_finish (&cm) ;
    return (TRUE) ;
#else
    /* CHOLMOD and its supporting packages (CAMD, CCOLAMD, COLAMD, METIS)
      not installed */
    return (FALSE) ;
#endif
}
