/* ========================================================================== */
/* === KLU_defaults ========================================================= */
/* ========================================================================== */

/* Sets default parameters for KLU */

#include "klu_internal.h"

Int KLU_defaults
(
    KLU_common *Common
)
{
    if (Common == NULL)
    {
        return (FALSE) ;
    }

    /* parameters */
    Common->tol = 0.001 ;       /* pivot tolerance for diagonal */
    Common->memgrow = 1.2;      /* realloc size ratio increase for LU factors */
    Common->initmem_amd = 1.2 ; /* init. mem with AMD:  c*nnz(L) + n */
    Common->initmem = 10 ;      /* init. mem otherwise: c*nnz(A) + n */
    Common->btf = TRUE ;        /* use BTF pre-ordering, or not */
    Common->maxwork = 0 ;       /* no limit to work done by btf_order */
    Common->ordering = 0 ;      /* 0: AMD, 1: COLAMD, 2: user-provided P and Q,
                                 * 3: user-provided function */
    Common->scale = 2 ;         /* scale: -1: none, and do not check for errors
                                 * in the input matrix in KLU_refactor.
                                 * 0: none, but check for errors,
                                 * 1: sum, 2: max */
    Common->halt_if_singular = TRUE ;   /* quick halt if matrix is singular */

    /* user ordering function and optional argument */
    Common->user_order = NULL ;
    Common->user_data = NULL ;

    /* statistics */
    Common->status = KLU_OK ;
    Common->nrealloc = 0 ;
    Common->structural_rank = EMPTY ;
    Common->numerical_rank = EMPTY ;
    Common->noffdiag = EMPTY ;
    Common->flops = EMPTY ;
    Common->rcond = EMPTY ;
    Common->condest = EMPTY ;
    Common->rgrowth = EMPTY ;
    Common->work = 0 ;          /* work done by btf_order */

    Common->memusage = 0 ;
    Common->mempeak = 0 ;

    return (TRUE) ;
}
