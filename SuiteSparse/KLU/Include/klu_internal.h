/* ========================================================================== */
/* === KLU/Include/klu_internal.h =========================================== */
/* ========================================================================== */

/* For internal use in KLU routines only, not for user programs */

#ifndef _KLU_INTERNAL_H
#define _KLU_INTERNAL_H

#include "klu.h" 
#include "btf.h"
#include "klu_version.h"

/* ========================================================================== */
/* make sure debugging and printing is turned off */

#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef NPRINT
#define NPRINT
#endif

/* To enable debugging and assertions, uncomment this line:
 #undef NDEBUG
 */

/* To enable diagnostic printing, uncomment this line:
 #undef NPRINT
 */

/* ========================================================================== */

#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>

#undef ASSERT
#ifndef NDEBUG
#define ASSERT(a) assert(a)
#else
#define ASSERT(a)
#endif

#define SCALAR_IS_NAN(x) ((x) != (x))

/* true if an integer (stored in double x) would overflow (or if x is NaN) */
#define INT_OVERFLOW(x) ((!((x) * (1.0+1e-8) <= (double) Int_MAX)) \
                        || SCALAR_IS_NAN (x))

#undef TRUE
#undef FALSE
#undef MAX
#undef MIN
#undef PRINTF
#undef FLIP

#ifndef NPRINT
#define PRINTF(s) SUITESPARSE_PRINTF (s)
#else
#define PRINTF(s)
#endif

#define TRUE 1
#define FALSE 0
#define MAX(a,b) (((a) > (b)) ?  (a) : (b))
#define MIN(a,b) (((a) < (b)) ?  (a) : (b))

/* FLIP is a "negation about -1", and is used to mark an integer i that is
 * normally non-negative.  FLIP (EMPTY) is EMPTY.  FLIP of a number > EMPTY
 * is negative, and FLIP of a number < EMTPY is positive.  FLIP (FLIP (i)) = i
 * for all integers i.  UNFLIP (i) is >= EMPTY. */
#define EMPTY (-1)
#define FLIP(i) (-(i)-2)
#define UNFLIP(i) (((i) < EMPTY) ? FLIP (i) : (i))


size_t KLU_kernel   /* final size of LU on output */
(
    /* input, not modified */
    Int n,              /* A is n-by-n */
    Int Ap [ ],         /* size n+1, column pointers for A */
    Int Ai [ ],         /* size nz = Ap [n], row indices for A */
    Entry Ax [ ],       /* size nz, values of A */
    Int Q [ ],          /* size n, optional input permutation */
    size_t lusize,      /* initial size of LU */

    /* output, not defined on input */
    Int Pinv [ ],       /* size n */
    Int P [ ],          /* size n */
    Unit **p_LU,        /* size lusize on input, size Uxp[n] on output*/
    Entry Udiag [ ],    /* size n, diagonal of U */
    Int Llen [ ],       /* size n, column length of L */
    Int Ulen [ ],       /* size n, column length of U */
    Int Lip [ ],        /* size n+1 */
    Int Uip [ ],        /* size n+1 */
    Int *lnz,           /* size of L */
    Int *unz,           /* size of U */

    /* workspace, not defined on input */
    Entry X [ ],   /* size n, zero on output */

    /* workspace, not defined on input or output */
    Int Stack [ ],  /* size n */
    Int Flag [ ],   /* size n */
    Int adj_pos [ ],    /* size n */
    
    /* workspace for pruning only */
    Int Lpend [ ],      /* size n workspace */

    /* inputs, not modified on output */
    Int k1,             /* the block of A is from k1 to k2-1 */
    Int PSinv [ ],      /* inverse of P from symbolic factorization */
    double Rs [ ],      /* scale factors for A */

    /* inputs, modified on output */
    Int Offp [ ],   /* off-diagonal matrix (modified by this routine) */
    Int Offi [ ],
    Entry Offx [ ],
    KLU_common *Common  /* the control input/output structure */
) ;


size_t KLU_kernel_factor            /* 0 if failure, size of LU if OK */
(
    /* inputs, not modified */
    Int n,          /* A is n-by-n. n must be > 0. */
    Int Ap [ ],     /* size n+1, column pointers for A */
    Int Ai [ ],     /* size nz = Ap [n], row indices for A */
    Entry Ax [ ],   /* size nz, values of A */
    Int Q [ ],      /* size n, optional column permutation */
    double Lsize,   /* initial size of L and U */

    /* outputs, not defined on input */
    Unit **p_LU,        /* row indices and values of L and U */
    Entry Udiag [ ],    /* size n, diagonal of U */
    Int Llen [ ],       /* size n, column length of L */
    Int Ulen [ ],       /* size n, column length of U */
    Int Lip [ ],        /* size n+1, column pointers of L */
    Int Uip [ ],        /* size n+1, column pointers of U */
    Int P [ ],          /* row permutation, size n */
    Int *lnz,           /* size of L */
    Int *unz,           /* size of U */

    /* workspace, undefined on input */
    Entry *X,       /* size n entries.  Zero on output */
    Int *Work,      /* size 5n Int's */

    /* inputs, not modified on output */
    Int k1,             /* the block of A is from k1 to k2-1 */
    Int PSinv [ ],      /* inverse of P from symbolic factorization */
    double Rs [ ],      /* scale factors for A */

    /* inputs, modified on output */
    Int Offp [ ],   /* off-diagonal matrix (modified by this routine) */
    Int Offi [ ],
    Entry Offx [ ],
    KLU_common *Common  /* the control input/output structure */
) ;

void KLU_lsolve
(
    /* inputs, not modified: */
    Int n,
    Int Lp [ ],
    Int Li [ ],
    Unit LU [ ],
    Int nrhs,
    /* right-hand-side on input, solution to Lx=b on output */
    Entry X [ ]
) ;

void KLU_ltsolve
(
    /* inputs, not modified: */
    Int n,
    Int Lp [ ],
    Int Li [ ],
    Unit LU [ ],
    Int nrhs,
#ifdef COMPLEX
    Int conj_solve,
#endif
    /* right-hand-side on input, solution to L'x=b on output */
    Entry X [ ]
) ;


void KLU_usolve
(
    /* inputs, not modified: */
    Int n,
    Int Up [ ],
    Int Ui [ ],
    Unit LU [ ],
    Entry Udiag [ ],
    Int nrhs,
    /* right-hand-side on input, solution to Ux=b on output */
    Entry X [ ]
) ;

void KLU_utsolve
(
    /* inputs, not modified: */
    Int n,
    Int Up [ ],
    Int Ui [ ],
    Unit LU [ ],
    Entry Udiag [ ],
    Int nrhs,
#ifdef COMPLEX
    Int conj_solve,
#endif
    /* right-hand-side on input, solution to U'x=b on output */
    Entry X [ ]
) ;

Int KLU_valid 
(
    Int n, 
    Int Ap [ ], 
    Int Ai [ ], 
    Entry Ax [ ]
) ;

Int KLU_valid_LU 
(
    Int n, 
    Int flag_test_start_ptr, 
    Int Xip [ ],
    Int Xlen [ ],  
    Unit LU [ ]
);

size_t KLU_add_size_t (size_t a, size_t b, Int *ok) ;

size_t KLU_mult_size_t (size_t a, size_t k, Int *ok) ;

KLU_symbolic *KLU_alloc_symbolic (Int n, Int *Ap, Int *Ai, KLU_common *Common) ;

#endif
