/* ========================================================================= */
/* === camd_internal.h ===================================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,                       */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/* This file is for internal use in CAMD itself, and does not normally need to
 * be included in user code (it is included in UMFPACK, however).   All others
 * should use camd.h instead.
 */

/* ========================================================================= */
/* === NDEBUG ============================================================== */
/* ========================================================================= */

/*
 * Turning on debugging takes some work (see below).   If you do not edit this
 * file, then debugging is always turned off, regardless of whether or not
 * -DNDEBUG is specified in your compiler options.
 *
 * If CAMD is being compiled as a mexFunction, then MATLAB_MEX_FILE is defined,
 * and mxAssert is used instead of assert.  If debugging is not enabled, no
 * MATLAB include files or functions are used.  Thus, the CAMD library libcamd.a
 * can be safely used in either a stand-alone C program or in another
 * mexFunction, without any change.
 */

/*
    CAMD will be exceedingly slow when running in debug mode.  The next three
    lines ensure that debugging is turned off.
*/
#ifndef NDEBUG
#define NDEBUG
#endif

/*
    To enable debugging, uncomment the following line:
#undef NDEBUG
*/


/* ------------------------------------------------------------------------- */
/* ANSI include files */
/* ------------------------------------------------------------------------- */

/* from stdlib.h:  size_t, malloc, free, realloc, and calloc */
#include <stdlib.h>

#if !defined(NPRINT) || !defined(NDEBUG)
/* from stdio.h:  printf.  Not included if NPRINT is defined at compile time.
 * fopen and fscanf are used when debugging. */
#include <stdio.h>
#endif

/* from limits.h:  INT_MAX and LONG_MAX */
#include <limits.h>

/* from math.h: sqrt */
#include <math.h>

/* ------------------------------------------------------------------------- */
/* MATLAB include files (only if being used in or via MATLAB) */
/* ------------------------------------------------------------------------- */

#ifdef MATLAB_MEX_FILE
#include "matrix.h"
#include "mex.h"
#endif

/* ------------------------------------------------------------------------- */
/* basic definitions */
/* ------------------------------------------------------------------------- */

#ifdef FLIP
#undef FLIP
#endif

#ifdef MAX
#undef MAX
#endif

#ifdef MIN
#undef MIN
#endif

#ifdef EMPTY
#undef EMPTY
#endif

#ifdef GLOBAL
#undef GLOBAL
#endif

#ifdef PRIVATE
#undef PRIVATE
#endif

/* FLIP is a "negation about -1", and is used to mark an integer i that is
 * normally non-negative.  FLIP (EMPTY) is EMPTY.  FLIP of a number > EMPTY
 * is negative, and FLIP of a number < EMTPY is positive.  FLIP (FLIP (i)) = i
 * for all integers i.  UNFLIP (i) is >= EMPTY. */
#define EMPTY (-1)
#define FLIP(i) (-(i)-2)
#define UNFLIP(i) ((i < EMPTY) ? FLIP (i) : (i))

/* for integer MAX/MIN, or for doubles when we don't care how NaN's behave: */
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

/* logical expression of p implies q: */
#define IMPLIES(p,q) (!(p) || (q))

/* Note that the IBM RS 6000 xlc predefines TRUE and FALSE in <types.h>. */
/* The Compaq Alpha also predefines TRUE and FALSE. */
#ifdef TRUE
#undef TRUE
#endif
#ifdef FALSE
#undef FALSE
#endif

#define TRUE (1)
#define FALSE (0)
#define PRIVATE static
#define GLOBAL
#define EMPTY (-1)

/* Note that Linux's gcc 2.96 defines NULL as ((void *) 0), but other */
/* compilers (even gcc 2.95.2 on Solaris) define NULL as 0 or (0).  We */
/* need to use the ANSI standard value of 0. */
#ifdef NULL
#undef NULL
#endif

#define NULL 0

/* largest value of size_t */
#ifndef SIZE_T_MAX
#ifdef SIZE_MAX
/* C99 only */
#define SIZE_T_MAX SIZE_MAX
#else
#define SIZE_T_MAX ((size_t) (-1))
#endif
#endif

/* ------------------------------------------------------------------------- */
/* integer type for CAMD: int or SuiteSparse_long */
/* ------------------------------------------------------------------------- */

#include "camd.h"

#if defined (DLONG) || defined (ZLONG)

#define Int SuiteSparse_long
#define ID  SuiteSparse_long_id
#define Int_MAX SuiteSparse_long_max

#define CAMD_order camd_l_order
#define CAMD_defaults camd_l_defaults
#define CAMD_control camd_l_control
#define CAMD_info camd_l_info
#define CAMD_1 camd_l1
#define CAMD_2 camd_l2
#define CAMD_valid camd_l_valid
#define CAMD_cvalid camd_l_cvalid
#define CAMD_aat camd_l_aat
#define CAMD_postorder camd_l_postorder
#define CAMD_post_tree camd_l_post_tree
#define CAMD_dump camd_l_dump
#define CAMD_debug camd_l_debug
#define CAMD_debug_init camd_l_debug_init
#define CAMD_preprocess camd_l_preprocess

#else

#define Int int
#define ID "%d"
#define Int_MAX INT_MAX

#define CAMD_order camd_order
#define CAMD_defaults camd_defaults
#define CAMD_control camd_control
#define CAMD_info camd_info
#define CAMD_1 camd_1
#define CAMD_2 camd_2
#define CAMD_valid camd_valid
#define CAMD_cvalid camd_cvalid
#define CAMD_aat camd_aat
#define CAMD_postorder camd_postorder
#define CAMD_post_tree camd_post_tree
#define CAMD_dump camd_dump
#define CAMD_debug camd_debug
#define CAMD_debug_init camd_debug_init
#define CAMD_preprocess camd_preprocess

#endif

/* ------------------------------------------------------------------------- */
/* CAMD routine definitions (not user-callable) */
/* ------------------------------------------------------------------------- */

GLOBAL size_t CAMD_aat
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],
    Int Len [ ],
    Int Tp [ ],
    double Info [ ]
) ;

GLOBAL void CAMD_1
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],
    Int P [ ],
    Int Pinv [ ],
    Int Len [ ],
    Int slen,
    Int S [ ],
    double Control [ ],
    double Info [ ],
    const Int C [ ]
) ;

GLOBAL Int CAMD_postorder
(
    Int j, Int k, Int n, Int head [], Int next [], Int post [], Int stack []
) ;

GLOBAL void CAMD_preprocess
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],
    Int Rp [ ],
    Int Ri [ ],
    Int W [ ],
    Int Flag [ ]
) ;

/* ------------------------------------------------------------------------- */
/* debugging definitions */
/* ------------------------------------------------------------------------- */

#ifndef NDEBUG

/* from assert.h:  assert macro */
#include <assert.h>

#ifndef EXTERN
#define EXTERN extern
#endif

EXTERN Int CAMD_debug ;

GLOBAL void CAMD_debug_init ( char *s ) ;

GLOBAL void CAMD_dump
(
    Int n,
    Int Pe [ ],
    Int Iw [ ],
    Int Len [ ],
    Int iwlen,
    Int pfree,
    Int Nv [ ],
    Int Next [ ],
    Int Last [ ],
    Int Head [ ],
    Int Elen [ ],
    Int Degree [ ],
    Int W [ ],
    Int nel,
    Int BucketSet [],
    const Int C [],
    Int Curc
) ;

#ifdef ASSERT
#undef ASSERT
#endif

/* Use mxAssert if CAMD is compiled into a mexFunction */
#ifdef MATLAB_MEX_FILE
#define ASSERT(expression) (mxAssert ((expression), ""))
#else
#define ASSERT(expression) (assert (expression))
#endif

#define CAMD_DEBUG0(params) { SUITESPARSE_PRINTF (params) ; }
#define CAMD_DEBUG1(params) \
    { if (CAMD_debug >= 1) SUITESPARSE_PRINTF (params) ; }
#define CAMD_DEBUG2(params) \
    { if (CAMD_debug >= 2) SUITESPARSE_PRINTF (params) ; }
#define CAMD_DEBUG3(params) \
    { if (CAMD_debug >= 3) SUITESPARSE_PRINTF (params) ; }
#define CAMD_DEBUG4(params) \
    { if (CAMD_debug >= 4) SUITESPARSE_PRINTF (params) ; }

#else

/* no debugging */
#define ASSERT(expression)
#define CAMD_DEBUG0(params)
#define CAMD_DEBUG1(params)
#define CAMD_DEBUG2(params)
#define CAMD_DEBUG3(params)
#define CAMD_DEBUG4(params)

#endif
