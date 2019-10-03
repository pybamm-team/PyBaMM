/* ========================================================================== */
/* === ssmult.h ============================================================= */
/* ========================================================================== */

/* Include file for ssmult.c and ssmultsym.c
 * Copyright 2007-2009, Timothy A. Davis, http://www.suitesparse.com
 */

#include "mex.h"
#include <stdlib.h>

/* NOTE: this code will FAIL abysmally if Int is mwIndex. */
#define Int mwSignedIndex

#define ERROR_DIMENSIONS (-1)
#define ERROR_TOO_LARGE (-2)

/* turn off debugging */
#ifndef NDEBUG
#define NDEBUG
#endif

#define MAX(a,b) (((a) > (b)) ? (a) : (b))

#define MXFREE(a) { \
    void *ptr ; \
    ptr = (void *) (a) ; \
    if (ptr != NULL) mxFree (ptr) ; \
}

mxArray *ssmult_transpose       /* returns C = A' or A.' */
(
    const mxArray *A,
    int conj                    /* compute A' if true, compute A.' if false */
) ;

int ssmult_nnz (const mxArray *A) ;

void ssmult_invalid (int error_code) ;

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
) ;

mxArray *ssmult_saxpy   /* returns C = A*B using the sparse saxpy method */
(
    const mxArray *A,
    const mxArray *B,
    int ac,                 /* if true use conj(A) */
    int bc,                 /* if true use conj(B) */
    int cc,                 /* if true compute conj(C) */
    int sorted              /* if true, return C with sorted columns */
) ;

mxArray *ssmult_dot     /* returns C = A'*B using sparse dot product method */
(
    const mxArray *A,
    const mxArray *B,
    int ac,             /* if true: conj(A)   if false: A. ignored if A real */
    int bc,             /* if true: conj(B)   if false: B. ignored if B real */
    int cc              /* if true: conj(C)   if false: C. ignored if C real */
) ;

void ssdump (const mxArray *A) ;
