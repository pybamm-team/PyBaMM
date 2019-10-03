/* ========================================================================== */
/* === RBio/Include/RBio.h: include file for RBio =========================== */
/* ========================================================================== */

/* Copyright 2009, Timothy A. Davis, All Rights Reserved.
   Refer to RBio/Doc/license.txt for the RBio license. */

#ifndef _RBIO_H

/* -------------------------------------------------------------------------- */
/* large file I/O support */
/* -------------------------------------------------------------------------- */

/* Definitions required for large file I/O, which must come before any other
 * #includes.  These are not used if -DNLARGEFILE is defined at compile time.
 * Large file support may not be portable across all platforms and compilers;
 * if you encounter an error here, compile your code with -DNLARGEFILE.  In
 * particular, you must use -DNLARGEFILE for MATLAB 6.5 or earlier (which does
 * not have the io64.h include file).   See also CHOLMOD/Include/cholmod_io64.h.
 */

/* skip all of this if NLARGEFILE is defined at the compiler command line */
#ifndef NLARGEFILE

#if defined(MATLAB_MEX_FILE) || defined(MATHWORKS)

/* RBio is being compiled as a MATLAB mexFunction, or for use in MATLAB */
#include "io64.h"

#else

/* RBio is being compiled in a stand-alone library */
#undef  _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64

#endif

#endif


/* -------------------------------------------------------------------------- */
/* include files */
/* -------------------------------------------------------------------------- */

#include "SuiteSparse_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#ifdef MATLAB_MEX_FILE
#include "mex.h"
#endif

/* -------------------------------------------------------------------------- */
/* error codes */
/* -------------------------------------------------------------------------- */

#define RBIO_OK (0)               /* matrix is OK */

/* data structure errors */
#define RBIO_CP_INVALID (-1)      /* column pointers are invalid */
#define RBIO_ROW_INVALID (-2)     /* row indices are out of range */
#define RBIO_DUPLICATE (-3)       /* duplicate entry */
#define RBIO_EXTRANEOUS (-4)      /* entries in upper tri part of sym matrix */
#define RBIO_TYPE_INVALID (-5)    /* matrix type (RUA, etc) invalid */
#define RBIO_DIM_INVALID (-6)     /* matrix dimensions invalid */
#define RBIO_JUMBLED (-7)         /* matrix contains unsorted columns */
#define RBIO_ARG_ERROR (-8)       /* input arguments invalid */
#define RBIO_OUT_OF_MEMORY (-9)   /* out of memory */
#define RBIO_MKIND_INVALID (-10)  /* mkind is invalid */
#define RBIO_UNSUPPORTED (-11)    /* finite-element form unsupported */

/* I/O errors */
#define RBIO_HEADER_IOERROR (-91) /* I/O error: header */
#define RBIO_CP_IOERROR (-92)     /* I/O error: column pointers */
#define RBIO_ROW_IOERROR (-93)    /* I/O error: row indices */
#define RBIO_VALUE_IOERROR (-94)  /* I/O error: numerical values */
#define RBIO_FILE_IOERROR (-95)   /* I/O error: cannot read/write the file */

#define RBIO_DATE "May 4, 2016"
#define RBIO_VER_CODE(main,sub) ((main) * 1000 + (sub))
#define RBIO_MAIN_VERSION 2
#define RBIO_SUB_VERSION 2
#define RBIO_SUBSUB_VERSION 6
#define RBIO_VERSION RBIO_VER_CODE(RBIO_MAIN_VERSION,RBIO_SUB_VERSION)


/* -------------------------------------------------------------------------- */
/* user-callable functions */
/* -------------------------------------------------------------------------- */

/*
    RBread:         read a Rutherford/Boeing matrix from a file
    RBwrite:        write a matrix to a file in R/B format

    RBkind:         determine the matrix type (RUA, RSA, etc)
    RBreadraw:      read the raw contents of a R/B file

    RBget_entry:    get a single numerical value from a matrix
    RBput_entry:    put a single numerical value into a matrix

    RBmalloc:       malloc-wrapper for RBio
    RBfree:         free-wrapper for RBio
    RBok:           test the validity of a sparse matrix

    Each function comes in two versions: one with "int" integers, the other
    with "SuiteSparse_long" integers.  SuiteSparse_long is "long", except for
    Windows (for which it is __int64).  The default type is SuiteSparse_long.
    Functions for "int" integers have the _i suffix appended to their names.
*/

int RBkind_i        /* 0: OK, < 0: error, > 0: warning */
(
    /* input */
    int nrow,       /* A is nrow-by-ncol */
    int ncol,
    int *Ap,        /* Ap [0...ncol]: column pointers */
    int *Ai,        /* Ai [0...nnz-1]: row indices */
    double *Ax,     /* Ax [0...nnz-1]: real values.  Az holds imaginary part */
    double *Az,     /* if real, Az is NULL. if complex, Az is non-NULL */
    int mkind_in,   /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */

    /* output */
    int *mkind,     /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    int *skind,     /* r: -1 (rectangular), u: 0 (unsymmetric), s: 1 symmetric,
                       h: 2 (Hermitian), z: 3 (skew symmetric) */
    char mtype [4], /* rua, psa, rra, cha, etc */
    double *xmin,   /* smallest value */
    double *xmax,   /* largest value */

    /* workspace: allocated internally if NULL */
    int *cp         /* workspace of size ncol+1, undefined on input and output*/
) ;

SuiteSparse_long RBkind (SuiteSparse_long nrow, SuiteSparse_long ncol,
    SuiteSparse_long *Ap, SuiteSparse_long *Ai, double *Ax, double *Az,
    SuiteSparse_long mkind_in, SuiteSparse_long *mkind, SuiteSparse_long *skind,
    char mtype [4], double *xmin, double *xmax, SuiteSparse_long *cp) ;


int RBread_i            /* 0: OK, < 0: error, > 0: warning */
(
    /* input */
    char *filename,     /* file to read from */
    int build_upper,    /* if true, construct upper part for sym. matrices */
    int zero_handling,  /* 0: do nothing, 1: prune zeros, 2: extract zeros */

    /* output */
    char title [73],
    char key [9],
    char mtype [4],     /* RUA, RSA, PUA, PSA, RRA, etc */
    int *nrow,          /* A is nrow-by-ncol */
    int *ncol,
    int *mkind,         /* R: 0, P: 1, C: 2, I: 3 */
    int *skind,         /* R: -1, U: 0, S: 1, H: 2, Z: 3 */
    int *asize,         /* Ai array has size asize*sizeof(double) */
    int *znz,           /* number of explicit zeros removed from A */

    /* output: these are malloc'ed below and must be freed by the caller */
    int **Ap,           /* column pointers of A */
    int **Ai,           /* row indices of A */
    double **Ax,        /* real values (ignored if NULL) of A */
    double **Az,        /* imaginary values (ignored if NULL) of A */
    int **Zp,           /* column pointers of Z */
    int **Zi            /* row indices of Z */
) ;

SuiteSparse_long RBread (char *filename, SuiteSparse_long build_upper,
    SuiteSparse_long zero_handling, char title [73], char key [9],
    char mtype [4], SuiteSparse_long *nrow, SuiteSparse_long *ncol,
    SuiteSparse_long *mkind, SuiteSparse_long *skind, SuiteSparse_long *asize,
    SuiteSparse_long *znz, SuiteSparse_long **Ap, SuiteSparse_long **Ai,
    double **Ax, double **Az, SuiteSparse_long **Zp, SuiteSparse_long **Zi) ;


int RBreadraw_i         /* 0: OK, < 0: error, > 0: warning */
(
    /* input */
    char *filename,     /* file to read from */

    /* output */
    char title [73],
    char key [9],
    char mtype [4],     /* RUA, RSA, PUA, PSA, RRA, etc */
    int *nrow,          /* A is nrow-by-ncol */
    int *ncol,
    int *nnz,           /* size of Ai */
    int *nelnz,
    int *mkind,         /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */
    int *skind,         /* R: -1, U: 0, S: 1, H: 2, Z: 3 */
    int *fem,           /* 0:__A, 1:__E */
    int *xsize,         /* size of Ax */

    /* output: these are malloc'ed below and must be freed by the caller */
    int **p_Ap,         /* size ncol+1, column pointers of A */
    int **p_Ai,         /* size nnz, row indices of A */
    double **p_Ax       /* size xsize, numerical values of A */
) ;


SuiteSparse_long RBreadraw (char *filename, char title [73], char key [9],
    char mtype[4], SuiteSparse_long *nrow, SuiteSparse_long *ncol,
    SuiteSparse_long *nnz, SuiteSparse_long *nelnz, SuiteSparse_long *mkind,
    SuiteSparse_long *skind, SuiteSparse_long *fem, SuiteSparse_long *xsize,
    SuiteSparse_long **p_Ap, SuiteSparse_long **p_Ai, double **p_Ax) ;


int RBwrite_i       /* 0:OK, < 0: error, > 0: warning */
(
    /* input */
    char *filename, /* filename to write to (stdout if NULL) */
    char *title,    /* title (72 char max), may be NULL */
    char *key,      /* key (8 char max), may be NULL */
    int nrow,       /* A is nrow-by-ncol */
    int ncol,
    int *Ap,        /* size ncol+1, column pointers */
    int *Ai,        /* size anz=Ap[ncol], row indices (sorted) */
    double *Ax,     /* size anz or 2*anz, numerical values (binary if NULL) */
    double *Az,     /* size anz, imaginary part (real if NULL) */
    int *Zp,        /* size ncol+1, column pointers for Z (or NULL) */
    int *Zi,        /* size znz=Zp[ncol], row indices for Z (or NULL) */
    int mkind_in,   /* 0:R, 1:P: 2:Csplit, 3:I, 4:Cmerged */

    /* output */
    char mtype [4]  /* matrix type (RUA, RSA, etc), may be NULL */
) ;

SuiteSparse_long RBwrite (char *filename, char *title, char *key,
    SuiteSparse_long nrow, SuiteSparse_long ncol, SuiteSparse_long *Ap,
    SuiteSparse_long *Ai, double *Ax, double *Az, SuiteSparse_long *Zp,
    SuiteSparse_long *Zi, SuiteSparse_long mkind_in, char mtype [4]) ;


void RBget_entry_i
(
    int mkind,          /* R: 0, P: 1, C: 2, I: 3 */
    double *Ax,         /* real part, or both if merged-complex */
    double *Az,         /* imaginary part if split-complex */
    int p,              /* index of the entry */
    double *xr,         /* real part */
    double *xz          /* imaginary part */
) ;

void RBget_entry (SuiteSparse_long mkind, double *Ax, double *Az,
    SuiteSparse_long p, double *xr, double *xz) ;


void RBput_entry_i
(
    int mkind,          /* R: 0, P: 1, C: 2, I: 3 */
    double *Ax,         /* real part, or both if merged-complex */
    double *Az,         /* imaginary part if split-complex */
    int p,              /* index of the entry */
    double xr,          /* real part */
    double xz           /* imaginary part */
) ;

void RBput_entry (SuiteSparse_long mkind, double *Ax, double *Az,
    SuiteSparse_long p, double xr, double xz) ;


int RBok_i          /* 0:OK, < 0: error, > 0: warning */
(
    /* inputs, not modified */
    int nrow,       /* number of rows */
    int ncol,       /* number of columns */
    int nzmax,      /* max # of entries */
    int *Ap,        /* size ncol+1, column pointers */
    int *Ai,        /* size nz = Ap [ncol], row indices */
    double *Ax,     /* real part, or both if merged-complex */
    double *Az,     /* imaginary part for split-complex */
    char *As,       /* logical matrices (useful for MATLAB caller only) */
    int mkind,      /* 0:real, 1:logical/pattern, 2:split-complex, 3:integer,
                       4:merged-complex */

    /* outputs, not defined on input */
    int *p_njumbled,   /* # of jumbled row indices (-1 if not computed) */
    int *p_nzeros      /* number of explicit zeros (-1 if not computed) */
) ;

SuiteSparse_long RBok (SuiteSparse_long nrow, SuiteSparse_long ncol,
    SuiteSparse_long nzmax, SuiteSparse_long *Ap, SuiteSparse_long *Ai,
    double *Ax, double *Az, char *As, SuiteSparse_long mkind,
    SuiteSparse_long *p_njumbled, SuiteSparse_long *p_nzeros) ;

#ifdef MATLAB_MEX_FILE
void RBerror (int status) ;     /* only for MATLAB mexFunctions */
#endif

#ifdef __cplusplus
}
#endif
#endif
