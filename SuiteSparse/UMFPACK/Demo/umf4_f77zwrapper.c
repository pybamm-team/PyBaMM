/* ========================================================================== */
/* === umf4_f77zwrapper ===================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* UMFPACK Copyright (c) 2005-2012 by Timothy A. Davis,                       */
/* http://www.suitesparse.com. All Rights Reserved.                           */
/* See ../Doc/License.txt for License.                                        */
/* -------------------------------------------------------------------------- */

/* FORTRAN interface for the C-callable UMFPACK library (complex / int version
 * only and complex / SuiteSparse_long versions only).  This is HIGHLY
 * non-portable.  You will need to modify this depending on how your FORTRAN
 * and C compilers behave.
 *
 * See umf4z_f77wrapper.c for more information.
 *
 * The complex values are provided in two separate arrays.  Ax contains the
 * real part and Az contains the imaginary part.  The solution vector is in
 * x (the real part) and xz (the imaginary part.  b is the real part of the
 * right-hand-side and bz is the imaginary part.  Does not support the
 * packed complex type.
 */

#include "umfpack.h"
#include <ctype.h>
#include <stdio.h>
#ifdef NULL
#undef NULL
#endif
#define NULL 0
#define LEN 200

/* -------------------------------------------------------------------------- */
/* integer type: int or SuiteSparse_long */
/* -------------------------------------------------------------------------- */

#if defined (ZLONG)

#define Int SuiteSparse_long
#define UMFPACK_defaults	 umfpack_zl_defaults
#define UMFPACK_free_numeric	 umfpack_zl_free_numeric
#define UMFPACK_free_symbolic	 umfpack_zl_free_symbolic
#define UMFPACK_numeric		 umfpack_zl_numeric
#define UMFPACK_report_control	 umfpack_zl_report_control
#define UMFPACK_report_info	 umfpack_zl_report_info
#define UMFPACK_save_numeric	 umfpack_zl_save_numeric
#define UMFPACK_save_symbolic	 umfpack_zl_save_symbolic
#define UMFPACK_load_numeric	 umfpack_zl_load_numeric
#define UMFPACK_load_symbolic	 umfpack_zl_load_symbolic
#define UMFPACK_scale		 umfpack_zl_scale
#define UMFPACK_solve		 umfpack_zl_solve
#define UMFPACK_symbolic	 umfpack_zl_symbolic

#else

#define Int int
#define UMFPACK_defaults	 umfpack_zi_defaults
#define UMFPACK_free_numeric	 umfpack_zi_free_numeric
#define UMFPACK_free_symbolic	 umfpack_zi_free_symbolic
#define UMFPACK_numeric		 umfpack_zi_numeric
#define UMFPACK_report_control	 umfpack_zi_report_control
#define UMFPACK_report_info	 umfpack_zi_report_info
#define UMFPACK_save_numeric	 umfpack_zi_save_numeric
#define UMFPACK_save_symbolic	 umfpack_zi_save_symbolic
#define UMFPACK_load_numeric	 umfpack_zi_load_numeric
#define UMFPACK_load_symbolic	 umfpack_zi_load_symbolic
#define UMFPACK_scale		 umfpack_zi_scale
#define UMFPACK_solve		 umfpack_zi_solve
#define UMFPACK_symbolic	 umfpack_zi_symbolic

#endif

/* -------------------------------------------------------------------------- */
/* construct a file name from a file number (not user-callable) */
/* -------------------------------------------------------------------------- */

static void make_filename (Int filenum, char *prefix, char *filename)
{
    char *psrc, *pdst ;
#ifdef ZLONG
    sprintf (filename, "%s%ld.umf", prefix, filenum) ;
#else
    sprintf (filename, "%s%d.umf", prefix, filenum) ;
#endif
    /* remove any spaces in the filename */
    pdst = filename ;
    for (psrc = filename ; *psrc ; psrc++)
    {
	if (!isspace (*psrc)) *pdst++ = *psrc ;
    }
    *pdst = '\0' ;
}

/* ========================================================================== */
/* === with underscore ====================================================== */
/* ========================================================================== */

/* Solaris, Linux, and SGI IRIX.  Probably Compaq Alpha as well. */

/* -------------------------------------------------------------------------- */
/* umf4zdef: set default control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4zdef (control) */

void umf4zdef_ (double Control [UMFPACK_CONTROL])
{
    UMFPACK_defaults (Control) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zpcon: print control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4zpcon (control) */

void umf4zpcon_ (double Control [UMFPACK_CONTROL])
{
    fflush (stdout) ;
    UMFPACK_report_control (Control) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zsym: pre-ordering and symbolic factorization */
/* -------------------------------------------------------------------------- */

/* call umf4zsym (m, n, Ap, Ai, Ax, Az, symbolic, control, info) */

void umf4zsym_ (Int *m, Int *n, Int Ap [ ], Int Ai [ ],
    double Ax [ ], double Az [ ], void **Symbolic,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_symbolic (*m, *n, Ap, Ai, Ax, Az, Symbolic, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4znum: numeric factorization */
/* -------------------------------------------------------------------------- */

/* call umf4znum (Ap, Ai, Ax, Az, symbolic, numeric, control, info) */

void umf4znum_ (Int Ap [ ], Int Ai [ ], double Ax [ ], double Az [ ],
    void **Symbolic, void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_numeric (Ap, Ai, Ax, Az, *Symbolic, Numeric, Control, Info);
}

/* -------------------------------------------------------------------------- */
/* umf4zsolr: solve a linear system with iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4zsolr (sys, Ap, Ai, Ax, Az, x, xz, b, bz, numeric, control, info) */

void umf4zsolr_ (Int *sys, Int Ap [ ], Int Ai [ ], double Ax [ ], double Az [ ],
    double x [ ], double xz [ ], double b [ ], double bz [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_solve (*sys, Ap, Ai, Ax, Az, x, xz, b, bz,
	*Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zsol: solve a linear system without iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4zsol (sys, x, xz, b, bz, numeric, control, info) */

void umf4zsol_ (Int *sys, double x [ ], double xz [ ], double b [ ],
    double bz [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    Control [UMFPACK_IRSTEP] = 0 ;
    (void) UMFPACK_solve (*sys, (Int *) NULL, (Int *) NULL, (double *) NULL,
	(double *) NULL, x, xz, b, bz, *Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zscal: scale a vector using UMFPACK's scale factors */
/* -------------------------------------------------------------------------- */

/* call umf4zscal (x, xz, b, bz, numeric, status) */

void umf4zscal_ (double x [ ], double xz [ ], double b [ ], double bz [ ],
    void **Numeric, Int *status)
{
    *status = UMFPACK_scale (x, xz, b, bz, *Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zpinf: print info */
/* -------------------------------------------------------------------------- */

/* call umf4zpinf (control) */

void umf4zpinf_ (double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    fflush (stdout) ;
    UMFPACK_report_info (Control, Info) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zfnum: free the Numeric object */
/* -------------------------------------------------------------------------- */

/* call umf4zfnum (numeric) */

void umf4zfnum_ (void **Numeric)
{
    UMFPACK_free_numeric (Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zfsym: free the Symbolic object */
/* -------------------------------------------------------------------------- */

/* call umf4zfsym (symbolic) */

void umf4zfsym_ (void **Symbolic)
{
    UMFPACK_free_symbolic (Symbolic) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zsnum: save the Numeric object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4zsnum (numeric, filenum, status) */

void umf4zsnum_ (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_save_numeric (*Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zssym: save the Symbolic object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4zssym (symbolic, filenum, status) */

void umf4zssym_ (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_save_symbolic (*Symbolic, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zlnum: load the Numeric object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4zlnum (numeric, filenum, status) */

void umf4zlnum_ (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_load_numeric (Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zlsym: load the Symbolic object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4zlsym (symbolic, filenum, status) */

void umf4zlsym_ (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_load_symbolic (Symbolic, filename) ;
}

/* ========================================================================== */
/* === with no underscore =================================================== */
/* ========================================================================== */

/* IBM AIX.  Probably Microsoft Windows and HP Unix as well.  */

/* -------------------------------------------------------------------------- */
/* umf4zdef: set default control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4zdef (control) */

void umf4zdef (double Control [UMFPACK_CONTROL])
{
    UMFPACK_defaults (Control) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zpcon: print control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4zpcon (control) */

void umf4zpcon (double Control [UMFPACK_CONTROL])
{
    fflush (stdout) ;
    UMFPACK_report_control (Control) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zsym: pre-ordering and symbolic factorization */
/* -------------------------------------------------------------------------- */

/* call umf4zsym (m, n, Ap, Ai, Ax, Az, symbolic, control, info) */

void umf4zsym (Int *m, Int *n, Int Ap [ ], Int Ai [ ],
    double Ax [ ], double Az [ ], void **Symbolic,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_symbolic (*m, *n, Ap, Ai, Ax, Az, Symbolic, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4znum: numeric factorization */
/* -------------------------------------------------------------------------- */

/* call umf4znum (Ap, Ai, Ax, Az, symbolic, numeric, control, info) */

void umf4znum (Int Ap [ ], Int Ai [ ], double Ax [ ], double Az [ ],
    void **Symbolic, void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_numeric (Ap, Ai, Ax, Az, *Symbolic, Numeric, Control, Info);
}

/* -------------------------------------------------------------------------- */
/* umf4zsolr: solve a linear system with iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4zsolr (sys, Ap, Ai, Ax, Az, x, xz, b, bz, numeric, control, info) */

void umf4zsolr (Int *sys, Int Ap [ ], Int Ai [ ], double Ax [ ], double Az [ ],
    double x [ ], double xz [ ], double b [ ], double bz [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_solve (*sys, Ap, Ai, Ax, Az, x, xz, b, bz,
	*Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zsol: solve a linear system without iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4zsol (sys, x, xz, b, bz, numeric, control, info) */

void umf4zsol (Int *sys, double x [ ], double xz [ ], double b [ ],
    double bz [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    Control [UMFPACK_IRSTEP] = 0 ;
    (void) UMFPACK_solve (*sys, (Int *) NULL, (Int *) NULL, (double *) NULL,
	(double *) NULL, x, xz, b, bz, *Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zscal: scale a vector using UMFPACK's scale factors */
/* -------------------------------------------------------------------------- */

/* call umf4zscal (x, xz, b, bz, numeric, status) */

void umf4zscal (double x [ ], double xz [ ], double b [ ], double bz [ ],
    void **Numeric, Int *status)
{
    *status = UMFPACK_scale (x, xz, b, bz, *Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zpinf: print info */
/* -------------------------------------------------------------------------- */

/* call umf4zpinf (control) */

void umf4zpinf (double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    fflush (stdout) ;
    UMFPACK_report_info (Control, Info) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zfnum: free the Numeric object */
/* -------------------------------------------------------------------------- */

/* call umf4zfnum (numeric) */

void umf4zfnum (void **Numeric)
{
    UMFPACK_free_numeric (Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zfsym: free the Symbolic object */
/* -------------------------------------------------------------------------- */

/* call umf4zfsym (symbolic) */

void umf4zfsym (void **Symbolic)
{
    UMFPACK_free_symbolic (Symbolic) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zsnum: save the Numeric object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4zsnum (numeric, filenum, status) */

void umf4zsnum (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_save_numeric (*Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zssym: save the Symbolic object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4zssym (symbolic, filenum, status) */

void umf4zssym (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_save_symbolic (*Symbolic, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zlnum: load the Numeric object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4zlnum (numeric, filenum, status) */

void umf4zlnum (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_load_numeric (Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zlsym: load the Symbolic object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4zlsym (symbolic, filenum, status) */

void umf4zlsym (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_load_symbolic (Symbolic, filename) ;
}

