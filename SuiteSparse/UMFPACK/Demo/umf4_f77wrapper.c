/* ========================================================================== */
/* === umf4_f77wrapper ====================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* UMFPACK Copyright (c) 2005-2012 by Timothy A. Davis,                       */
/* http://www.suitesparse.com. All Rights Reserved.                           */
/* See ../Doc/License.txt for License.                                        */
/* -------------------------------------------------------------------------- */

/* FORTRAN interface for the C-callable UMFPACK library (double / int version
 * only and double / SuiteSparse_long versions only).  This is HIGHLY
 * non-portable.  You will need to modify this depending on how your FORTRAN
 * and C compilers behave.  This has been tested in Linux, Sun Solaris, SGI
 * IRIX, and IBM AIX, with various compilers.  It has not been exhaustively
 * tested on all possible combinations of C and FORTRAN compilers.  The
 * SuiteSparse_long version works on Solaris, SGI IRIX, and IBM AIX when the
 * UMFPACK library is compiled in 64-bit mode.
 *
 * Only a subset of UMFPACK's capabilities are provided.  Refer to the UMFPACK
 * User Guide for details.
 *
 * For some C and FORTRAN compilers, the FORTRAN compiler appends a single
 * underscore ("_") after each routine name.  C doesn't do this, so the
 * translation is made here.  Other FORTRAN compilers treat underscores
 * differently.  For example, a FORTRAN call to a_b gets translated to a call
 * to a_b__ by g77, and to a_b_ by most other FORTRAN compilers.  Thus, the
 * FORTRAN names here do not use underscores.  The xlf compiler in IBM AIX
 * doesn't add an underscore.
 *
 * The matrix A is passed to UMFPACK in compressed column form, with 0-based
 * indices.  In FORTRAN, for an m-by-n matrix A with nz entries, the row
 * indices of the first column (column 1) are in Ai (Ap (1) + 1 ... Ap (2)),
 * with values in Ax (Ap (1) + 1 ... Ap (2)).  The last column (column n) is
 * in Ai (Ap (n) + 1 ... Ap (n+1)) and Ax (Ap (n) + 1 ... Ap (n+1)).  The row
 * indices in Ai are in the range 0 to m-1.  They must be sorted, with no
 * duplicate entries allowed.  Refer to umfpack_di_triplet_to_col for a more
 * flexible format for the input matrix.  The following defintions apply
 * for each of the routines in this file:
 *
 *	integer m, n, Ap (n+1), Ai (nz), symbolic, numeric, filenum, status
 *	double precision Ax (nz), control (20), info (90), x (n), b (n)
 *
 * UMFPACK's status is returned in either a status argument, or in info (1).
 * It is zero if everything is OK, 1 if the matrix is singular (this is a
 * warning, not an error), and negative if an error occurred.  See umfpack.h
 * for more details on the contents of the control and info arrays, and the
 * value of the sys argument.
 *
 * For the Numeric and Symbolic handles, it's probably safe to assume that a
 * FORTRAN integer is sufficient to store a C pointer.  If that doesn't work,
 * try defining numeric and symbolic as integer arrays of size 2, or as
 * integer*8, in the FORTRAN routine that calls these wrapper routines.
 * The latter is required on Solaris, SGI IRIX, and IBM AIX when UMFPACK is
 * compiled in 64-bit mode.
 *
 * If you want to use 64-bit integers, try compiling this file with the -DDLONG
 * compiler option (via "make fortran64").  First modify your
 * SuiteSparse_config.mk
 * file to compile UMFPACK in LP64 mode (see the User Guide for details).
 * Your FORTRAN code should use integer*8.  See umf4hb64.f for an example.
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

#if defined (DLONG)

#define Int SuiteSparse_long
#define UMFPACK_defaults	 umfpack_dl_defaults
#define UMFPACK_free_numeric	 umfpack_dl_free_numeric
#define UMFPACK_free_symbolic	 umfpack_dl_free_symbolic
#define UMFPACK_numeric		 umfpack_dl_numeric
#define UMFPACK_report_control	 umfpack_dl_report_control
#define UMFPACK_report_info	 umfpack_dl_report_info
#define UMFPACK_save_numeric	 umfpack_dl_save_numeric
#define UMFPACK_save_symbolic	 umfpack_dl_save_symbolic
#define UMFPACK_load_numeric	 umfpack_dl_load_numeric
#define UMFPACK_load_symbolic	 umfpack_dl_load_symbolic
#define UMFPACK_scale		 umfpack_dl_scale
#define UMFPACK_solve		 umfpack_dl_solve
#define UMFPACK_symbolic	 umfpack_dl_symbolic

#else

#define Int int
#define UMFPACK_defaults	 umfpack_di_defaults
#define UMFPACK_free_numeric	 umfpack_di_free_numeric
#define UMFPACK_free_symbolic	 umfpack_di_free_symbolic
#define UMFPACK_numeric		 umfpack_di_numeric
#define UMFPACK_report_control	 umfpack_di_report_control
#define UMFPACK_report_info	 umfpack_di_report_info
#define UMFPACK_save_numeric	 umfpack_di_save_numeric
#define UMFPACK_save_symbolic	 umfpack_di_save_symbolic
#define UMFPACK_load_numeric	 umfpack_di_load_numeric
#define UMFPACK_load_symbolic	 umfpack_di_load_symbolic
#define UMFPACK_scale		 umfpack_di_scale
#define UMFPACK_solve		 umfpack_di_solve
#define UMFPACK_symbolic	 umfpack_di_symbolic

#endif

/* -------------------------------------------------------------------------- */
/* construct a file name from a file number (not user-callable) */
/* -------------------------------------------------------------------------- */

static void make_filename (Int filenum, char *prefix, char *filename)
{
    char *psrc, *pdst ;
#ifdef DLONG
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
/* umf4def: set default control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4def (control) */

void umf4def_ (double Control [UMFPACK_CONTROL])
{
    UMFPACK_defaults (Control) ;
}

/* -------------------------------------------------------------------------- */
/* umf4pcon: print control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4pcon (control) */

void umf4pcon_ (double Control [UMFPACK_CONTROL])
{
    fflush (stdout) ;
    UMFPACK_report_control (Control) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4sym: pre-ordering and symbolic factorization */
/* -------------------------------------------------------------------------- */

/* call umf4sym (m, n, Ap, Ai, Ax, symbolic, control, info) */

void umf4sym_ (Int *m, Int *n, Int Ap [ ], Int Ai [ ],
    double Ax [ ], void **Symbolic,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_symbolic (*m, *n, Ap, Ai, Ax, Symbolic, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4num: numeric factorization */
/* -------------------------------------------------------------------------- */

/* call umf4num (Ap, Ai, Ax, symbolic, numeric, control, info) */

void umf4num_ (Int Ap [ ], Int Ai [ ], double Ax [ ],
    void **Symbolic, void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_numeric (Ap, Ai, Ax, *Symbolic, Numeric, Control, Info);
}

/* -------------------------------------------------------------------------- */
/* umf4solr: solve a linear system with iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4solr (sys, Ap, Ai, Ax, x, b, numeric, control, info) */

void umf4solr_ (Int *sys, Int Ap [ ], Int Ai [ ], double Ax [ ],
    double x [ ], double b [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_solve (*sys, Ap, Ai, Ax, x, b, *Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4sol: solve a linear system without iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4sol (sys, x, b, numeric, control, info) */

void umf4sol_ (Int *sys, double x [ ], double b [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    Control [UMFPACK_IRSTEP] = 0 ;
    (void) UMFPACK_solve (*sys, (Int *) NULL, (Int *) NULL, (double *) NULL,
	x, b, *Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4scal: scale a vector using UMFPACK's scale factors */
/* -------------------------------------------------------------------------- */

/* call umf4scal (x, b, numeric, status) */

void umf4scal_ (double x [ ], double b [ ], void **Numeric, Int *status)
{
    *status = UMFPACK_scale (x, b, *Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4pinf: print info */
/* -------------------------------------------------------------------------- */

/* call umf4pinf (control) */

void umf4pinf_ (double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    fflush (stdout) ;
    UMFPACK_report_info (Control, Info) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4fnum: free the Numeric object */
/* -------------------------------------------------------------------------- */

/* call umf4fnum (numeric) */

void umf4fnum_ (void **Numeric)
{
    UMFPACK_free_numeric (Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4fsym: free the Symbolic object */
/* -------------------------------------------------------------------------- */

/* call umf4fsym (symbolic) */

void umf4fsym_ (void **Symbolic)
{
    UMFPACK_free_symbolic (Symbolic) ;
}

/* -------------------------------------------------------------------------- */
/* umf4snum: save the Numeric object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4snum (numeric, filenum, status) */

void umf4snum_ (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_save_numeric (*Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4ssym: save the Symbolic object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4ssym (symbolic, filenum, status) */

void umf4ssym_ (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_save_symbolic (*Symbolic, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4lnum: load the Numeric object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4lnum (numeric, filenum, status) */

void umf4lnum_ (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_load_numeric (Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4lsym: load the Symbolic object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4lsym (symbolic, filenum, status) */

void umf4lsym_ (void **Symbolic, Int *filenum, Int *status)
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
/* umf4def: set default control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4def (control) */

void umf4def (double Control [UMFPACK_CONTROL])
{
    UMFPACK_defaults (Control) ;
}

/* -------------------------------------------------------------------------- */
/* umf4pcon: print control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4pcon (control) */

void umf4pcon (double Control [UMFPACK_CONTROL])
{
    fflush (stdout) ;
    UMFPACK_report_control (Control) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4sym: pre-ordering and symbolic factorization */
/* -------------------------------------------------------------------------- */

/* call umf4sym (m, n, Ap, Ai, Ax, symbolic, control, info) */

void umf4sym (Int *m, Int *n, Int Ap [ ], Int Ai [ ],
    double Ax [ ], void **Symbolic,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_symbolic (*m, *n, Ap, Ai, Ax, Symbolic, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4num: numeric factorization */
/* -------------------------------------------------------------------------- */

/* call umf4num (Ap, Ai, Ax, symbolic, numeric, control, info) */

void umf4num (Int Ap [ ], Int Ai [ ], double Ax [ ],
    void **Symbolic, void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_numeric (Ap, Ai, Ax, *Symbolic, Numeric, Control, Info);
}

/* -------------------------------------------------------------------------- */
/* umf4solr: solve a linear system with iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4solr (sys, Ap, Ai, Ax, x, b, numeric, control, info) */

void umf4solr (Int *sys, Int Ap [ ], Int Ai [ ], double Ax [ ],
    double x [ ], double b [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_solve (*sys, Ap, Ai, Ax, x, b, *Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4sol: solve a linear system without iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4sol (sys, x, b, numeric, control, info) */

void umf4sol (Int *sys, double x [ ], double b [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    Control [UMFPACK_IRSTEP] = 0 ;
    (void) UMFPACK_solve (*sys, (Int *) NULL, (Int *) NULL, (double *) NULL,
	x, b, *Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4scal: scale a vector using UMFPACK's scale factors */
/* -------------------------------------------------------------------------- */

/* call umf4scal (x, b, numeric, status) */

void umf4scal (double x [ ], double b [ ], void **Numeric, Int *status)
{
    *status = UMFPACK_scale (x, b, *Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4pinf: print info */
/* -------------------------------------------------------------------------- */

/* call umf4pinf (control) */

void umf4pinf (double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    fflush (stdout) ;
    UMFPACK_report_info (Control, Info) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4fnum: free the Numeric object */
/* -------------------------------------------------------------------------- */

/* call umf4fnum (numeric) */

void umf4fnum (void **Numeric)
{
    UMFPACK_free_numeric (Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4fsym: free the Symbolic object */
/* -------------------------------------------------------------------------- */

/* call umf4fsym (symbolic) */

void umf4fsym (void **Symbolic)
{
    UMFPACK_free_symbolic (Symbolic) ;
}

/* -------------------------------------------------------------------------- */
/* umf4snum: save the Numeric object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4snum (numeric, filenum, status) */

void umf4snum (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_save_numeric (*Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4ssym: save the Symbolic object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4ssym (symbolic, filenum, status) */

void umf4ssym (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_save_symbolic (*Symbolic, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4lnum: load the Numeric object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4lnum (numeric, filenum, status) */

void umf4lnum (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_load_numeric (Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4lsym: load the Symbolic object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4lsym (symbolic, filenum, status) */

void umf4lsym (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_load_symbolic (Symbolic, filename) ;
}
