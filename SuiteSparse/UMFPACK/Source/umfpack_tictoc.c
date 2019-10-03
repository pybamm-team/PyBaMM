/* ========================================================================== */
/* === umfpack_tictoc ======================================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Returns the current wall clock time.  BE CAREFUL:  if you
    compare the run time of UMFPACK with other sparse matrix packages, be sure
    to use the same timer.  See umfpack_tictoc.h for details.

    These routines conform to the POSIX standard.  See umf_config.h for
    more details.

    In a prior version, two timings were returned: wall clock and cpu time.
    In the current version, both times are set to the wall clock time.
*/

#include "umf_internal.h"

void umfpack_tic (double stats [2])
{
    stats [0] = SuiteSparse_time ( ) ;
    stats [1] = stats [0] ;
}

void umfpack_toc (double stats [2])
{
    stats [0] = SuiteSparse_time ( ) - stats [0] ;
    stats [1] = stats [0] ;
}
