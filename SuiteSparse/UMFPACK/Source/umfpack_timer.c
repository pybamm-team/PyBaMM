/* ========================================================================== */
/* === umfpack_timer ======================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Returns the time in seconds used by the process.  BE
    CAREFUL:  if you compare the run time of UMFPACK with other sparse matrix
    packages, be sure to use the same timer.  See umfpack_timer.h for details.
    See umfpack_tictoc.h, which is the timer used internally by UMFPACK.
*/

#include "umfpack_timer.h"
#include "SuiteSparse_config.h"

double umfpack_timer ( void )
{
    return (SuiteSparse_time ( )) ;
}
