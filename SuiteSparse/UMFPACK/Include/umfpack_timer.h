/* ========================================================================== */
/* === umfpack_timer ======================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

double umfpack_timer ( void ) ;

/*
Syntax (for all versions: di, dl, zi, and zl):

    #include "umfpack.h"
    double t ;
    t = umfpack_timer ( ) ;

Purpose:

    Returns the current wall clock time on POSIX C 1993 systems.

Arguments:

    None.
*/
