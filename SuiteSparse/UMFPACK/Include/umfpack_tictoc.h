/* ========================================================================== */
/* === umfpack_tictoc ======================================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

void umfpack_tic (double stats [2]) ;

void umfpack_toc (double stats [2]) ;


/*
Syntax (for all versions: di, dl, zi, and zl):

    #include "umfpack.h"
    double stats [2] ;
    umfpack_tic (stats) ;
    ...
    umfpack_toc (stats) ;

Purpose:

    umfpack_tic returns the wall clock time.
    umfpack_toc returns the wall clock time since the
    last call to umfpack_tic with the same stats array.

    Typical usage:

	umfpack_tic (stats) ;
	... do some work ...
	umfpack_toc (stats) ;

    then stats [0] contains the elapsed wall clock time in seconds between
    umfpack_tic and umfpack_toc.

Arguments:

    double stats [2]:

	stats [0]:  wall clock time, in seconds
	stats [1]:  (same; was CPU time in prior versions)
*/
