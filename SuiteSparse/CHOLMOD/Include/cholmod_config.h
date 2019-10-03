/* ========================================================================== */
/* === Include/cholmod_config.h ============================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Include/cholmod_config.h.
 * Copyright (C) 2005-2013, Univ. of Florida.  Author: Timothy A. Davis
 * -------------------------------------------------------------------------- */

/* CHOLMOD configuration file, for inclusion in user programs.
 *
 * You do not have to edit any CHOLMOD files to compile and install CHOLMOD.
 * However, if you do not use all of CHOLMOD's modules, you need to compile
 * with the appropriate flag, or edit this file to add the appropriate #define.
 *
 * Compiler flags for CHOLMOD:
 *
 * -DNCHECK	    do not include the Check module.
 * -DNCHOLESKY	    do not include the Cholesky module.
 * -DNPARTITION	    do not include the Partition module.
 * -DNCAMD          do not include the interfaces to CAMD,
 *                  CCOLAMD, CSYMAND in Partition module.
 * -DNMATRIXOPS	    do not include the MatrixOps module.
 * -DNMODIFY	    do not include the Modify module.
 * -DNSUPERNODAL    do not include the Supernodal module.
 *
 * -DNPRINT	    do not print anything
 *
 * -D'LONGBLAS=long' or -DLONGBLAS='long long' defines the integers used by
 *		    LAPACK and the BLAS.  Use LONGBLAS=long on Solaris to use
 *		    the 64-bit Sun Performance BLAS in cholmod_l_* routines.
 *		    You may need to use -D'LONGBLAS=long long' on the SGI
 *		    (this is not tested).
 *
 * -DNSUNPERF	    for Solaris only.  If defined, do not use the Sun
 *		    Performance Library.  The default is to use SunPerf.
 *		    You must compile CHOLMOD with -xlic_lib=sunperf.
 *
 * The Core Module is always included in the CHOLMOD library.
 */

#ifndef CHOLMOD_CONFIG_H
#define CHOLMOD_CONFIG_H

/* Use the compiler flag, or uncomment the definition(s), if you want to use
 * one or more non-default installation options: */

/*
#define NCHECK
#define NCHOLESKY
#define NCAMD
#define NPARTITION

#define NMATRIXOPS
#define NMODIFY
#define NSUPERNODAL

#define NPRINT

#define LONGBLAS long
#define LONGBLAS long long
#define NSUNPERF
*/

/* The option disables the MatrixOps, Modify, and Supernodal modules.  The
    existence of this #define here, and its use in these 3 modules, does not
    affect the license itself; see CHOLMOD/Doc/License.txt for your actual
    license.
 */
#ifdef NGPL
#define NMATRIXOPS
#define NMODIFY
#define NSUPERNODAL
#endif

#endif
