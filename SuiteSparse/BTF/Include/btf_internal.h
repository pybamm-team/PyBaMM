/* ========================================================================== */
/* === btf_internal include file ============================================ */
/* ========================================================================== */

#ifndef _BTF_INTERNAL_H
#define _BTF_INTERNAL_H

/*
 * By Tim Davis.  Copyright (c) 2004-2007, University of Florida.
 * with support from Sandia National Laboratories.  All Rights Reserved.
 */

/* Not to be included in any user program. */

#ifdef DLONG
#define Int SuiteSparse_long
#define Int_id SuiteSparse_long_id
#define BTF(name) btf_l_ ## name
#else
#define Int int
#define Int_id "%d"
#define BTF(name) btf_ ## name
#endif

/* ========================================================================== */
/* make sure debugging and printing is turned off */

#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef NPRINT
#define NPRINT
#endif

/* To enable debugging and assertions, uncomment this line: 
 #undef NDEBUG
*/
/* To enable diagnostic printing, uncomment this line: 
 #undef NPRINT
*/

/* ========================================================================== */

#include <stdio.h>
#include <assert.h>
#define ASSERT(a) assert(a)

#undef TRUE
#undef FALSE
#undef PRINTF
#undef MIN

#ifndef NPRINT
#define PRINTF(s) { printf s ; } ;
#else
#define PRINTF(s)
#endif

#define TRUE 1
#define FALSE 0
#define EMPTY (-1)
#define MIN(a,b) (((a) < (b)) ?  (a) : (b))

#endif
