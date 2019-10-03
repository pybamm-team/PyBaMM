/* ========================================================================== */
/* === CHOLMOD/Include/cholmod_function.h ================================ */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Include/cholmod_function.h
 * Copyright (C) 2014, Timothy A. Davis
 * This specific file (CHOLMOD/Include/cholmod_function.h) has no license
 * restrictions at all.  You may freely include this in your applications, and
 * modify it at will.
 * -------------------------------------------------------------------------- */

/* Memory management, printing, and math function pointers were removed from
   the CHOLMOD Common struct as of version 2.2.0 and later.  They now appear in
   SuiteSparse_config.h instead.  This file assists in backward compatibility,
   so that you can use either old or new versions of CHOLMOD and SuiteSparse in
   an application that uses the function pointers.  You can copy the file into
   your own application that uses older versions of CHOLMOD, or the current
   version, so that you have a transparent method for setting these function
   pointers for any version of CHOLMOD and SuiteSparse.

   In both old and new versions of CHOLMOD (and SuiteSparse), the intent of
   these function pointers is that they are not to be called directly.
   Instead, you should use (for example), the cholmod_malloc function.  That
   function is a wrapper that then uses the cc->malloc_memory or
   SuiteSparse_config.malloc_func function pointers.

   In each of the macros below, 'cc' is a pointer to the CHOLMOD Common struct. 

   Usage:  to assign, say, 'malloc' as your memory allocator, use this:

        #include "cholmod_function.h"
        ...
        cholmod_common *cc, Common ;
        cc = &Common ;
        cholmod_start (cc) ;
        ...
        CHOLMOD_FUNCTION_DEFAULTS ;
        CHOLMOD_FUNCTION_MALLOC (cc) = mymalloc ;

    instead of this, in older versions of CHOLMOD:

        cc->malloc_memory = mymalloc ;

    or in newer versions of CHOLMOD:

        SuiteSparse_config.malloc_func = mymalloc ;
*/

#ifndef CHOLMOD_FUNCTION_H
#define CHOLMOD_FUNCTION_H

#include "cholmod.h"

/* -------------------------------------------------------------------------- */
/* location of function pointers, depending on the CHOLMOD version */
/* -------------------------------------------------------------------------- */

#if (CHOLMOD_VERSION < (CHOLMOD_VER_CODE(2,2)))

    #define CHOLMOD_FUNCTION_MALLOC(cc)     cc->malloc_memory
    #define CHOLMOD_FUNCTION_REALLOC(cc)    cc->realloc_memory
    #define CHOLMOD_FUNCTION_FREE(cc)       cc->free_memory
    #define CHOLMOD_FUNCTION_CALLOC(cc)     cc->calloc_memory
    #define CHOLMOD_FUNCTION_PRINTF(cc)     cc->print_function
    #define CHOLMOD_FUNCTION_DIVCOMPLEX(cc) cc->complex_divide
    #define CHOLMOD_FUNCTION_HYPOTENUSE(cc) cc->hypotenuse

#else

    #include "SuiteSparse_config.h"
    #define CHOLMOD_FUNCTION_MALLOC(cc)     SuiteSparse_config.malloc_func
    #define CHOLMOD_FUNCTION_REALLOC(cc)    SuiteSparse_config.realloc_func
    #define CHOLMOD_FUNCTION_FREE(cc)       SuiteSparse_config.free_func
    #define CHOLMOD_FUNCTION_CALLOC(cc)     SuiteSparse_config.calloc_func
    #define CHOLMOD_FUNCTION_PRINTF(cc)     SuiteSparse_config.printf_func
    #define CHOLMOD_FUNCTION_DIVCOMPLEX(cc) SuiteSparse_config.divcomplex_func
    #define CHOLMOD_FUNCTION_HYPOTENUSE(cc) SuiteSparse_config.hypot_func

#endif

/* -------------------------------------------------------------------------- */
/* default math functions, depending on the CHOLMOD version */
/* -------------------------------------------------------------------------- */

#if (CHOLMOD_VERSION < (CHOLMOD_VER_CODE(2,2)))

    #define CHOLMOD_FUNCTION_DEFAULT_DIVCOMPLEX cholmod_l_divcomplex
    #define CHOLMOD_FUNCTION_DEFAULT_HYPOTENUSE cholmod_l_hypot

#else

    #define CHOLMOD_FUNCTION_DEFAULT_DIVCOMPLEX SuiteSparse_divcomplex
    #define CHOLMOD_FUNCTION_DEFAULT_HYPOTENUSE SuiteSparse_hypot

#endif

/* -------------------------------------------------------------------------- */
/* default memory manager functions */
/* -------------------------------------------------------------------------- */

#ifndef NMALLOC
    #ifdef MATLAB_MEX_FILE
        /* MATLAB mexFunction */
        #define CHOLMOD_FUNCTION_DEFAULT_MALLOC  mxMalloc
        #define CHOLMOD_FUNCTION_DEFAULT_CALLOC  mxCalloc
        #define CHOLMOD_FUNCTION_DEFAULT_REALLOC mxRealloc
        #define CHOLMOD_FUNCTION_DEFAULT_FREE    mxFree
    #else
        /* standard ANSI C */
        #define CHOLMOD_FUNCTION_DEFAULT_MALLOC  malloc
        #define CHOLMOD_FUNCTION_DEFAULT_CALLOC  calloc
        #define CHOLMOD_FUNCTION_DEFAULT_REALLOC realloc
        #define CHOLMOD_FUNCTION_DEFAULT_FREE    free
    #endif
#else
    /* no memory manager defined at compile time */
    #define CHOLMOD_FUNCTION_DEFAULT_MALLOC  NULL
    #define CHOLMOD_FUNCTION_DEFAULT_CALLOC  NULL
    #define CHOLMOD_FUNCTION_DEFAULT_REALLOC NULL
    #define CHOLMOD_FUNCTION_DEFAULT_FREE    NULL
#endif

/* -------------------------------------------------------------------------- */
/* default printf function */
/* -------------------------------------------------------------------------- */

#ifdef MATLAB_MEX_FILE
    #define CHOLMOD_FUNCTION_DEFAULT_PRINTF mexPrintf
#else
    #define CHOLMOD_FUNCTION_DEFAULT_PRINTF printf
#endif

/* -------------------------------------------------------------------------- */
/* set all the defaults */
/* -------------------------------------------------------------------------- */

/* Use this macro to initialize all the function pointers to their defaults 
   for any version of CHOLMOD.  For CHOLMD 2.2.0 and later, it sets function
   pointers in the SuiteSparse_config struct.  For older versions, it sets
   function pointers in the CHOLMOD Common.  This assignment is not
   thread-safe, and should be done before launching any threads. */

#define CHOLMOD_FUNCTION_DEFAULTS \
{ \
    CHOLMOD_FUNCTION_MALLOC (cc)     = CHOLMOD_FUNCTION_DEFAULT_MALLOC ; \
    CHOLMOD_FUNCTION_REALLOC (cc)    = CHOLMOD_FUNCTION_DEFAULT_REALLOC ; \
    CHOLMOD_FUNCTION_FREE (cc)       = CHOLMOD_FUNCTION_DEFAULT_FREE ; \
    CHOLMOD_FUNCTION_CALLOC (cc)     = CHOLMOD_FUNCTION_DEFAULT_CALLOC ; \
    CHOLMOD_FUNCTION_PRINTF (cc)     = CHOLMOD_FUNCTION_DEFAULT_PRINTF ; \
    CHOLMOD_FUNCTION_DIVCOMPLEX (cc) = CHOLMOD_FUNCTION_DEFAULT_DIVCOMPLEX ; \
    CHOLMOD_FUNCTION_HYPOTENUSE (cc) = CHOLMOD_FUNCTION_DEFAULT_HYPOTENUSE ; \
}

#endif
