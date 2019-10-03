/* ========================================================================== */
/* === Core/cholmod_version ================================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Core Module.  Copyright (C) 2005-2013,
 * Univ. of Florida.  Author: Timothy A. Davis
 * -------------------------------------------------------------------------- */

/* Return the current version of CHOLMOD.  Unlike all other functions in
   CHOLMOD, this function does not require the CHOLMOD Common. */

#include "cholmod_internal.h"
#include "cholmod_core.h"

int CHOLMOD(version)        /* returns CHOLMOD_VERSION */
(
    /* output, contents not defined on input.  Not used if NULL.
        version [0] = CHOLMOD_MAIN_VERSION ;
        version [1] = CHOLMOD_SUB_VERSION ;
        version [2] = CHOLMOD_SUBSUB_VERSION ;
    */
    int version [3]
)
{
    if (version != NULL)
    {
        version [0] = CHOLMOD_MAIN_VERSION ;
        version [1] = CHOLMOD_SUB_VERSION ;
        version [2] = CHOLMOD_SUBSUB_VERSION ;
    }
    return (CHOLMOD_VERSION) ;
}

