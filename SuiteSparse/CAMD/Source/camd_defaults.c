/* ========================================================================= */
/* === CAMD_defaults ======================================================= */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,			     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/* User-callable.  Sets default control parameters for CAMD.  See camd.h
 * for details.
 */

#include "camd_internal.h"

/* ========================================================================= */
/* === CAMD defaults ======================================================= */
/* ========================================================================= */

GLOBAL void CAMD_defaults
(
    double Control [ ]
)
{
    Int i ;
    if (Control != (double *) NULL)
    {
	for (i = 0 ; i < CAMD_CONTROL ; i++)
	{
	    Control [i] = 0 ;
	}
	Control [CAMD_DENSE] = CAMD_DEFAULT_DENSE ;
	Control [CAMD_AGGRESSIVE] = CAMD_DEFAULT_AGGRESSIVE ;
    }
}
