/* ========================================================================= */
/* === CAMD_control ======================================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,			     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/* User-callable.  Prints the control parameters for CAMD.  See camd.h
 * for details.  If the Control array is not present, the defaults are
 * printed instead.
 */

#include "camd_internal.h"

GLOBAL void CAMD_control
(
    double Control [ ]
)
{
    double alpha ;
    Int aggressive ;

    if (Control != (double *) NULL)
    {
	alpha = Control [CAMD_DENSE] ;
	aggressive = Control [CAMD_AGGRESSIVE] != 0 ;
    }
    else
    {
	alpha = CAMD_DEFAULT_DENSE ;
	aggressive = CAMD_DEFAULT_AGGRESSIVE ;
    }

    SUITESPARSE_PRINTF ((
        "\ncamd version %d.%d, %s:  approximate minimum degree ordering:\n"
	"    dense row parameter: %g\n", CAMD_MAIN_VERSION, CAMD_SUB_VERSION,
	CAMD_DATE, alpha)) ;

    if (alpha < 0)
    {
	SUITESPARSE_PRINTF (("    no rows treated as dense\n")) ;
    }
    else
    {
	SUITESPARSE_PRINTF ((
	"    (rows with more than max (%g * sqrt (n), 16) entries are\n"
	"    considered \"dense\", and placed last in output permutation)\n",
	alpha)) ;
    }

    if (aggressive)
    {
	SUITESPARSE_PRINTF (("    aggressive absorption:  yes\n")) ;
    }
    else
    {
	SUITESPARSE_PRINTF (("    aggressive absorption:  no\n")) ;
    }

    SUITESPARSE_PRINTF (("    size of CAMD integer: %d\n\n", sizeof (Int))) ;
}
