/* ========================================================================== */
/* === UMF_valid_numeric ==================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/* Returns TRUE if the Numeric object is valid, FALSE otherwise. */
/* Does not check everything.  UMFPACK_report_numeric checks more. */

#include "umf_internal.h"
#include "umf_valid_numeric.h"

GLOBAL Int UMF_valid_numeric
(
    NumericType *Numeric
)
{
    /* This routine does not check the contents of the individual arrays, so */
    /* it can miss some errors.  All it checks for is the presence of the */
    /* arrays, and the Numeric "valid" entry. */

    if (!Numeric)
    {
	return (FALSE) ;
    }

    if (Numeric->valid != NUMERIC_VALID)
    {
	/* Numeric does not point to a NumericType object */
	return (FALSE) ;
    }

    if (Numeric->n_row <= 0 || Numeric->n_col <= 0 || !Numeric->D ||
	!Numeric->Rperm || !Numeric->Cperm ||
	!Numeric->Lpos || !Numeric->Upos ||
	!Numeric->Lilen || !Numeric->Uilen || !Numeric->Lip || !Numeric->Uip ||
	!Numeric->Memory || (Numeric->ulen > 0 && !Numeric->Upattern))
    {
	return (FALSE) ;
    }

    return (TRUE) ;
}
