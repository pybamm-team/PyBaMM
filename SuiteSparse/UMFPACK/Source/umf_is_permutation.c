/* ========================================================================== */
/* === UMF_is_permutation =================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

/* Return TRUE if P is a r-permutation vector, FALSE otherwise */
/* P [0..r-1] must be an r-permutation of 0..n-1 */

#include "umf_internal.h"
#include "umf_is_permutation.h"

GLOBAL Int UMF_is_permutation
(
    const Int P [ ],	/* permutation of size r */
    Int W [ ],		/* workspace of size n */
    Int n,
    Int r
)
{
    Int i, k ;

    if (!P)
    {
	/* if P is (Int *) NULL, this is the identity permutation */
	return (TRUE) ;
    }

    ASSERT (W != (Int *) NULL) ;

    for (i = 0 ; i < n ; i++)
    {
	W [i] = FALSE ;
    }
    for (k = 0 ; k < r ; k++)
    {
	i = P [k] ;
	DEBUG5 (("k "ID" i "ID"\n", k, i)) ;
	if (i < 0 || i >= n)
	{
	    DEBUG0 (("i out of range "ID" "ID"\n", i, n)) ;
	    return (FALSE) ;
	}
	if (W [i])
	{
	    DEBUG0 (("i duplicate "ID"\n", i)) ;
	    return (FALSE) ;
	}
	W [i] = TRUE ;
    }
    return (TRUE) ;
}
