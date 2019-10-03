// =============================================================================
// === spqr_cumsum =============================================================
// =============================================================================

#include "spqr.hpp"

//  Overwrite a vector of length n with its cumulative sum of length n+1.
//  On input, X [0:n-1] contains the counts.  On output, X [k] is replaced with
//  the sum of the input counts X [0:k-1].  No work is done if n < 0 or if
//  X is NULL.  Long overflow is not checked (SuiteSparseQR doesn't need it;
//  it has already been checked whereever this function is used).

Long spqr_cumsum
(
    // input, not modified
    Long n,

    // input/output
    Long *X                         // size n+1. X = cumsum ([0 X])
)
{
    Long itot, t, x, k ;

    // -------------------------------------------------------------------------
    // X = cumsum ([0 X])
    // -------------------------------------------------------------------------

    itot = 0 ;
    if (X != NULL)
    {
        for (k = 0 ; k < n ; k++)
        {
            t = itot ;              // t = sum (X [0:k-1])
            x = X [k] ;
            itot += x ;
            X [k] = t ;
        }
        X [n] = itot ;
    }

    return (itot) ;
}
