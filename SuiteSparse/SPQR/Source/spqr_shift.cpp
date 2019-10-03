// =============================================================================
// === spqr_shift ==============================================================
// =============================================================================

// Insert a zero as the first entry in a vector, shifting all the other entries
// down by one position.  Does nothing if X is NULL.

#include "spqr.hpp"

void spqr_shift
(
    // input, not modified
    Long n,

    // input/output
    Long *X                     // size n+1
)
{
    Long k ;
    if (X != NULL)
    {
        for (k = n ; k >= 1 ; k--)
        {
            X [k] = X [k-1] ;
        }
        X [0] = 0 ;
    }
}
