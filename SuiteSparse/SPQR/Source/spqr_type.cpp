// =============================================================================
// === spqr_type ===============================================================
// =============================================================================

// Return the CHOLMOD type, based on the SuiteSparseQR template Entry type.
// Note that CHOLMOD_REAL is 1 an CHOLMOD_COMPLEX is 2.

#include "spqr.hpp"

template <> int spqr_type <double> (void)
{
    return (CHOLMOD_REAL) ;
}

template <> int spqr_type <Complex> (void)
{
    return (CHOLMOD_COMPLEX) ;
}
