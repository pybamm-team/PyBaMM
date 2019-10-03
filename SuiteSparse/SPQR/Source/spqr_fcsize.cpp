// =============================================================================
// === spqr_fcsize =============================================================
// =============================================================================

#include "spqr.hpp"

Long spqr_fcsize    // returns # of entries in C of current front F
(
    // input, not modified
    Long m,                 // # of rows in F
    Long n,                 // # of columns in F
    Long npiv,              // number of pivotal columns in F
    Long rank               // the C block starts at F (rank,npiv)
)
{
    Long cm, cn, csize ;
    ASSERT (m >= 0 && n >= 0 && npiv >= 0 && npiv <= n) ;
    ASSERT (rank >= 0 && rank <= MIN (m,npiv)) ;
    cn = n - npiv ;                         // number of columns of C
    cm = MIN (m-rank, cn) ;                 // number of rows of C
    ASSERT (cm <= cn) ;
    // Note that this is safe from Long overflow:
    csize = (cm * (cm+1)) / 2 + cm * (cn - cm) ;
    return (csize) ;                        // return # of entries in C
}
