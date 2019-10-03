// =============================================================================
// === spqr_rmap ===============================================================
// =============================================================================

// R is squeezed, find the mapping that permutes it to trapezoidal form

// Rmap is a permutation that converts R from squeezed to upper trapezoidal.
// If R is already in upper triangular form, then Rmap is NULL (an implicit
// identity; Rmap [0:n-1] = 0:n-1), and this function is not used.

// If R is rank deficient, Rmap [j] = k if column j of the squeezed R is the
// kth column in the trapezoidal R.  If j is a live column then
// k = Rmap [j] < QR->rank; otherwise k = Rmap [j] > QR->rank.  RmapInv is
// the inverse of Rmap.

// Example:  Suppose R has the following format:
//
//      0 1 2 3 4 5 6
//      X x x x x x x
//      . X x x x x x
//      . . . X x x x
//      . . . . . X x
//      . . . . . . X
//  
// Then Rmap is [0 1 5 2 6 3 4] and RmapInv is [0 1 3 5 6 2 4].  The rank of R
// is 5, and thus columns 2 and 4 (with Rmap [2] = 5 and Rmap [4] = 6) are both
// dead.

#include "spqr.hpp"

template <typename Entry> int spqr_rmap
(
    SuiteSparseQR_factorization <Entry> *QR,
    cholmod_common *cc
)
{
    Long n, j, i, p, n1rows, n1cols ;
    Long *Rmap, *RmapInv, *R1p, *R1j ;

    n = QR->nacols ;
    Rmap = QR->Rmap ;
    RmapInv = QR->RmapInv ;

    if (Rmap == NULL)
    {
        ASSERT (RmapInv == NULL) ;
        QR->Rmap    = Rmap    = (Long *) cholmod_l_malloc (n, sizeof(Long), cc);
        QR->RmapInv = RmapInv = (Long *) cholmod_l_malloc (n, sizeof(Long), cc);
        if (cc->status < CHOLMOD_OK)
        {
            // out of memory
            return (FALSE) ;
        }
    }

    for (j = 0 ; j < n ; j++)
    {
        Rmap [j] = EMPTY ;
    }

    R1p = QR->R1p ;
    R1j = QR->R1j ;
    n1rows = QR->n1rows ;
    n1cols = QR->n1cols ;

    // find the mapping for the singleton rows
    for (i = 0 ; i < n1rows ; i++)
    {
        // The ith row of R is a singleton row; find its corresponding
        // pivotal column.
        p = R1p [i] ;
        ASSERT (R1p [i] < R1p [i+1]) ;
        j = R1j [p] ;
        ASSERT (j >= 0 && j < n1cols) ;
        Rmap [j] = i ;
    }

    // find the mapping for the pivotal rows of the multifrontal R
    char *Rdead = QR->QRnum->Rdead ;
    for (j = n1cols ; j < n ; j++)
    {
        if (!Rdead [j-n1cols])
        {
            Rmap [j] = i++ ;
        }
    }
    ASSERT (i == QR->rank) ;

    // finish the mapping with the dead columns of R, both in the singleton
    // part of R and the multifrontal part of R
    for (j = 0 ; j < n ; j++)
    {
        if (Rmap [j] == EMPTY)
        {
            Rmap [j] = i++ ;
        }
        PR (("Rmap [%ld] = %ld (%d),  rank = %ld\n",
            j, Rmap [j], Rmap [j] >= QR->rank, QR->rank)) ;
    }
    ASSERT (i == n) ;

    // construct the inverse of Rmap
    for (j = 0 ; j < n ; j++)
    {
        i = Rmap [j] ;
        RmapInv [i] = j ;
    }
    return (TRUE) ;
}

template int spqr_rmap <double>
(
    SuiteSparseQR_factorization <double> *QR,
    cholmod_common *cc
) ;

template int spqr_rmap <Complex>
(
    SuiteSparseQR_factorization <Complex> *QR,
    cholmod_common *cc
) ;
