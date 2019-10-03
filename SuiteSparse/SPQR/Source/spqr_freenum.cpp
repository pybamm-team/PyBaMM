// =============================================================================
// === spqr_freenum ============================================================
// =============================================================================

// Frees the contents of the QR Numeric object

#include "spqr.hpp"

template <typename Entry> void spqr_freenum
(
    spqr_numeric <Entry> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
)
{
    spqr_numeric <Entry> *QRnum ;
    Long nf, n, m, rjsize, hisize, ns, stack, maxstack ;

    if (QRnum_handle == NULL || *QRnum_handle == NULL)
    {
        // nothing to do; caller probably ran out of memory
        return ;
    }
    QRnum = *QRnum_handle ;

    n  = QRnum->n ;
    m  = QRnum->m ;
    nf = QRnum->nf ;
    rjsize = QRnum->rjsize ;
    hisize = QRnum->hisize ;
    ns = QRnum->ns ;
    maxstack = QRnum->maxstack ;

    cholmod_l_free (nf, sizeof (Entry *), QRnum->Rblock, cc) ;
    cholmod_l_free (n,  sizeof (char),    QRnum->Rdead,  cc) ;

    if (QRnum->keepH)
    {
        // QRnum->H* items are present only if H is kept
        cholmod_l_free (rjsize, sizeof (Long),  QRnum->HStair,  cc) ;
        cholmod_l_free (rjsize, sizeof (Entry), QRnum->HTau,    cc) ;
        cholmod_l_free (nf,     sizeof (Long),  QRnum->Hm,      cc) ;
        cholmod_l_free (nf,     sizeof (Long),  QRnum->Hr,      cc) ;
        cholmod_l_free (hisize, sizeof (Long),  QRnum->Hii,     cc) ;
        cholmod_l_free (m,      sizeof (Long),  QRnum->HPinv,   cc) ;
    }

    // free each stack
    if (QRnum->Stacks != NULL)
    {
        Long *Stack_size = QRnum->Stack_size ;
        for (stack = 0 ; stack < ns ; stack++)
        {
            size_t s = Stack_size ? (Stack_size [stack]) : maxstack ;
            cholmod_l_free (s, sizeof (Entry), QRnum->Stacks [stack], cc) ;
        }
    }
    cholmod_l_free (ns, sizeof (Entry *), QRnum->Stacks, cc) ;
    cholmod_l_free (ns, sizeof (Long), QRnum->Stack_size, cc) ;

    cholmod_l_free (1, sizeof (spqr_numeric<Entry>), QRnum, cc) ;
    *QRnum_handle = NULL ;
}

// =============================================================================

template void spqr_freenum <double>
(
    spqr_numeric <double> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
) ;

// =============================================================================

template void spqr_freenum <Complex>
(
    spqr_numeric <Complex> **QRnum_handle,

    // workspace and parameters
    cholmod_common *cc
) ;
