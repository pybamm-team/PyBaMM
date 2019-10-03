// =============================================================================
// === spqr_hpinv ==============================================================
// =============================================================================

// Finalizes the row permutation that is implicit in the pattern of H.  This
// must be done sequentially, after all threads have finished factorizing the
// matrix and finding all the dead columns.  Also determines QRnum->maxfm.

#include "spqr.hpp"

template <typename Entry> void spqr_hpinv
(
    // input
    spqr_symbolic *QRsym,
    // input/output
    spqr_numeric <Entry> *QRnum,
    // workspace
    Long *W              // size QRnum->m
)
{
    Long *Hi, *Hii, *Hip, *HPinv, *Hr, *Super, *Rp, *Hm, *Sleft, *PLinv ;
    Long nf, m, n, f, rm, i, row1, row2, fm, fn, fp, cm, cn, maxfm ;

    // -------------------------------------------------------------------------
    // get the contents of the QRsym and QRnum objects
    // -------------------------------------------------------------------------

    // this function must not be called if Householder vectors weren't kept
    ASSERT (QRnum->keepH) ;

    nf = QRsym->nf ;
    m = QRsym->m ;
    n = QRsym->n ;
    Hr = QRnum->Hr ;
    Hm = QRnum->Hm ;
    Hii = QRnum->Hii ;
    Hip = QRsym->Hip ;
    HPinv = QRnum->HPinv ;
    Super = QRsym->Super ;
    Rp = QRsym->Rp ;
    Sleft = QRsym->Sleft ;
    PLinv = QRsym->PLinv ;
    maxfm = 0 ;

#ifndef NDEBUG
    for (f = 0 ; f < nf ; f++)
    {
        Long j ;
        rm = 0 ;
        for (j = Super [f] ; j < Super [f+1] ; j++)
        {
            if (!(QRnum->Rdead [j]))
            {
                rm++ ;                      // column j is not dead
            }
        }
        ASSERT (Hr [f] == rm) ;             // # rows in R block
    }
    for (i = 0 ; i < m ; i++)
    {
        W [i] = EMPTY ;
        PR (("For S, PLinv row perm (%ld) = %ld\n", i, PLinv [i])) ;
    }
#endif

    // -------------------------------------------------------------------------
    // extract the inverse permutation for R1
    // -------------------------------------------------------------------------

    row1 = 0 ;                              // number of squeezed rows of R
    row2 = m ;                              // for ordering empty rows of R

    // order the empty rows of S
    ASSERT (Sleft [n+1] == m) ;
    PR (("Sleft [%ld] = %ld, m = %ld\n", n, Sleft [n], m)) ;
    for (i = Sleft [n] ; i < m ; i++)
    {
        // row i of S is empty, and so it does not appear as a row in the Hi
        // pattern of any front
        PR (("empty row %ld\n", i)) ;
        W [i] = (--row2) ;
    }

    // order the non-empty rows of S that appear in frontal matrices
    for (f = 0 ; f < nf ; f++)              // traverse in natural order
    {
        Hi = &Hii [Hip [f]] ;               // list of row indices of H

        // order the pivotal rows of F
        rm = Hr [f] ;                       // number of rows in R block
        for (i = 0 ; i < rm ; i++)
        {
            // row1 is a row of R; it is row Hi [i] of H and S
            ASSERT (Hi [i] >= 0 && Hi [i] < m) ;
            ASSERT (W [Hi [i]] == EMPTY) ;
            W [Hi [i]] = row1++ ;
        }

        // order the non-pivotal rows of F which are not in the C block
        fp = Super [f+1] - Super [f] ;
        fn = Rp [f+1] - Rp [f] ;
        fm = Hm [f] ;
        maxfm = MAX (maxfm, fm) ;
        cn = fn - fp ;
        cm = MIN (fm - rm, cn) ;
        for (i = fm-1 ; i >= rm + cm ; i--)
        {
            // row2 is an empty row of R (does not appear in the rank-size R);
            // it is row Hi [i] of H and S
            ASSERT (Hi [i] >= 0 && Hi [i] < m) ;
            ASSERT (W [Hi [i]] == EMPTY) ;
            W [Hi [i]] = (--row2) ;
        }
    }
    ASSERT (row1 == QRnum->rank) ;
    ASSERT (row1 == row2) ;

    QRnum->maxfm = maxfm ;

#ifndef NDEBUG
    for (i = 0 ; i < m ; i++)
    {
        PR (("H row perm W (%ld) = %ld\n", i, W [i])) ;
        ASSERT (W [i] >= 0 && W [i] < m) ;
    }
#endif

    // -------------------------------------------------------------------------
    // combine the permutations
    // -------------------------------------------------------------------------

    // At this point, W [i] = k if row i of S is row k of R, and
    // PLinv [i] = k if row i of A is row k of S.  Combine the two permutations
    // into a single one, HPinv [row i of A] = row k of R

    for (i = 0 ; i < m ; i++)
    {
        HPinv [i] = W [PLinv [i]] ;
        PR (("Combined H row perm HPinv (%ld) = %ld\n", i, HPinv [i])) ;
    }

    // -------------------------------------------------------------------------
    // revise the pattern of the frontal matrices
    // -------------------------------------------------------------------------

    for (f = 0 ; f < nf ; f++)
    {
        Hi = &Hii [Hip [f]] ;                   // list of row indices of H
        fm = Hm [f] ;
        for (i = 0 ; i < fm ; i++)
        {
            ASSERT (Hi [i] >= 0 && Hi [i] < m) ;
            Hi [i] = W [Hi [i]] ;
        }
        // Now Hi [0..fm-1] contains a list of row indices for front F which
        // correspond to rows of R.  The row indices are also sorted.
#ifndef NDEBUG
        for (i = 1 ; i < fm ; i++)
        {
            ASSERT (Hi [i] > Hi [i-1]) ;
        }
#endif
    }
}


// =============================================================================

template void spqr_hpinv <double>
(
    // input
    spqr_symbolic *QRsym,
    // input/output
    spqr_numeric <double> *QRnum,
    // workspace
    Long *W              // size QRnum->m
) ;

// =============================================================================

template void spqr_hpinv <Complex>
(
    // input
    spqr_symbolic *QRsym,
    // input/output
    spqr_numeric <Complex> *QRnum,
    // workspace
    Long *W              // size QRnum->m
) ;
