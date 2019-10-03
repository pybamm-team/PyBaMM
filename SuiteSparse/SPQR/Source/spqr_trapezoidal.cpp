// =============================================================================
// === spqr_trapezoidal ========================================================
// =============================================================================

// Permute the columns of a "squeezed" R, possibly rank-deficient, into
// upper trapezoidal form.  On input, Qfill gives the column permutation of
// A that gave the factor R (Q*R = A(:,Qfill).  On output, T is upper
// trapezoidal, and Qtrap is the adjusted column ordering so that
// Q*T = A(:,Qtrap).  T is permuted so that T = [T1 T2], where T1 is square
// and upper triangular (or upper trapezoidal), with nnz (diag (T1)) == rank.
// On input, the row indices of the matrix R must be sorted.

#include "spqr.hpp"

template <typename Entry> Long spqr_trapezoidal // rank of R; EMPTY on failure
(
    // inputs, not modified

    // FUTURE : make R and T cholmod_sparse:
    Long n,         // R is m-by-n (m is not needed here; can be economy R)
    Long *Rp,       // size n+1, column pointers of R
    Long *Ri,       // size rnz = Rp [n], row indices of R
    Entry *Rx,      // size rnz, numerical values of R

    Long bncols,    // number of columns of B

    Long *Qfill,    // size n+bncols, fill-reducing ordering.  Qfill [k] = j if
                    // the jth column of A is the kth column of R.  If Qfill is
                    // NULL, then it is assumed to be the identity
                    // permutation.

    int skip_if_trapezoidal,        // if R is already in trapezoidal form,
                                    // and skip_if_trapezoidal is TRUE, then
                                    // the matrix T is not created.

    // outputs, not allocated on input
    Long **p_Tp,    // size n+1, column pointers of T
    Long **p_Ti,    // size rnz, row indices of T
    Entry **p_Tx,   // size rnz, numerical values of T

    Long **p_Qtrap,  // size n+bncols, modified Qfill

    // workspace and parameters
    cholmod_common *cc
)
{
    Entry *Tx ;
    Long *Tp, *Ti, *Qtrap ;
    Long rnz, i, rank, k, p, pend, len, t1nz, t2nz, k1, k2, p1, p2, found_dead,
        is_trapezoidal ;

    // -------------------------------------------------------------------------
    // find the rank of R, nnz(T1), and nnz(T2) 
    // -------------------------------------------------------------------------

    rank = 0 ;              // rank of R
    t1nz = 0 ;              // number of nonzeros in T1
    t2nz = 0 ;              // number of nonzeros in T2
    found_dead = FALSE ;    // true when first dead column is found
    is_trapezoidal = TRUE ; // becomes false if is R not in trapezoidal form

    *p_Tp = NULL ;
    *p_Ti = NULL ;
    *p_Tx = NULL ;
    *p_Qtrap = NULL ;

    for (k = 0 ; k < n ; k++)
    {
        // get the row index of the last entry in R (:,k)
        p = Rp [k] ;
        pend = Rp [k+1] ;
        len = pend - p ;
        i = (len > 0) ? Ri [pend - 1] : EMPTY ;

        // determine status of column k
        if (i > rank)
        {
            // R is not upper triangular, squeezed or otherwise.  Do not create
            // T and Qtrap; R is left in its original non-trapezoidal form.
            PR (("R not upper, k = %ld\n", k)) ;
            return (EMPTY) ;
        }
        else if (i == rank)
        {
            // column k is live
            rank++ ;
            t1nz += len ;
            if (found_dead)
            {
                // If all live columns appear first (if any), then all dead
                // columns, then the matrix is already in upper trapezoidal
                // form.  We just found a live column after one or more dead
                // columns, so the matrix R is not upper trapezoidal.
                is_trapezoidal = FALSE ;
            }
        }
        else
        {
            // column k is dead
            found_dead = TRUE ;
            t2nz += len ;
        }
    }

    // -------------------------------------------------------------------------
    // quick return if already trapezoidal
    // -------------------------------------------------------------------------

    if (is_trapezoidal)
    {
        PR (("already trapezoidal\n")) ;
        if (skip_if_trapezoidal)
        {
            // do not create T
            return (rank) ;
        }
    }

    // -------------------------------------------------------------------------
    // allocate the results (T and Qtrap)
    // -------------------------------------------------------------------------

    rnz = Rp [n] ;

    Tp    = (Long  *) cholmod_l_malloc (n+1,      sizeof (Long),  cc) ;
    Ti    = (Long  *) cholmod_l_malloc (rnz,      sizeof (Long),  cc) ;
    Tx    = (Entry *) cholmod_l_malloc (rnz,      sizeof (Entry), cc) ;
    Qtrap = (Long  *) cholmod_l_malloc (n+bncols, sizeof (Long),  cc) ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        cholmod_l_free (n+1,      sizeof (Long),  Tp,    cc) ;
        cholmod_l_free (rnz,      sizeof (Long),  Ti,    cc) ;
        cholmod_l_free (rnz,      sizeof (Entry), Tx,    cc) ;
        cholmod_l_free (n+bncols, sizeof (Long),  Qtrap, cc) ;
        return (EMPTY) ;
    }

    PR (("rank %ld of T, nnz(T1) %ld nnz(T2) %ld\n", rank, t1nz, t2nz)) ;

    // -------------------------------------------------------------------------
    // find the column pointers Tp and permutation Qtrap
    // -------------------------------------------------------------------------

    k1 = 0 ;                // first column of T1
    k2 = rank ;             // first column of T2
    p1 = 0 ;                // T1 starts at Ti,Tx [0]
    p2 = t1nz ;             // T2 starts at Ti,Tx [p2]
    rank = 0 ;              // rank of R

    for (k = 0 ; k < n ; k++)
    {
        // get the row index of the last entry in R (:,k)
        p = Rp [k] ;
        pend = Rp [k+1] ;
        len = pend - p ;
        i = (len > 0) ? Ri [pend - 1] : EMPTY ;

        // column k is live if i is a new row index
        ASSERT (! (i > rank)) ;
        if (i == rank)
        {
            // column k is live; place it in T1
            rank++ ;
            Tp [k1] = p1 ;
            Qtrap [k1] = Qfill ? Qfill [k] : k ;
            k1++ ;
            for ( ; p < pend ; p++)
            {
                Ti [p1] = Ri [p] ;
                Tx [p1] = Rx [p] ;
                p1++ ;
            }
        }
        else
        {
            // column k is dead; place it in T2
            Tp [k2] = p2 ;
            Qtrap [k2] = Qfill ? Qfill [k] : k ;
            k2++ ;
            for ( ; p < pend ; p++)
            {
                Ti [p2] = Ri [p] ;
                Tx [p2] = Rx [p] ;
                p2++ ;
            }
        }
    }

    for ( ; k < n+bncols ; k++)
    {
        Qtrap [k] = Qfill ? Qfill [k] : k ;
    }

    // -------------------------------------------------------------------------
    // finalize the column pointers and return the results
    // -------------------------------------------------------------------------

    ASSERT (k1 == rank) ;
    ASSERT (k2 == n) ;
    ASSERT (p1 == t1nz) ;
    ASSERT (p2 == rnz) ;
    Tp [n] = rnz ;
    *p_Tp = Tp ;
    *p_Ti = Ti ;
    *p_Tx = Tx ;
    *p_Qtrap = Qtrap ;
    return (rank) ;
}


// =============================================================================

template Long spqr_trapezoidal <double>      // rank of R, or EMPTY on failure
(
    // inputs, not modified

    Long n,         // R is m-by-n (m is not needed here; can be economy R)
    Long *Rp,       // size n+1, column pointers of R
    Long *Ri,       // size rnz = Rp [n], row indices of R
    double *Rx,     // size rnz, numerical values of R

    Long bncols,    // number of columns of B

    Long *Qfill,    // size n+bncols, fill-reducing ordering.  Qfill [k] = j if
                    // the jth column of A is the kth column of R.  If Qfill is
                    // NULL, then it is assumed to be the identity
                    // permutation.

    int skip_if_trapezoidal,        // if R is already in trapezoidal form,
                                    // and skip_if_trapezoidal is TRUE, then
                                    // the matrix T is not created.

    // outputs, not allocated on input
    Long **p_Tp,    // size n+1, column pointers of T
    Long **p_Ti,    // size rnz, row indices of T
    double **p_Tx,  // size rnz, numerical values of T

    Long **p_Qtrap,  // size n+bncols, modified Qfill

    // workspace and parameters
    cholmod_common *cc
) ;

// =============================================================================

template Long spqr_trapezoidal <Complex>     // rank of R, or EMPTY on failure
(
    // inputs, not modified

    Long n,         // R is m-by-n (m is not needed here; can be economy R)
    Long *Rp,       // size n+1, column pointers of R
    Long *Ri,       // size rnz = Rp [n], row indices of R
    Complex *Rx,    // size rnz, numerical values of R

    Long bncols,    // number of columns of B

    Long *Qfill,    // size n+bncols, fill-reducing ordering.  Qfill [k] = j if
                    // the jth column of A is the kth column of R.  If Qfill is
                    // NULL, then it is assumed to be the identity
                    // permutation.

    int skip_if_trapezoidal,        // if R is already in trapezoidal form,
                                    // and skip_if_trapezoidal is TRUE, then
                                    // the matrix T is not created.

    // outputs, not allocated on input
    Long **p_Tp,    // size n+1, column pointers of T
    Long **p_Ti,    // size rnz, row indices of T
    Complex **p_Tx, // size rnz, numerical values of T

    Long **p_Qtrap, // size n+bncols, modified Qfill

    // workspace and parameters
    cholmod_common *cc
) ;
