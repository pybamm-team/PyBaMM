// =============================================================================
// === spqr_append =============================================================
// =============================================================================

// Appends a dense column X onto a sparse matrix A, increasing nnzmax(A) as
// needed.  The column pointer array is not modified; it must be large enough
// to accomodate the new column.

#include "spqr.hpp"

template <typename Entry> int spqr_append       // TRUE/FALSE if OK or not
(
    // inputs, not modified
    Entry *X,           // size m-by-1
    Long *P,            // size m, or NULL; permutation to apply to X.
                        // P [k] = i if row k of A is row i of X

    // input/output
    cholmod_sparse *A,  // size m-by-(A->ncol) where A->ncol > n must hold
    Long *p_n,          // n = # of columns of A so far; increased one

    // workspace and parameters
    cholmod_common *cc
)
{
    Entry *Ax ;
    Long *Ai, *Ap ;
    Long nzmax, nz, i, k, nznew, n, m, nz2 ;
    int ok = TRUE ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    m = A->nrow ;
    n = *p_n ;
    Ap = (Long *) A->p ;

    if (m == 0)
    {
        // quick return
        n++ ;
        *p_n = n ;
        Ap [n] = 0 ;
        return (TRUE) ;
    }

    Ai = (Long *) A->i ;
    Ax = (Entry *) A->x ;
    nzmax = A->nzmax ;      // current nzmax(A)
    nz = Ap [n] ;           // current nnz(A)
    PR (("nz %ld nzmax %ld\n", nz, nzmax)) ;
    ASSERT (nz <= nzmax) ;

    // -------------------------------------------------------------------------
    // append X onto A
    // -------------------------------------------------------------------------

    nz2 = spqr_add (nz, m, &ok) ;

    if (ok && nz2 <= nzmax)
    {

        // ---------------------------------------------------------------------
        // A is large enough to hold all of X without reallocating
        // ---------------------------------------------------------------------

        for (k = 0 ; k < m ; k++)
        {
            i = P ? P [k] : k ;
            if (X [i] != (Entry) 0)
            {
                Ai [nz] = k ;
                Ax [nz] = X [i] ;
                nz++ ;
            }
        }

    }
    else
    {

        // ---------------------------------------------------------------------
        // A might need to be increased in size
        // ---------------------------------------------------------------------

        for (k = 0 ; k < m ; k++)
        {
            i = P ? P [k] : k ;
            if (X [i] != (Entry) 0)
            {
                if (nz >= nzmax)
                {
                    // Ai and Ax are not big enough; increase their size.
                    // nznew = 2*nzmax + m ;
                    nznew = spqr_mult (2, nzmax, &ok) ;
                    nznew = spqr_add (nznew, m, &ok) ;
                    if (!ok || !cholmod_l_reallocate_sparse (nznew, A, cc))
                    {
                        // out of memory
                        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
                        return (FALSE) ;
                    }
                    // Ai and Ax have moved, reaquire the pointers
                    Ai = (Long *) A->i ;
                    Ax = (Entry *) A->x ;
                    PR (("reallocated from %ld to %ld\n", nzmax, nznew)) ;
                    nzmax = nznew ;
                }
                Ai [nz] = k ;
                Ax [nz] = X [i] ;
                nz++ ;
            }
        }
    }

    // -------------------------------------------------------------------------
    // finalize column pointers
    // -------------------------------------------------------------------------

    PR (("new nz %ld\n", nz)) ;
    n++ ;
    *p_n = n ;

    A->nzmax = nzmax ;
    A->i = Ai ;
    A->x = Ax ;
    Ap [n] = nz ;
    return (TRUE) ;
}


// =============================================================================

template int spqr_append <double>
(
    // inputs, not modified
    double *X,      // size m-by-1
    Long *P,        // size m, or NULL; permutation to apply to X.
                    // P [k] = i if row k of A is row i of X

    // input/output
    cholmod_sparse *A,    // size m-by-n2 where n2 > n
    Long *p_n,       // number of columns of A; increased by one

    // workspace and parameters
    cholmod_common *cc
) ;

// =============================================================================

template int spqr_append <Complex>
(
    // inputs, not modified
    Complex *X,     // size m-by-1
    Long *P,        // size m, or NULL; permutation to apply to X.
                    // P [k] = i if row k of A is row i of X

    // input/output
    cholmod_sparse *A,    // size m-by-n2 where n2 > n
    Long *p_n,      // number of columns of A; increased by one

    // workspace and parameters
    cholmod_common *cc
) ;
