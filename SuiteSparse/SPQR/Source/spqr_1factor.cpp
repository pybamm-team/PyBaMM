// =============================================================================
// === spqr_1factor ============================================================
// =============================================================================

/* Compute the QR factorization of a sparse matrix, both symbolic and numeric.

    This function exploits column singletons, and thus cannot be
    split into symbolic and numeric phases.

    Outline:

        function [C,R,E,X,H] = sparseQR (A,B)
        E = fill reducing ordering of A, with singletons coming first
        S = A*E
        Y = [S B] or [S2 B2]
        QR factorization of Y, obtaining factor R, Householder vectors H,
            and applying H to B2 to get C2
        X = E*(R\C)

        ordering options:
            0 or 3: fixed
            1: natural (only look for singletons)
            2: colamd after finding singletons
            4: CHOLMOD fter finding singletons
            5: amd(A'*A) after finding singletons
            6: metis(A'*A) after finding singletons
            7: SuiteSparseQR default (selects COLAMD, AMD, or METIS)

    First, the column singletons of the sparse matrix A are found.  If the
    column singletons and their corresponding rows are permuted to the left and
    top, respectively, of a matrix, and the matrix is not rank deficient, the
    result looks like:

        c x x x x x x x
        . c x x x x x x
        . . c x x x x x
        . . . c x x x x
        . . . . a a a a
        . . . . a a a a
        . . . . a a a a
        . . . . a a a a
        . . . . a a a a

    where c denotes a column singleton j and its corresponding row i, x denotes
    an entry in a singleton row, and a denotes the rest of the matrix.  No
    column in the S2 submatrix (the "a" entries) has just one entry in it
    unless that entry falls below tol.

    A singleton entry c is not accepted if its magnitude falls below tol.  If
    tol is negative, any entry present in the data structure of A is
    acceptable, even an explicitly-stored zero entry).

    If the matrix is structurally rank-deficient, then the permuted matrix can
    look like the following, where in this case the 3rd column is "dead":

        c x x x x x x x
        . c x x x x x x
        . . . c x x x x
        . . . . a a a a
        . . . . a a a a
        . . . . a a a a
        . . . . a a a a
        . . . . a a a a
        . . . . a a a a

    After the column singletons are found, the remaining S2 matrix (and R12) is
    ordered with a fill-reducing ordering, resulting in the ordering E.  The
    columns of B are not permuted.

    The matrix [A*E B] is split into a block 2-by-3 form:

        R11 R12 B1
        0   S2  B2

    where R1 = [R11 R12] are the singleton rows of A, and [R11 ; 0] are the
    singleton columns.  R1 is stored in row-oriented form (with sorted rows)
    and Y = [S2 B2] is stored in column-oriented form (with sorted columns if
    the input matrix A has sorted columns).  S2 contains no empty columns, and
    any columns in S2 with just one entry have a 2-norm less than tol.  Note
    that the rows of Y are permuted according to the singleton rows.
*/

#include "spqr.hpp"

template <typename Entry> SuiteSparseQR_factorization <Entry> *spqr_1factor
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // only accept singletons above tol.  If tol <= -2,
                            // then use the default tolerance
    Long bncols,            // number of columns of B
    int keepH,              // if TRUE, keep the Householder vectors
    cholmod_sparse *A,      // m-by-n sparse matrix
    Long ldb,               // if dense, the leading dimension of B
    Long *Bp,               // size bncols+1, column pointers of B
    Long *Bi,               // size bnz = Bp [bncols], row indices of B
    Entry *Bx,              // size bnz, numerical values of B

    // workspace and parameters
    cholmod_common *cc
)
{
    spqr_symbolic *QRsym ;
    spqr_numeric <Entry> *QRnum ;
    SuiteSparseQR_factorization <Entry> *QR ;
    Long *Yp, *Yi, *Q1fill, *R1p, *R1j, *P1inv, *Ap, *Ai ;
    Entry *Yx, *R1x, *Ax ;
    Long noY, anz, a2nz, r1nz, ynz, i, j, k, p, p2, bnz, py, n1rows,
        n1cols, n2, Bsparse, d, iold, inew, m, n ;
    cholmod_sparse *Y = NULL ;

    double t0 = SuiteSparse_time ( ) ;
    double t1, t2 ;

    // -------------------------------------------------------------------------
    // get inputs and allocate result
    // -------------------------------------------------------------------------

    m = A->nrow ;
    n = A->ncol ;
    Ap = (Long *) A->p ;
    Ai = (Long *) A->i ;
    Ax = (Entry *) A->x ;

    QR = (SuiteSparseQR_factorization <Entry> *)
        cholmod_l_malloc (1, sizeof (SuiteSparseQR_factorization <Entry>), cc) ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        return (NULL) ;
    }

    QR->QRsym = NULL ;
    QR->QRnum = NULL ;

    QR->R1p = NULL ;
    QR->R1j = NULL ;
    QR->R1x = NULL ;
    QR->P1inv = NULL ;
    QR->Q1fill = NULL ;
    QR->Rmap = NULL ;
    QR->RmapInv = NULL ;
    QR->HP1inv = NULL ;

    QR->narows = m ;
    QR->nacols = n ;
    QR->n1rows = 0 ;
    QR->n1cols = 0 ;

    QR->r1nz = 0 ;
    r1nz = 0 ;

    // B is an optional input.  It can be sparse or dense
    Bsparse = (Bp != NULL && Bi != NULL) ;
    if (Bx == NULL)
    {
        // B is not present; force bncols to be zero
        bncols = 0 ;
    }

    QR->bncols = bncols ;

    // -------------------------------------------------------------------------
    // find the default tol, if requested
    // -------------------------------------------------------------------------

    if (tol <= SPQR_DEFAULT_TOL)
    {
        tol = spqr_tol <Entry> (A, cc) ; 
    }
    if (tol < 0)
    {
        // no rank detection will be performed
        QR->allow_tol = FALSE ;
        tol = EMPTY ;
    }
    else
    {
        QR->allow_tol = TRUE ;
    }
    QR->tol = tol ;

    // -------------------------------------------------------------------------
    // find singletons and construct column pointers for the A part of Y
    // -------------------------------------------------------------------------

    // These return R1p, P1inv, and Y; but they are all NULL if out of memory.
    // Note that only Y->p is allocated (Y->i and Y->x are dummy placeholders
    // of one Long and one Entry, each, actually).  The entries of Y are
    // allocated later, below.

    if (ordering == SPQR_ORDERING_GIVEN)
    {
        ordering = SPQR_ORDERING_FIXED ;
    }

    if (ordering == SPQR_ORDERING_FIXED)
    {
        // fixed ordering: find column singletons without permuting columns
        Q1fill = NULL ;
        spqr_1fixed <Entry> (tol, bncols, A,
            &R1p, &P1inv, &Y, &n1cols, &n1rows, cc) ;
    }
    else
    {
        // natural or fill-reducing ordering: find column singletons with
        // column permutations allowed, then permute the pruned submatrix with
        // a fill-reducing ordering if ordering is not SPQR_ORDERING_NATURAL.
        spqr_1colamd <Entry> (ordering, tol, bncols, A, &Q1fill,
            &R1p, &P1inv, &Y, &n1cols, &n1rows, cc) ;
        ordering = cc->SPQR_istat [7]  ;
    }

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        spqr_freefac (&QR, cc) ;
        return (NULL) ;
    }

    QR->R1p = R1p ;
    QR->P1inv = P1inv ;
    QR->Q1fill = Q1fill ;
    QR->n1rows = n1rows ;
    QR->n1cols = n1cols ;

    noY = (Y == NULL) ;                         // A will be factorized, not Y
    ASSERT (noY == (n1cols == 0 && bncols == 0)) ;
    Yp = noY ? NULL : (Long *) Y->p ;
    anz = Ap [n] ;                              // nonzeros in A
    a2nz = noY ? anz : Yp [n-n1cols] ;          // nonzeros in S2
    n2 = n - n1cols ;                           // number of columns of S2

    // Y is NULL, or of size (m-n1rows)-by-(n-n1cols+bncols)
    ASSERT (IMPLIES (Y != NULL, ((Long) Y->nrow == m-n1rows))) ;
    ASSERT (IMPLIES (Y != NULL, ((Long) Y->ncol == n-n1cols+bncols))) ;

    // Y, if allocated, has no space for any entries yet
    ynz = 0 ;

    // -------------------------------------------------------------------------
    // construct the column pointers for the B or B2 part of Y
    // -------------------------------------------------------------------------

    if (noY)
    {

        // A will be factorized instead of Y.  There is no B.  C or X can exist
        // as empty matrices with rows but no columns.
        // There are no singletons.
        ASSERT (Yp == NULL) ;
        ASSERT (R1p == NULL) ;
        ASSERT (P1inv == NULL) ;
        ASSERT (n1rows == 0) ;
        ASSERT (n1cols == 0) ;
        ASSERT (a2nz == Ap [n]) ;
        ASSERT (bncols == 0) ;

    }
    else if (n1cols == 0)
    {

        // ---------------------------------------------------------------------
        // construct the column pointers for the B part of Y = [S B]
        // ---------------------------------------------------------------------

        ASSERT (R1p == NULL) ;
        ASSERT (P1inv == NULL) ;
        ASSERT (n1rows == 0) ;
        ASSERT (a2nz == Ap [n]) ;

        ynz = a2nz ;
        if (Bsparse)
        {
            // B is sparse
            for (k = 0 ; k < bncols ; k++)
            {
                Yp [(n-n1cols)+k] = ynz ;
                d = Bp [k+1] - Bp [k] ;
                ynz += d ;
            }
        }
        else
        {
            // B is dense
            Entry *B1 = Bx ;
            for (k = 0 ; k < bncols ; k++)
            {
                // count the nonzero entries in column k of B
                Yp [(n-n1cols)+k] = ynz ;
                d = 0 ;
                for (i = 0 ; i < m ; i++)
                {
                    if (B1 [i] != (Entry) 0)
                    {
                        d++ ;
                    }
                }
                B1 += ldb ;
                ynz += d ;
            }
        }
        Yp [(n-n1cols)+bncols] = ynz ;

    }
    else
    {

        // ---------------------------------------------------------------------
        // construct the column pointers for the B2 part of Y = [S2 B2]
        // ---------------------------------------------------------------------

        ynz = a2nz ;
        if (Bsparse)
        {
            // B is sparse
            for (k = 0 ; k < bncols ; k++)
            {
                // count the nonzero entries in column k of B2
                Yp [(n-n1cols)+k] = ynz ;
                d = 0 ;
                for (p = Bp [k] ; p < Bp [k+1] ; p++)
                {
                    iold = Bi [p] ;
                    inew = P1inv [iold] ;
                    if (inew >= n1rows)
                    {
                        d++ ;
                    }
                }
                ynz += d ;
            }
        }
        else
        {
            // B is dense
            Entry *B1 = Bx ;
            for (k = 0 ; k < bncols ; k++)
            {
                // count the nonzero entries in column k of B2
                Yp [(n-n1cols)+k] = ynz ;
                d = 0 ;
                for (iold = 0 ; iold < m ; iold++)
                {
                    inew = P1inv [iold] ;
                    if (inew >= n1rows && B1 [iold] != (Entry) 0)
                    {
                        d++ ;
                    }
                }
                B1 += ldb ;
                ynz += d ;
            }
        }
        Yp [(n-n1cols)+bncols] = ynz ;
    }


    // -------------------------------------------------------------------------
    // allocate the nonzeros for Y
    // -------------------------------------------------------------------------

    if (noY)
    {
        // no singletons found, and B is empty.  pass Y=A to QR factorization,
        // and pass in Q1fill as the "user-provided" ordering
        ASSERT (Yp == NULL) ;
        Yi = NULL ;
        Yx = NULL ;
    }
    else
    {
        cholmod_l_reallocate_sparse (ynz, Y, cc) ;
        Yi = (Long  *) Y->i ;
        Yx = (Entry *) Y->x ;
    }

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        spqr_freefac (&QR, cc) ;
        cholmod_l_free_sparse (&Y, cc) ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // create the pattern and values of Y and R1
    // -------------------------------------------------------------------------

    if (noY)
    {

        // ---------------------------------------------------------------------
        // R1 does not exist
        // ---------------------------------------------------------------------

        ASSERT (R1p == NULL) ;
        R1j = NULL ;
        R1x = NULL ;

    }
    else if (n1cols == 0)
    {

        // ---------------------------------------------------------------------
        // R1 does not exist
        // ---------------------------------------------------------------------

        ASSERT (R1p == NULL) ;
        R1j = NULL ;
        R1x = NULL ;

        // ---------------------------------------------------------------------
        // construct the A part of Y = [S B]
        // ---------------------------------------------------------------------

        ASSERT (anz == a2nz) ;
        py = 0 ;
        for (k = 0 ; k < n ; k++)
        {
            j = Q1fill ? Q1fill [k] : k ;
            ASSERT (py == Yp [k]) ;
            for (p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                Yi [py] = Ai [p] ;
                Yx [py] = Ax [p] ;
                py++ ;
            }
        }
        ASSERT (py == anz) ;
        ASSERT (py == Yp [n]) ;

        // ---------------------------------------------------------------------
        // construct the B part of Y = [S B]
        // ---------------------------------------------------------------------

        if (Bsparse)
        {
            // B is sparse
            bnz = Bp [bncols] ;
            for (p = 0 ; p < bnz ; p++)
            {
                Yi [py++] = Bi [p] ;
            }
            py = anz ;
            for (p = 0 ; p < bnz ; p++)
            {
                Yx [py++] = Bx [p] ;
            }
        }
        else
        {
            // B is dense
            Entry *B1 = Bx ;
            for (k = 0 ; k < bncols ; k++)
            {
                ASSERT (py == Yp [n+k]) ;
                for (i = 0 ; i < m ; i++)
                {
                    Entry bij = B1 [i] ;
                    if (bij != (Entry) 0)
                    {
                        Yi [py] = i ;
                        Yx [py] = bij ;
                        py++ ;
                    }
                }
                B1 += ldb ;
            }
        }
        ASSERT (py == ynz) ;

    }
    else
    {

        // ---------------------------------------------------------------------
        // R1p = cumsum ([0 R1p])
        // ---------------------------------------------------------------------

        r1nz = spqr_cumsum (n1rows, R1p) ;      // Long overflow cannot occur
        PR (("total nonzeros in R1: %ld\n", r1nz)) ;

        // ---------------------------------------------------------------------
        // allocate R1
        // ---------------------------------------------------------------------

        R1j = (Long  *) cholmod_l_malloc (r1nz, sizeof (Long ), cc) ;
        R1x = (Entry *) cholmod_l_malloc (r1nz, sizeof (Entry), cc) ;
        QR->R1j = R1j ;
        QR->R1x = R1x ;
        QR->r1nz = r1nz ;

        if (cc->status < CHOLMOD_OK)
        {
            // out of memory
            spqr_freefac (&QR, cc) ;
            cholmod_l_free_sparse (&Y, cc) ;
            return (NULL) ;
        }

        // ---------------------------------------------------------------------
        // scan A and construct R11
        // ---------------------------------------------------------------------

        // At this point, R1p [i] points to the start of row i:
        // for (Long t = 0 ; t <= n1rows ; t++) Rsave [t] = R1p [t] ;

        for (k = 0 ; k < n1cols ; k++)
        {
            j = Q1fill ? Q1fill [k] : k ;
            for (p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                // row i of A is row inew after singleton permutation
                i = Ai [p] ;
                inew = P1inv [i] ;
                ASSERT (inew < n1rows) ;
                // A (i,j) is in a singleton row.  It becomes R1 (inew,k)
                p2 = R1p [inew]++ ;
                ASSERT (p2 < R1p [inew+1]) ;
                R1j [p2] = k ;
                R1x [p2] = Ax [p] ;
            }
        }

        // ---------------------------------------------------------------------
        // scan A and construct R12 and the S2 part of Y = [S2 B2]
        // ---------------------------------------------------------------------

        py = 0 ;
        for ( ; k < n ; k++)
        {
            j = Q1fill ? Q1fill [k] : k ;
            ASSERT (py == Yp [k-n1cols]) ;
            for (p = Ap [j] ; p < Ap [j+1] ; p++)
            {
                // row i of A is row inew after singleton permutation
                i = Ai [p] ;
                inew = P1inv [i] ;
                if (inew < n1rows)
                {
                    // A (i,j) is in a singleton row.  It becomes R1 (inew,k)
                    p2 = R1p [inew]++ ;
                    ASSERT (p2 < R1p [inew+1]) ;
                    R1j [p2] = k ;
                    R1x [p2] = Ax [p] ;
                }
                else
                {
                    // A (i,j) is not in a singleton row.  Place it in
                    // Y (inew-n1rows, k-n1cols)
                    Yi [py] = inew - n1rows ;
                    Yx [py] = Ax [p] ;
                    py++ ;
                }
            }
        }
        ASSERT (py == Yp [n-n1cols]) ;

        // ---------------------------------------------------------------------
        // restore the row pointers for R1
        // ---------------------------------------------------------------------

        spqr_shift (n1rows, R1p) ;

        // the row pointers are back to what they were:
        // for (Long t = 0 ; t <= n1rows ; t++) ASSERT (Rsave [t] == R1p [t]) ;

        // ---------------------------------------------------------------------
        // construct the B2 part of Y = [S2 B2]
        // ---------------------------------------------------------------------

        if (Bsparse)
        {
            // B is sparse
            for (k = 0 ; k < bncols ; k++)
            {
                // construct the nonzero entries in column k of B2
                ASSERT (py == Yp [k+(n-n1cols)]) ;
                for (p = Bp [k] ; p < Bp [k+1] ; p++)
                {
                    iold = Bi [p] ;
                    inew = P1inv [iold] ;
                    if (inew >= n1rows)
                    {
                        Yi [py] = inew - n1rows ;
                        Yx [py] = Bx [p] ;
                        py++ ;
                    }
                }
            }
        }
        else
        {
            // B is dense
            Entry *B1 = Bx ;
            for (k = 0 ; k < bncols ; k++)
            {
                // construct the nonzero entries in column k of B2
                ASSERT (py == Yp [k+(n-n1cols)]) ;
                for (iold = 0 ; iold < m ; iold++)
                {
                    inew = P1inv [iold] ;
                    if (inew >= n1rows)
                    {
                        Entry bij = B1 [iold] ;
                        if (bij != (Entry) 0)
                        {
                            Yi [py] = inew - n1rows ;
                            Yx [py] = bij ;
                            py++ ;
                        }
                    }
                }
                B1 += ldb ;
            }
        }
        ASSERT (py == ynz) ;
    }

    // -------------------------------------------------------------------------
    // QR factorization of A or Y
    // -------------------------------------------------------------------------

    if (noY)
    {
        // factorize A, with fill-reducing ordering already given in Q1fill
        QRsym = spqr_analyze (A, SPQR_ORDERING_GIVEN, Q1fill,
            tol >= 0, keepH, cc) ;
        t1 = SuiteSparse_time ( ) ;
        QRnum = spqr_factorize <Entry> (&A, FALSE, tol, n, QRsym, cc) ;
    }
    else
    {
        // fill-reducing ordering is already applied to Y; free Y when loaded
        QRsym = spqr_analyze (Y, SPQR_ORDERING_FIXED, NULL,
            tol >= 0, keepH, cc) ;
        t1 = SuiteSparse_time ( ) ;
        QRnum = spqr_factorize <Entry> (&Y, TRUE, tol, n2, QRsym, cc) ;
        // Y has been freed
        ASSERT (Y == NULL) ;
    }

    // record the actual ordering used (this will have been changed to GIVEN
    // or FIXED, in spqr_analyze, but change it back to the ordering used by
    // spqr_1fixed or spqr_1colamd.
    cc->SPQR_istat [7] = ordering ;

    QR->QRsym = QRsym ;
    QR->QRnum = QRnum ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        spqr_freefac (&QR, cc) ;
        return (NULL) ;
    }

    // singletons do not take part in the symbolic analysis, and any columns
    // of B are tacked on and do take part.
    ASSERT (QRsym->n == n - n1cols + bncols) ;

    cc->SPQR_istat [0] += r1nz ;       // nnz (R)

    // rank estimate of A, including singletons but excluding the columns of
    // of B, in case [A B] was factorized.
    QR->rank = n1rows + QRnum->rank1 ;
    PR (("rank estimate of A: QR->rank = %ld = %ld + %ld\n",
    QR->rank, n1rows, QRnum->rank1)) ;

    // -------------------------------------------------------------------------
    // construct global row permutation if H is kept and singletons exist
    // -------------------------------------------------------------------------

    // If there are no singletons, then HP1inv [0:m-1] and HPinv [0:m-1] would
    // be identical, so HP1inv is not needed.

    ASSERT ((n1cols == 0) == (P1inv == NULL)) ;
    ASSERT (IMPLIES (n1cols == 0, n1rows == 0)) ;
    ASSERT (n1cols >= n1rows) ;

    if (keepH && n1cols > 0)
    {
        // construct the global row permutation.  Currently, the row indices
        // in H reflect the global R.  P1inv is the singleton permutation,
        // where a row index of Y = (P1inv (row of A) - n1rows), and
        // row of R2 = QRnum->HPinv (row of Y).   Combine these two into
        // HP1inv, where a global row of R = HP1inv (a row of A)

        Long kk ;
        Long *HP1inv, *HPinv ;
        QR->HP1inv = HP1inv = (Long *) cholmod_l_malloc (m, sizeof (Long), cc) ;
        HPinv = QRnum->HPinv ;

        if (cc->status < CHOLMOD_OK)
        {
            // out of memory
            spqr_freefac (&QR, cc) ;
            return (NULL) ;
        }

        for (i = 0 ; i < m ; i++)
        {
            // i is a row of A, k is a row index after row singletons are
            // permuted.  Then kk is a row index of the global R.
            k = P1inv ? P1inv [i] : i ;
            ASSERT (k >= 0 && k < m) ;
            if (k < n1rows)
            {
                kk = k ;
            }
            else
            {
                // k-n1rows is a row index of Y, the matrix factorized by
                // the QR factorization kernels (in QRsym and QRnum).
                // HPinv [k-n1rows] gives a row index of R2, to which n1rows
                // must be added to give a row of the global R.
                kk = HPinv [k - n1rows] + n1rows ;
            }
            ASSERT (kk >= 0 && kk < m) ;
            HP1inv [i] = kk ;
        }
    }

    // -------------------------------------------------------------------------
    // find the mapping for the squeezed R, if A is rank deficient
    // -------------------------------------------------------------------------

    if (QR->rank < n && !spqr_rmap <Entry> (QR, cc))
    {
        // out of memory
        spqr_freefac (&QR, cc) ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // output statistics
    // -------------------------------------------------------------------------

    cc->SPQR_istat [4] = QR->rank ;         // estimated rank of A
    cc->SPQR_istat [5] = n1cols ;           // number of columns singletons
    cc->SPQR_istat [6] = n1rows ;           // number of singleton rows
    cc->SPQR_tol_used = tol ;               // tol used

    t2 = SuiteSparse_time ( ) ;
    cc->SPQR_analyze_time = t1 - t0 ;   // analyze time, including singletons
    cc->SPQR_factorize_time = t2 - t1 ; // factorize time

    return (QR) ;
}


// =============================================================================

template SuiteSparseQR_factorization <double> *spqr_1factor <double>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // only accept singletons above tol
    Long bncols,            // number of columns of B
    int keepH,              // if TRUE, keep the Householder vectors
    cholmod_sparse *A,      // m-by-n sparse matrix
    Long ldb,               // if dense, the leading dimension of B
    Long *Bp,               // size bncols+1, column pointers of B
    Long *Bi,               // size bnz = Bp [bncols], row indices of B
    double *Bx,             // size bnz, numerical values of B

    // workspace and parameters
    cholmod_common *cc
) ;

// =============================================================================

template SuiteSparseQR_factorization <Complex> *spqr_1factor <Complex>
(
    // inputs, not modified
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // only accept singletons above tol
    Long bncols,            // number of columns of B
    int keepH,              // if TRUE, keep the Householder vectors
    cholmod_sparse *A,      // m-by-n sparse matrix
    Long ldb,               // if dense, the leading dimension of B
    Long *Bp,               // size bncols+1, column pointers of B
    Long *Bi,               // size bnz = Bp [bncols], row indices of B
    Complex *Bx,            // size bnz, numerical values of B

    // workspace and parameters
    cholmod_common *cc
) ;
