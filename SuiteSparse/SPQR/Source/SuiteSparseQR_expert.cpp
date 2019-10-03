// =============================================================================
// === SuiteSparseQR_expert ====================================================
// =============================================================================

// This file includes a suite of "expert" user-callable routines that allow the
// reuse of a QR factorization, including the ability to find the min 2-norm
// solution of an under-determined system of equations:
//
// The QR factorization can be computed in one of two ways:
//
//      SuiteSparseQR_factorize  QR factorization, returning a QR object
//                               This performs both symbolic analysis and
//                               numeric factorization, and exploits singletons.
//                               Note that H is always kept.
//
//      SuiteSparseQR_symbolic   symbolic QR factorization, based purely on the
//                               nonzero pattern of A; to be followed by:
//      SuiteSparseQR_numeric    numeric QR factorization.  Does not exploit
//                               singletons.  Note that H is always kept.
//
//      SuiteSparseQR_solve      forward/backsolve using R from the QR object
//      SuiteSparseQR_qmult      multiply by Q or Q', using Q from the QR object
//      SuiteSparseQR_min2norm   min 2-norm solution for x=A\b
//      SuiteSparseQR_free       free the QR object
//
// All of these functions keep the Householder vectors.  The
// SuiteSparseQR_solve function does not require the Householder vectors, but
// in the current version, it is only used in that case.  Since these functions
// require the Householder vectors, the GPU is never used in these functions.
//
// If all you need is to solve a least-squares problem, or to compute a QR
// factorization and return the results in vanilla sparse matrix format, see
// the routines in SuiteSparseQR.cpp and SuiteSparseQR_qmult.cpp instead.
//
// NOTE: Q the matrix is not formed; H the Householder vectors are always kept.

#ifndef NEXPERT
#include "spqr.hpp"

// =============================================================================
// === SuiteSparseQR_symbolic ==================================================
// =============================================================================

// This returns a QR factorization object with a NULL numeric part.  It must
// be followed by a numeric factorization, by SuiteSparseQR_numeric.

template <typename Entry> 
SuiteSparseQR_factorization <Entry> *SuiteSparseQR_symbolic
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    int allow_tol,          // if FALSE, tol is ignored by the numeric
                            // factorization, and no rank detection is performed
    cholmod_sparse *A,      // sparse matrix to factorize (A->x ignored)
    cholmod_common *cc      // workspace and parameters
)
{
    double t0 = SuiteSparse_time ( ) ;

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (A, NULL) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (A, NULL) ;
    cc->status = CHOLMOD_OK ;

    // -------------------------------------------------------------------------
    // get inputs and allocate result
    // -------------------------------------------------------------------------

    SuiteSparseQR_factorization <Entry> *QR ;
    spqr_symbolic *QRsym ;

    QR = (SuiteSparseQR_factorization <Entry> *)
        cholmod_l_malloc (1, sizeof (SuiteSparseQR_factorization <Entry>), cc) ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // perform the symbolic analysis
    // -------------------------------------------------------------------------

    // Using SuiteSparseQR_symbolic followed by SuiteSparseQR_numeric requires
    // that the Householder vectors be kept, and thus the GPU will not be used.
    int keepH = TRUE ;
    QR->QRsym = QRsym = spqr_analyze (A, ordering, NULL, allow_tol, keepH, cc) ;

    QR->QRnum = NULL ;          // allocated later, by numeric factorization

    // singleton information:
    QR->R1p = NULL ;            // no singletons; these always remain NULL or 0
    QR->R1j = NULL ;
    QR->R1x = NULL ;
    QR->P1inv = NULL ;
    QR->HP1inv = NULL ;     
    QR->r1nz = 0 ;
    QR->n1rows = 0 ;
    QR->n1cols = 0 ;
    cc->SPQR_istat [5] = 0 ;    // number of columns singletons
    cc->SPQR_istat [6] = 0 ;    // number of singleton rows

    QR->Q1fill = NULL ;         // may change below

    QR->Rmap = NULL ;           // may be allocated by numeric factorization
    QR->RmapInv = NULL ;

    QR->narows = A->nrow ;
    QR->nacols = A->ncol ;
    QR->bncols = 0 ;            // [A B] is not factorized

    QR->allow_tol = (allow_tol != 0) ;
    QR->tol = QR->allow_tol ? SPQR_DEFAULT_TOL : EMPTY ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        spqr_freefac (&QR, cc) ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // copy the fill-reducing ordering
    // -------------------------------------------------------------------------

    if (QRsym->Qfill != NULL)
    {
        Long n, k, *Qfill, *Q1fill ;
        Qfill = QRsym->Qfill ;
        n = A->ncol ;
        Q1fill = (Long *) cholmod_l_malloc (n, sizeof (Long), cc) ;
        QR->Q1fill = Q1fill ;
        if (cc->status < CHOLMOD_OK)
        {
            // out of memory
            spqr_freefac (&QR, cc) ;
            return (NULL) ;
        }
        for (k = 0 ; k < n ; k++)
        {
            Q1fill [k] = Qfill [k] ;
        }
    }

    double t1 = SuiteSparse_time ( ) ;
    cc->SPQR_analyze_time = t1 - t0 ;

    return (QR) ;
}

template
SuiteSparseQR_factorization <double> *SuiteSparseQR_symbolic <double>
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    int allow_tol,          // if FALSE, tol is ignored by the numeric
                            // factorization, and no rank detection is performed
    cholmod_sparse *A,      // sparse matrix to factorize (A->x ignored)
    cholmod_common *cc      // workspace and parameters
) ;

template
SuiteSparseQR_factorization <Complex> *SuiteSparseQR_symbolic <Complex>
(
    // inputs:
    int ordering,           // all, except 3:given treated as 0:fixed
    int allow_tol,          // if FALSE, tol is ignored by the numeric
                            // factorization, and no rank detection is performed
    cholmod_sparse *A,      // sparse matrix to factorize (A->x ignored)
    cholmod_common *cc      // workspace and parameters
) ;

// =============================================================================
// === SuiteSparseQR_numeric ===================================================
// =============================================================================

// numeric QR factorization; must be preceded by a call to
// SuiteSparseQR_symbolic.  Alternatively, if SuiteSparseQR_factorize
// did not find any singletons (QR->n1cols == 0), a call to
// SuiteSparseQR_factorize can be followed by a factorization by this function.
//
// Returns TRUE if successful, FALSE otherwise.
//
// The GPU is not used by this function, since it requires the Householder
// vectors to be kept.

template <typename Entry> int SuiteSparseQR_numeric
(
    // inputs:
    double tol,             // treat columns with 2-norm <= tol as zero
    cholmod_sparse *A,      // sparse matrix to factorize
    // input/output
    SuiteSparseQR_factorization <Entry> *QR,
    cholmod_common *cc      // workspace and parameters
)
{
    double t0 = SuiteSparse_time ( ) ;

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    RETURN_IF_NULL (QR, FALSE) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (A, FALSE) ;
    cc->status = CHOLMOD_OK ;

    if (QR->n1cols > 0 || QR->bncols > 0)
    {
        // this numeric factorization phase cannot be used if singletons were
        // found, or if [A B] was factorized.
        ERROR (CHOLMOD_INVALID, "cannot refactorize w/singletons or [A B]") ;
        return (FALSE) ;
    }

    Long n = A->ncol ;

    // -------------------------------------------------------------------------
    // get the column 2-norm tolerance
    // -------------------------------------------------------------------------

    if (QR->allow_tol)
    {
        // compute default tol, if requested
        if (tol <= SPQR_DEFAULT_TOL)
        {
            tol = spqr_tol <Entry> (A, cc) ;
        }
    }
    else
    {
        // tol is ignored; no rank detection performed
        tol = EMPTY ;
    }
    QR->tol = tol ;

    // -------------------------------------------------------------------------
    // numeric factorization
    // -------------------------------------------------------------------------

    // free the existing numeric factorization, if any
    spqr_freenum (&(QR->QRnum), cc) ;

    // compute the new factorization
    QR->QRnum = spqr_factorize <Entry> (&A, FALSE, tol, n, QR->QRsym, cc) ;

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory; QR factorization remains a symbolic-only object
        ASSERT (QR->QRnum == NULL) ;
        return (FALSE) ;
    }

    QR->rank = QR->QRnum->rank1 ;

    // -------------------------------------------------------------------------
    // find the mapping for the squeezed R, if A is rank deficient
    // -------------------------------------------------------------------------

    if (QR->rank < n && !spqr_rmap (QR, cc))
    {
        // out of memory; QR factorization remains a symbolic-only object
        spqr_freenum (&(QR->QRnum), cc) ;
        return (FALSE) ;
    }

    // -------------------------------------------------------------------------
    // output statistics
    // -------------------------------------------------------------------------

    cc->SPQR_istat [4] = QR->rank ;         // estimated rank of A
    cc->SPQR_tol_used = tol ;               // tol used

    double t1 = SuiteSparse_time ( ) ;
    cc->SPQR_factorize_time = t1 - t0 ;

    return (TRUE) ;
}

template int SuiteSparseQR_numeric <double>
(
    // inputs:
    double tol,             // treat columns with 2-norm <= tol as zero
    cholmod_sparse *A,      // sparse matrix to factorize
    // input/output
    SuiteSparseQR_factorization <double> *QR,
    cholmod_common *cc      // workspace and parameters
) ;

template int SuiteSparseQR_numeric <Complex>
(
    // inputs:
    double tol,             // treat columns with 2-norm <= tol as zero
    cholmod_sparse *A,      // sparse matrix to factorize
    // input/output
    SuiteSparseQR_factorization <Complex> *QR,
    cholmod_common *cc      // workspace and parameters
) ;

// =============================================================================
// === SuiteSparseQR_factorize =================================================
// =============================================================================

// QR factorization of a sparse matrix A, including symbolic ordering/analysis,
// and numeric factorization.  This function is a combined symbolic+numeric
// factorization, because it exploits singletons.  It factorizes A*E into Q*R,
// where E is the fill-reducing ordering, Q is in Householder form, and R is
// composed of a set of singleton rows followed by the rows of the multifrontal
// factorization of the singleton-free submatrix.
//
// The MATLAB equivalent of this function is [Q,R,E]=spqr(A,opts) where opts.Q
// = 'Householder', except that Q, R, and E are all stored in a single QR
// object (of type SuiteSparseQR_factorization), rather than being put into
// MATLAB format.  That extraction is done by the overloaded SuiteSparseQR
// function, which also frees its QR object after extracting Q, R, and E.
//
// If you have a least-squares problem with a single right-hand-side B (where B
// is a vector or matrix, or sparse or dense), and do not need to reuse the QR
// factorization of A, then use the SuiteSparseQR function instead (which
// corresponds to the MATLAB statement x=A\B, using a sparse QR factorization).
//
// The GPU is not used because the Householder vectors are always kept.

template <typename Entry>
SuiteSparseQR_factorization <Entry> *SuiteSparseQR_factorize
(
    // inputs, not modified:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // treat columns with 2-norm <= tol as zero
    cholmod_sparse *A,      // sparse matrix to factorize
    cholmod_common *cc      // workspace and parameters
)
{
    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (A, NULL) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (A, NULL) ;
    cc->status = CHOLMOD_OK ;
    // B is not present, and always keep H:
    int keepH = TRUE ;
    return (spqr_1factor <Entry> (ordering, tol, 0, keepH, A,
        0, NULL, NULL, NULL, cc)) ;
}

template SuiteSparseQR_factorization <double> *SuiteSparseQR_factorize <double>
(
    // inputs, not modified:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // treat columns with 2-norm <= tol as zero
    cholmod_sparse *A,      // sparse matrix to factorize
    // workspace and parameters
    cholmod_common *cc
) ;

template SuiteSparseQR_factorization <Complex> *SuiteSparseQR_factorize<Complex>
(
    // inputs, not modified:
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,             // treat columns with 2-norm <= tol as zero
    cholmod_sparse *A,      // sparse matrix to factorize
    // workspace and parameters
    cholmod_common *cc
) ;


// =============================================================================
// === spqr_private_rtsolve ====================================================
// =============================================================================

// Solve X = R'\(E'*B) or X = R'\B using the QR factorization from spqr_1factor
// or SuiteSparseQR_factorize (including the singleton-row R and the
// multifrontal R).  A is m-by-n.  B is n-by-nrhs, and X is m-by-nrhs.

template <typename Entry> void spqr_private_rtsolve
(
    // inputs
    SuiteSparseQR_factorization <Entry> *QR,
    int use_Q1fill,

    Long nrhs,              // number of columns of B
    Long ldb,               // leading dimension of B
    Entry *B,               // size n-by-nrhs with leading dimension ldb

    // output, contents undefined on input
    Entry *X,               // size m-by-nrhs with leading dimension m

    cholmod_common *cc
)
{
    Entry xi ;
    spqr_symbolic *QRsym ;
    spqr_numeric <Entry> *QRnum ;
    Long *R1p, *R1j, *Rmap, *Rp, *Rj, *Super, *HStair, *Hm, *Stair, *Q1fill,
        *RmapInv ;
    Entry *R1x, **Rblock, *R, *X1, *X2 ;
    char *Rdead ;
    Long i, j, k, m, n, p, kk, n1rows, n1cols, rank, nf, f, col1, col2, fp, pr,
        fn, rm, row1, keepH, fm, h, t, live, jj ;

    // -------------------------------------------------------------------------
    // get the contents of the QR factorization
    // -------------------------------------------------------------------------

    QRsym = QR->QRsym ;
    QRnum = QR->QRnum ;
    n1rows = QR->n1rows ;
    n1cols = QR->n1cols ;
    n = QR->nacols ;
    m = QR->narows ;
    Q1fill = use_Q1fill ? QR->Q1fill : NULL ;
    R1p = QR->R1p ;
    R1j = QR->R1j ;
    R1x = QR->R1x ;
    Rmap = QR->Rmap ;
    RmapInv = QR->RmapInv ;
    rank = QR->rank ;
    ASSERT ((Rmap == NULL) == (rank == n)) ;
    ASSERT ((RmapInv == NULL) == (rank == n)) ;
    ASSERT (rank <= MIN (m,n)) ;

    keepH = QRnum->keepH ;
    PR (("\nrtsolve keepH %ld n1rows %ld n1cols %ld\n", keepH, n1rows, n1cols));

    // This code is only used when keepH is true.  keepH is tested below,
    // however, in case this changes in the future (the following assert would
    // also need to be removed).
    ASSERT (keepH) ;

    nf = QRsym->nf ;
    Rblock = QRnum->Rblock ;
    Rp = QRsym->Rp ;
    Rj = QRsym->Rj ;
    Super = QRsym->Super ;
    Rdead = QRnum->Rdead ;

    HStair = QRnum->HStair ;
    Hm = QRnum->Hm ;

    ASSERT (ldb >= n) ;

    // -------------------------------------------------------------------------
    // X = E'*B or X = B
    // -------------------------------------------------------------------------

    X1 = X ;
    if (rank == n)
    {
        // R is in upper triangular form
        ASSERT (n <= m) ;
        for (kk = 0 ; kk < nrhs ; kk++)
        {
            for (k = 0 ; k < n ; k++)
            {
                X1 [k] = B [Q1fill ? Q1fill [k] : k] ;
            }
            for ( ; k < m ; k++)
            {
                X1 [k] = 0 ;
            }
            X1 += m ;
            B += ldb ;
        }
    }
    else
    {
        // R is in squeezed form; use the mapping to trapezoidal
        for (kk = 0 ; kk < nrhs ; kk++)
        {
            for (i = 0 ; i < rank ; i++)
            {
                k = RmapInv [i] ;
                ASSERT (k >= 0 && k < n) ;
                Long knew = Q1fill ? Q1fill [k] : k ;
                ASSERT (knew >= 0 && knew < n) ;
                X1 [i] = B [knew] ;
            }
            for ( ; i < m ; i++)
            {
                X1 [i] = 0 ;
            }
            X1 += m ;
            B += ldb ;
        }
    }

    // =========================================================================
    // === solve with the singleton rows of R ==================================
    // =========================================================================

    X1 = X ;
    for (kk = 0 ; kk < nrhs ; kk++)
    {
        for (i = 0 ; i < n1rows ; i++)
        {
            // divide by the diagonal (the singleton entry itself)
            p = R1p [i] ;
            xi = spqr_divide (X1 [i], spqr_conj (R1x [p]), cc) ;
            X1 [i] = xi ;

            // solve with the off-diagonal entries
            for (p++ ; p < R1p [i+1] ; p++)
            {
                j = R1j [p] ;
                ASSERT (j >= i && j < n) ;
                k = Rmap ? Rmap [j] : j ;
                ASSERT (k >= 0 && k < n) ;
                if (k < rank)
                {
                    // the jth column of R is live, and is the kth column of
                    // the squeezed R
                    X1 [k] -= spqr_conj (R1x [p]) * xi ;
                }
            }
        }
        X1 += m ;
    }

    // =========================================================================
    // === solve with the multifrontal rows of R ===============================
    // =========================================================================

    Stair = NULL ;
    fm = 0 ;
    h = 0 ;
    t = 0 ;

    // start with row1 = n1rows, the first non-singleton row, which is the
    // first row of the multifrontal R.

    row1 = n1rows ;
    for (f = 0 ; f < nf ; f++)
    {

        // ---------------------------------------------------------------------
        // get the R block for front F
        // ---------------------------------------------------------------------

        R = Rblock [f] ;
        col1 = Super [f] ;                  // first pivot column in front F
        col2 = Super [f+1] ;                // col2-1 is last pivot col
        fp = col2 - col1 ;                  // number of pivots in front F
        pr = Rp [f] ;                       // pointer to row indices for F
        fn = Rp [f+1] - pr ;                // # of columns in front F

        if (keepH)
        {
            Stair = HStair + pr ;           // staircase of front F
            fm = Hm [f] ;                   // # of rows in front F
            h = 0 ;                         // H vector starts in row h
        }

        // ---------------------------------------------------------------------
        // solve with the squeezed upper triangular part of R
        // ---------------------------------------------------------------------

        rm = 0 ;                            // number of rows in R block
        for (k = 0 ; k < fp ; k++)
        {
            j = col1 + k ;
            ASSERT (Rj [pr + k] == j) ;
            if (j+n1cols >= n) return ;     // in case [A Binput] was factorized
            if (keepH)
            {
                t = Stair [k] ;             // length of R+H vector
                ASSERT (t >= 0 && t <= fm) ;
                if (t == 0)
                {
                    live = FALSE ;          // column k is dead
                    t = rm ;                // dead col, R only, no H
                    h = rm ;
                }
                else
                {
                    live = (rm < fm) ;      // k is live, unless we hit the wall
                    h = rm + 1 ;            // H vector starts in row h
                }
                ASSERT (t >= h) ;
            }
            else
            {
                // Ironically, this statement is itself dead code.  In the
                // current version, keepH is always TRUE whenever rtsolve is
                // used.  If H was not kept, this statement would determine
                // whether the the pivot column is live or dead.
                live = (!Rdead [j]) ;                                   // DEAD
            }

            if (live)
            {
                // This is a live pivot column of the R block.  rm has not been
                // incremented yet, so at this point it is equal to the number
                // of entries in the pivot column above the diagonal entry.
                // The corresponding rows of the global R (and the solution X)
                // are row1:row1+rm, and row1+rm is the "diagonal".
                X1 = X + row1 ;
                ASSERT (IMPLIES (Rmap != NULL, Rmap [j+n1cols] == row1 + rm)) ;
                ASSERT (IMPLIES (Rmap == NULL, j+n1cols == row1 + rm)) ;
                for (kk = 0 ; kk < nrhs ; kk++)
                {
                    xi = X1 [rm] ;      // xi = X (row1+rm,kk)
                    for (i = 0 ; i < rm ; i++)
                    {
                        xi -= spqr_conj (R [i]) * X1 [i] ;
                    }
                    X1 [rm] = spqr_divide (xi, spqr_conj (R [rm]), cc) ;
                    X1 += m ;
                }
                // After rm is incremented, it represents the number of rows
                // so far in the R block
                rm++ ;
            }

            // advance to the next column of R in the R block
            R += rm + (keepH ? (t-h) : 0) ;
        }

        // ---------------------------------------------------------------------
        // solve with the rectangular part of R
        // ---------------------------------------------------------------------

        for ( ; k < fn ; k++)
        {
            j = Rj [pr + k] ;
            ASSERT (j >= col2 && j < QRsym->n) ;
            jj = j + n1cols ;
            if (jj >= n) break ;            // in case [A Binput] was factorized

            Long ii = Rmap ? Rmap [jj] : jj ;
            ASSERT (ii >= n1rows && ii < n) ;
            if (ii < rank)
            {
                // X (ii,:) -= R (row1:row+rm-1,:)' * X (row1:row1+rm-1,:)
                X2 = X + row1 ;                     // pointer to X (row1,0)
                X1 = X ;
                for (kk = 0 ; kk < nrhs ; kk++)
                {
                    xi = X1 [ii] ;                  // X (ii,kk)
                    for (i = 0 ; i < rm ; i++)
                    {
                        xi -= spqr_conj (R [i]) * X2 [i] ;
                    }
                    X1 [ii] = xi ;
                    X1 += m ;
                    X2 += m ;
                }
            }

            // go to the next column of R
            R += rm ;
            if (keepH)
            {
                t = Stair [k] ;             // length of R+H vector
                ASSERT (t >= 0 && t <= fm) ;
                h = MIN (h+1, fm) ;         // H vector starts in row h
                ASSERT (t >= h) ;
                R += (t-h) ;
            }
        }

        row1 += rm ;    // advance past the rm rows of R in this front
    }
    ASSERT (row1 == rank) ;
}


// =============================================================================
// === SuiteSparseQR_solve (dense case) ========================================
// =============================================================================

// Use the R factor from a QR object returned by SuiteSparseQR_factorize to
// solve an upper or lower triangular system.  SuiteSparseQR_factorize computes
// a QR object whose MATLAB equivalent is [Q,R,E]=spqr(A,opts) where opts.Q =
// 'Householder' (QR contains Q in Householder form, R, and the permutation
// vector E of 0:n-1).  This function performs one of the following operations,
// where A is m-by-n:
//
// system=SPQR_RX_EQUALS_B    (0): X = R\B         B is m-by-k and X is n-by-k
// system=SPQR_RETX_EQUALS_B  (1): X = E*(R\B)     as above, E is a permutation
// system=SPQR_RTX_EQUALS_B   (2): X = R'\B        B is n-by-k and X is m-by-k
// system=SPQR_RTX_EQUALS_ETB (3): X = R'\(E'*B)   as above, E is a permutation
//
// Both X and B are dense matrices.  Only the first r rows and columns of R are
// used, where r is the rank estimate of A found by SuiteSparseQR_factorize.

template <typename Entry> cholmod_dense *SuiteSparseQR_solve    // returns X
(
    // inputs, not modified:
    int system,                 // which system to solve (0,1,2,3)
    SuiteSparseQR_factorization <Entry> *QR,    // of an m-by-n sparse matrix A
    cholmod_dense *B,           // right-hand-side, m-by-nrhs or n-by-nrhs
    // workspace and parameters
    cholmod_common *cc
)
{
    Entry *Bx ;
    cholmod_dense *W, *X ;
    Long m, n, nrhs, ldb, ok ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (B, NULL) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (B, NULL) ;
    RETURN_IF_NULL (QR, NULL) ;
    RETURN_IF_NULL (QR->QRnum, NULL) ;
    if (system < SPQR_RX_EQUALS_B || system > SPQR_RTX_EQUALS_ETB)
    {
        ERROR (CHOLMOD_INVALID, "Invalid system") ;
        return (NULL) ;
    }
    m = QR->narows ;
    n = QR->nacols ;
    if ((Long) B->nrow != ((system <= SPQR_RETX_EQUALS_B) ? m : n))
    {
        ERROR (CHOLMOD_INVALID, "invalid dimensions") ;
        return (NULL) ;
    }

    ASSERT (QR->bncols == 0) ;

    cc->status = CHOLMOD_OK ;

    nrhs = B->ncol ;
    Bx = (Entry *) B->x ;
    ldb = B->d ;

    if (system == SPQR_RX_EQUALS_B || system == SPQR_RETX_EQUALS_B)
    {

        // ---------------------------------------------------------------------
        // X = E*(R\B) or X=R\B
        // ---------------------------------------------------------------------

        Long *Rlive ;
        Entry **Rcolp ;
        X = cholmod_l_allocate_dense (n, nrhs, n, xtype, cc) ;
        Long maxfrank = QR->QRnum->maxfrank  ;
        W = cholmod_l_allocate_dense (maxfrank, nrhs, maxfrank, xtype, cc) ;
        Rlive = (Long *)   cholmod_l_malloc (maxfrank, sizeof (Long),    cc) ;
        Rcolp = (Entry **) cholmod_l_malloc (maxfrank, sizeof (Entry *), cc) ;
        ok = (X != NULL) && (W != NULL) && (cc->status == CHOLMOD_OK) ;
        if (ok)
        {
            spqr_rsolve (QR, system == SPQR_RETX_EQUALS_B, nrhs, ldb, Bx,
                (Entry *) X->x, Rcolp, Rlive, (Entry *) W->x, cc) ;
        }
        cholmod_l_free (maxfrank, sizeof (Long),    Rlive, cc) ;
        cholmod_l_free (maxfrank, sizeof (Entry *), Rcolp, cc) ;
        cholmod_l_free_dense (&W, cc) ;

    }
    else
    {

        // ---------------------------------------------------------------------
        // X = R'\(E'*B) or R'\B
        // ---------------------------------------------------------------------

        X = cholmod_l_allocate_dense (m, nrhs, m, xtype, cc) ;
        ok = (X != NULL) ;
        if (ok)
        {
            spqr_private_rtsolve (QR, system == SPQR_RTX_EQUALS_ETB, nrhs, ldb,
                Bx, (Entry *) X->x, cc) ;
        }
    }

    if (!ok)
    {
        // out of memory
        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
        cholmod_l_free_dense (&X, cc) ;
        return (NULL) ;
    }

    return (X) ;
}


template cholmod_dense *SuiteSparseQR_solve <Complex>
(
    // inputs, not modified:
    int system,                 // which system to solve
    SuiteSparseQR_factorization <Complex> *QR,  // of an m-by-n sparse matrix A
    cholmod_dense *B,           // right-hand-side, m-by-nrhs or n-by-nrhs
    // workspace and parameters
    cholmod_common *cc
) ;

template cholmod_dense *SuiteSparseQR_solve <double>
(
    // inputs, not modified:
    int system,                 // which system to solve
    SuiteSparseQR_factorization <double> *QR,   // of an m-by-n sparse matrix A
    cholmod_dense *B,           // right-hand-side, m-by-nrhs or n-by-nrhs
    // workspace and parameters
    cholmod_common *cc
) ;


// =============================================================================
// === SuiteSparseQR_solve (sparse case) =======================================
// =============================================================================

template <typename Entry> cholmod_sparse *SuiteSparseQR_solve    // returns X
(
    // inputs, not modified:
    int system,                 // which system to solve (0,1,2,3)
    SuiteSparseQR_factorization <Entry> *QR,    // of an m-by-n sparse matrix A
    cholmod_sparse *Bsparse,    // right-hand-side, m-by-nrhs or n-by-nrhs
    // workspace and parameters
    cholmod_common *cc
)
{
    cholmod_dense *Bdense, *Xdense ;
    cholmod_sparse *Xsparse = NULL ;
    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (QR, NULL) ;
    RETURN_IF_NULL (Bsparse, NULL) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (Bsparse, NULL) ;
    cc->status = CHOLMOD_OK ;

    Bdense = cholmod_l_sparse_to_dense (Bsparse, cc) ;
    Xdense = SuiteSparseQR_solve <Entry> (system, QR, Bdense, cc) ;
    cholmod_l_free_dense (&Bdense, cc) ;
    Xsparse = cholmod_l_dense_to_sparse (Xdense, TRUE, cc) ;
    cholmod_l_free_dense (&Xdense, cc) ;

    if (Xsparse == NULL)
    {
        cc->status = CHOLMOD_OUT_OF_MEMORY ;
    }
    return (Xsparse) ;
}

template cholmod_sparse *SuiteSparseQR_solve <double>
(
    // inputs, not modified:
    int system,                 // which system to solve (0,1,2,3)
    SuiteSparseQR_factorization <double> *QR,   // of an m-by-n sparse matrix A
    cholmod_sparse *Bsparse,    // right-hand-side, m-by-nrhs or n-by-nrhs
    // workspace and parameters
    cholmod_common *cc
) ;

template cholmod_sparse *SuiteSparseQR_solve <Complex>
(
    // inputs, not modified:
    int system,                 // which system to solve (0,1,2,3)
    SuiteSparseQR_factorization <Complex> *QR,  // of an m-by-n sparse matrix A
    cholmod_sparse *Bsparse,    // right-hand-side, m-by-nrhs or n-by-nrhs
    // workspace and parameters
    cholmod_common *cc
) ;


// =============================================================================
// === spqr_private_get_H_vectors ==============================================
// =============================================================================

// Get pointers to the Householder vectors in a single front.
// Returns # Householder vectors in F.

template <typename Entry> Long spqr_private_get_H_vectors
(
    // inputs
    Long f,                 // front to operate on
    SuiteSparseQR_factorization <Entry> *QR,

    // outputs
    Entry *H_Tau,           // size QRsym->maxfn
    Long *H_start,          // size QRsym->maxfn
    Long *H_end,            // size QRsym->maxfn
    cholmod_common *cc
)
{
    spqr_symbolic *QRsym ;
    spqr_numeric <Entry> *QRnum ;
    Entry *Tau ;
    Long *Rj, *Stair ;
    Long col1, col2, fp, pr, fn, fm, h, nh, p, rm, k, j, t, n1cols, n ;

    // -------------------------------------------------------------------------
    // get the R block for front F
    // -------------------------------------------------------------------------

    QRsym = QR->QRsym ;
    QRnum = QR->QRnum ;
    n1cols = QR->n1cols ;
    Rj = QRsym->Rj ;
    n = QR->nacols ;

    col1 = QRsym->Super [f] ;           // first pivot column in front F
    col2 = QRsym->Super [f+1] ;         // col2-1 is last pivot col
    fp = col2 - col1 ;                  // number of pivots in front F
    pr = QRsym->Rp [f] ;                // pointer to row indices for F
    fn = QRsym->Rp [f+1] - pr ;         // # of columns in front F

    Stair = QRnum->HStair + pr ;        // staircase of front F
    Tau = QRnum->HTau + pr ;            // Householder coeff. for front F

    fm = QRnum->Hm [f] ;                // # of rows in front F
    h = 0 ;                             // H vector starts in row h
    nh = 0 ;                            // number of Householder vectors
    p = 0 ;                             // index into R+H block

    // -------------------------------------------------------------------------
    // find all the Householder vectors in the R+H block
    // -------------------------------------------------------------------------

    rm = 0 ;                            // number of rows in R block
    for (k = 0 ; k < fn && nh < fm ; k++)
    {
        if (k < fp)
        {
            // a pivotal column of front F
            j = col1 + k ;
            ASSERT (Rj [pr + k] == j) ;
            t = Stair [k] ;                 // length of R+H vector
            ASSERT (t >= 0 && t <= fm) ;
            if (t == 0)
            {
                p += rm ;                   // no H vector, skip over R
                continue ;
            }
            else if (rm < fm)
            {
                rm++ ;                      // column k is not dead
            }
            h = rm ;                        // H vector starts in row h
            ASSERT (t >= h) ;
        }
        else
        {
            // a non-pivotal column of front F
            j = Rj [pr + k] ;
            ASSERT (j >= col2 && j < QRsym->n) ;
            t = Stair [k] ;                 // length of R+H vector
            ASSERT (t >= rm && t <= fm) ;
            h = MIN (h+1,fm) ;              // one more row of C to skip
            ASSERT (t >= h) ;
        }
        if (j+n1cols >= n) break ;          // in case [A Binput] was factorized
        p += rm ;                           // skip over R
        H_Tau [nh] = Tau [k] ;              // one more H vector found
        H_start [nh] = p ;                  // H vector starts here
        p += (t-h) ;                        // skip over the H vector
        H_end [nh] = p ;                    // H vector ends here
        nh++ ;
        if (h == fm) break ;                // that was the last H vector
    }

    return (nh) ;
}


// =============================================================================
// === spqr_private_load_H_vectors =============================================
// =============================================================================

// Load Householder vectors h1:h2-1 into the panel V.  Return # of rows in V.

template <typename Entry> Long spqr_private_load_H_vectors
(
    // input
    Long h1,            // load vectors h1 to h2-1
    Long h2,
    Long *H_start,      // vector h starts at R [H_start [h]]
    Long *H_end,        // vector h ends at R [H_end [h]-1]
    Entry *R,           // Rblock [f]
    // output
    Entry *V,           // V is v-by-(h2-h1) and lower triangular
    cholmod_common *cc
)
{
    // v = length of last H vector
    Long v = H_end [h2-1] - H_start [h2-1] + (h2-h1) ;
    Entry *V1 = V ;
    for (Long h = h1 ; h < h2 ; h++)
    {
        Long i ;
        // This part of V is not accessed, for testing only:
        // for (i = 0 ; i < h-h1 ; i++) V1 [i] = 0 ;
        i = h-h1 ;
        V1 [i++] = 1 ;
        for (Long p = H_start [h] ; p < H_end [h] ; p++)
        {
            V1 [i++] = R [p] ;
        }
        for ( ; i < v ; i++)
        {
            V1 [i] = 0 ;
        }
        V1 += v ;
    }
    return (v) ;
}


// =============================================================================
// === spqr_private_Happly =====================================================
// =============================================================================

// Given a QR factorization from spqr_1factor, apply the Householder vectors
// to a dense matrix X.

template <typename Entry> void spqr_private_Happly
(
    // inputs
    int method,             // 0,1,2,3
    SuiteSparseQR_factorization <Entry> *QR,
    Long hchunk,            // apply hchunk Householder vectors at a time

    // input/output
    Long m,
    Long n,
    Entry *X,               // size m-by-n with leading dimension m; only
                            // X (n1rows:m-1,:) or X (:,n1rows:n-1) is modified

    // workspace, not defined on input or output
    Entry *H_Tau,           // size QRsym->maxfn
    Long *H_start,          // size QRsym->maxfn
    Long *H_end,            // size QRsym->maxfn
    Entry *V,               // size v-by-hchunk, where v = QRnum->maxfm
    Entry *C,               // size: method 0,1: v*n,     method 2,3: m*v
    Entry *W,               // size: method 0,1: h*h+n*h, method 2,3: h*h+m*h
                            // where h = hchunk
    cholmod_common *cc
)
{

    spqr_symbolic *QRsym ;
    spqr_numeric <Entry> *QRnum ;
    Entry **Rblock, *R, *X2 ;
    Long *Hii, *Hip, *Hi ;
    Long nf, f, nh, h1, h2, v, n1rows, m2, n2 ;

    // -------------------------------------------------------------------------
    // get the contents of the QR factorization
    // -------------------------------------------------------------------------

    QRsym = QR->QRsym ;
    QRnum = QR->QRnum ;
    ASSERT (QRnum->keepH) ;
    nf = QRsym->nf ;
    Rblock = QRnum->Rblock ;
    Hii = QRnum->Hii ;
    Hip = QRsym->Hip ;
    ASSERT (QR->narows == ((method <= SPQR_QX) ? m : n)) ;
    n1rows = QR->n1rows ;

    // -------------------------------------------------------------------------
    // operate on X (n1rows:m-1,:) or X (:,n1rows:n-1)
    // -------------------------------------------------------------------------

    // The Householder vectors must be shifted downwards by n1rows, past the
    // singleton rows which are not part of the multifrontal part of the
    // QR factorization.

    if (method == SPQR_QTX || method == SPQR_QX)
    {
        // Q*X or Q'*X
        X2 = X + n1rows ;
        n2 = n ;
        m2 = m - n1rows ;
    }
    else
    {
        X2 = X + n1rows * m ;
        n2 = n - n1rows ; 
        m2 = m ;
    }

    // -------------------------------------------------------------------------
    // apply the Householder vectors
    // -------------------------------------------------------------------------

    if (method == SPQR_QTX || method == SPQR_XQ)
    {

        // ---------------------------------------------------------------------
        // apply in forward direction
        // ---------------------------------------------------------------------

        for (f = 0 ; f < nf ; f++)
        {
            // get the Householder vectors for front F
            nh = spqr_private_get_H_vectors (f, QR, H_Tau, H_start, H_end, cc) ;
            R = Rblock [f] ;
            Hi = &Hii [Hip [f]] ;               // list of row indices of H

            // apply the Householder vectors, one panel at a time
            for (h1 = 0 ; h1 < nh ; h1 = h2)
            {
                // load vectors h1:h2-1 from R into the panel V and apply them
                h2 = MIN (h1 + hchunk, nh) ;
                v = spqr_private_load_H_vectors (h1, h2, H_start, H_end, R, V,
                    cc) ;
                ASSERT (v+h1 <= QR->QRnum->Hm [f]) ;
                spqr_panel (method, m2, n2, v, h2-h1, Hi+h1, V, H_Tau+h1,
                    m, X2, C, W, cc) ;
            }
        }
    }
    else
    {

        // ---------------------------------------------------------------------
        // apply in backward direction
        // ---------------------------------------------------------------------

        for (f = nf-1 ; f >= 0 ; f--)
        {

            // get the Householder vectors for front F
            nh = spqr_private_get_H_vectors (f, QR, H_Tau, H_start, H_end, cc) ;
            R = Rblock [f] ;
            Hi = &Hii [Hip [f]] ;               // list of row indices of H

            // apply the Householder vectors, one panel at a time
            for (h2 = nh ; h2 > 0 ; h2 = h1)
            {
                // load vectors h1:h2-1 from R into the panel V and apply them
                h1 = MAX (h2 - hchunk, 0) ;
                v = spqr_private_load_H_vectors (h1, h2, H_start, H_end, R, V,
                    cc) ;
                ASSERT (v+h1 <= QR->QRnum->Hm [f]) ;
                spqr_panel (method, m2, n2, v, h2-h1, Hi+h1, V, H_Tau+h1,
                    m, X2, C, W, cc) ;
            }
        }
    }
}


// =============================================================================
// === SuiteSparseQR_qmult (dense case) ========================================
// =============================================================================

// Applies Q in Householder form (as stored in the QR factorization object
// returned by SuiteSparseQR_factorize) to a dense matrix X.
//
//  method SPQR_QTX (0): Y = Q'*X
//  method SPQR_QX  (1): Y = Q*X
//  method SPQR_XQT (2): Y = X*Q'
//  method SPQR_XQ  (3): Y = X*Q

#define HCHUNK 32        // FUTURE: make this an input parameter

#define FREE_WORK \
{ \
    cholmod_l_free_dense (&Zdense, cc) ; \
    cholmod_l_free_dense (&Vdense, cc) ; \
    cholmod_l_free_dense (&Wdense, cc) ; \
    cholmod_l_free_dense (&Cdense, cc) ; \
    cholmod_l_free (maxfn, sizeof (Entry), H_Tau,   cc) ; \
    cholmod_l_free (maxfn, sizeof (Long),  H_start, cc) ; \
    cholmod_l_free (maxfn, sizeof (Long),  H_end,   cc) ; \
}

// returns Y of size m-by-n, or NULL on failure
template <typename Entry> cholmod_dense *SuiteSparseQR_qmult
(
    // inputs, not modified
    int method,             // 0,1,2,3
    SuiteSparseQR_factorization <Entry> *QR,
    cholmod_dense *Xdense,  // size m-by-n with leading dimension ldx

    // workspace and parameters
    cholmod_common *cc
)
{
    cholmod_dense *Ydense, *Cdense, *Vdense, *Wdense, *Zdense ;
    Entry *X, *Y, *X1, *Y1, *Z1, *C, *V, *Z, *W, *H_Tau ;
    Long *HPinv, *H_start, *H_end ;
    Long i, k, mh, v, hchunk, ldx, m, n, maxfn, ok ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (QR, NULL) ;
    RETURN_IF_NULL (QR->QRnum, NULL) ;
    RETURN_IF_NULL (QR->QRnum->Hm, NULL) ;
    RETURN_IF_NULL (Xdense, NULL) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (Xdense, NULL) ;
    cc->status = CHOLMOD_OK ;

    // get HPinv from QR->HP1inv if singletons exist, else QR->QRnum->HPinv
    HPinv = (QR->n1cols > 0) ? QR->HP1inv : QR->QRnum->HPinv ;

    v = QR->QRnum->maxfm ;
    mh = QR->narows ;
    maxfn = QR->QRsym->maxfn ;

    X = (Entry *) Xdense->x ;
    m = Xdense->nrow ;
    n = Xdense->ncol ;
    ldx = Xdense->d ;
    ASSERT (ldx >= m) ;

    if (method == SPQR_QTX || method == SPQR_QX)
    {
        // rows of H and X must be the same
        if (mh != m)
        {
            ERROR (CHOLMOD_INVALID, "mismatched dimensions") ;
            return (NULL) ;
        }
    }
    else if (method == SPQR_XQT || method == SPQR_XQ)
    {
        // rows of H and columns of X must be the same
        if (mh != n)
        {
            ERROR (CHOLMOD_INVALID, "mismatched dimensions") ;
            return (NULL) ;
        }
    }
    else
    {
        ERROR (CHOLMOD_INVALID, "invalid method") ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // allocate result Y
    // -------------------------------------------------------------------------

    Ydense = cholmod_l_allocate_dense (m, n, m, xtype, cc) ;
    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        return (NULL) ;
    }
    Y = (Entry *) Ydense->x ;

    if (m == 0 || n == 0)
    {
        // nothing to do
        return (Ydense) ;    
    }

    // -------------------------------------------------------------------------
    // allocate workspace
    // -------------------------------------------------------------------------

    Z = NULL ;
    Zdense = NULL ;
    ok = TRUE ;
    if (method == SPQR_QX || method == SPQR_XQT)
    {
        // Z of size m-by-n is needed only for Q*X and X*Q'
        Zdense = cholmod_l_allocate_dense (m, n, m, xtype, cc) ;
        ok = (Zdense != NULL) ;
    }

    hchunk = HCHUNK ;
    ASSERT (v <= mh) ;

    // C is workspace of size v-by-n or m-by-v
    Cdense = cholmod_l_allocate_dense (v, (method <= SPQR_QX) ? n : m,
        v, xtype, cc) ;
    Vdense = NULL ;
    Wdense = NULL ;

    H_Tau   = (Entry *) cholmod_l_malloc (maxfn, sizeof (Entry), cc) ;
    H_start = (Long *)  cholmod_l_malloc (maxfn, sizeof (Long),  cc) ;
    H_end   = (Long *)  cholmod_l_malloc (maxfn, sizeof (Long),  cc) ;

    if (!ok || Cdense == NULL || cc->status < CHOLMOD_OK)
    {
        // out of memory; free workspace and result Y
        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
        cholmod_l_free_dense (&Ydense, cc) ;
        FREE_WORK ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // copy X into Z
    // -------------------------------------------------------------------------

    if (method == SPQR_QX || method == SPQR_XQT)
    {
        // Z = X
        Z = (Entry *) Zdense->x ;
        Z1 = Z ;
        X1 = X ;
        for (k = 0 ; k < n ; k++)
        {
            for (i = 0 ; i < m ; i++)
            {
                Z1 [i] = X1 [i] ;
            }
            X1 += ldx ;
            Z1 += m ;
        }
    }

    // -------------------------------------------------------------------------
    // allocate O(hchunk) workspace
    // -------------------------------------------------------------------------

    // V is workspace of size v-by-hchunk
    Vdense = cholmod_l_allocate_dense (v, hchunk, v, xtype, cc) ;

    // W is workspace of size h*h+n*h or h*h+m*h where h = hchunk
    Wdense = cholmod_l_allocate_dense (hchunk,
        hchunk + ((method <= SPQR_QX) ? n : m), hchunk, xtype, cc) ;

    // -------------------------------------------------------------------------
    // punt if out of memory
    // -------------------------------------------------------------------------

    if (Vdense == NULL || Wdense == NULL)
    {
        // PUNT: out of memory; try again with hchunk = 1
        cc->status = CHOLMOD_OK ;
        hchunk = 1 ;

        cholmod_l_free_dense (&Vdense, cc) ;
        cholmod_l_free_dense (&Wdense, cc) ;

        Vdense = cholmod_l_allocate_dense (v, hchunk, v, xtype, cc) ;
        Wdense = cholmod_l_allocate_dense (hchunk,
            hchunk + ((method <= SPQR_QX) ? n : m), hchunk, xtype, cc) ;

        if (Vdense == NULL || Wdense == NULL)
        {
            // out of memory; free workspace and result Y
            ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
            cholmod_l_free_dense (&Ydense, cc) ;
            FREE_WORK ;
            return (NULL) ;
        }
    }

    // the dimension (->nrow, ->ncol, and ->d) of this workspace is not used,
    // just the arrays themselves.
    V = (Entry *) Vdense->x ;
    C = (Entry *) Cdense->x ;
    W = (Entry *) Wdense->x ;

    // -------------------------------------------------------------------------
    // Y = Q'*X, Q*X, X*Q, or X*Q'
    // -------------------------------------------------------------------------

    PR (("Qfmult Method %d m %ld n %ld X %p Y %p\n", method, m, n, X, Y)) ;

    if (method == SPQR_QTX)
    {

        // ---------------------------------------------------------------------
        // Y = Q'*X
        // ---------------------------------------------------------------------

        // Y (P,:) = X, and change leading dimension from ldx to m
        X1 = X ;
        Y1 = Y ;
        for (k = 0 ; k < n ; k++)
        {
            for (i = 0 ; i < m ; i++)
            {
                Y1 [HPinv [i]] = X1 [i] ;
            }
            X1 += ldx ;
            Y1 += m ;
        }

        // apply H to Y
        spqr_private_Happly (method, QR, hchunk, m, n, Y, H_Tau, H_start, H_end,
            V, C, W, cc) ;

    }
    else if (method == SPQR_QX)
    {

        // ---------------------------------------------------------------------
        // Y = Q*X
        // ---------------------------------------------------------------------

        // apply H to Z
        spqr_private_Happly (method, QR, hchunk, m, n, Z, H_Tau, H_start, H_end,
            V, C, W, cc) ;

        // Y = Z (P,:)
        Z1 = Z ;
        Y1 = Y ;
        for (k = 0 ; k < n ; k++)
        {
            for (i = 0 ; i < m ; i++)
            {
                Y1 [i] = Z1 [HPinv [i]] ;
            }
            Z1 += m ;
            Y1 += m ;
        }

    }
    else if (method == SPQR_XQT)
    {

        // ---------------------------------------------------------------------
        // Y = X*Q'
        // ---------------------------------------------------------------------

        // apply H to Z
        spqr_private_Happly (method, QR, hchunk, m, n, Z, H_Tau, H_start, H_end,
            V, C, W, cc) ;

        // Y = Z (:,P)
        Y1 = Y ;
        for (k = 0 ; k < n ; k++)
        {
            Z1 = Z + HPinv [k] * m ;    // m = leading dimension of Z
            for (i = 0 ; i < m ; i++)
            {
                Y1 [i] = Z1 [i] ;
            }
            Y1 += m ;
        }

    }
    else if (method == SPQR_XQ)
    {

        // ---------------------------------------------------------------------
        // Y = X*Q
        // ---------------------------------------------------------------------

        // Y (:,P) = X and change leading dimension from ldx to m
        X1 = X ;
        for (k = 0 ; k < n ; k++)
        {
            Y1 = Y + HPinv [k] * m ;    // m = leading dimension of Y
            for (i = 0 ; i < m ; i++)
            {
                Y1 [i] = X1 [i] ;
            }
            X1 += ldx ;
        }

        // apply H to Y
        spqr_private_Happly (method, QR, hchunk, m, n, Y, H_Tau, H_start, H_end,
            V, C, W, cc) ;

    }

    // -------------------------------------------------------------------------
    // free workspace and return Y
    // -------------------------------------------------------------------------

    FREE_WORK ;

    if (CHECK_BLAS_INT && !cc->blas_ok)
    {
        ERROR (CHOLMOD_INVALID, "problem too large for the BLAS") ;
        cholmod_l_free_dense (&Ydense, cc) ;
        return (NULL) ;
    }

    return (Ydense) ;
}


template cholmod_dense *SuiteSparseQR_qmult <double>
(
    // inputs, not modified
    int method,             // 0,1,2,3
    SuiteSparseQR_factorization <double> *QR,
    cholmod_dense *Xdense,  // size m-by-n with leading dimension ldx

    // workspace and parameters
    cholmod_common *cc
) ;


template cholmod_dense *SuiteSparseQR_qmult <Complex>
(
    // inputs, not modified
    int method,             // 0,1,2,3
    SuiteSparseQR_factorization <Complex> *QR,
    cholmod_dense *Xdense,  // size m-by-n with leading dimension ldx

    // workspace and parameters
    cholmod_common *cc
) ;


// =============================================================================
// === SuiteSparseQR_qmult (sparse case) =======================================
// =============================================================================

// returns Y of size m-by-n, or NULL on failure
template <typename Entry> cholmod_sparse *SuiteSparseQR_qmult
(
    // inputs, not modified
    int method,                 // 0,1,2,3
    SuiteSparseQR_factorization <Entry> *QR,
    cholmod_sparse *Xsparse,    // size m-by-n
    // workspace and parameters
    cholmod_common *cc
)
{
    cholmod_dense *Xdense, *Ydense ;
    cholmod_sparse *Ysparse = NULL ;
    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (QR, NULL) ;
    RETURN_IF_NULL (Xsparse, NULL) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (Xsparse, NULL) ;
    cc->status = CHOLMOD_OK ;

    Xdense = cholmod_l_sparse_to_dense (Xsparse, cc) ;
    Ydense = SuiteSparseQR_qmult <Entry> (method, QR, Xdense, cc) ;
    cholmod_l_free_dense (&Xdense, cc) ;
    Ysparse = cholmod_l_dense_to_sparse (Ydense, TRUE, cc) ;
    cholmod_l_free_dense (&Ydense, cc) ;

    if (Ysparse == NULL)
    {
        cc->status = CHOLMOD_OUT_OF_MEMORY ;
    }
    return (Ysparse) ;
}

template cholmod_sparse *SuiteSparseQR_qmult <double>
(
    // inputs, not modified
    int method,                 // 0,1,2,3
    SuiteSparseQR_factorization <double> *QR,
    cholmod_sparse *Xsparse,    // size m-by-n
    // workspace and parameters
    cholmod_common *cc
) ;

template cholmod_sparse *SuiteSparseQR_qmult <Complex>
(
    // inputs, not modified
    int method,                 // 0,1,2,3
    SuiteSparseQR_factorization <Complex> *QR,
    cholmod_sparse *Xsparse,    // size m-by-n
    // workspace and parameters
    cholmod_common *cc
) ;


// =============================================================================
// === SuiteSparseQR_free ======================================================
// =============================================================================

// Free the QR object; this is just a user-callable wrapper for spqr_freefac.

template <typename Entry> int SuiteSparseQR_free
(
    SuiteSparseQR_factorization <Entry> **QR,
    cholmod_common *cc
)
{
    RETURN_IF_NULL_COMMON (FALSE) ;
    spqr_freefac <Entry> (QR, cc) ;
    return (TRUE) ;
}

template int SuiteSparseQR_free <double>
(
    SuiteSparseQR_factorization <double> **QR,
    cholmod_common *cc
) ;

template int SuiteSparseQR_free <Complex>
(
    SuiteSparseQR_factorization <Complex> **QR,
    cholmod_common *cc
) ;


// =============================================================================
// === SuiteSparseQR_min2norm ==================================================
// =============================================================================

// Find the min 2-norm solution for underdetermined systems (A is m-by-n with
// m<n), or find a least-squares solution otherwise.

template <typename Entry> cholmod_dense *SuiteSparseQR_min2norm
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    cholmod_sparse *A,
    cholmod_dense *B,
    cholmod_common *cc
)
{
    cholmod_dense *X ; 
    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (A, NULL) ;
    RETURN_IF_NULL (B, NULL) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (A, NULL) ;
    RETURN_IF_XTYPE_INVALID (B, NULL) ;
    cc->status = CHOLMOD_OK ;

    if (A->nrow < A->ncol)
    {

        double t0 = SuiteSparse_time ( ) ;

        // x=A\B, using a QR factorization of A'
        SuiteSparseQR_factorization <Entry> *QR ;
        cholmod_sparse *AT ;
        cholmod_dense *Y ; 
        // [Q,R,E] = qr (A')
        AT = cholmod_l_transpose (A, 2, cc) ;
        QR = SuiteSparseQR_factorize <Entry> (ordering, tol, AT, cc);
        cholmod_l_free_sparse (&AT, cc) ;
        // solve Y = R'\(E'*B)
        Y = SuiteSparseQR_solve (SPQR_RTX_EQUALS_ETB, QR, B, cc) ;
        // X = Q*Y
        X = SuiteSparseQR_qmult (SPQR_QX, QR, Y, cc) ;
        cholmod_l_free_dense (&Y, cc) ;
        spqr_freefac (&QR, cc) ;

        double t3 = SuiteSparse_time ( ) ;
        double total_time = t3 - t0 ;
        cc->SPQR_solve_time =
            total_time - cc->SPQR_analyze_time - cc->SPQR_factorize_time ;

    }
    else
    {
        // x=A\B, using a QR factorization of A
        SuiteSparseQR <Entry> (ordering, tol, 0, 2, A, NULL, B, NULL, &X,
            NULL, NULL, NULL, NULL, NULL, cc) ;
    }

    if (X == NULL)
    {
        // After A and B are valid, the only way any of the above functions can
        // fail is by running out of memory.  However, if (say) AT is NULL
        // because of insufficient memory, SuiteSparseQR_factorize will report
        // CHOLMOD_INVALID since its input is NULL.  In that case, restore the
        // error back to CHOLMOD_OUT_OF_MEMORY.
        cc->status = CHOLMOD_OUT_OF_MEMORY ;
    }

    return (X) ;
}

template cholmod_dense *SuiteSparseQR_min2norm <double>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    cholmod_sparse *A,
    cholmod_dense *B,
    cholmod_common *cc
) ;

template cholmod_dense *SuiteSparseQR_min2norm <Complex>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    cholmod_sparse *A,
    cholmod_dense *B,
    cholmod_common *cc
) ;


// =============================================================================
// === SuiteSparseQR_min2norm (sparse case) ====================================
// =============================================================================

template <typename Entry> cholmod_sparse *SuiteSparseQR_min2norm    // returns X
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    cholmod_sparse *A,
    cholmod_sparse *Bsparse,
    cholmod_common *cc
)
{
    double t0 = SuiteSparse_time ( ) ;

    cholmod_dense *Bdense, *Xdense ;
    cholmod_sparse *Xsparse = NULL ;
    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (A, NULL) ;
    RETURN_IF_NULL (Bsparse, NULL) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (A, NULL) ;
    RETURN_IF_XTYPE_INVALID (Bsparse, NULL) ;
    cc->status = CHOLMOD_OK ;

    Bdense = cholmod_l_sparse_to_dense (Bsparse, cc) ;
    Xdense = SuiteSparseQR_min2norm <Entry> (ordering, tol, A, Bdense, cc) ;
    cholmod_l_free_dense (&Bdense, cc) ;
    Xsparse = cholmod_l_dense_to_sparse (Xdense, TRUE, cc) ;
    cholmod_l_free_dense (&Xdense, cc) ;

    if (Xsparse == NULL)
    {
        cc->status = CHOLMOD_OUT_OF_MEMORY ;
    }

    double t3 = SuiteSparse_time ( ) ;
    double total_time = t3 - t0 ;
    cc->SPQR_solve_time =
        total_time - cc->SPQR_analyze_time - cc->SPQR_factorize_time ;

    return (Xsparse) ;
}

template cholmod_sparse *SuiteSparseQR_min2norm <double>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    cholmod_sparse *A,
    cholmod_sparse *Bsparse,
    cholmod_common *cc
) ;

template cholmod_sparse *SuiteSparseQR_min2norm <Complex>
(
    int ordering,           // all, except 3:given treated as 0:fixed
    double tol,
    cholmod_sparse *A,
    cholmod_sparse *Bsparse,
    cholmod_common *cc
) ;
#endif
