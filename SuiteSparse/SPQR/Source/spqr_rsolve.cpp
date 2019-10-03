// =============================================================================
// === spqr_rsolve =============================================================
// =============================================================================

// Solve X = E*(R\B) or X=R\B using the QR factorization from spqr_1factor
// (including the singleton-row R and the multifrontal R).

/*
   Let A by an m-by-n matrix.

   If just A was factorized, then the QR factorization in QRsym and QRnum
   contains the global R factor:

        R22
         0

   where n1cols = 0, n1rows = 0, A is m-by-n, R22 is (QRnum->rank1)-by-n.
   R22 is the multifrontal part of R.

   If [A Binput] was factorized and no singletons were removed prior to
   factorization, then the QR factorization in QRsym and QRnum contains the
   global R factor:

        R22 C2
         0  C3

   where R22 is (QRnum->rank1)-by-n and where A is m-by-n and [C2 ; C2] is
   m-by-bncols.

   If [A Binput] was factorized and singletons were removed prior to
   factorization, then the QR factorization of the [S2 B2] matrix was computed
   (the global R):

       R11 R12 C1
        0  R22 C2
        0   0  C3

    where R11 has n1cols columns and n1rows rows.  [R11 R12] is the singleton-
    row part of R.  The R1 = [R11 R12] matrix (not in QRsym and QRnum) contains
    the singleton rows.  The QR factorization in QRsym and QRnum contains only
    the R factor:

        R22 C2
         0  C3

    To solve with R22, the columns of R22 must be shifted by n1cols and then
    permuted via Q1fill to get a column index in the range n1cols to n-1, where
    A is m-by-n.  However, the QR factorization (and Q1fill) also contain column
    indices >= n.  If a column index in the QR factorization is >= n, then it
    refers to a column of C and is ignored in this solve phase.

    The row indices of R22 must also be shifted, by n1rows.

    Note that in all cases, the number of rows of R22 is given by QRnum->rank1.
*/

#include "spqr.hpp"

template <typename Entry> void spqr_rsolve
(
    // inputs
    SuiteSparseQR_factorization <Entry> *QR,
    int use_Q1fill,         // if TRUE, do X=E*(R\B), otherwise do X=R\B

    Long nrhs,              // number of columns of B
    Long ldb,               // leading dimension of B
    Entry *B,               // size m-by-nrhs with leading dimesion ldb

    // output
    Entry *X,               // size n-by-nrhs with leading dimension n

    // workspace
    Entry **Rcolp,          // size QRnum->maxfrank
    Long *Rlive,            // size QRnum->maxfrank
    Entry *W,               // size QRnum->maxfrank * nrhs

    cholmod_common *cc
)
{
    spqr_symbolic *QRsym ;
    spqr_numeric <Entry> *QRnum ;
    Long n1rows, n1cols, n ;
    Long *Q1fill, *R1p, *R1j ;
    Entry *R1x ;

    Entry xi ;
    Entry **Rblock, *R, *W1, *B1, *X1 ;
    Long *Rp, *Rj, *Super, *HStair, *Hm, *Stair ;
    char *Rdead ;
    Long nf, // m,
        rank, j, f, col1, col2, fp, pr, fn, rm, k, i, row1, row2, ii,
        keepH, fm, h, t, live, kk ;

    // -------------------------------------------------------------------------
    // get the contents of the QR object
    // -------------------------------------------------------------------------

    QRsym = QR->QRsym ;
    QRnum = QR->QRnum ;
    n1rows = QR->n1rows ;
    n1cols = QR->n1cols ;
    n = QR->nacols ;
    Q1fill = use_Q1fill ? QR->Q1fill : NULL ;
    R1p = QR->R1p ;
    R1j = QR->R1j ;
    R1x = QR->R1x ;

    keepH = QRnum->keepH ;
    PR (("rsolve keepH %ld\n", keepH)) ;
    nf = QRsym->nf ;
    // m = QRsym->m ;
    Rblock = QRnum->Rblock ;
    Rp = QRsym->Rp ;
    Rj = QRsym->Rj ;
    Super = QRsym->Super ;
    Rdead = QRnum->Rdead ;
    rank = QR->rank ;   // R22 is R(n1rows:rank-1,n1cols:n-1) of
                        // the global R.
    HStair = QRnum->HStair ;
    Hm = QRnum->Hm ;

    // -------------------------------------------------------------------------
    // X = 0
    // -------------------------------------------------------------------------

    X1 = X ;
    for (kk = 0 ; kk < nrhs ; kk++)
    {
        for (i = 0 ; i < n ; i++)
        {
            X1 [i] = 0 ;
        }
        X1 += n ;
    }

    // =========================================================================
    // === solve with the multifrontal rows of R ===============================
    // =========================================================================

    Stair = NULL ;
    fm = 0 ;
    h = 0 ;
    t = 0 ;

    // start with row2 = QR-num->rank + n1rows, the last row of the combined R
    // factor of [A Binput]

    row2 = QRnum->rank + n1rows ;
    for (f = nf-1 ; f >= 0 ; f--)
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
        // find the live pivot columns in this R or RH block
        // ---------------------------------------------------------------------

        rm = 0 ;                            // number of rows in R block
        for (k = 0 ; k < fp ; k++)
        {
            j = col1 + k ;
            ASSERT (Rj [pr + k] == j) ;
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
                live = (!Rdead [j])  ;
            }

            if (live)
            {
                // R (rm,k) is a "diagonal"; rm and k are local indices.
                // Keep track of a pointer to the first entry R(0,k)
                Rcolp [rm] = R ;
                Rlive [rm] = j ;
                rm++ ;
            }
            else
            {
                // compute the basic solution; dead columns are zero
                ii = Q1fill ? Q1fill [j+n1cols] : j+n1cols ;
                if (ii < n)
                {
                    for (kk = 0 ; kk < nrhs ; kk++)
                    {
                        // X (ii,kk) = 0; note this is stride n
                        X [INDEX (ii,kk,n)] = 0 ;
                    }
                }
            }

            // advance to the next column of R in the R block
            R += rm + (keepH ? (t-h) : 0) ;
        }

        // There are rm rows in this R block, corresponding to the rm live
        // columns in the range col1:col2-1.  The list of live global column
        // indices is given in Rlive [0:rm-1].  Pointers to the numerical
        // entries for each of these columns in this R block are given in
        // Rcolp [0:rm-1].   The rm rows in this R block correspond to
        // row1:row2-1 of R and b.

        row1 = row2 - rm ;

        // ---------------------------------------------------------------------
        // get the right-hand sides for these rm equations
        // ---------------------------------------------------------------------

        // W = B (row1:row2-1,:)
        ASSERT (rm <= QRnum->maxfrank) ;
        W1 = W ;
        B1 = B ;
        for (kk = 0 ; kk < nrhs ; kk++)
        {
            for (i = 0 ; i < rm ; i++)
            {
                ii = row1 + i ;
                ASSERT (ii >= n1rows) ;
                W1 [i] = (ii < rank) ? B1 [ii] : 0 ;
            }
            W1 += rm ;
            B1 += ldb ;
        }

        // ---------------------------------------------------------------------
        // solve with the rectangular part of R (W = W - R2*x2)
        // ---------------------------------------------------------------------

        for ( ; k < fn ; k++)
        {
            j = Rj [pr + k] ;
            ASSERT (j >= col2 && j < QRsym->n) ;
            ii = Q1fill ? Q1fill [j+n1cols] : j+n1cols ;
            ASSERT ((ii < n) == (j+n1cols < n)) ;
            // break if past the last column of A in QR of [A Binput]
            if (ii >= n) break ;

            if (!Rdead [j])
            {
                // global column j is live
                W1 = W ;
                for (kk = 0 ; kk < nrhs ; kk++)
                {
                    xi = X [INDEX (ii,kk,n)] ;        // xi = X (ii,kk)
                    if (xi != (Entry) 0)
                    {
                        FLOP_COUNT (2*rm) ;
                        for (i = 0 ; i < rm ; i++)
                        {
                            W1 [i] -= R [i] * xi ;
                        }
                    }
                    W1 += rm ;
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

        // ---------------------------------------------------------------------
        // solve with the squeezed upper triangular part of R
        // ---------------------------------------------------------------------

        for (k = rm-1 ; k >= 0 ; k--)
        {
            R = Rcolp [k] ;                 // kth live pivot column
            j = Rlive [k] ;                 // is jth global column
            ii = Q1fill ? Q1fill [j+n1cols] : j+n1cols ;
            ASSERT ((ii < n) == (j+n1cols < n)) ;
            if (ii < n)
            {
                W1 = W ;
                for (kk = 0 ; kk < nrhs ; kk++)
                {
                    // divide by the "diagonal"
                    // xi = W1 [k] / R [k] ;
                    xi = spqr_divide (W1 [k], R [k], cc) ;
                    FLOP_COUNT (1) ;
                    X [INDEX(ii,kk,n)] = xi ;
                    if (xi != (Entry) 0)
                    {
                        FLOP_COUNT (2*k) ;
                        for (i = 0 ; i < k ; i++)
                        {
                            W1 [i] -= R [i] * xi ;
                        }
                    }
                    W1 += rm ;
                }
            }
        }

        // ---------------------------------------------------------------------
        // prepare for the R block for front f-1
        // ---------------------------------------------------------------------

        row2 = row1 ;
    }
    ASSERT (row2 == n1rows) ;

    // =========================================================================
    // === solve with the singleton rows of R ==================================
    // =========================================================================

    FLOP_COUNT ((n1rows <= 0) ? 0 :
        nrhs * (n1rows + (2 * (R1p [n1rows] - n1rows)))) ;

    for (kk = 0 ; kk < nrhs ; kk++)
    {
        for (i = n1rows-1 ; i >= 0 ; i--)
        {
            // get the right-hand side for this ith singleton row
            Entry x = B [i] ;
            // solve with the "off-diagonal" entries, x = x-R(i,:)*x2
            for (Long p = R1p [i] + 1 ; p < R1p [i+1] ; p++)
            {
                Long jnew = R1j [p] ;
                ASSERT (jnew >= i && jnew < n) ;
                Long jold = Q1fill ? Q1fill [jnew] : jnew ;
                ASSERT (jold >= 0 && jold < n) ;
                x -= R1x [p] * X [jold] ;
            }
            // divide by the "diagonal" (the singleton entry itself)
            Long p = R1p [i] ;
            Long jnew = R1j [p] ;
            Long jold = Q1fill ? Q1fill [jnew] : jnew ;
            ASSERT (jold >= 0 && jold < n) ;
            // X [jold] = x / R1x [p] ; using cc->complex_divide
            X [jold] = spqr_divide (x, R1x [p], cc) ;
        }
        B += ldb ;
        X += n ;
    }
}

// =============================================================================

template void spqr_rsolve <double>
(
    // inputs
    SuiteSparseQR_factorization <double> *QR,
    int use_Q1fill,

    Long nrhs,              // number of columns of B
    Long ldb,               // leading dimension of B
    double *B,              // size m-by-nrhs with leading dimesion ldb

    // output
    double *X,              // size n-by-nrhs with leading dimension n

    // workspace
    double **Rcolp,
    Long *Rlive,
    double *W,

    cholmod_common *cc
) ;


template void spqr_rsolve <Complex>
(
    // inputs
    SuiteSparseQR_factorization <Complex> *QR,
    int use_Q1fill,

    Long nrhs,              // number of columns of B
    Long ldb,               // leading dimension of B
    Complex *B,             // size m-by-nrhs with leading dimesion ldb

    // output
    Complex *X,             // size n-by-nrhs with leading dimension n

    // workspace
    Complex **Rcolp,
    Long *Rlive,
    Complex *W,

    cholmod_common *cc
) ;

