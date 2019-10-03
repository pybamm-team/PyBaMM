// =============================================================================
// === spqr_stranspose =========================================================
// =============================================================================

// Given a fill-reducing ordering, Qfill, and the user's input matrix A in 
// standard compressed-column form, construct S=A(P,Qfill)' if S is considered
// to be in compressed-column form (S is A(P,Qfill) in compressed-row form).
// The permutation P sorts the rows of S according to the leftmost column index
// in each row.  Empty rows of A are placed last in S.  The input matrix A need
// not have sorted columns, but it cannot have duplicate entries.  Note that P
// itself is not returned; its inverse PLinv is returned instead.
//
// The time taken by this function is O(nnz(A)+m+n).  In practice, it takes
// slightly more time than computing the transpose, S=A'.

#include "spqr.hpp"

void spqr_stranspose1
(
    // input, not modified
    cholmod_sparse *A,  // m-by-n
    Long *Qfill,        // size n, fill-reducing column permutation;
                        // Qfill [k] = j
                        // if the kth column of S is the jth column of A.
                        // Identity permutation is used if Qfill is NULL.

    // output, contents not defined on input
    // FUTURE : make S cholmod_sparse
    Long *Sp,           // size m+1, row pointers of S
    Long *Sj,           // size nz, column indices of S
    Long *PLinv,        // size m, inverse row permutation, PLinv [i] = k
    Long *Sleft,        // size n+2, Sleft [j] ... Sleft [j+1]-1 is the list of
                        // rows of S whose leftmost column index is j.  The list
                        // can be empty (that is, Sleft [j] == Sleft [j+1]).
                        // Sleft [n] is the number of non-empty rows of S, and
                        // Sleft [n+1] is always m.  That is, Sleft [n] ...
                        // Sleft [n+1]-1 gives the empty rows of S.

    // workspace, not defined on input or output
    Long *W              // size m
)
{
    Long i, j, p, pend, t, k, row, col, kstart, s, m, n, *Ap, *Ai ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    m = A->nrow ;
    n = A->ncol ;
    Ap = (Long *) A->p ;
    Ai = (Long *) A->i ;

    // -------------------------------------------------------------------------
    // clear the inverse permutation
    // -------------------------------------------------------------------------

    for (i = 0 ; i < m ; i++)
    {
        PLinv [i] = EMPTY ;             // position of A(i,:) undefined in S
        DEBUG (W [i] = EMPTY) ;
    }

    // -------------------------------------------------------------------------
    // count the entries in each row of A and find PLinv
    // -------------------------------------------------------------------------

    k = 0 ;
    for (col = 0 ; col < n ; col++)     // for each column of A(:,Qfill)
    {
        j = Qfill ? Qfill [col] : col ;         // col of S is column j of A
        ASSERT (j >= 0 && j < n) ;
        kstart = k ;
        pend = Ap [j+1] ;
        for (p = Ap [j] ; p < pend ; p++)
        {
            i = Ai [p] ;                // the entry A(i,j)
            row = PLinv [i] ;           // row of S is row i of A
            if (row == EMPTY)
            {
                // column j is the first row index in A(i,Qfill),
                // so this row becomes the next row in S
                row = k++ ;
                PLinv [i] = row ;
                W [row] = 1 ;           // initialize row count
            }
            else
            {
                ASSERT (W [row] >= 1) ;
                W [row]++ ;             // one more entry S(row,col)
            }
        }
        Sleft [col] = k - kstart ;      // number of rows of S with leftmost
                                        // column in col
    }

    // -------------------------------------------------------------------------
    // compute Sleft = cumsum ([0 Sleft])
    // -------------------------------------------------------------------------

    s = 0 ;
    for (col = 0 ; col < n ; col++)     // for each column of A(:,Qfill)
    {
        t = s ;
        s += Sleft [col] ;
        Sleft [col] = t ;
    }
    Sleft [n] = k ;
    Sleft [n+1] = m ;

    // -------------------------------------------------------------------------
    // finish up the row permutation, in case A has empty rows
    // -------------------------------------------------------------------------

    if (k < m)
    {
        for (i = 0 ; i < m ; i++)
        {
            if (PLinv [i] == EMPTY)     // ith row of A is empty
            {
                row = k++ ;             // it becomes a row of S
                PLinv [i] = row ;
                W [row] = 0 ;           // with no entries
            }
        }
    }
    ASSERT (k == m) ;
    DEBUG (for (i = 0 ; i < m ; i++) ASSERT (W [i] >= 0)) ;

    // -------------------------------------------------------------------------
    // compute the row pointers of S (and a copy in W)
    // -------------------------------------------------------------------------

    p = 0 ;
    for (row = 0 ; row < m ; row++)
    {
        t = p ;
        p += W [row] ;
        W [row] = t ;                   // replace W with cumsum ([0 W])
        Sp [row] = t ;                  // and copy into Sp as well
    }
    Sp [m] = p ;

    // -------------------------------------------------------------------------
    // create S = A (p,q)', or S=A(p,q) if S is considered to be in row-form
    // -------------------------------------------------------------------------

    for (col = 0 ; col < n ; col++)     // for each column of A(:,Qfill)
    {
        j = Qfill ? Qfill [col] : col ; // col of S is column j of A
        pend = Ap [j+1] ;
        for (p = Ap [j] ; p < pend ; p++)
        {
            i = Ai [p] ;                // the entry A(i,j)
            row = PLinv [i] ;           // row of S is row i of A
            s = W [row]++ ;             // place S(row,col) in position
            Sj [s] = col ;
        }
    }
}
