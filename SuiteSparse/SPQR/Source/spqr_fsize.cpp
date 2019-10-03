// =============================================================================
// === spqr_fsize ==============================================================
// =============================================================================

// Compute the number of rows in front F, initialize its staircase, and 
// create its Fmap

#include "spqr.hpp"

Long spqr_fsize     // returns # of rows of F
(
    // inputs, not modified
    Long f,
    Long *Super,            // size nf, from QRsym
    Long *Rp,               // size nf, from QRsym
    Long *Rj,               // size rjsize, from QRsym
    Long *Sleft,            // size n+2, from QRsym
    Long *Child,            // size nf, from QRsym
    Long *Childp,           // size nf+1, from QRsym
    Long *Cm,               // size nf

    // outputs, not defined on input
    Long *Fmap,             // size n
    Long *Stair             // size fn
)
{
    Long col1, col2, p1, p2, fp, fn, fm, col, p, j, c, pc, cm, ci, t, fpc ;

    // -------------------------------------------------------------------------
    // get the front F
    // -------------------------------------------------------------------------

    // pivotal columns Super [f] ... Super [f+1]-1
    col1 = Super [f] ;      // front F has columns col1:col2-1
    col2 = Super [f+1] ;
    p1 = Rp [f] ;           // Rj [p1:p2-1] = columns in F
    p2 = Rp [f+1] ;
    fp = col2 - col1 ;      // first fp columns are pivotal
    fn = p2 - p1 ;          // exact number of columns of F
    ASSERT (fp > 0) ;       // all fronts have at least one pivot column
    ASSERT (fp <= fn) ;

    PR (("\n---- Front: %ld\n", f)) ;
    PR (("Get Fsize: col1 %ld col2 %ld fp %ld fn %ld\n", col1, col2, fp, fn)) ;

    // -------------------------------------------------------------------------
    // create the Fmap for front F
    // -------------------------------------------------------------------------

    for (p = p1, j = 0 ; p < p2 ; p++, j++)
    {
        col = Rj [p] ;              // global column col is jth col of F
        Fmap [col] = j ;
        PR (("Fmap (%ld): %ld \n", col, j)) ;
    }

    // -------------------------------------------------------------------------
    // initialize the staircase for front F
    // -------------------------------------------------------------------------

    // initialize the staircase with original rows of S
    for (j = 0 ; j < fp ; j++)
    {
        // global column j+col1 is the jth pivot column of front F
        col = j + col1 ;
        Stair [j] = Sleft [col+1] - Sleft [col] ;
        PR (("init rows, j: %ld count %ld\n", j, Stair [j])) ;
    }

    // contribution blocks from children will be added here
    for ( ; j < fn ; j++)
    {
        Stair [j] = 0 ;
    }

    // -------------------------------------------------------------------------
    // construct the staircase for each child
    // -------------------------------------------------------------------------

    for (p = Childp [f] ; p < Childp [f+1] ; p++)
    {
        c = Child [p] ;                 // get the child c of front F
        PR (("child %ld\n", c)) ;
        pc = Rp [c] ;                   // get the pattern of child R
        cm = Cm [c] ;                   // # of rows in child C
        // fnc = (Rp [c+1] - pc) ;      // total # cols in child F
        fpc = Super [c+1] - Super [c] ; // # of pivot cols in child
        // cn = (fnc - fpc) ;           // # of cols in child C
        // ASSERT (cm >= 0 && cm <= cn) ;
        pc += fpc ;                     // pointer to column indices in C
        // ASSERT (pc + cn == Rp [c+1]) ;
        // PR (("  cm %ld cn %ld\n", cm, cn)) ;

        // add the child rows to the staircase
        for (ci = 0 ; ci < cm ; ci++)
        {
            col = Rj [pc + ci] ;        // leftmost col of this row of C
            j = Fmap [col] ;            // global col is jth col of F
            PR (("  child col %ld j %ld\n", col, j)) ;
            ASSERT (j >= 0 && j < fn) ;
            Stair [j]++ ;               // add this row to jth staircase
        }
    }

    // -------------------------------------------------------------------------
    // replace Stair with cumsum ([0 Stair]), and find # rows of F
    // -------------------------------------------------------------------------

    fm = 0 ;
    for (j = 0 ; j < fn ; j++)
    {
        t = fm ;
        fm += Stair [j] ;
        Stair [j] = t ;
    }
    PR (("fm %ld %ld\n", fm, Stair [fn-1])) ;

    // Note that the above code differs from the symbolic analysis.  The
    // nonzero values in column j of F will reside in row 0 of F to row
    // Stair [j+1]-1 of F.  Once the rows of S and the children of F are
    // assembled, this will change to Stair [j]-1.
    //
    // When a row is assembled into F whose leftmost column is j, it is placed
    // as row Stair [j], which is then incremented to accommodate the next row
    // with leftmost column j.  At that point, Stair [0:fn-1] will then "equal"
    // Stair [0:fn-1] in the symbolic analysis (except that the latter is an
    // upper bound if rank detection has found failed pivot columns).

    return (fm) ;
}
