// =============================================================================
// === spqr_assemble ===========================================================
// =============================================================================

/* Assemble the children into the current front.

    Example:  Suppose there are 3 pivot columns in F (5, 6, 7), and 4
    non-pivotal columns of F (8, 9, 11, 12).  We have fn = 7.

    col1 = 5.  fp = 3, so pivot columns are [5 6 7].
    Rj [Rp [f]...Rp[f+1]-1] = [5 6 7 8 9 11 12]. 

    Seven rows of S are to be assembled (rows 16 to 22).  The rows of S are
    stored in compressed-row form, and the rows themselves appear in S in
    increasing order of leftmost column index.  Column indices in the rows
    of S are in ascending order, but this is not required for assembly.

            5 6 7 8 9 11 12
        16: * . x x . x x       The pattern of these rows is a subset of
        17: # x . . . . x       [pivot columns] [non pivot columns].
        18: # x . x x x .       The leftmost columns are marked with #.
        19:   * x . x . x       Suppose Sleft [5] = 16.  Then
        20:   # . x . . x       Sleft [6] = 16 + 3 = 19.
        21:     * . . x x       Sleft [7] = 19 + 2 = 21. 
        22:     # x x x x       Sleft [8] = 21 + 2 = 23.
        23:       *

    The values of Sleft [5 6 7 8] are marked with "*", above.

    Three children are to be assembled, one of which happens to be trapezoidal.
    They are stored in packed column-major upper trapezoidal form.

            6 8 11          6 7 8 12            5 7 9 11
        b1: # x x       c1: # x x x         d1: # x x x 
        b2:   # x       c2:   # x x         d2:   # x x
        b3:     #       c3:     # x         d3:     # x
                        c4:       #

    The number of rows of these 4 blocks (S, b, c, and d) with leftmost
    column in j = 0:6 is [4 4 4 2 1 1 1], which is Stair [0:fn-1]
    before the cumulative sum.  After the cumulative sum,
    Stair [0:fn-1] is [0 4 8 12 14 15 16].

    Lining up the rows of each block (not done except conceptually) gives:

            5 6 7 8 9 11 12
        16: # . x x . x x
        17: # x . . . . x
        18: # x . x x x .
        19:   # x . x . x
        20:   # . x . . x
        21:     # . . x x
        22:     # x x x x

        b1:   # . x . x .
        b2:       # . x .
        b3:           # .

        c1:   # x x . . x
        c2:     # x . . x
        c3:       # . . x
        c4:             #

        d1: # . x . x x .
        d2:     # . x x .
        d3:         # x .

    These rows are assembled into F in order of increasing leftmost column (the
    # symbols in each row).  The resulting F is given below.  Note the local
    row indices which correspond to the initial and final values of Stair.

            5 6 7 8 9 11 12
        16: # . x x . x x   <- row 0 of F
        17: # x . . . . x
        18: # x . x x x .
        d1: # . x . x x .
        19:   # x . x . x   <- 4
        20:   # . x . . x
        b1:   # . x . x .
        c1:   # x x . . x
        21:     # . . x x   <- 8
        22:     # x x x x
        d2:     # . x x .
        c2:     # x . . x
        b2:       # . x .   <- 12
        c3:       # . . x
        d3:         # x .   <- 14
        b3:           # .   <- 15
        c4:             #   <- 16

    After the front is assembled, Stair is [4 8 12 14 15 16 17], because Stair
    [j] is incremented once for each row whose leftmost column is column j in
    F.  That is, ignoring explicit zeros stored within the staircase:

        0    # x x x x x x
        1    # x x x x x x
        2    # x x x x x x
        3    # x x x x x x
        4    - # x x x x x   <- Stair [0] = 4
        5      # x x x x x
        6      # x x x x x
        7      # x x x x x
        8      - # x x x x   <- Stair [1] = 8
        9        # x x x x
       10        # x x x x
       11        # x x x x
       12        - # x x x   <- Stair [2] = 12
       13          # x x x
       14          - # x x   <- Stair [3] = 14
       15            - # x   <- Stair [4] = 15
       16              - #   <- Stair [5] = 16
                         -   <- Stair [6] = 17

    The rows of S are assembled first.  For this example, after assembling the
    rows of S, the front contains:

                5 6 7 8 9 11 12
        0   16: # . x x . x x
        1   17: # x . . . . x
        2   18: # x . x x x .
        3     :                 <- Stair [0] = 3 (was 0)
        4   19:   # x . x . x
        5   20:   # . x . . x
        6     :                 <- Stair [1] = 6 (was 4)
        7     :
        8   21:     # . . x x
        9   22:     # x x x x
       10     :                 <- Stair [2] = 10 (was 8)
       11     :
       12     :                 <- Stair [3] = 12
       13     :
       14     :                 <- Stair [4] = 14
       15     :                 <- Stair [5] = 15
       16     :                 <- Stair [6] = 16

   Next, the C block of each child is assembled.  Suppose the children are
   assembled in order b, c, and d.  After child b:

            b1:   # . x . x .
            b2:       # . x .
            b3:           # .

                5 6 7 8 9 11 12
        0   16: # . x x . x x
        1   17: # x . . . . x
        2   18: # x . x x x .
        3     :                 <- Stair [0] = 3
        4   19:   # x . x . x
        5   20:   # . x . . x
        6   b1:   # . x . x .
        7     :                 <- Stair [1] = 7 (was 6)
        8   21:     # . . x x
        9   22:     # x x x x
       10     :                 <- Stair [2] = 10
       11     :
       12   b2:       # . x .
       13     :                 <- Stair [3] = 13 (was 12)
       14     :                 <- Stair [4] = 14
       15   b3:           # .
       16     :                 <- Stair [5] = 16 (was 15)
                                   Stair [5] = 16

    Assembling child c:

            c1:   # x x . . x
            c2:     # x . . x
            c3:       # . . x
            c4:             #

                5 6 7 8 9 11 12
        0   16: # . x x . x x
        1   17: # x . . . . x
        2   18: # x . x x x .
        3     :                 <- Stair [0] = 3
        4   19:   # x . x . x
        5   20:   # . x . . x
        6   b1:   # . x . x .
        7   c1:   # x x . . x
        8   21:     # . . x x   <- Stair [1] = 8 (was 7)
        9   22:     # x x x x
       10   c2:     # x . . x
       11     :                 <- Stair [2] = 11 (was 10)
       12   b2:       # . x .
       13   c3:       # . . x
       14     :                 <- Stair [3] = 14 (was 13)
                                   Stair [4] = 14
       15   b3:           # .
       16   c4:             #   <- Stair [5] = 16
                                   Stair [6] = 17 (was 16)

    Assembling child d gives the final front.

            d1: # . x . x x .
            d2:     # . x x .
            d3:         # x .

    The final front is shown below, where (-) denotes the zero entry
    pointed to by Stair.

                5 6 7 8 9 11 12
        0   16: # . x x . x x
        1   17: # x . . . . x
        2   18: # x . x x x .
        3   d1: # . x . x x .
        4   19: - # x . x . x   <- Stair [0] = 4 (was 3)
        5   20:   # . x . . x
        6   b1:   # . x . x .
        7   c1:   # x x . . x
        8   21:   - # . . x x   <- Stair [1] = 8
        9   22:     # x x x x
       10   c2:     # x . . x
       11   d2:     # . x x .
       12   b2:     - # . x .   <- Stair [2] = 12 (was 11)
       13   c3:       # . . x
       14   d3:       - # x .   <- Stair [3] = 14
       15   b3:         - # .   <- Stair [4] = 15 (was 14)
       16   c4:           - #   <- Stair [5] = 16
                            -   <- Stair [6] = 17

    The final Stair [0:fn-1] = [4 8 12 14 15 16 17].  Note that column j in F
    has entries in rows 0 through Stair [j]-1, inclusive.  The staircase of the
    last column will always be fm.
 */

#include "spqr.hpp"

template <typename Entry> void spqr_assemble
(
    /* inputs, not modified */
    Long f,                 /* front to assemble F */
    Long fm,                /* number of rows of F */
    int keepH,              /* if TRUE, then construct row pattern of H */
    Long *Super,
    Long *Rp,
    Long *Rj,
    Long *Sp,
    Long *Sj,
    Long *Sleft,
    Long *Child,
    Long *Childp,
    Entry *Sx,
    Long *Fmap,
    Long *Cm,
    Entry **Cblock,
#ifndef NDEBUG
    char *Rdead,
#endif
    Long *Hr,

    /* input/output */
    Long *Stair,
    Long *Hii,              /* if keepH, construct list of row indices for F */
    // input only
    Long *Hip,

    /* outputs, not defined on input */
    Entry *F,

    /* workspace, not defined on input or output */
    Long *Cmap
)
{
    Entry *Fi, *Fj, *C ;
    Long k, fsize, fn, col1, col2, p, p1, p2, fp, j, leftcol, row, col, i,
        cm, cn, ci, cj, c, pc, fnc, fpc, rmc ;
    Long *Hi = NULL, *Hichild = NULL ;

    /* ---------------------------------------------------------------------- */
    /* get the front F */
    /* ---------------------------------------------------------------------- */

    /* pivotal columns Super [f] ... Super [f+1]-1 */
    col1 = Super [f] ;      /* front F has pivot columns col1:col2-1 */
    col2 = Super [f+1] ;
    p1 = Rp [f] ;           /* Rj [p1:p2-1] = columns in F */
    p2 = Rp [f+1] ;
    fp = col2 - col1 ;      /* first fp columns are pivotal */
    fn = p2 - p1 ;          /* exact number of columns of F */

    ASSERT (fp > 0) ;       /* all fronts have at least one pivot column */
    ASSERT (fp <= fn) ;

    /* set F to zero */
    fsize = fm * fn ;
    for (k = 0 ; k < fsize ; k++)
    {
        F [k] = 0 ;
    }

    if (keepH)
    {
        Hi = &Hii [Hip [f]] ;   // list of rows for front F will be constructed
    }

#ifndef NDEBUG
    for (k = 0 ; k < fn ; k++) PR (("stair [%ld] = %ld\n", k, Stair [k])) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* assemble the rows of S whose leftmost column is a pivot column in F */
    /* ---------------------------------------------------------------------- */

    for (k = 0 ; k < fp ; k++)
    {
        /* assemble all rows whose leftmost global column index is k+col1 */
        leftcol = k + col1 ;

        /* for each row of S whose leftmost column is leftcol */
        for (row = Sleft [leftcol] ; row < Sleft [leftcol+1] ; row++)
        {
            /* get the location of this row in F, and advance the staircase */
            i = Stair [k]++ ;
            Fi = F + i ;                /* pointer to F (i,:) */
            PR (("assemble row: %ld i: %ld j %ld\n", row, i, k)) ;

            /* scatter the row into F */
            ASSERT (Sj [Sp [row]] == leftcol) ;
            for (p = Sp [row] ; p < Sp [row+1] ; p++)
            {
                col = Sj [p] ;          /* global column index of S */
                ASSERT (col >= leftcol /* && col < n */) ;
                j = Fmap [col] ;        /* column index in F */
                ASSERT (j >= 0 && j < fn) ;
                Fi [j*fm] = Sx [p] ;    /* F (i,j) = S (row,col) ; */
            }

            /* construct the pattern of the Householder vectors */
            if (keepH)
            {
                Hi [i] = row ;         /* this row of S is row i of F */
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* assemble each child */
    /* ---------------------------------------------------------------------- */

    for (p = Childp [f] ; p < Childp [f+1] ; p++)
    {

        /* ------------------------------------------------------------------ */
        /* get the child */
        /* ------------------------------------------------------------------ */

        c = Child [p] ;                 /* get the child c of front F */
        pc = Rp [c] ;                   /* get the pattern of child R */
        cm = Cm [c] ;                   /* # of rows in child C */
        fnc = Rp [c+1] - pc ;           /* total # cols in child F */
        fpc = Super [c+1] - Super [c] ; /* # of pivot cols in child */
        cn = fnc - fpc ;                /* # of cols in child C */
        ASSERT (cm >= 0 && cm <= cn) ;
        pc += fpc ;                     /* pointer to column indices in C */
        ASSERT (pc + cn == Rp [c+1]) ;
        C = Cblock [c] ;                /* pointer to numerical C block */

        if (keepH)
        {
            /* get the list of row indices of the C block of the child */
            rmc = Hr [c] ;
#ifndef NDEBUG
            {
                Long rmc2 = 0 ;
                for (j = Super [c] ; j < Super [c+1] ; j++)
                {
                    if (!Rdead [j])
                    {
                        rmc2++ ;        /* count # of rows in child R */
                    }
                }
                ASSERT (rmc2 == rmc) ;
            }
#endif
            // Hchild = HHi [c] + rmc ;    /* list of rows in child C block */
            Hichild = &Hii [Hip [c] + rmc] ;
        }

        /* ------------------------------------------------------------------ */
        /* construct the Cmap */
        /* ------------------------------------------------------------------ */

        /* The contents of Cmap are currently undefined.  Cmap [0..cm-1] gives
           the mapping of the rows of C to the rows of F.  If row ci of C is
           row i of F, then Cmap [ci] = i */

        for (ci = 0 ; ci < cm ; ci++)
        {
            col = Rj [pc + ci] ;        /* leftmost col of row i */
            j = Fmap [col] ;            /* global col is jth col of F */
            ASSERT (j >= 0 && j < fn) ;
            i = Stair [j]++ ;           /* add row F(i,:) to jth staircase */
            ASSERT (i >= 0 && i < fm) ;
            Cmap [ci] = i ;             /* keep track of the mapping */
            if (keepH)
            {
                /* get global row index for C(ci,:) from child Householder,
                   which becomes the local row index F(i,:) */
                Hi [i] = Hichild [ci] ;
                ASSERT (Hi [i] >= 0 /* && Hi [i] < m */) ;
            }
        }

        /* ------------------------------------------------------------------ */
        /* copy the triangular part of C into F (cm-by-cm, upper triangular) */
        /* ------------------------------------------------------------------ */

        for (cj = 0 ; cj < cm ; cj++)
        {
            /* copy C (:,cj) into F */
            col = Rj [pc + cj] ;            /* global col of column cj of C */
            j = Fmap [col] ;                /* and also column j of F */
            Fj = F + fm * j ;               /* pointer to F(:,j) */
            for (ci = 0 ; ci <= cj ; ci++)
            {
                i = Cmap [ci] ;             /* row ci of C is row fi of F */
                Fj [i] = *(C++) ;           /* F (Cmap (ci), j) = C(ci,cj) */
            }
        }

        /* ------------------------------------------------------------------ */
        /* copy the rectangular part of C into F (cm-by-(cn-cm) in size) */
        /* ------------------------------------------------------------------ */

        for ( ; cj < cn ; cj++)
        {
            /* copy C (:,cj) into F */
            col = Rj [pc + cj] ;            /* global col of column cj of C */
            j = Fmap [col] ;                /* and also column j of F */
            Fj = F + fm * j ;               /* pointer to F(:,j) */
            for (ci = 0 ; ci < cm ; ci++)
            {
                i = Cmap [ci] ;             /* row ci of C is row fi of F */
                Fj [i] = *(C++) ;           /* F (Cmap (ci), j) = C(ci,cj) */
            }
        }

        /* the contents of Cmap are no longer needed */
    }

    /* the Stair is now complete */
    ASSERT (Stair [fn-1] == fm) ;
}


/* ========================================================================== */

template void spqr_assemble <double>
(
    /* inputs, not modified */
    Long f,                 /* front to assemble F */
    Long fm,                /* number of rows of F */
    int keepH,              /* if TRUE, then construct row pattern of H */
    Long *Super,
    Long *Rp,
    Long *Rj,
    Long *Sp,
    Long *Sj,
    Long *Sleft,
    Long *Child,
    Long *Childp,
    double *Sx,
    Long *Fmap,
    Long *Cm,
    double **Cblock,
#ifndef NDEBUG
    char *Rdead,
#endif
    Long *Hr,

    /* input/output */
    Long *Stair,
    Long *Hii,              /* if keepH, construct list of row indices for F */
    // input only
    Long *Hip,

    /* outputs, not defined on input */
    double *F,

    /* workspace, not defined on input or output */
    Long *Cmap
) ;

/* ========================================================================== */

template void spqr_assemble <Complex>
(
    /* inputs, not modified */
    Long f,                 /* front to assemble F */
    Long fm,                /* number of rows of F */
    int keepH,              /* if TRUE, then construct row pattern of H */
    Long *Super,
    Long *Rp,
    Long *Rj,
    Long *Sp,
    Long *Sj,
    Long *Sleft,
    Long *Child,
    Long *Childp,
    Complex *Sx,
    Long *Fmap,
    Long *Cm,
    Complex **Cblock,
#ifndef NDEBUG
    char *Rdead,
#endif
    Long *Hr,

    /* input/output */
    Long *Stair,
    Long *Hii,              /* if keepH, construct list of row indices for F */
    // input only
    Long *Hip,

    /* outputs, not defined on input */
    Complex *F,

    /* workspace, not defined on input or output */
    Long *Cmap
) ;
