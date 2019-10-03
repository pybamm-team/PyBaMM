/* ========================================================================== */
/* === BTF_MAXTRANS ========================================================= */
/* ========================================================================== */

/* Finds a column permutation that maximizes the number of entries on the
 * diagonal of a sparse matrix.  See btf.h for more information.
 *
 * This function is identical to cs_maxtrans in CSparse, with the following
 * exceptions:
 *
 *  (1) cs_maxtrans finds both jmatch and imatch, where jmatch [i] = j and
 *      imatch [j] = i if row i is matched to column j.  This function returns
 *      just jmatch (the Match array).  The MATLAB interface to cs_maxtrans
 *      (the single-output cs_dmperm) returns imatch, not jmatch to the MATLAB
 *      caller.
 *
 *  (2) cs_maxtrans includes a pre-pass that counts the number of non-empty
 *      rows and columns (m2 and n2, respectively), and computes the matching
 *      using the transpose of A if m2 < n2.  cs_maxtrans also returns quickly
 *      if the diagonal of the matrix is already zero-free.  This pre-pass
 *      allows cs_maxtrans to be much faster than maxtrans, if the use of the
 *      transpose is warranted.
 *
 *      However, for square structurally non-singular matrices with one or more
 *      zeros on the diagonal, the pre-pass is a waste of time, and for these
 *      matrices, maxtrans can be twice as fast as cs_maxtrans.  Since the
 *      maxtrans function is intended primarily for square matrices that are
 *      typically structurally nonsingular, the pre-pass is not included here.
 *      If this maxtrans function is used on a matrix with many more columns
 *      than rows, consider passing the transpose to this function, or use
 *      cs_maxtrans instead.
 *
 *  (3) cs_maxtrans can operate as a randomized algorithm, to help avoid
 *      rare cases of excessive run-time.
 *
 *  (4) this maxtrans function includes an option that limits the total work
 *      performed.  If this limit is reached, the maximum transveral might not
 *      be found.
 *
 * Thus, for general usage, cs_maxtrans is preferred.  For square matrices that
 * are typically structurally non-singular, maxtrans is preferred.  A partial
 * maxtrans can still be very useful when solving a sparse linear system.
 *
 * By Tim Davis.  Copyright (c) 2004-2007, University of Florida.
 * with support from Sandia National Laboratories.  All Rights Reserved.
 */ 

#include "btf.h"
#include "btf_internal.h"


/* ========================================================================== */
/* === augment ============================================================== */
/* ========================================================================== */

/* Perform a depth-first-search starting at column k, to find an augmenting
 * path.  An augmenting path is a sequence of row/column pairs (i1,k), (i2,j1),
 * (i3,j2), ..., (i(s+1), js), such that all of the following properties hold:
 *
 *      * column k is not matched to any row
 *      * entries in the path are nonzero
 *      * the pairs (i1,j1), (i2,j2), (i3,j3) ..., (is,js) have been 
 *          previously matched to each other
 *      * (i(s+1), js) is nonzero, and row i(s+1) is not matched to any column
 *
 * Once this path is found, the matching can be changed to the set of pairs
 * path.  An augmenting path is a sequence of row/column pairs
 *
 *      (i1,k), (i2,j1), (i3,j2), ..., (i(s+1), js)
 *
 * Once a row is matched with a column it remains matched with some column, but
 * not necessarily the column it was first matched with.
 *
 * In the worst case, this function can examine every nonzero in A.  Since it
 * is called n times by maxtrans, the total time of maxtrans can be as high as
 * O(n*nnz(A)).  To limit this work, pass a value of maxwork > 0.  Then at
 * most O((maxwork+1)*nnz(A)) work will be performed; the maximum matching might
 * not be found, however.
 *
 * This routine is very similar to the dfs routine in klu_kernel.c, in the
 * KLU sparse LU factorization package.  It is essentially identical to the
 * cs_augment routine in CSparse, and its recursive version (augment function
 * in cs_maxtransr_mex.c), except that this routine allows for the search to be
 * terminated early if too much work is being performed.
 *
 * The algorithm is based on the paper "On Algorithms for obtaining a maximum
 * transversal" by Iain Duff, ACM Trans. Mathematical Software, vol 7, no. 1,
 * pp. 315-330, and "Algorithm 575: Permutations for a zero-free diagonal",
 * same issue, pp. 387-390.  The code here is a new implementation of that
 * algorithm, with different data structures and control flow.  After writing
 * this code, I carefully compared my algorithm with MC21A/B (ACM Algorithm 575)
 * Some of the comparisons are partial because I didn't dig deeply into all of
 * the details of MC21A/B, such as how the stack is maintained.  The following
 * arguments are essentially identical between this code and MC21A:
 *
 * maxtrans     MC21A,B
 * --------     -------
 * n            N           identical
 * k            JORD        identical
 * Ap           IP          column / row pointers
 * Ai           ICN         row / column indices
 * Ap[n]        LICN        length of index array (# of nonzeros in A)
 * Match        IPERM       output column / row permutation
 * nmatch       NUMNZ       # of nonzeros on diagonal of permuted matrix
 * Flag         CV          mark a node as visited by the depth-first-search
 *
 * The following are different, but analogous:
 *
 * Cheap        ARP         indicates what part of the a column / row has
 *                          already been matched.
 *
 * The following arguments are very different:
 *
 * -            LENR        # of entries in each row/column (unused in maxtrans)
 * Pstack       OUT         Pstack keeps track of where we are in the depth-
 *                          first-search scan of column j.  I think that OUT
 *                          plays a similar role in MC21B, but I'm unsure.
 * Istack       PR          keeps track of the rows in the path.  PR is a link
 *                          list, though, whereas Istack is a stack.  Maxtrans
 *                          does not use any link lists.
 * Jstack       OUT? PR?    the stack for nodes in the path (unsure)
 *
 * The following control structures are roughly comparable:
 *
 * maxtrans                     MC21B
 * --------                     -----
 * for (k = 0 ; k < n ; k++)    DO 100 JORD=1,N
 * while (head >= 0)            DO 70 K=1,JORD
 * for (p = Cheap [j] ; ...)    DO 20 II=IN1,IN2
 * for (p = head ; ...)         DO 90 K=1,JORD
 */

static Int augment
(
    Int k,              /* which stage of the main loop we're in */
    Int Ap [ ],         /* column pointers, size n+1 */
    Int Ai [ ],         /* row indices, size nz = Ap [n] */
    Int Match [ ],      /* size n,  Match [i] = j if col j matched to i */
    Int Cheap [ ],      /* rows Ai [Ap [j] .. Cheap [j]-1] alread matched */
    Int Flag [ ],       /* Flag [j] = k if j already visited this stage */
    Int Istack [ ],     /* size n.  Row index stack. */
    Int Jstack [ ],     /* size n.  Column index stack. */
    Int Pstack [ ],     /* size n.  Keeps track of position in adjacency list */
    double *work,       /* work performed by the depth-first-search */
    double maxwork      /* maximum work allowed */
)
{
    /* local variables, but "global" to all DFS levels: */
    Int found ; /* true if match found.  */
    Int head ;  /* top of stack */

    /* variables that are purely local to any one DFS level: */
    Int j2 ;    /* the next DFS goes to node j2 */
    Int pend ;  /* one past the end of the adjacency list for node j */
    Int pstart ;
    Int quick ;

    /* variables that need to be pushed then popped from the stack: */
    Int i ;     /* the row tentatively matched to i if DFS successful */
    Int j ;     /* the DFS is at the current node j */
    Int p ;     /* current index into the adj. list for node j */
    /* the variables i, j, and p are stacked in Istack, Jstack, and Pstack */

    quick = (maxwork > 0) ;

    /* start a DFS to find a match for column k */
    found = FALSE ;
    i = EMPTY ;
    head = 0 ;
    Jstack [0] = k ;
    ASSERT (Flag [k] != k) ;

    while (head >= 0)
    {
        j = Jstack [head] ;
        pend = Ap [j+1] ;

        if (Flag [j] != k)          /* a node is not yet visited */
        {

            /* -------------------------------------------------------------- */
            /* prework for node j */
            /* -------------------------------------------------------------- */

            /* first time that j has been visited */
            Flag [j] = k ;
            /* cheap assignment: find the next unmatched row in col j.  This
             * loop takes at most O(nnz(A)) time for the sum total of all
             * calls to augment. */
            for (p = Cheap [j] ; p < pend && !found ; p++)
            {
                i = Ai [p] ;
                found = (Match [i] == EMPTY) ;
            }
            Cheap [j] = p ;

            /* -------------------------------------------------------------- */

            /* prepare for DFS */
            if (found)
            {
                /* end of augmenting path, column j matched with row i */
                Istack [head] = i ;
                break ;
            }
            /* set Pstack [head] to the first entry in column j to scan */
            Pstack [head] = Ap [j] ;
        }

        /* ------------------------------------------------------------------ */
        /* quick return if too much work done */
        /* ------------------------------------------------------------------ */

        if (quick && *work > maxwork)
        {
            /* too much work has been performed; abort the search */
            return (EMPTY) ;
        }

        /* ------------------------------------------------------------------ */
        /* DFS for nodes adjacent to j */
        /* ------------------------------------------------------------------ */

        /* If cheap assignment not made, continue the depth-first search.  All
         * rows in column j are already matched.  Add the adjacent nodes to the
         * stack by iterating through until finding another non-visited node.
         *
         * It is the following loop that can force maxtrans to take
         * O(n*nnz(A)) time. */

        pstart = Pstack [head] ;
        for (p = pstart ; p < pend ; p++)
        {
            i = Ai [p] ;
            j2 = Match [i] ;
            ASSERT (j2 != EMPTY) ;
            if (Flag [j2] != k)
            {
                /* Node j2 is not yet visited, start a depth-first search on
                 * node j2.  Keep track of where we left off in the scan of adj
                 * list of node j so we can restart j where we left off. */
                Pstack [head] = p + 1 ;
                /* Push j2 onto the stack and immediately break so we can
                 * recurse on node j2.  Also keep track of row i which (if this
                 * search for an augmenting path works) will be matched with the
                 * current node j. */
                Istack [head] = i ;
                Jstack [++head] = j2 ;
                break ;
            }
        }

        /* ------------------------------------------------------------------ */
        /* determine how much work was just performed */
        /* ------------------------------------------------------------------ */

        *work += (p - pstart + 1) ;

        /* ------------------------------------------------------------------ */
        /* node j is done, but the postwork is postponed - see below */
        /* ------------------------------------------------------------------ */

        if (p == pend)
        {
            /* If all adjacent nodes of j are already visited, pop j from
             * stack and continue.  We failed to find a match. */
            head-- ;
        }
    }

    /* postwork for all nodes j in the stack */
    /* unwind the path and make the corresponding matches */
    if (found)
    {
        for (p = head ; p >= 0 ; p--)
        {
            j = Jstack [p] ;
            i = Istack [p] ;

            /* -------------------------------------------------------------- */
            /* postwork for node j */
            /* -------------------------------------------------------------- */
            /* if found, match row i with column j */
            Match [i] = j ;
        }
    }
    return (found) ;
}


/* ========================================================================== */
/* === maxtrans ============================================================= */
/* ========================================================================== */

Int BTF(maxtrans)   /* returns # of columns in the matching */
(
    /* --- input --- */
    Int nrow,       /* A is nrow-by-ncol in compressed column form */
    Int ncol,
    Int Ap [ ],     /* size ncol+1 */
    Int Ai [ ],     /* size nz = Ap [ncol] */
    double maxwork, /* do at most maxwork*nnz(A) work; no limit if <= 0.  This
                     * work limit excludes the O(nnz(A)) cheap-match phase. */

    /* --- output --- */
    double *work,   /* work = -1 if maxwork > 0 and the total work performed
                     * reached the maximum of maxwork*nnz(A)).
                     * Otherwise, work = the total work performed. */

    Int Match [ ],  /* size nrow.  Match [i] = j if column j matched to row i */

    /* --- workspace --- */
    Int Work [ ]    /* size 5*ncol */
)
{
    Int *Cheap, *Flag, *Istack, *Jstack, *Pstack ;
    Int i, j, k, nmatch, work_limit_reached, result ;

    /* ---------------------------------------------------------------------- */
    /* get workspace and initialize */
    /* ---------------------------------------------------------------------- */

    Cheap  = Work ; Work += ncol ;
    Flag   = Work ; Work += ncol ;

    /* stack for non-recursive depth-first search in augment function */
    Istack = Work ; Work += ncol ;
    Jstack = Work ; Work += ncol ;
    Pstack = Work ;

    /* in column j, rows Ai [Ap [j] .. Cheap [j]-1] are known to be matched */
    for (j = 0 ; j < ncol ; j++)
    {
        Cheap [j] = Ap [j] ;
        Flag [j] = EMPTY ; 
    }

    /* all rows and columns are currently unmatched */
    for (i = 0 ; i < nrow ; i++)
    {
        Match [i] = EMPTY ;
    }

    if (maxwork > 0)
    {
        maxwork *= Ap [ncol] ;
    }
    *work = 0 ;

    /* ---------------------------------------------------------------------- */
    /* find a matching row for each column k */
    /* ---------------------------------------------------------------------- */

    nmatch = 0 ;
    work_limit_reached = FALSE ;
    for (k = 0 ; k < ncol ; k++)
    {
        /* find an augmenting path to match some row i to column k */
        result = augment (k, Ap, Ai, Match, Cheap, Flag, Istack, Jstack, Pstack,
            work, maxwork) ;
        if (result == TRUE)
        {
            /* we found it.  Match [i] = k for some row i has been done. */
            nmatch++ ;
        }
        else if (result == EMPTY)
        {
            /* augment gave up because of too much work, and no match found */
            work_limit_reached = TRUE ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* return the Match, and the # of matches made */
    /* ---------------------------------------------------------------------- */

    /* At this point, row i is matched to j = Match [i] if j >= 0.  i is an
     * unmatched row if Match [i] == EMPTY. */

    if (work_limit_reached)
    {
        /* return -1 if the work limit of maxwork*nnz(A) was reached */
        *work = EMPTY ;
    }

    return (nmatch) ;
}
