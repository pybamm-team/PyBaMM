/* ========================================================================== */
/* === BTF_STRONGCOMP ======================================================= */
/* ========================================================================== */

/* Finds the strongly connected components of a graph, or equivalently, permutes
 * the matrix into upper block triangular form.  See btf.h for more details.
 * Input matrix and Q are not checked on input.
 *
 * By Tim Davis.  Copyright (c) 2004-2007, University of Florida.
 * with support from Sandia National Laboratories.  All Rights Reserved.
 */

#include "btf.h"
#include "btf_internal.h"

#define UNVISITED (-2)      /* Flag [j] = UNVISITED if node j not visited yet */
#define UNASSIGNED (-1)     /* Flag [j] = UNASSIGNED if node j has been visited,
                             * but not yet assigned to a strongly-connected
                             * component (aka block).  Flag [j] = k (k in the
                             * range 0 to nblocks-1) if node j has been visited
                             * (and completed, with its postwork done) and
                             * assigned to component k. */

/* This file contains two versions of the depth-first-search, a recursive one
 * and a non-recursive one.  By default, the non-recursive one is used. */

#ifndef RECURSIVE

/* ========================================================================== */
/* === dfs: non-recursive version (default) ================================= */
/* ========================================================================== */

/* Perform a depth-first-search of a graph, stored in an adjacency-list form.
 * The row indices of column j (equivalently, the out-adjacency list of node j)
 * are stored in Ai [Ap[j] ... Ap[j+1]-1].  Self-edge (diagonal entries) are
 * ignored.  Ap[0] must be zero, and thus nz = Ap[n] is the number of entries
 * in the matrix (or edges in the graph).  The row indices in each column need
 * not be in any particular order.  If an input column permutation is given,
 * node j (in the permuted matrix A*Q) is located in
 * Ai [Ap[Q[j]] ... Ap[Q[j]+1]-1].  This Q can be the same as the Match array
 * output from the maxtrans routine, for a square matrix that is structurally
 * full rank.
 *
 * The algorithm is from the paper by Robert E. Tarjan, "Depth-first search and
 * linear graph algorithms," SIAM Journal on Computing, vol. 1, no. 2,
 * pp. 146-160, 1972.  The time taken by strongcomp is O(nnz(A)).
 *
 * See also MC13A/B in the Harwell subroutine library (Iain S. Duff and John
 * K. Reid, "Algorithm 529: permutations to block triangular form," ACM Trans.
 * on Mathematical Software, vol. 4, no. 2, pp. 189-192, 1978, and "An
 * implementation of Tarjan's algorithm for the block triangular form of a
 * matrix," same journal, pp. 137-147.  This code is implements the same
 * algorithm as MC13A/B, except that the data structures are very different.
 * Also, unlike MC13A/B, the output permutation preserves the natural ordering
 * within each block.
 */

static void dfs
(
    /* inputs, not modified on output: */
    Int j,              /* start the DFS at node j */
    Int Ap [ ],         /* size n+1, column pointers for the matrix A */
    Int Ai [ ],         /* row indices, size nz = Ap [n] */
    Int Q [ ],          /* input column permutation */

    /* inputs, modified on output (each array is of size n): */
    Int Time [ ],       /* Time [j] = "time" that node j was first visited */
    Int Flag [ ],       /* Flag [j]: see above */
    Int Low [ ],        /* Low [j]: see definition below */
    Int *p_nblocks,     /* number of blocks (aka strongly-connected-comp.)*/
    Int *p_timestamp,   /* current "time" */

    /* workspace, not defined on input or output: */
    Int Cstack [ ],     /* size n, output stack to hold nodes of components */
    Int Jstack [ ],     /* size n, stack for the variable j */
    Int Pstack [ ]      /* size n, stack for the variable p */
)
{
    /* ---------------------------------------------------------------------- */
    /* local variables, and initializations */
    /* ---------------------------------------------------------------------- */

    /* local variables, but "global" to all DFS levels: */
    Int chead ;     /* top of Cstack */
    Int jhead ;     /* top of Jstack and Pstack */

    /* variables that are purely local to any one DFS level: */
    Int i ;         /* edge (j,i) considered; i can be next node to traverse */
    Int parent ;    /* parent of node j in the DFS tree */
    Int pend ;      /* one past the end of the adjacency list for node j */
    Int jj ;        /* column j of A*Q is column jj of the input matrix A */

    /* variables that need to be pushed then popped from the stack: */
    Int p ;         /* current index into the adj. list for node j */
    /* the variables j and p are stacked in Jstack and Pstack */

    /* local copies of variables in the calling routine */
    Int nblocks   = *p_nblocks ;
    Int timestamp = *p_timestamp ;

    /* ---------------------------------------------------------------------- */
    /* start a DFS at node j (same as the recursive call dfs (EMPTY, j)) */
    /* ---------------------------------------------------------------------- */

    chead = 0 ;             /* component stack is empty */
    jhead = 0 ;             /* Jstack and Pstack are empty */
    Jstack [0] = j ;        /* put the first node j on the Jstack */
    ASSERT (Flag [j] == UNVISITED) ;

    while (jhead >= 0)
    {
        j = Jstack [jhead] ;        /* grab the node j from the top of Jstack */

        /* determine which column jj of the A is column j of A*Q */
        jj = (Q == (Int *) NULL) ? (j) : (BTF_UNFLIP (Q [j])) ;
        pend = Ap [jj+1] ;          /* j's row index list ends at Ai [pend-1] */

        if (Flag [j] == UNVISITED)
        {

            /* -------------------------------------------------------------- */
            /* prework at node j */
            /* -------------------------------------------------------------- */

            /* node j is being visited for the first time */
            Cstack [++chead] = j ;          /* push j onto the stack */
            timestamp++ ;                   /* get a timestamp */
            Time [j] = timestamp ;          /* give the timestamp to node j */
            Low [j] = timestamp ;
            Flag [j] = UNASSIGNED ;         /* flag node j as visited */

            /* -------------------------------------------------------------- */
            /* set Pstack [jhead] to the first entry in column j to scan */
            /* -------------------------------------------------------------- */

            Pstack [jhead] = Ap [jj] ;
        }

        /* ------------------------------------------------------------------ */
        /* DFS rooted at node j (start it, or continue where left off) */
        /* ------------------------------------------------------------------ */

        for (p = Pstack [jhead] ; p < pend ; p++)
        {
            i = Ai [p] ;    /* examine the edge from node j to node i */
            if (Flag [i] == UNVISITED)
            {
                /* Node i has not been visited - start a DFS at node i.
                 * Keep track of where we left off in the scan of adjacency list
                 * of node j so we can restart j where we left off. */
                Pstack [jhead] = p + 1 ;
                /* Push i onto the stack and immediately break
                 * so we can recurse on node i. */
                Jstack [++jhead] = i ;
                ASSERT (Time [i] == EMPTY) ;
                ASSERT (Low [i] == EMPTY) ;
                /* break here to do what the recursive call dfs (j,i) does */
                break ;
            }
            else if (Flag [i] == UNASSIGNED)
            {
                /* Node i has been visited, but still unassigned to a block
                 * this is a back or cross edge if Time [i] < Time [j].
                 * Note that i might equal j, in which case this code does
                 * nothing. */
                ASSERT (Time [i] > 0) ;
                ASSERT (Low [i] > 0) ;
                Low [j] = MIN (Low [j], Time [i]) ;
            }
        }

        if (p == pend)
        {
            /* If all adjacent nodes of j are already visited, pop j from
             * Jstack and do the post work for node j.  This also pops p
             * from the Pstack. */
            jhead-- ;

            /* -------------------------------------------------------------- */
            /* postwork at node j */
            /* -------------------------------------------------------------- */

            /* determine if node j is the head of a component */
            if (Low [j] == Time [j])
            {
                /* pop all nodes in this SCC from Cstack */
                while (TRUE)
                {
                    ASSERT (chead >= 0) ;       /* stack not empty (j in it) */
                    i = Cstack [chead--] ;      /* pop a node from the Cstack */
                    ASSERT (i >= 0) ;
                    ASSERT (Flag [i] == UNASSIGNED) ;
                    Flag [i] = nblocks ;        /* assign i to current block */
                    if (i == j) break ;         /* current block ends at j */
                }
                nblocks++ ;     /* one more block has been found */
            }
            /* update Low [parent], if the parent exists */
            if (jhead >= 0)
            {
                parent = Jstack [jhead] ;
                Low [parent] = MIN (Low [parent], Low [j]) ;
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* cleanup: update timestamp and nblocks */
    /* ---------------------------------------------------------------------- */

    *p_timestamp = timestamp ;
    *p_nblocks   = nblocks ;
}

#else

/* ========================================================================== */
/* === dfs: recursive version (only for illustration) ======================= */
/* ========================================================================== */

/* The following is a recursive version of dfs, which computes identical results
 * as the non-recursive dfs.  It is included here because it is easier to read.
 * Compare the comments in the code below with the identical comments in the
 * non-recursive code above, and that will help you see the correlation between
 * the two routines.
 *
 * This routine can cause stack overflow, and is thus not recommended for heavy
 * usage, particularly for large matrices.  To help in delaying stack overflow,
 * global variables are used, reducing the amount of information each call to
 * dfs places on the call/return stack (the integers i, j, p, parent, and the
 * return address).  Note that this means the recursive code is not thread-safe.
 * To try this version, compile the code with -DRECURSIVE or include the
 * following line at the top of this file:

#define RECURSIVE

 */

static Int  /* for recursive illustration only, not for production use */
    chead, timestamp, nblocks, n, *Ap, *Ai, *Flag, *Cstack, *Time, *Low,
    *P, *R, *Q ;

static void dfs
(
    Int parent,         /* came from parent node */
    Int j               /* at node j in the DFS */
)
{
    Int p ;         /* current index into the adj. list for node j */
    Int i ;         /* edge (j,i) considered; i can be next node to traverse */
    Int jj ;        /* column j of A*Q is column jj of the input matrix A */

    /* ---------------------------------------------------------------------- */
    /* prework at node j */
    /* ---------------------------------------------------------------------- */

    /* node j is being visited for the first time */
    Cstack [++chead] = j ;          /* push j onto the stack */
    timestamp++ ;                   /* get a timestamp */
    Time [j] = timestamp ;          /* give the timestamp to node j */
    Low [j] = timestamp ;
    Flag [j] = UNASSIGNED ;         /* flag node j as visited */

    /* ---------------------------------------------------------------------- */
    /* DFS rooted at node j */
    /* ---------------------------------------------------------------------- */

    /* determine which column jj of the A is column j of A*Q */
    jj = (Q == (Int *) NULL) ? (j) : (BTF_UNFLIP (Q [j])) ;
    for (p = Ap [jj] ; p < Ap [jj+1] ; p++)
    {
        i = Ai [p] ;    /* examine the edge from node j to node i */
        if (Flag [i] == UNVISITED)
        {
            /* Node i has not been visited - start a DFS at node i. */
            dfs (j, i) ;
        }
        else if (Flag [i] == UNASSIGNED)
        {
            /* Node i has been visited, but still unassigned to a block
             * this is a back or cross edge if Time [i] < Time [j].
             * Note that i might equal j, in which case this code does
             * nothing. */
            Low [j] = MIN (Low [j], Time [i]) ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* postwork at node j */
    /* ---------------------------------------------------------------------- */

    /* determine if node j is the head of a component */
    if (Low [j] == Time [j])
    {
        /* pop all nodes in this strongly connected component from Cstack */
        while (TRUE)
        {
            i = Cstack [chead--] ;      /* pop a node from the Cstack */
            Flag [i] = nblocks ;        /* assign node i to current block */
            if (i == j) break ;         /* current block ends at node j */
        }
        nblocks++ ;     /* one more block has been found */
    }
    /* update Low [parent] */
    if (parent != EMPTY)
    {
        /* Note that this could be done with Low[j] = MIN(Low[j],Low[i]) just
         * after the dfs (j,i) statement above, and then parent would not have
         * to be an input argument.  Putting it here places all the postwork
         * for node j in one place, thus making the non-recursive DFS easier. */
        Low [parent] = MIN (Low [parent], Low [j]) ;
    }
}

#endif

/* ========================================================================== */
/* === btf_strongcomp ======================================================= */
/* ========================================================================== */

#ifndef RECURSIVE

Int BTF(strongcomp) /* return # of strongly connected components */
(
    /* input, not modified: */
    Int n,          /* A is n-by-n in compressed column form */
    Int Ap [ ],     /* size n+1 */
    Int Ai [ ],     /* size nz = Ap [n] */

    /* optional input, modified (if present) on output: */
    Int Q [ ],      /* size n, input column permutation.  The permutation Q can
                     * include a flag which indicates an unmatched row.
                     * jold = BTF_UNFLIP (Q [jnew]) is the permutation;
                     * this function ingnores these flags.  On output, it is
                     * modified according to the permutation P. */

    /* output, not defined on input: */
    Int P [ ],      /* size n.  P [k] = j if row and column j are kth row/col
                     * in permuted matrix. */
    Int R [ ],      /* size n+1.  kth block is in rows/cols R[k] ... R[k+1]-1
                     * of the permuted matrix. */

    /* workspace, not defined on input or output: */
    Int Work [ ]    /* size 4n */
)

#else

Int BTF(strongcomp) /* recursive version - same as above except for Work size */
(
    Int n_in,
    Int Ap_in [ ],
    Int Ai_in [ ],
    Int Q_in [ ],
    Int P_in [ ],
    Int R_in [ ],
    Int Work [ ]    /* size 2n */
)

#endif

{
    Int j, k, b ;

#ifndef RECURSIVE
    Int timestamp, nblocks, *Flag, *Cstack, *Time, *Low, *Jstack, *Pstack ;
#else
    n = n_in ;
    Ap = Ap_in ;
    Ai = Ai_in ;
    Q = Q_in ;
    P = P_in ;
    R = R_in ;
    chead = EMPTY ;
#endif

    /* ---------------------------------------------------------------------- */
    /* get and initialize workspace */
    /* ---------------------------------------------------------------------- */

    /* timestamp is incremented each time a new node is visited.
     *
     * Time [j] is the timestamp given to node j.
     *
     * Low [j] is the lowest timestamp of any node reachable from j via either
     * a path to any descendent of j in the DFS tree, or via a single edge to
     * an either an ancestor (a back edge) or another node that's neither an
     * ancestor nor a descendant (a cross edge).  If Low [j] is equal to
     * the timestamp of node j (Time [j]), then node j is the "head" of a
     * strongly connected component (SCC).  That is, it is the first node
     * visited in its strongly connected component, and the DFS subtree rooted
     * at node j spans all the nodes of the strongly connected component.
     *
     * The term "block" and "component" are used interchangebly in this code;
     * "block" being a matrix term and "component" being a graph term for the
     * same thing.
     *
     * When a node is visited, it is placed on the Cstack (for "component"
     * stack).  When node j is found to be an SCC head, all the nodes from the
     * top of the stack to node j itself form the nodes in the SCC.  This Cstack
     * is used for both the recursive and non-recursive versions.
     */

    Time   = Work ; Work += n ;
    Flag   = Work ; Work += n ;
    Low    = P ;                /* use output array P as workspace for Low */
    Cstack = R ;                /* use output array R as workspace for Cstack */

#ifndef RECURSIVE
    /* stack for non-recursive dfs */
    Jstack = Work ; Work += n ;     /* stack for j */
    Pstack = Work ;                 /* stack for p */
#endif

    for (j = 0 ; j < n ; j++)
    {
        Flag [j] = UNVISITED ;
        Low [j] = EMPTY ;
        Time [j] = EMPTY ;
#ifndef NDEBUG
        Cstack [j] = EMPTY ;
#ifndef RECURSIVE
        Jstack [j] = EMPTY ;
        Pstack [j] = EMPTY ;
#endif
#endif
    }

    timestamp = 0 ;     /* each node given a timestamp when it is visited */
    nblocks = 0 ;       /* number of blocks found so far */

    /* ---------------------------------------------------------------------- */
    /* find the connected components via a depth-first-search */
    /* ---------------------------------------------------------------------- */

    for (j = 0 ; j < n ; j++)
    {
        /* node j is unvisited or assigned to a block. Cstack is empty. */
        ASSERT (Flag [j] == UNVISITED || (Flag [j] >= 0 && Flag [j] < nblocks));
        if (Flag [j] == UNVISITED)
        {
#ifndef RECURSIVE
            /* non-recursive dfs (default) */
            dfs (j, Ap, Ai, Q, Time, Flag, Low, &nblocks, &timestamp,
                    Cstack, Jstack, Pstack) ;
#else
            /* recursive dfs (for illustration only) */
            ASSERT (chead == EMPTY) ;
            dfs (EMPTY, j) ;
            ASSERT (chead == EMPTY) ;
#endif
        }
    }
    ASSERT (timestamp == n) ;

    /* ---------------------------------------------------------------------- */
    /* construct the block boundary array, R */
    /* ---------------------------------------------------------------------- */

    for (b = 0 ; b < nblocks ; b++)
    {
        R [b] = 0 ;
    }
    for (j = 0 ; j < n ; j++)
    {
        /* node j has been assigned to block b = Flag [j] */
        ASSERT (Time [j] > 0 && Time [j] <= n) ;
        ASSERT (Low [j] > 0 && Low [j] <= n) ;
        ASSERT (Flag [j] >= 0 && Flag [j] < nblocks) ;
        R [Flag [j]]++ ;
    }
    /* R [b] is now the number of nodes in block b.  Compute cumulative sum
     * of R, using Time [0 ... nblocks-1] as workspace. */
    Time [0] = 0 ;
    for (b = 1 ; b < nblocks ; b++)
    {
        Time [b] = Time [b-1] + R [b-1] ;
    }
    for (b = 0 ; b < nblocks ; b++)
    {
        R [b] = Time [b] ;
    }
    R [nblocks] = n ;

    /* ---------------------------------------------------------------------- */
    /* construct the permutation, preserving the natural order */
    /* ---------------------------------------------------------------------- */

#ifndef NDEBUG
    for (k = 0 ; k < n ; k++)
    {
        P [k] = EMPTY ;
    }
#endif

    for (j = 0 ; j < n ; j++)
    {
        /* place column j in the permutation */
        P [Time [Flag [j]]++] = j ;
    }

#ifndef NDEBUG
    for (k = 0 ; k < n ; k++)
    {
        ASSERT (P [k] != EMPTY) ;
    }
#endif

    /* Now block b consists of the nodes k1 to k2-1 in the permuted matrix,
     * where k1 = R [b] and k2 = R [b+1].  Row and column j of the original
     * matrix becomes row and column P [k] of the permuted matrix.  The set of
     * of rows/columns (nodes) in block b is given by P [k1 ... k2-1], and this
     * set is sorted in ascending order.  Thus, if the matrix consists of just
     * one block, P is the identity permutation. */

    /* ---------------------------------------------------------------------- */
    /* if Q is present on input, set Q = Q*P' */
    /* ---------------------------------------------------------------------- */

    if (Q != (Int *) NULL)
    {
        /* We found a symmetric permutation P for the matrix A*Q.  The overall
         * permutation is thus P*(A*Q)*P'.  Set Q=Q*P' so that the final
         * permutation is P*A*Q.  Use Time as workspace.  Note that this
         * preserves the negative values of Q if the matrix is structurally
         * singular. */
        for (k = 0 ; k < n ; k++)
        {
            Time [k] = Q [P [k]] ;
        }
        for (k = 0 ; k < n ; k++)
        {
            Q [k] = Time [k] ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* how to traverse the permuted matrix */
    /* ---------------------------------------------------------------------- */

    /* If Q is not present, the following code can be used to traverse the
     * permuted matrix P*A*P'
     *
     *      // compute the inverse of P
     *      for (knew = 0 ; knew < n ; knew++)
     *      {
     *          // row and column kold in the old matrix is row/column knew
     *          // in the permuted matrix P*A*P'
     *          kold = P [knew] ;
     *          Pinv [kold] = knew ;
     *      }
     *      for (b = 0 ; b < nblocks ; b++)
     *      {
     *          // traverse block b of the permuted matrix P*A*P'
     *          k1 = R [b] ;
     *          k2 = R [b+1] ;
     *          nk = k2 - k1 ;
     *          for (jnew = k1 ; jnew < k2 ; jnew++)
     *          {
     *              jold = P [jnew] ;
     *              for (p = Ap [jold] ; p < Ap [jold+1] ; p++)
     *              {
     *                  iold = Ai [p] ;
     *                  inew = Pinv [iold] ;
     *                  // Entry in the old matrix is A (iold, jold), and its
     *                  // position in the new matrix P*A*P' is (inew, jnew).
     *                  // Let B be the bth diagonal block of the permuted
     *                  // matrix.  If inew >= k1, then this entry is in row/
     *                  // column (inew-k1, jnew-k1) of the nk-by-nk matrix B.
     *                  // Otherwise, the entry is in the upper block triangular
     *                  // part, not in any diagonal block.
     *              }
     *          }
     *      }
     *
     * If Q is present replace the above statement
     *          jold = P [jnew] ;
     * with
     *          jold = Q [jnew] ;
     * or
     *          jold = BTF_UNFLIP (Q [jnew]) ;
     *
     * then entry A (iold,jold) in the old (unpermuted) matrix is at (inew,jnew)
     * in the permuted matrix P*A*Q.  Everything else remains the same as the
     * above (simply replace P*A*P' with P*A*Q in the above comments).
     */

    /* ---------------------------------------------------------------------- */
    /* return # of blocks / # of strongly connected components */
    /* ---------------------------------------------------------------------- */

    return (nblocks) ;
}
