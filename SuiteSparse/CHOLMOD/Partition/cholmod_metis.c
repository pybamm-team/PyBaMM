/* ========================================================================== */
/* === Partition/cholmod_metis ============================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Partition Module.
 * Copyright (C) 2005-2006, Univ. of Florida.  Author: Timothy A. Davis
 * -------------------------------------------------------------------------- */

/* CHOLMOD interface to the METIS package (Version 5.1.0):
 *
 * cholmod_metis_bisector:
 *
 *	Wrapper for the METIS node separator function,
 *      METIS_ComputeVertexSeparator (METIS 5.1).
 *
 *      Finds a set of nodes that partitions the graph into two parts.  METIS
 *      4.0 (the function METIS_ComputeVertexSeparator) allowed for edge
 *      weights to be passed to the bisector.  This feature is removed in METIS
 *      5.1.  CHOLMOD itself does not rely on this feature (it calls the METIS
 *      bisector with edge weights of all 1s).  However, user code can call
 *      cholmod_metis_bisector directly, and pass in edge weights.  If you use
 *      METIS 5.1, these edge weights are now ignored; if you pass a non-NULL
 *      entry for edge weights, an error will be returned.
 *
 * cholmod_metis:
 *
 *	Wrapper for METIS_NodeND, METIS's own nested dissection algorithm.
 *	Typically faster than cholmod_nested_dissection, mostly because it
 *	uses minimum degree on just the leaves of the separator tree, rather
 *	than the whole matrix.
 *
 * Note that METIS does not return an error if it runs out of memory.  Instead,
 * it terminates the program.  This interface attempts to avoid that problem
 * by preallocating space that should be large enough for any memory allocations
 * within METIS, and then freeing that space, just before the call to METIS.
 * While this is not guaranteed to work, it is very unlikely to fail.  If you
 * encounter this problem, increase Common->metis_memory.  If you don't mind
 * having your program terminated, set Common->metis_memory to zero (a value of
 * 2.0 is usually safe).  Several other METIS workarounds are made in the
 * routines in this file.  See the description of metis_memory_ok, just below,
 * for more details.
 *
 * FUTURE WORK: interfaces to other partitioners (CHACO, SCOTCH, JOSTLE, ... )
 *
 * workspace: several size-nz and size-n temporary arrays.  Uses no workspace
 * in Common.
 *
 * Supports any xtype (pattern, real, complex, or zomplex).
 */

#ifndef NPARTITION

#include "cholmod_internal.h"
#include "metis.h"
#include "cholmod_partition.h"
#include "cholmod_cholesky.h"

/* ========================================================================== */
/* === dumpgraph ============================================================ */
/* ========================================================================== */

/* For dumping the input graph to METIS_NodeND, to check with METIS's onmetis
 * and graphchk programs.  For debugging only.  To use, uncomment this #define:
#define DUMP_GRAPH
 */

#ifdef DUMP_GRAPH
#include <stdio.h>
/* After dumping the graph with this routine, run "onmetis metisgraph" */
static void dumpgraph (idx_t *Mp, idx_t *Mi, SuiteSparse_long n,
    cholmod_common *Common)
{
    SuiteSparse_long i, j, p, nz ;
    FILE *f ;
    nz = Mp [n] ;
    printf ("Dumping METIS graph n %ld nz %ld\n", n, nz) ;    /* DUMP_GRAPH */
    f = fopen ("metisgraph", "w") ;
    if (f == NULL)
    {
	ERROR (-99, "cannot open metisgraph") ;
	return ;
    }
    fprintf (f, "%ld %ld\n", n, nz/2) ;			    /* DUMP_GRAPH */
    for (j = 0 ; j < n ; j++)
    {
	for (p = Mp [j] ; p < Mp [j+1] ; p++)
	{
	    i = Mi [p] ;
	    fprintf (f, " %ld", i+1) ;			    /* DUMP_GRAPH */
	}
	fprintf (f, "\n") ;				    /* DUMP_GRAPH */
    }
    fclose (f) ;
}
#endif


/* ========================================================================== */
/* === metis_memory_ok ====================================================== */
/* ========================================================================== */

/* METIS will terminate your program if
 * they run out of memory.  In an attempt to workaround METIS' behavior, this
 * routine allocates a single block of memory of size equal to an observed
 * upper bound on METIS' memory usage.  It then immediately deallocates the
 * block.  If the allocation fails, METIS is not called.
 *
 * Median memory usage for a graph with n nodes and nz edges (counting each
 * edge twice, or both upper and lower triangular parts of a matrix) is
 * 4*nz + 40*n + 4096 integers.  A "typical" upper bound is 10*nz + 50*n + 4096
 * integers.  Nearly all matrices tested fit within that upper bound, with the
 * exception two in the UF sparse matrix collection: Schenk_IBMNA/c-64 and
 * Gupta/gupta2.  The latter exceeds the "upper bound" by a factor of just less
 * than 2.
 *
 * If you do not mind having your program terminated if it runs out of memory,
 * set Common->metis_memory to zero.  Its default value is 2, which allows for
 * some memory fragmentation, and also accounts for the Gupta/gupta2 matrix.
 */

#define GUESS(nz,n) (10 * (nz) + 50 * (n) + 4096)

static int metis_memory_ok
(
    Int n,
    Int nz,
    cholmod_common *Common
)
{
    double s ;
    void *p ;
    size_t metis_guard ;

    if (Common->metis_memory <= 0)
    {
	/* do not prevent METIS from running out of memory */
	return (TRUE) ;
    }

    n  = MAX (1, n) ;
    nz = MAX (0, nz) ;

    /* compute in double, to avoid integer overflow */
    s = GUESS ((double) nz, (double) n) ;
    s *= Common->metis_memory ;

    if (s * sizeof (idx_t) >= ((double) Size_max))
    {
	/* don't even attempt to malloc such a large block */
	return (FALSE) ;
    }

    /* recompute in size_t */
    metis_guard = GUESS ((size_t) nz, (size_t) n) ;
    metis_guard *= Common->metis_memory ;

    /* attempt to malloc the block */
    p = CHOLMOD(malloc) (metis_guard, sizeof (idx_t), Common) ;
    if (p == NULL)
    {
	/* failure - return out-of-memory condition */
	return (FALSE) ;
    }

    /* success - free the block */
    CHOLMOD(free) (metis_guard, sizeof (idx_t), p, Common) ;
    return (TRUE) ;
}


/* ========================================================================== */
/* === cholmod_metis_bisector =============================================== */
/* ========================================================================== */

/* Finds a set of nodes that bisects the graph of A or AA' (direct interface
 * to METIS_ComputeVertexSeparator.
 *
 * The input matrix A must be square, symmetric (with both upper and lower
 * parts present) and with no diagonal entries.  These conditions are NOT
 * checked.
 */

SuiteSparse_long CHOLMOD(metis_bisector)	/* returns separator size */
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to bisect */
    Int *Anw,		/* size A->nrow, node weights, can be NULL, */
                        /* which means the graph is unweighted. */ 
    Int *Aew,		/* size nz, edge weights (silently ignored). */
                        /* This option was available with METIS 4, but not */
                        /* in METIS 5.  This argument is now unused, but */
                        /* it remains for backward compatibilty, so as not */
                        /* to change the API for cholmod_metis_bisector. */
    /* ---- output --- */
    Int *Partition,	/* size A->nrow */
    /* --------------- */
    cholmod_common *Common
)
{
    Int *Ap, *Ai ;
    idx_t *Mp, *Mi, *Mnw, *Mpart ;
    Int n, nleft, nright, j, p, csep, total_weight, lightest, nz ;
    idx_t nn, csp ;
    size_t n1 ;
    int ok ;
    DEBUG (Int nsep) ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    RETURN_IF_NULL_COMMON (EMPTY) ;
    RETURN_IF_NULL (A, EMPTY) ;
    /* RETURN_IF_NULL (Anw, EMPTY) ; */
    /* RETURN_IF_NULL (Aew, EMPTY) ; */
    RETURN_IF_NULL (Partition, EMPTY) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, EMPTY) ;
    if (A->stype || A->nrow != A->ncol)
    {
	/* A must be square, with both upper and lower parts present */
	ERROR (CHOLMOD_INVALID, "matrix must be square, symmetric,"
		" and with both upper/lower parts present") ;
	return (EMPTY) ;
    }
    Common->status = CHOLMOD_OK ;
    ASSERT (CHOLMOD(dump_sparse) (A, "A for bisector", Common) >= 0) ;

    /* ---------------------------------------------------------------------- */
    /* quick return */
    /* ---------------------------------------------------------------------- */

    n = A->nrow ;
    if (n == 0)
    {
	return (0) ;
    }
    n1 = ((size_t) n) + 1 ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    Ap = A->p ;
    Ai = A->i ;
    nz = Ap [n] ;

    if (Anw != NULL) DEBUG (for (j = 0 ; j < n ; j++) ASSERT (Anw [j] > 0)) ;

    /* ---------------------------------------------------------------------- */
    /* copy Int to METIS idx_t, if necessary */
    /* ---------------------------------------------------------------------- */

    if (sizeof (Int) == sizeof (idx_t))
    {
	/* this is the typical case */
	Mi    = (idx_t *) Ai ;
	Mp    = (idx_t *) Ap ;
	Mnw   = (idx_t *) Anw ;
	Mpart = (idx_t *) Partition ;
    }
    else
    {
	/* idx_t and Int differ; copy the graph into the METIS idx_t */
	Mi    = CHOLMOD(malloc) (nz, sizeof (idx_t), Common) ;
	Mp    = CHOLMOD(malloc) (n1, sizeof (idx_t), Common) ;
	Mnw   = Anw ? (CHOLMOD(malloc) (n,  sizeof (idx_t), Common)) : NULL ;
	Mpart = CHOLMOD(malloc) (n,  sizeof (idx_t), Common) ;
	if (Common->status < CHOLMOD_OK)
	{
	    CHOLMOD(free) (nz, sizeof (idx_t), Mi,    Common) ;
	    CHOLMOD(free) (n1, sizeof (idx_t), Mp,    Common) ;
	    CHOLMOD(free) (n,  sizeof (idx_t), Mnw,   Common) ;
	    CHOLMOD(free) (n,  sizeof (idx_t), Mpart, Common) ;
	    return (EMPTY) ;
	}
	for (p = 0 ; p < nz ; p++)
	{
	    Mi [p] = Ai [p] ;
	}
	for (j = 0 ; j <= n ; j++)
	{
	    Mp [j] = Ap [j] ;
	}
        if (Anw != NULL)
        {
            for (j = 0 ; j <  n ; j++)
            {
                Mnw [j] = Anw [j] ;
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* METIS workaround: try to ensure METIS doesn't run out of memory */
    /* ---------------------------------------------------------------------- */

    if (!metis_memory_ok (n, nz, Common))
    {
	/* METIS might ask for too much memory and thus terminate the program */
	if (sizeof (Int) != sizeof (idx_t))
	{
	    CHOLMOD(free) (nz, sizeof (idx_t), Mi,    Common) ;
	    CHOLMOD(free) (n1, sizeof (idx_t), Mp,    Common) ;
	    CHOLMOD(free) (n,  sizeof (idx_t), Mnw,   Common) ;
	    CHOLMOD(free) (n,  sizeof (idx_t), Mpart, Common) ;
	}
	return (EMPTY) ;
    }

    /* ---------------------------------------------------------------------- */
    /* partition the graph */
    /* ---------------------------------------------------------------------- */

#ifndef NDEBUG
    PRINT1 (("Metis graph, n = "ID"\n", n)) ;
    for (j = 0 ; j < n ; j++)
    {
	Int ppp, nodeweight = (Mnw ? Mnw [j] : 1) ;
	PRINT2 (("M(:,"ID") node weight "ID"\n", j, nodeweight)) ;
	ASSERT (nodeweight > 0) ;
	for (ppp = Mp [j] ; ppp < Mp [j+1] ; ppp++)
	{
	    PRINT3 ((" "ID "\n", (Int) Mi [ppp])) ;
	    ASSERT (Mi [ppp] != j) ;
	}
    }
#endif

    /* 
    METIS_ComputeVertexSeparator(
            idx_t *nvtxs,       number of nodes
            idx_t *xadj,        column pointers
            idx_t *adjncy,      row indices
            idx_t *vwgt,        vertex weights (NULL means unweighted)
            idx_t *options,     options (NULL means defaults)
            idx_t *sepsize,     separator size
            idx_t *part);       partition.  part [i] = 0,1,2, where:
                                0:left, 1:right, 2:separator
    */

    nn = n ;
    ok = METIS_ComputeVertexSeparator (&nn, Mp, Mi, Mnw, NULL, &csp, Mpart) ;
    csep = csp ;

    PRINT1 (("METIS csep "ID"\n", csep)) ;

    /* ---------------------------------------------------------------------- */
    /* copy the results back from idx_t, if required */
    /* ---------------------------------------------------------------------- */

    if (ok == METIS_OK && (sizeof (Int) != sizeof (idx_t)))
    {
	for (j = 0 ; j < n ; j++)
	{
	    Partition [j] = Mpart [j] ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* free the workspace for METIS, if allocated */
    /* ---------------------------------------------------------------------- */

    if (sizeof (Int) != sizeof (idx_t))
    {
	CHOLMOD(free) (nz, sizeof (idx_t), Mi,    Common) ;
	CHOLMOD(free) (n1, sizeof (idx_t), Mp,    Common) ;
	CHOLMOD(free) (n,  sizeof (idx_t), Mnw,   Common) ;
	CHOLMOD(free) (n,  sizeof (idx_t), Mpart, Common) ;
    }

    if (ok == METIS_ERROR_MEMORY)
    {
        ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory in METIS") ;
	return (EMPTY) ;
    }
    else if (ok == METIS_ERROR_INPUT)
    {
        ERROR (CHOLMOD_INVALID, "invalid input to METIS") ;
	return (EMPTY) ;
    }
    else if (ok == METIS_ERROR)
    {
        ERROR (CHOLMOD_INVALID, "unspecified METIS error") ;
	return (EMPTY) ;
    }

    /* ---------------------------------------------------------------------- */
    /* ensure a reasonable separator */
    /* ---------------------------------------------------------------------- */

    /* METIS can return a valid separator with no nodes in (for example) the
     * left part.  In this case, there really is no separator.  CHOLMOD
     * prefers, in this case, for all nodes to be in the separator (and both
     * left and right parts to be empty).  Also, if the graph is unconnected,
     * METIS can return a valid empty separator.  CHOLMOD prefers at least one
     * node in the separator.  Note that cholmod_nested_dissection only calls
     * this routine on connected components, but cholmod_bisect can call this
     * routine for any graph. */

    if (csep == 0)
    {
	/* The separator is empty, select lightest node as separator.  If
	 * ties, select the highest numbered node. */
        if (Anw == NULL)
        {
	    lightest = n-1 ;
        }
        else
        {
	    lightest = 0 ;
            for (j = 0 ; j < n ; j++)
            {
                if (Anw [j] <= Anw [lightest])
                {
                    lightest = j ;
                }
            }
        }
	PRINT1 (("Force "ID" as sep\n", lightest)) ;
	Partition [lightest] = 2 ;
	csep = (Anw ? (Anw [lightest]) : 1) ;
    }

    /* determine the node weights in the left and right part of the graph */
    nleft = 0 ;
    nright = 0 ;
    DEBUG (nsep = 0) ;
    for (j = 0 ; j < n ; j++)
    {
	PRINT1 (("Partition ["ID"] = "ID"\n", j, Partition [j])) ;
	if (Partition [j] == 0)
	{
	    nleft += (Anw ? (Anw [j]) : 1) ;
	}
	else if (Partition [j] == 1)
	{
	    nright += (Anw ? (Anw [j]) : 1) ;
	}
#ifndef NDEBUG
	else
	{
	    ASSERT (Partition [j] == 2) ;
	    nsep += (Anw ? (Anw [j]) : 1) ;
	}
#endif
    }
    ASSERT (csep == nsep) ;

    total_weight = nleft + nright + csep ;

    if (csep < total_weight)
    {
	/* The separator is less than the whole graph.  Make sure the left and
	 * right parts are either both empty or both non-empty. */
	PRINT1 (("nleft "ID" nright "ID" csep "ID" tot "ID"\n",
		nleft, nright, csep, total_weight)) ;
	ASSERT (nleft + nright + csep == total_weight) ;
	ASSERT (nleft > 0 || nright > 0) ;
	if ((nleft == 0 && nright > 0) || (nleft > 0 && nright == 0))
	{
	    /* left or right is empty; put all nodes in the separator */
	    PRINT1 (("Force all in sep\n")) ;
	    csep = total_weight ;
	    for (j = 0 ; j < n ; j++)
	    {
		Partition [j] = 2 ;
	    }
	}
    }

    ASSERT (CHOLMOD(dump_partition) (n, Ap, Ai, Anw, Partition, csep, Common)) ;

    /* ---------------------------------------------------------------------- */
    /* return the sum of the weights of nodes in the separator */
    /* ---------------------------------------------------------------------- */

    return (csep) ;
}


/* ========================================================================== */
/* === cholmod_metis ======================================================== */
/* ========================================================================== */

/* CHOLMOD wrapper for the METIS_NodeND ordering routine.  Creates A+A',
 * A*A' or A(:,f)*A(:,f)' and then calls METIS_NodeND on the resulting graph.
 * This routine is comparable to cholmod_nested_dissection, except that it
 * calls METIS_NodeND directly, and it does not return the separator tree.
 *
 * workspace:  Flag (nrow), Iwork (4*n+uncol)
 *	Allocates a temporary matrix B=A*A' or B=A.
 */

int CHOLMOD(metis)
(
    /* ---- input ---- */
    cholmod_sparse *A,	/* matrix to order */
    Int *fset,		/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
    int postorder,	/* if TRUE, follow with etree or coletree postorder */
    /* ---- output --- */
    Int *Perm,		/* size A->nrow, output permutation */
    /* --------------- */
    cholmod_common *Common
)
{
    double d ;
    Int *Iperm, *Iwork, *Bp, *Bi ;
    idx_t *Mp, *Mi, *Mperm, *Miperm ;
    cholmod_sparse *B ;
    Int i, j, n, nz, p, identity, uncol ;
    idx_t nn, zero = 0 ;
    size_t n1, s ;
    int ok = TRUE ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    RETURN_IF_NULL_COMMON (FALSE) ;
    RETURN_IF_NULL (A, FALSE) ;
    RETURN_IF_NULL (Perm, FALSE) ;
    RETURN_IF_XTYPE_INVALID (A, CHOLMOD_PATTERN, CHOLMOD_ZOMPLEX, FALSE) ;
    Common->status = CHOLMOD_OK ;

    /* ---------------------------------------------------------------------- */
    /* quick return */
    /* ---------------------------------------------------------------------- */

    n = A->nrow ;
    if (n == 0)
    {
	return (TRUE) ;
    }
    n1 = ((size_t) n) + 1 ;

    /* ---------------------------------------------------------------------- */
    /* allocate workspace */
    /* ---------------------------------------------------------------------- */

    /* s = 4*n + uncol */
    uncol = (A->stype == 0) ? A->ncol : 0 ;
    s = CHOLMOD(mult_size_t) (n, 4, &ok) ;
    s = CHOLMOD(add_size_t) (s, uncol, &ok) ;
    if (!ok)
    {
	ERROR (CHOLMOD_TOO_LARGE, "problem too large") ;
	return (FALSE) ;
    }

    CHOLMOD(allocate_work) (n, s, 0, Common) ;
    if (Common->status < CHOLMOD_OK)
    {
	return (FALSE) ;
    }
    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, Common)) ;

    /* ---------------------------------------------------------------------- */
    /* convert the matrix to adjacency list form */
    /* ---------------------------------------------------------------------- */

    /* The input graph for METIS must be symmetric, with both upper and lower
     * parts present, and with no diagonal entries.  The columns need not be
     * sorted.
     * B = A+A', A*A', or A(:,f)*A(:,f)', upper and lower parts present */
    if (A->stype)
    {
	/* Add the upper/lower part to a symmetric lower/upper matrix by
	 * converting to unsymmetric mode */
	/* workspace: Iwork (nrow) */
	B = CHOLMOD(copy) (A, 0, -1, Common) ;
    }
    else
    {
	/* B = A*A' or A(:,f)*A(:,f)', no diagonal */
	/* workspace: Flag (nrow), Iwork (max (nrow,ncol)) */
	B = CHOLMOD(aat) (A, fset, fsize, -1, Common) ;
    }
    ASSERT (CHOLMOD(dump_sparse) (B, "B for NodeND", Common) >= 0) ;
    if (Common->status < CHOLMOD_OK)
    {
	return (FALSE) ;
    }
    ASSERT (B->nrow == A->nrow) ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    Iwork = Common->Iwork ;
    Iperm = Iwork ;		/* size n (i/i/l) */

    Bp = B->p ;
    Bi = B->i ;
    nz = Bp [n] ;

    /* B does not include the diagonal, and both upper and lower parts.
     * Common->anz includes the diagonal, and just the lower part of B */
    Common->anz = nz / 2 + n ;

    /* ---------------------------------------------------------------------- */
    /* allocate the METIS input arrays, if needed */
    /* ---------------------------------------------------------------------- */

    if (sizeof (Int) == sizeof (idx_t))
    {
	/* This is the typical case. */
	Miperm = (idx_t *) Iperm ;
	Mperm  = (idx_t *) Perm ;
	Mp     = (idx_t *) Bp ;
	Mi     = (idx_t *) Bi ;
    }
    else
    {
	/* allocate graph for METIS only if Int and idx_t differ */
	Miperm = CHOLMOD(malloc) (n,  sizeof (idx_t), Common) ;
	Mperm  = CHOLMOD(malloc) (n,  sizeof (idx_t), Common) ;
	Mp     = CHOLMOD(malloc) (n1, sizeof (idx_t), Common) ;
	Mi     = CHOLMOD(malloc) (nz, sizeof (idx_t), Common) ;
	if (Common->status < CHOLMOD_OK)
	{
	    /* out of memory */
	    CHOLMOD(free_sparse) (&B, Common) ;
	    CHOLMOD(free) (n,  sizeof (idx_t), Miperm, Common) ;
	    CHOLMOD(free) (n,  sizeof (idx_t), Mperm, Common) ;
	    CHOLMOD(free) (n1, sizeof (idx_t), Mp, Common) ;
	    CHOLMOD(free) (nz, sizeof (idx_t), Mi, Common) ;
	    return (FALSE) ;
	}
	for (j = 0 ; j <= n ; j++)
	{
	    Mp [j] = Bp [j] ;
	}
	for (p = 0 ; p < nz ; p++)
	{
	    Mi [p] = Bi [p] ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* METIS workarounds */
    /* ---------------------------------------------------------------------- */

    identity = FALSE ;
    if (nz == 0)
    {
	/* The matrix has no off-diagonal entries.  METIS_NodeND fails in this
	 * case, so avoid using it.  The best permutation is identity anyway,
	 * so this is an easy fix. */
	identity = TRUE ;
	PRINT1 (("METIS:: no nz\n")) ;
    }
    else if (Common->metis_nswitch > 0)
    {
	/* METIS_NodeND in METIS 4.0.1 gives a seg fault with one matrix of
	 * order n = 3005 and nz = 6,036,025, including the diagonal entries.
	 * The workaround is to return the identity permutation instead of using
	 * METIS for matrices of dimension 3000 or more and with density of 66%
	 * or more - admittedly an uncertain fix, but such matrices are so dense
	 * that any reasonable ordering will do, even identity (n^2 is only 50%
	 * higher than nz in this case).  CHOLMOD's nested dissection method
	 * (cholmod_nested_dissection) has no problems with the same matrix,
	 * even though it too uses METIS_ComputeVertexSeparator.  The matrix is
	 * derived from LPnetlib/lpi_cplex1 in the UF sparse matrix collection.
	 * If C is the lpi_cplex matrix (of order 3005-by-5224), A = (C*C')^2
	 * results in the seg fault.  The seg fault also occurs in the stand-
	 * alone onmetis program that comes with METIS.  If a future version of
	 * METIS fixes this problem, then set Common->metis_nswitch to zero.
	 */
	d = ((double) nz) / (((double) n) * ((double) n)) ;
	if (n > (Int) (Common->metis_nswitch) && d > Common->metis_dswitch)
	{
	    identity = TRUE ;
	    PRINT1 (("METIS:: nswitch/dswitch activated\n")) ;
	}
    }

    if (!identity && !metis_memory_ok (n, nz, Common))
    {
	/* METIS might ask for too much memory and thus terminate the program */
	identity = TRUE ;
    }

    /* ---------------------------------------------------------------------- */
    /* find the permutation */
    /* ---------------------------------------------------------------------- */

    if (identity)
    {
	/* no need to do the postorder */
	postorder = FALSE ;
	for (i = 0 ; i < n ; i++)
	{
	    Mperm [i] = i ;
	}
    }
    else
    {
#ifdef DUMP_GRAPH
	/* DUMP_GRAPH */ printf ("Calling METIS_NodeND n "ID" nz "ID""
	"density %g\n", n, nz, ((double) nz) / (((double) n) * ((double) n)));
	dumpgraph (Mp, Mi, n, Common) ;
#endif

        /*
        int METIS_NodeND(
            idx_t *nvtxs,       number of nodes
            idx_t *xadj,        column pointers
            idx_t *adjncy,      row indices
            idx_t *vwgt,        vertex weights (NULL means unweighted)
            idx_t *options,     options (NULL means defaults)
            idx_t *perm,        fill-reducing ordering
            idx_t *iperm);      inverse of perm
        */

	nn = n ;
	METIS_NodeND (&nn, Mp, Mi, NULL, NULL, Mperm, Miperm) ;

	PRINT0 (("METIS_NodeND done\n")) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free the METIS input arrays */
    /* ---------------------------------------------------------------------- */

    if (sizeof (Int) != sizeof (idx_t))
    {
	for (i = 0 ; i < n ; i++)
	{
	    Perm [i] = (Int) (Mperm [i]) ;
	}
	CHOLMOD(free) (n,   sizeof (idx_t), Miperm, Common) ;
	CHOLMOD(free) (n,   sizeof (idx_t), Mperm, Common) ;
	CHOLMOD(free) (n+1, sizeof (idx_t), Mp, Common) ;
	CHOLMOD(free) (nz,  sizeof (idx_t), Mi, Common) ;
    }

    CHOLMOD(free_sparse) (&B, Common) ;

    /* ---------------------------------------------------------------------- */
    /* etree or column-etree postordering, using the Cholesky Module */
    /* ---------------------------------------------------------------------- */

    if (postorder)
    {
	Int *Parent, *Post, *NewPerm ;
	Int k ;

	Parent = Iwork + 2*((size_t) n) + uncol ;   /* size n = nrow */
	Post   = Parent + n ;			    /* size n */

	/* workspace: Iwork (2*nrow+uncol), Flag (nrow), Head (nrow+1) */
	CHOLMOD(analyze_ordering) (A, CHOLMOD_METIS, Perm, fset, fsize,
		Parent, Post, NULL, NULL, NULL, Common) ;
	if (Common->status == CHOLMOD_OK)
	{
	    /* combine the METIS permutation with its postordering */
	    NewPerm = Parent ;	    /* use Parent as workspace */
	    for (k = 0 ; k < n ; k++)
	    {
		NewPerm [k] = Perm [Post [k]] ;
	    }
	    for (k = 0 ; k < n ; k++)
	    {
		Perm [k] = NewPerm [k] ;
	    }
	}
    }

    ASSERT (CHOLMOD(dump_work) (TRUE, TRUE, 0, Common)) ;
    PRINT1 (("cholmod_metis done\n")) ;
    return (Common->status == CHOLMOD_OK) ;
}
#endif
