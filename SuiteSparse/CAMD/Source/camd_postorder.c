/* ========================================================================= */
/* === CAMD_postorder ====================================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,			     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/* Perform a postordering (via depth-first search) of an assembly tree. */

#include "camd_internal.h"

GLOBAL Int CAMD_postorder
(
    Int j,	    /* start at node j, a root of the assembly tree */
    Int k,	    /* on input, next node is the kth node */
    Int n,	    /* normal nodes 0 to n-1, place-holder node n */
    Int head [],    /* head of link list of children of each node */
    Int next [],    /* next[i] is the next child after i in link list */
    Int post [],    /* postordering, post [k] = p if p is the kth node */
    Int stack []    /* recursion stack */
)
{
    int i, p, top = 0 ;
    stack [0] = j ;	  /* place j on the stack, maybe place-holder node n */
    while (top >= 0)	  /* while (stack is not empty) */
    {
	p = stack [top] ;	/* p = top of stack */
	i = head [p] ;		/* i = youngest child of p */
	if (i == -1)
	{
	    top-- ;		/* p has no unordered children left */
	    if (p != n)
	    {
		/* node p is the kth postordered node.  Do not postorder the
		 * place-holder node n, which is the root of a subtree
		 * containing all dense and empty nodes. */
		post [k++] = p ;
	    }
	}
	else
	{
	    head [p] = next [i] ;   /* remove i from children of p */
	    stack [++top] = i ;	    /* start dfs on child node i */
	}
    }
    return (k) ;
}
