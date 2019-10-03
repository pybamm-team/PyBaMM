/* ========================================================================== */
/* === Tcov/leak ============================================================ */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Look for CHOLMOD memory leaks.  Run cm with
 * cholmod_dump >= cholmod_dump_malloc (see Check/cholmod_check.c),
 * to get output file.  Then grep "cnt:" output | leak
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LEN 2048
char line [LEN] ;
char operation [LEN] ;
char s_cnt [LEN] ;
char s_inuse [LEN] ;

int block [LEN] ;
int blocksize [LEN] ;

#define FALSE 0
#define TRUE 1

int main (void)
{
    int p, size, cnt2, inuse2, nblocks, found, b, bfound, nlines ;

    nblocks = 0 ;
    nlines = 0 ;

    while (fgets (line, LEN, stdin) != NULL)
    {
	sscanf (line, "%s %x %d %s %d %s %d\n",
		operation, &p, &size, s_cnt, &cnt2, s_inuse, &inuse2) ;
	nlines++ ;

	printf ("%d:: %s %x %d %s %d %s %d\n",
		nlines, operation, p, size, s_cnt, cnt2, s_inuse, inuse2) ;

	/* determine operation */
	if (strcmp (operation, "cholmod_malloc") == 0 ||
	    strcmp (operation, "cholmod_realloc_new:") == 0 ||
	    strcmp (operation, "cholmod_calloc") == 0)
	{

	    /* p = malloc (size) */
	    for (b = 0 ; b < nblocks ; b++)
	    {
		if (p == block [b])
		{
		    printf ("duplicate!\n") ;
		    abort ( ) ;
		}
	    }

	    if (nblocks >= LEN)
	    {
		printf ("out of space!\n") ;
		abort ( ) ;
	    }

	    /* add the new block to the list */
	    block [nblocks] = p ;
	    blocksize [nblocks] = size ;
	    nblocks++ ;

	}

	else if (strcmp (operation, "cholmod_free") == 0 ||
	    strcmp (operation, "cholmod_realloc_old:") == 0)
	{

	    /* p = free (size) */
	    found = FALSE ;
	    for (b = 0 ; !found && b < nblocks ; b++)
	    {
		if (p == block [b])
		{
		    bfound = b ;
		    found = TRUE ;
		}
	    }
	    if (!found)
	    {
		printf ("not found!\n") ;
		abort ( ) ;
	    }

	    if (size != blocksize [bfound])
	    {
		printf ("wrong size! %x : %d vs %d\n",
			p, size, blocksize[bfound]) ;
		abort ( ) ;
	    }

	    /* remove the block from the list */
	    --nblocks ;
	    block [bfound] = block [nblocks] ;
	    blocksize [bfound] = blocksize [nblocks] ;

	}
	else
	{
	    printf ("unrecognized!\n") ;
	    abort ( ) ;
	}

	if (cnt2 != nblocks)
	{
	    printf ("nblocks wrong! %d %d\n", nblocks, cnt2) ;
	}

    }
    return (0) ;
}
