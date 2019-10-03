/* ========================================================================= */
/* === CAMD demo main program (jumbled matrix version) ===================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,			     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/* A simple C main program that illustrates the use of the ANSI C interface
 * to CAMD.
 *
 * Identical to camd_demo.c, except that it operates on an input matrix that has
 * unsorted columns and duplicate entries.
 */

#include "camd.h"
#include <stdio.h>
#include <stdlib.h>

int main (void)
{
    /* The symmetric can_24 Harwell/Boeing matrix (jumbled, and not symmetric).
     * Since CAMD operates on A+A', only A(i,j) or A(j,i) need to be specified,
     * or both.  The diagonal entries are optional (some are missing).
     * There are many duplicate entries, which must be removed. */
    int n = 24, nz,
    Ap [ ] = { 0, 9, 14, 20, 28, 33, 37, 44, 53, 58, 63, 63, 66, 69, 72, 75,
	      78, 82, 86, 91, 97, 101, 112, 112, 116 },
    Ai [ ] = {
	/* column  0: */    0, 17, 18, 21, 5, 12, 5, 0, 13,
	/* column  1: */    14, 1, 8, 13, 17,
	/* column  2: */    2, 20, 11, 6, 11, 22,
	/* column  3: */    3, 3, 10, 7, 18, 18, 15, 19,
	/* column  4: */    7, 9, 15, 14, 16,
	/* column  5: */    5, 13, 6, 17,
	/* column  6: */    5, 0, 11, 6, 12, 6, 23,
	/* column  7: */    3, 4, 9, 7, 14, 16, 15, 17, 18,
	/* column  8: */    1, 9, 14, 14, 14,
	/* column  9: */    7, 13, 8, 1, 17,
	/* column 10: */
	/* column 11: */    2, 12, 23,
	/* column 12: */    5, 11, 12,
	/* column 13: */    0, 13, 17,
	/* column 14: */    1, 9, 14,
	/* column 15: */    3, 15, 16,
	/* column 16: */    16, 4, 4, 15,
	/* column 17: */    13, 17, 19, 17,
	/* column 18: */    15, 17, 19, 9, 10,
	/* column 19: */    17, 19, 20, 0, 6, 10,
	/* column 20: */    22, 10, 20, 21,
	/* column 21: */    6, 2, 10, 19, 20, 11, 21, 22, 22, 22, 22,
	/* column 22: */
	/* column 23: */    12, 11, 12, 23 } ;

    int P [24], Pinv [24], i, j, k, jnew, p, inew, result ;
    double Control [CAMD_CONTROL], Info [CAMD_INFO] ;
    char A [24][24] ;
    int C [ ] = { 3, 0, 4, 0, 1, 1, 2, 2, 2, 2, 3, 4, 5, 5, 3, 4, 5, 2,
		    8, 10, 4, 2, 2, 0 } ;

    printf ("CAMD demo, with a jumbled version of the 24-by-24\n") ;
    printf ("Harwell/Boeing matrix, can_24:\n") ;

    /* get the default parameters, and print them */
    camd_defaults (Control) ;
    camd_control  (Control) ;

    /* print the input matrix */
    nz = Ap [n] ;
    printf ("\nJumbled input matrix:  %d-by-%d, with %d entries.\n"
	   "   Note that for a symmetric matrix such as this one, only the\n"
	   "   strictly lower or upper triangular parts would need to be\n"
	   "   passed to CAMD, since CAMD computes the ordering of A+A'.  The\n"
	   "   diagonal entries are also not needed, since CAMD ignores them.\n"
	   "   This version of the matrix has jumbled columns and duplicate\n"
	   "   row indices.\n", n, n, nz) ;
    for (j = 0 ; j < n ; j++)
    {
	printf ("\nColumn: %d, number of entries: %d, with row indices in"
		" Ai [%d ... %d]:\n    row indices:",
		j, Ap [j+1] - Ap [j], Ap [j], Ap [j+1]-1) ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;
	    printf (" %d", i) ;
	}
	printf ("\n") ;
    }

    /* print a character plot of the input matrix.  This is only reasonable
     * because the matrix is small. */
    printf ("\nPlot of (jumbled) input matrix pattern:\n") ;
    for (j = 0 ; j < n ; j++)
    {
	for (i = 0 ; i < n ; i++) A [i][j] = '.' ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;
	    A [i][j] = 'X' ;
	}
    }
    printf ("    ") ;
    for (j = 0 ; j < n ; j++) printf (" %1d", j % 10) ;
    printf ("\n") ;
    for (i = 0 ; i < n ; i++)
    {
	printf ("%2d: ", i) ;
	for (j = 0 ; j < n ; j++)
	{
	    printf (" %c", A [i][j]) ;
	}
	printf ("\n") ;
    }

    /* print a character plot of the matrix A+A'. */
    printf ("\nPlot of symmetric matrix to be ordered by camd_order:\n") ;
    for (j = 0 ; j < n ; j++)
    {
	for (i = 0 ; i < n ; i++) A [i][j] = '.' ;
    }
    for (j = 0 ; j < n ; j++)
    {
	A [j][j] = 'X' ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;
	    A [i][j] = 'X' ;
	    A [j][i] = 'X' ;
	}
    }
    printf ("    ") ;
    for (j = 0 ; j < n ; j++) printf (" %1d", j % 10) ;
    printf ("\n") ;
    for (i = 0 ; i < n ; i++)
    {
	printf ("%2d: ", i) ;
	for (j = 0 ; j < n ; j++)
	{
	    printf (" %c", A [i][j]) ;
	}
	printf ("\n") ;
    }

    /* order the matrix */
    result = camd_order (n, Ap, Ai, P, Control, Info, C) ;
    printf ("return value from camd_order: %d (should be %d)\n",
	result, CAMD_OK_BUT_JUMBLED) ;

    /* print the statistics */
    camd_info (Info) ;

    if (result != CAMD_OK_BUT_JUMBLED)
    {
	printf ("CAMD failed\n") ;
	exit (1) ;
    }

    /* print the permutation vector, P, and compute the inverse permutation */
    printf ("Permutation vector:\n") ;
    for (k = 0 ; k < n ; k++)
    {
	/* row/column j is the kth row/column in the permuted matrix */
	j = P [k] ;
	Pinv [j] = k ;
	printf (" %2d", j) ;
    }
    printf ("\nPermuted constraints:\n") ;
    for (k = 0 ; k < n ; k++)
    {
	/* row/column j is the kth row/column in the permuted matrix */
	printf (" %2d", C [P [k]]) ;
    }
    printf ("\n\n") ;

    printf ("Inverse permutation vector:\n") ;
    for (j = 0 ; j < n ; j++)
    {
	k = Pinv [j] ;
	printf (" %2d", k) ;
    }
    printf ("\n\n") ;

    /* print a character plot of the permuted matrix. */
    printf ("\nPlot of (symmetrized) permuted matrix pattern:\n") ;
    for (j = 0 ; j < n ; j++)
    {
	for (i = 0 ; i < n ; i++) A [i][j] = '.' ;
    }
    for (jnew = 0 ; jnew < n ; jnew++)
    {
	j = P [jnew] ;
	A [jnew][jnew] = 'X' ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    inew = Pinv [Ai [p]] ;
	    A [inew][jnew] = 'X' ;
	    A [jnew][inew] = 'X' ;
	}
    }
    printf ("    ") ;
    for (j = 0 ; j < n ; j++) printf (" %1d", j % 10) ;
    printf ("\n") ;
    for (i = 0 ; i < n ; i++)
    {
	printf ("%2d: ", i) ;
	for (j = 0 ; j < n ; j++)
	{
	    printf (" %c", A [i][j]) ;
	}
	printf ("\n") ;
    }

    return (0) ;
}
