/* ========================================================================= */
/* === CAMD demo main program ============================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,			     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/* A simple C main program that illustrates the use of the ANSI C interface
 * to CAMD.
 */

#include "camd.h"
#include <stdio.h>
#include <stdlib.h>

int main (void)
{
    /* The symmetric can_24 Harwell/Boeing matrix, including upper and lower
     * triangular parts, and the diagonal entries.  Note that this matrix is
     * 0-based, with row and column indices in the range 0 to n-1. */
    int n = 24, nz,
    Ap [ ] = { 0, 9, 15, 21, 27, 33, 39, 48, 57, 61, 70, 76, 82, 88, 94, 100,
	106, 110, 119, 128, 137, 143, 152, 156, 160 },
    Ai [ ] = {
	/* column  0: */    0, 5, 6, 12, 13, 17, 18, 19, 21,
	/* column  1: */    1, 8, 9, 13, 14, 17,
	/* column  2: */    2, 6, 11, 20, 21, 22,
	/* column  3: */    3, 7, 10, 15, 18, 19,
	/* column  4: */    4, 7, 9, 14, 15, 16,
	/* column  5: */    0, 5, 6, 12, 13, 17,
	/* column  6: */    0, 2, 5, 6, 11, 12, 19, 21, 23,
	/* column  7: */    3, 4, 7, 9, 14, 15, 16, 17, 18,
	/* column  8: */    1, 8, 9, 14,
	/* column  9: */    1, 4, 7, 8, 9, 13, 14, 17, 18,
	/* column 10: */    3, 10, 18, 19, 20, 21,
	/* column 11: */    2, 6, 11, 12, 21, 23,
	/* column 12: */    0, 5, 6, 11, 12, 23,
	/* column 13: */    0, 1, 5, 9, 13, 17,
	/* column 14: */    1, 4, 7, 8, 9, 14,
	/* column 15: */    3, 4, 7, 15, 16, 18,
	/* column 16: */    4, 7, 15, 16,
	/* column 17: */    0, 1, 5, 7, 9, 13, 17, 18, 19,
	/* column 18: */    0, 3, 7, 9, 10, 15, 17, 18, 19,
	/* column 19: */    0, 3, 6, 10, 17, 18, 19, 20, 21,
	/* column 20: */    2, 10, 19, 20, 21, 22,
	/* column 21: */    0, 2, 6, 10, 11, 19, 20, 21, 22,
	/* column 22: */    2, 20, 21, 22,
	/* column 23: */    6, 11, 12, 23 } ;

    int P [24], Pinv [24], i, j, k, jnew, p, inew, result ;
    double Control [CAMD_CONTROL], Info [CAMD_INFO] ;
    char A [24][24] ;
    int C [ ] = { 0, 0, 4, 0, 1, 0, 2, 2, 1, 1, 3, 4, 5, 5, 3, 4,
		    5, 2, 5, 3, 4, 2, 1, 0 } ;

    printf ("CAMD version %d.%d, date: %s\n", CAMD_MAIN_VERSION,
	CAMD_SUB_VERSION, CAMD_DATE) ;
    printf ("CAMD demo, with the 24-by-24 Harwell/Boeing matrix, can_24:\n") ;

    /* get the default parameters, and print them */
    camd_defaults (Control) ;
    camd_control  (Control) ;

    /* print the input matrix */
    nz = Ap [n] ;
    printf ("\nInput matrix:  %d-by-%d, with %d entries.\n"
	   "   Note that for a symmetric matrix such as this one, only the\n"
	   "   strictly lower or upper triangular parts would need to be\n"
	   "   passed to CAMD, since CAMD computes the ordering of A+A'.  The\n"
	   "   diagonal entries are also not needed, since CAMD ignores them.\n"
	   , n, n, nz) ;
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
    printf ("\nPlot of input matrix pattern:\n") ;
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

    /* order the matrix */
    result = camd_order (n, Ap, Ai, P, Control, Info, C) ;
    printf ("return value from camd_order: %d (should be %d)\n",
	result, CAMD_OK) ;

    /* print the statistics */
    camd_info (Info) ;

    if (result != CAMD_OK)
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
    printf ("\n\n") ;

    printf ("Inverse permutation vector:\n") ;
    for (j = 0 ; j < n ; j++)
    {
	k = Pinv [j] ;
	printf (" %2d", k) ;
    }
    printf ("\n\n") ;

    /* print a character plot of the permuted matrix. */
    printf ("\nPlot of permuted matrix pattern:\n") ;
    for (jnew = 0 ; jnew < n ; jnew++)
    {
	j = P [jnew] ;
	for (inew = 0 ; inew < n ; inew++) A [inew][jnew] = '.' ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    inew = Pinv [Ai [p]] ;
	    A [inew][jnew] = 'X' ;
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
