/* ========================================================================== */
/* === Tcov/cmread ========================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Tcov Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Read in a matrix from a file and print it out.
 *
 * Usage:
 *	cmread matrixfile
 *	cmread < matrixfile
 */

#include "cholmod.h"

#ifdef DLONG
#define CHOLMOD(routine) cholmod_l_ ## routine
#define Int SuiteSparse_long
#else
#define CHOLMOD(routine) cholmod_ ## routine
#define Int int
#endif

int main (int argc, char **argv)
{
    cholmod_sparse *A, *C, *Z ;
    cholmod_dense *X ;
    cholmod_triplet *T ;
    void *V ;
    FILE *f, *f2 ;
    cholmod_common Common, *cm ;
    int mtype, prefer, option ;

    /* ---------------------------------------------------------------------- */
    /* get the file containing the input matrix */
    /* ---------------------------------------------------------------------- */

    if (argc > 1)
    {
	if ((f = fopen (argv [1], "r")) == NULL)
	{
	    printf ("cannot open file: %s\n", argv [1]) ;
	    return (0) ;
	}
    }
    else
    {
	f = stdin ;
    }

    /* ---------------------------------------------------------------------- */
    /* start CHOLMOD, read the matrix, print it, and free it */
    /* ---------------------------------------------------------------------- */

    cm = &Common ;
    CHOLMOD (start) (cm) ;
    cm->print = 5 ;
    A = CHOLMOD (read_sparse) (f, cm) ;
    if (argc > 1) fclose (f) ;
    CHOLMOD (print_sparse) (A, "A", cm) ;

    if (argc > 1)
    {
	for (prefer = 0 ; prefer <= 2 ; prefer++)
	{
	    printf ("\n---------------------- Prefer: %d\n", prefer) ;
	    f = fopen (argv [1], "r") ;
	    V = CHOLMOD (read_matrix) (f, prefer, &mtype, cm) ;
	    if (V != NULL) switch (mtype)
	    {
		case CHOLMOD_TRIPLET:
		    T = V ;
		    CHOLMOD (print_triplet) (T, "T", cm) ;
		    CHOLMOD (free_triplet) (&T, cm) ;
		    break ;
		case CHOLMOD_SPARSE:
		    C = V ;
		    CHOLMOD (print_sparse) (C, "C", cm) ;
		    Z = CHOLMOD (speye) (C->nrow, C->ncol, CHOLMOD_PATTERN, cm);
		    for (option = 0 ; option <= 2 ; option++)
		    {
			int asym ;
			Int xmatch = 0, pmatch = 0, nzoff = 0, nzd = 0 ;
			asym = CHOLMOD (symmetry) (C, option, 
			    &xmatch, &pmatch, &nzoff, &nzd, cm) ;
			f2 = fopen ("temp5.mtx", "w") ;
			CHOLMOD (write_sparse) (f2, C, Z, NULL, cm) ;
			fclose (f2) ;
			printf ("asym %d\n", asym) ;
		    }
		    CHOLMOD (free_sparse) (&C, cm) ;
		    CHOLMOD (free_sparse) (&Z, cm) ;
		    break ;
		case CHOLMOD_DENSE:
		    X = V ;
		    CHOLMOD (print_dense) (X, "X", cm) ;
		    f2 = fopen ("temp5.mtx", "w") ;
		    CHOLMOD (write_dense) (f2, X, NULL, cm) ;
		    fclose (f2) ;
		    CHOLMOD (free_dense) (&X, cm) ;
		    break ;
	    }
	    fclose (f) ;
	}
    }

    CHOLMOD (free_sparse) (&A, cm) ;
    CHOLMOD (finish) (cm) ;
    return (0) ;
}
