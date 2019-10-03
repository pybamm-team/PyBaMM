/* ========================================================================= */
/* === CAMD_info =========================================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,			     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

/* User-callable.  Prints the output statistics for CAMD.  See camd.h
 * for details.  If the Info array is not present, nothing is printed.
 */

#include "camd_internal.h"

#define PRI(format,x) { if (x >= 0) { SUITESPARSE_PRINTF ((format, x)) ; }}

GLOBAL void CAMD_info
(
    double Info [ ]
)
{
    double n, ndiv, nmultsubs_ldl, nmultsubs_lu, lnz, lnzd ;

    SUITESPARSE_PRINTF (("\nCAMD version %d.%d.%d, %s, results:\n",
	CAMD_MAIN_VERSION, CAMD_SUB_VERSION, CAMD_SUBSUB_VERSION, CAMD_DATE)) ;

    if (!Info)
    {
	return ;
    }

    n = Info [CAMD_N] ;
    ndiv = Info [CAMD_NDIV] ;
    nmultsubs_ldl = Info [CAMD_NMULTSUBS_LDL] ;
    nmultsubs_lu = Info [CAMD_NMULTSUBS_LU] ;
    lnz = Info [CAMD_LNZ] ;
    lnzd = (n >= 0 && lnz >= 0) ? (n + lnz) : (-1) ;

    /* CAMD return status */
    SUITESPARSE_PRINTF (("    status: ")) ;
    if (Info [CAMD_STATUS] == CAMD_OK)
    {
	SUITESPARSE_PRINTF (("OK\n")) ;
    }
    else if (Info [CAMD_STATUS] == CAMD_OUT_OF_MEMORY)
    {
	SUITESPARSE_PRINTF (("out of memory\n")) ;
    }
    else if (Info [CAMD_STATUS] == CAMD_INVALID)
    {
	SUITESPARSE_PRINTF (("invalid matrix\n")) ;
    }
    else if (Info [CAMD_STATUS] == CAMD_OK_BUT_JUMBLED)
    {
	SUITESPARSE_PRINTF (("OK, but jumbled\n")) ;
    }
    else
    {
	SUITESPARSE_PRINTF (("unknown\n")) ;
    }

    /* statistics about the input matrix */
    PRI ("    n, dimension of A:                                  %.20g\n", n);
    PRI ("    nz, number of nonzeros in A:                        %.20g\n",
	Info [CAMD_NZ]) ;
    PRI ("    symmetry of A:                                      %.4f\n",
	Info [CAMD_SYMMETRY]) ;
    PRI ("    number of nonzeros on diagonal:                     %.20g\n",
	Info [CAMD_NZDIAG]) ;
    PRI ("    nonzeros in pattern of A+A' (excl. diagonal):       %.20g\n",
	Info [CAMD_NZ_A_PLUS_AT]) ;
    PRI ("    # dense rows/columns of A+A':                       %.20g\n",
	Info [CAMD_NDENSE]) ;

    /* statistics about CAMD's behavior  */
    PRI ("    memory used, in bytes:                              %.20g\n",
	Info [CAMD_MEMORY]) ;
    PRI ("    # of memory compactions:                            %.20g\n",
	Info [CAMD_NCMPA]) ;

    /* statistics about the ordering quality */
    SUITESPARSE_PRINTF (("\n"
	"    The following approximate statistics are for a subsequent\n"
	"    factorization of A(P,P) + A(P,P)'.  They are slight upper\n"
	"    bounds if there are no dense rows/columns in A+A', and become\n"
	"    looser if dense rows/columns exist.\n\n")) ;

    PRI ("    nonzeros in L (excluding diagonal):                 %.20g\n",
	lnz) ;
    PRI ("    nonzeros in L (including diagonal):                 %.20g\n",
	lnzd) ;
    PRI ("    # divide operations for LDL' or LU:                 %.20g\n",
	ndiv) ;
    PRI ("    # multiply-subtract operations for LDL':            %.20g\n",
	nmultsubs_ldl) ;
    PRI ("    # multiply-subtract operations for LU:              %.20g\n",
	nmultsubs_lu) ;
    PRI ("    max nz. in any column of L (incl. diagonal):        %.20g\n",
	Info [CAMD_DMAX]) ;

    /* total flop counts for various factorizations */

    if (n >= 0 && ndiv >= 0 && nmultsubs_ldl >= 0 && nmultsubs_lu >= 0)
    {
	SUITESPARSE_PRINTF (("\n"
	"    chol flop count for real A, sqrt counted as 1 flop: %.20g\n"
	"    LDL' flop count for real A:                         %.20g\n"
	"    LDL' flop count for complex A:                      %.20g\n"
	"    LU flop count for real A (with no pivoting):        %.20g\n"
	"    LU flop count for complex A (with no pivoting):     %.20g\n\n",
	n + ndiv + 2*nmultsubs_ldl,
	    ndiv + 2*nmultsubs_ldl,
	  9*ndiv + 8*nmultsubs_ldl,
	    ndiv + 2*nmultsubs_lu,
	  9*ndiv + 8*nmultsubs_lu)) ;
    }
}
