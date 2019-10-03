/* ========================================================================== */
/* === MATLAB/rand48.c ====================================================== */
/* ========================================================================== */

/* METIS uses drand48 and srand48, which conform to the SVID standard.  The
 * lcc compiler for Windows shipped with MATLAB does not include the drand48
 * and srand48 routines.  This file is a cheap replacement using the
 * more-common rand and srand instead (which lcc does have).
 *
 * This file is only used when compiling CHOLMOD with the cholmod_make.m,
 * on the PC, with the lcc compiler.
 */

#include <stdlib.h>

double drand48 (void)
{
    return (((double) (rand ( ))) / ((double) RAND_MAX)) ;
}

void srand48 (long int seed)
{
    srand ((unsigned int) seed) ;
}
