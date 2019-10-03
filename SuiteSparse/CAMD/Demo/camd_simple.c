/* ------------------------------------------------------------------------- */
/* CAMD, Copyright (c) Timothy A. Davis, Yanqing Chen,			     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: DrTimothyAldenDavis@gmail.com                                      */
/* ------------------------------------------------------------------------- */

#include <stdio.h>
#include "camd.h"

int n = 5 ;
int Ap [ ] = { 0,   2,       6,       10,  12, 14} ;
int Ai [ ] = { 0,1, 0,1,2,4, 1,2,3,4, 2,3, 1,4   } ;
int C [ ] = { 2, 0, 0, 0, 1 } ;
int P [5] ;

int main (void)
{
    int k ;
    (void) camd_order (n, Ap, Ai, P, (double *) NULL, (double *) NULL, C) ;
    for (k = 0 ; k < n ; k++) printf ("P [%d] = %d\n", k, P [k]) ;
    return (0) ;
}

