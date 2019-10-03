#include "waitmex.h"

/* The MATLAB equivalent of this function is give in waitex.m.
   Compile with:

    mex waitexample.c waitmex.c
 */

void useless (double *x) ;

void useless (double *x) { (*x)++ ; }

void mexFunction (int nargout, mxArray *pargout [ ],
    int nargin, const mxArray *pargin [ ])
{
    int i, j ;
    double x = 0 ;
    waitbar *h ;

    /* just like h = waitbar (0, 'Please wait...') in MATLAB */
    h = waitbar_create (0, "Please wait...") ;

    for (i = 0 ; i <= 100 ; i++)
    {
        if (i == 50)
        {
            /* just like waitbar (i/100, h, 'over half way there') in MATLAB */
            waitbar_update (((double) i) / 100., h, "over half way there") ;
        }
        else
        {
            /* just like waitbar (i/100, h) in MATLAB */
            waitbar_update (((double) i) / 100., h, NULL) ;
        }

        /* do some useless work */
        for (j = 0 ; j <= 10000000 ; j++) useless (&x) ;
    }
    if (nargout > 0)
    {
        /* return the handle to the waitbar, if requested */
        pargout [0] = waitbar_return (h) ;
    }
    else
    {
        /* just like close (h) in MATLAB */
        waitbar_destroy (h) ;
    }
}
