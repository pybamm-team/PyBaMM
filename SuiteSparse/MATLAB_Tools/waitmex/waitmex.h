/* -------------------------------------------------------------------------- */
/* waitmex include file */
/* -------------------------------------------------------------------------- */

#ifndef _WAITMEX_H
#define _WAITMEX_H
#include "mex.h"

typedef struct waitbar_struct
{
    mxArray *inputs [3] ;       /* waitbar inputs */
    mxArray *outputs [2] ;      /* waitbar outputs */
    mxArray *handle ;           /* handle from waitbar */
    mxArray *fraction ;         /* fraction from 0 to 1 (a scalar) */
    mxArray *message ;          /* waitbar message */

} waitbar ;

waitbar *waitbar_create (double, char *) ;
void waitbar_update (double, waitbar *, char *) ;
void waitbar_destroy (waitbar *) ;
mxArray *waitbar_return (waitbar *) ;
#endif
