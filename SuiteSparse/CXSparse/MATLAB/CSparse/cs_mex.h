#include "cs.h"
#include "mex.h"

void cs_mex_check (CS_INT nel, CS_INT m, CS_INT n, int square,
    int sparse, int values, const mxArray *A) ;

CS_INT *cs_dl_mex_get_int (CS_INT n, const mxArray *Imatlab, CS_INT *imax,
    int lo);
mxArray *cs_dl_mex_put_int (CS_INT *p, CS_INT n, CS_INT offset, int do_free) ;

double *cs_dl_mex_get_double (CS_INT n, const mxArray *X) ;
cs_dl *cs_dl_mex_get_sparse (cs_dl *A, int square, int values,
    const mxArray *Amatlab) ;
double *cs_dl_mex_put_double (CS_INT n, const double *b, mxArray **X) ;
mxArray *cs_dl_mex_put_sparse (cs_dl **A) ;

#ifndef NCOMPLEX
cs_complex_t *cs_cl_mex_get_double (CS_INT n, const mxArray *X) ;
cs_cl *cs_cl_mex_get_sparse (cs_cl *A, int square, const mxArray *Amatlab) ;
mxArray *cs_cl_mex_put_double (CS_INT n, cs_complex_t *b) ;
mxArray *cs_cl_mex_put_sparse (cs_cl **Ahandle) ;
#endif
