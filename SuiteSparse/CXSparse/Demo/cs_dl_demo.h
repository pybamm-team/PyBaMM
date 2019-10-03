#include "cs.h"
typedef struct problem_struct
{
    cs_dl *A ;
    cs_dl *C ;
    cs_long_t sym ;
    double *x ;
    double *b ;
    double *resid ;
} problem ;

problem *get_problem (FILE *f, double tol) ;
cs_long_t demo2 (problem *Prob) ;
cs_long_t demo3 (problem *Prob) ;
problem *free_problem (problem *Prob) ;
