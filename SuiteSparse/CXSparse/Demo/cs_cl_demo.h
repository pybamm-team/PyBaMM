#include "cs.h"
typedef struct problem_struct
{
    cs_cl *A ;
    cs_cl *C ;
    cs_long_t sym ;
    cs_complex_t *x ;
    cs_complex_t *b ;
    cs_complex_t *resid ;
} problem ;

problem *get_problem (FILE *f, double tol) ;
cs_long_t demo2 (problem *Prob) ;
cs_long_t demo3 (problem *Prob) ;
problem *free_problem (problem *Prob) ;
