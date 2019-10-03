#include "cs.h"
typedef struct problem_struct
{
    cs_ci *A ;
    cs_ci *C ;
    int sym ;
    cs_complex_t *x ;
    cs_complex_t *b ;
    cs_complex_t *resid ;
} problem ;

problem *get_problem (FILE *f, double tol) ;
int demo2 (problem *Prob) ;
int demo3 (problem *Prob) ;
problem *free_problem (problem *Prob) ;
