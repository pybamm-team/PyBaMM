#include "cs.h"
int main (void)
{
    cs_cl *T, *A, *Eye, *AT, *C, *D ;
    cs_long_t i, m ;
    T = cs_cl_load (stdin) ;               /* load triplet matrix T from stdin */
    printf ("T:\n") ; cs_cl_print (T, 0) ; /* print T */
    A = cs_cl_compress (T) ;               /* A = compressed-column form of T */
    printf ("A:\n") ; cs_cl_print (A, 0) ; /* print A */
    cs_cl_spfree (T) ;                     /* clear T */
    AT = cs_cl_transpose (A, 1) ;          /* AT = A' */
    printf ("AT:\n") ; cs_cl_print (AT, 0) ; /* print AT */
    m = A ? A->m : 0 ;                  /* m = # of rows of A */
    T = cs_cl_spalloc (m, m, m, 1, 1) ;    /* create triplet identity matrix */
    for (i = 0 ; i < m ; i++) cs_cl_entry (T, i, i, 1) ;
    Eye = cs_cl_compress (T) ;             /* Eye = speye (m) */
    cs_cl_spfree (T) ;
    C = cs_cl_multiply (A, AT) ;           /* C = A*A' */
    D = cs_cl_add (C, Eye, 1, cs_cl_norm (C)) ;   /* D = C + Eye*norm (C,1) */
    printf ("D:\n") ; cs_cl_print (D, 0) ; /* print D */
    cs_cl_spfree (A) ;                     /* clear A AT C D Eye */
    cs_cl_spfree (AT) ;
    cs_cl_spfree (C) ;
    cs_cl_spfree (D) ;
    cs_cl_spfree (Eye) ;
    return (0) ;
}
