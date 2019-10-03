#include "cs.h"
int main (void)
{
    cs_di *T, *A, *Eye, *AT, *C, *D ;
    int i, m ;
    T = cs_di_load (stdin) ;               /* load triplet matrix T from stdin */
    printf ("T:\n") ; cs_di_print (T, 0) ; /* print T */
    A = cs_di_compress (T) ;               /* A = compressed-column form of T */
    printf ("A:\n") ; cs_di_print (A, 0) ; /* print A */
    cs_di_spfree (T) ;                     /* clear T */
    AT = cs_di_transpose (A, 1) ;          /* AT = A' */
    printf ("AT:\n") ; cs_di_print (AT, 0) ; /* print AT */
    m = A ? A->m : 0 ;                  /* m = # of rows of A */
    T = cs_di_spalloc (m, m, m, 1, 1) ;    /* create triplet identity matrix */
    for (i = 0 ; i < m ; i++) cs_di_entry (T, i, i, 1) ;
    Eye = cs_di_compress (T) ;             /* Eye = speye (m) */
    cs_di_spfree (T) ;
    C = cs_di_multiply (A, AT) ;           /* C = A*A' */
    D = cs_di_add (C, Eye, 1, cs_di_norm (C)) ;   /* D = C + Eye*norm (C,1) */
    printf ("D:\n") ; cs_di_print (D, 0) ; /* print D */
    cs_di_spfree (A) ;                     /* clear A AT C D Eye */
    cs_di_spfree (AT) ;
    cs_di_spfree (C) ;
    cs_di_spfree (D) ;
    cs_di_spfree (Eye) ;
    return (0) ;
}
