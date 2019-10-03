#include "cs.h"
int main (void)
{
    cs_dl *T, *A, *Eye, *AT, *C, *D ;
    cs_long_t i, m ;
    T = cs_dl_load (stdin) ;               /* load triplet matrix T from stdin */
    printf ("T:\n") ; cs_dl_print (T, 0) ; /* print T */
    A = cs_dl_compress (T) ;               /* A = compressed-column form of T */
    printf ("A:\n") ; cs_dl_print (A, 0) ; /* print A */
    cs_dl_spfree (T) ;                     /* clear T */
    AT = cs_dl_transpose (A, 1) ;          /* AT = A' */
    printf ("AT:\n") ; cs_dl_print (AT, 0) ; /* print AT */
    m = A ? A->m : 0 ;                  /* m = # of rows of A */
    T = cs_dl_spalloc (m, m, m, 1, 1) ;    /* create triplet identity matrix */
    for (i = 0 ; i < m ; i++) cs_dl_entry (T, i, i, 1) ;
    Eye = cs_dl_compress (T) ;             /* Eye = speye (m) */
    cs_dl_spfree (T) ;
    C = cs_dl_multiply (A, AT) ;           /* C = A*A' */
    D = cs_dl_add (C, Eye, 1, cs_dl_norm (C)) ;   /* D = C + Eye*norm (C,1) */
    printf ("D:\n") ; cs_dl_print (D, 0) ; /* print D */
    cs_dl_spfree (A) ;                     /* clear A AT C D Eye */
    cs_dl_spfree (AT) ;
    cs_dl_spfree (C) ;
    cs_dl_spfree (D) ;
    cs_dl_spfree (Eye) ;
    return (0) ;
}
