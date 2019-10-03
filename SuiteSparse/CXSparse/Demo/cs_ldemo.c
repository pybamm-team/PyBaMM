#include "cs.h"
/* test real/complex conversion routines (cs_long_t version) */
int main (void)
{
    cs_cl *T, *A, *A1, *A2, *B ;
    cs_dl *C1, *C2, *Treal, *Timag ;

    printf ("\n--- cs_ldemo, size of CS_INT: %d\n", (int) sizeof (CS_INT)) ;

    T = cs_cl_load (stdin) ;            /* load a complex triplet matrix, T */
    printf ("\nT:\n") ;
    cs_cl_print (T, 0) ;

    Treal = cs_l_real (T, 1) ;          /* Treal = real part of T */
    printf ("\nTreal:\n") ;
    cs_dl_print (Treal, 0) ;

    Timag = cs_l_real (T, 0) ;          /* Treal = imaginary part of T */
    printf ("\nTimag:\n") ;
    cs_dl_print (Timag, 0) ;

    A = cs_cl_compress (T) ;            /* A = compressed-column form of T */
    printf ("\nA:\n") ;
    cs_cl_print (A, 0) ;

    C1 = cs_l_real (A, 1) ;             /* C1 = real (A) */
    printf ("\nC1 = real(A):\n") ;
    cs_dl_print (C1, 0) ;

    C2 = cs_l_real (A, 0) ;             /* C2 = imag (A) */
    printf ("\nC2 = imag(A):\n") ;
    cs_dl_print (C2, 0) ;

    A1 = cs_l_complex (C1, 1) ;         /* A1 = complex version of C1 */
    printf ("\nA1:\n") ;
    cs_cl_print (A1, 0) ;

    A2 = cs_l_complex (C2, 0) ;         /* A2 = complex version of C2 (imag.) */
    printf ("\nA2:\n") ;
    cs_cl_print (A2, 0) ;

    B = cs_cl_add (A1, A2, 1., -1.) ;   /* B = A1 - A2 */
    printf ("\nB = conj(A):\n") ;
    cs_cl_print (B, 0) ;

    cs_cl_spfree (T) ;
    cs_cl_spfree (A) ;
    cs_cl_spfree (A1) ;
    cs_cl_spfree (A2) ;
    cs_cl_spfree (B) ;
    cs_dl_spfree (C1) ;
    cs_dl_spfree (C2) ;
    cs_dl_spfree (Treal) ;
    cs_dl_spfree (Timag) ;

    return (0) ;
}
