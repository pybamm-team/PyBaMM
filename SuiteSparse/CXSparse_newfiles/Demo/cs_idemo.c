#include "cs.h"
/* test real/complex conversion routines (int version) */
int main (void)
{
    cs_ci *T, *A, *A1, *A2, *B ;
    cs_di *C1, *C2, *Treal, *Timag ;

    printf ("\n--- cs_idemo, size of CS_INT: %d\n", (int) sizeof (CS_INT)) ;

    T = cs_ci_load (stdin) ;            /* load a complex triplet matrix, T */
    printf ("\nT:\n") ;
    cs_ci_print (T, 0) ;

    Treal = cs_i_real (T, 1) ;          /* Treal = real part of T */
    printf ("\nTreal:\n") ;
    cs_di_print (Treal, 0) ;

    Timag = cs_i_real (T, 0) ;          /* Treal = imaginary part of T */
    printf ("\nTimag:\n") ;
    cs_di_print (Timag, 0) ;

    A = cs_ci_compress (T) ;            /* A = compressed-column form of T */
    printf ("\nA:\n") ;
    cs_ci_print (A, 0) ;

    C1 = cs_i_real (A, 1) ;             /* C1 = real (A) */
    printf ("\nC1 = real(A):\n") ;
    cs_di_print (C1, 0) ;

    C2 = cs_i_real (A, 0) ;             /* C2 = imag (A) */
    printf ("\nC2 = imag(A):\n") ;
    cs_di_print (C2, 0) ;

    A1 = cs_i_complex (C1, 1) ;         /* A1 = complex version of C1 */
    printf ("\nA1:\n") ;
    cs_ci_print (A1, 0) ;

    A2 = cs_i_complex (C2, 0) ;         /* A2 = complex version of C2 (imag.) */
    printf ("\nA2:\n") ;
    cs_ci_print (A2, 0) ;

    B = cs_ci_add (A1, A2, 1., -1.) ;   /* B = A1 - A2 */
    printf ("\nB = conj(A):\n") ;
    cs_ci_print (B, 0) ;

    cs_ci_spfree (T) ;
    cs_ci_spfree (A) ;
    cs_ci_spfree (A1) ;
    cs_ci_spfree (A2) ;
    cs_ci_spfree (B) ;
    cs_di_spfree (C1) ;
    cs_di_spfree (C2) ;
    cs_di_spfree (Treal) ;
    cs_di_spfree (Timag) ;

    return (0) ;
}
