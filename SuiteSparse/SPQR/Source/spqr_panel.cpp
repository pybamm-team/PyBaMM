// =============================================================================
// === spqr_panel ==============================================================
// =============================================================================

// Applies a panel of Householder vectors to a dense m-by-n matrix X.
//
//  Let H(k) = I - Tau(k) * V(k) * V(k)', then either of the following is done:
//
//  method SPQR_QTX (0): X = Q'*X
//      X = H(h-1) * ... * H(1) * H(0) * X         where H is m-by-h
//
//  method SPQR_QX  (1): X = Q*X
//      X = H(0)' * H(1)' * ... * H(h-1)' * X      where H is m-by-h
//
//  method SPQR_XQT (2): X = X*Q'
//      X = X * H(h-1) * ... * H(1) * H(0)         where H is n-by-h
//
//  method SPQR_XQ  (3): X = X*Q
//      X = X * H(0)' * H(1) * ... * H(h-1)'       where H is n-by-h
//
//  The first nonzero entry in each column of H is assumed to be equal to 1.
//  This function does not apply the row permutation vector (the Q.P part of
//  the Q struct in the MATLAB interface).
//
//  The Householder vectors are stored in a sparse block format.  The vectors
//  are held in V, an array of size v-by-h.  The nonzero patterns of each of
//  these vectors is the same; it is held in Vi [0:v-1].  The array V is lower
//  triangular with implicit unit diagonal (the unit need not be actually
//  present).

#include "spqr.hpp"

template <typename Entry> void spqr_panel
(
    // input
    int method,         // 0,1,2,3
    Long m,
    Long n,
    Long v,             // length of the first vector in V
    Long h,             // number of Householder vectors in the panel
    Long *Vi,           // Vi [0:v-1] defines the pattern of the panel
    Entry *V,           // v-by-h, panel of Householder vectors
    Entry *Tau,         // size h, Householder coefficients for the panel
    Long ldx,

    // input/output
    Entry *X,           // m-by-n with leading dimension ldx

    // workspace
    Entry *C,           // method 0,1: v-by-n;  method 2,3: m-by-v
    Entry *W,           // method 0,1: h*h+n*h; method 2,3: h*h+m*h

    cholmod_common *cc
)
{
    Entry *C1, *X1 ;
    Long k, p, i ;

    // -------------------------------------------------------------------------
    // gather X into workspace C
    // -------------------------------------------------------------------------

    if (method == SPQR_QTX || method == SPQR_QX)
    {
        // X is m-by-n with leading dimension ldx
        // C is v-by-n with leading dimension v
        C1 = C ;
        X1 = X ;
        for (k = 0 ; k < n ; k++)
        {
            for (p = 0 ; p < v ; p++)
            {
                i = Vi [p] ;
                C1 [p] = X1 [i] ;
            }
            C1 += v ;
            X1 += ldx ;
        }
    }
    else // if (method == SPQR_XQT || method == SPQR_XQ)
    {
        // X is m-by-n with leading dimension ldx
        // C is m-by-v with leading dimension m
        C1 = C ;
        for (p = 0 ; p < v ; p++)
        {
            i = Vi [p] ;
            X1 = X + i*ldx ;
            for (k = 0 ; k < m ; k++)
            {
                C1 [k] = X1 [k] ;
            }
            C1 += m ;
        }
    }

    // -------------------------------------------------------------------------
    // apply the Householder panel to C
    // -------------------------------------------------------------------------

    if (method == SPQR_QTX || method == SPQR_QX)
    {
        spqr_larftb (method, v, n, h, v, v, V, Tau, C, W, cc) ;
    }
    else // if (method == SPQR_XQT || method == SPQR_XQ)
    {
        spqr_larftb (method, m, v, h, m, v, V, Tau, C, W, cc) ;
    }

    // -------------------------------------------------------------------------
    // scatter C back into X
    // -------------------------------------------------------------------------

    if (method == SPQR_QTX || method == SPQR_QX)
    {
        C1 = C ;
        X1 = X ;
        for (k = 0 ; k < n ; k++)
        {
            for (p = 0 ; p < v ; p++)
            {
                i = Vi [p] ;
                X1 [i] = C1 [p] ;
            }
            C1 += v ;
            X1 += ldx ;
        }
    }
    else // if (method == SPQR_XQT || method == SPQR_XQ)
    {
        C1 = C ;
        for (p = 0 ; p < v ; p++)
        {
            i = Vi [p] ;
            X1 = X + i*ldx ;
            for (k = 0 ; k < m ; k++)
            {
                X1 [k] = C1 [k] ;
            }
            C1 += m ;
        }
    }
}


// =============================================================================

template void spqr_panel <double>
(
    // input
    int method,
    Long m,
    Long n,
    Long v,
    Long h,             // number of Householder vectors in the panel
    Long *Vi,           // Vi [0:v-1] defines the pattern of the panel
    double *V,          // v-by-h, panel of Householder vectors
    double *Tau,        // size h, Householder coefficients for the panel
    Long ldx,

    // input/output
    double *X,          // m-by-n with leading dimension m 

    // workspace
    double *C,          // method 0,1: v-by-n;  method 2,3: m-by-v
    double *W,          // method 0,1: k*k+n*k; method 2,3: k*k+m*k

    cholmod_common *cc
) ;

template void spqr_panel <Complex>
(
    // input
    int method,
    Long m,
    Long n,
    Long v,
    Long h,             // number of Householder vectors in the panel
    Long *Vi,           // Vi [0:v-1] defines the pattern of the panel
    Complex *V,         // v-by-h, panel of Householder vectors
    Complex *Tau,       // size h, Householder coefficients for the panel
    Long ldx,

    // input/output
    Complex *X,         // m-by-n with leading dimension m

    // workspace
    Complex *C,         // method 0,1: v-by-n;  method 2,3: m-by-v
    Complex *W,         // method 0,1: k*k+n*k; method 2,3: k*k+m*k

    cholmod_common *cc
) ;
