// =============================================================================
// === spqr_happly =============================================================
// =============================================================================

// Applies a set n of sparse Householder vectors to a dense m-by-n matrix X.
//
//  Let H(k) = I - Tau(k) * V(k) * V(k)', then either of the following is done:
//
//  method 0: X = Q'*X
//      X = H(nh-1) * ... * H(1) * H(0) * X         where H is m-by-nh
//
//  method 1: X = Q*X
//      X = H(0)' * H(1)' * ... * H(nh-1)' * X      where H is m-by-nh
//
//  method 2: X = X*Q'
//      X = X * H(nh-1) * ... * H(1) * H(0)         where H is n-by-nh
//
//  method 3: X = X*Q
//      X = X * H(0)' * H(1) * ... * H(nh-1)'       where H is n-by-nh
//
//  The first nonzero entry in each column of H is assumed to be equal to 1.
//  This function does not apply the row permutation vector (the Q.P part of
//  the Q struct in the MATLAB interface).

#include "spqr.hpp"

// =============================================================================
// === spqr_private_do_panel ===================================================
// =============================================================================

// Loads V with a panel of Householder vectors and applies them to X

template <typename Entry> void spqr_private_do_panel
(
    // inputs, not modified
    int method,         // which method to use (0,1,2,3)
    Long m,
    Long n,
    Long v,             // number of Householder vectors in the panel
    Long *Wi,           // Wi [0:v-1] defines the pattern of the panel
    Long h1,            // load H (h1) to H (h2-1) into V
    Long h2,

    // FUTURE : make H cholmod_sparse:
    Long *Hp,           // Householder vectors: mh-by-nh sparse matrix
    Long *Hi,
    Entry *Hx,

    Entry *Tau,         // Householder coefficients (size nh)

    // input/output
    Long *Wmap,         // inverse of Wi on input, set to all EMPTY on output
    Entry *X,           // m-by-n with leading dimension m

    // workspace, undefined on input and output
    Entry *V,           // dense panel
    Entry *C,           // workspace
    Entry *W,           // workspace
    cholmod_common *cc
)
{
    Entry *V1 ;
    Long h, k, p, i ;

    // -------------------------------------------------------------------------
    // load the panel with Householder vectors h1 ... h2-1
    // -------------------------------------------------------------------------

    // Wi [0 .. v-1] defines the pattern of the panel, and Wmap gives its
    // inverse (Wmap [Wi [k]] == k for k = 0 to v-1).  The V matrix is v-by-k.

#ifndef NDEBUG
    for (k = 0 ; k < v ; k++) ASSERT (Wmap [Wi [k]] == k) ;
#endif

    V1 = V ;
    for (h = h1 ; h < h2 ; h++)
    {
        PR (("loading %ld\n", h)) ;
        for (k = 0 ; k < v ; k++)
        {
            V1 [k] = 0 ;
        }
        for (p = Hp [h] ; p < Hp [h+1] ; p++)
        {
            i = Hi [p] ;
            ASSERT (Wmap [i] >= 0 && Wmap [i] < v) ;
            V1 [Wmap [i]] = Hx [p] ;
        }
        V1 += v ;
    }

    // -------------------------------------------------------------------------
    // apply the panel
    // -------------------------------------------------------------------------

    spqr_panel (method, m, n, v, h2-h1, Wi, V, Tau+h1, m, X, C, W, cc) ;

    // -------------------------------------------------------------------------
    // clear the panel mapping
    // -------------------------------------------------------------------------

    for (k = 0 ; k < v ; k++)
    {
        i = Wi [k] ;
        Wmap [i] = EMPTY ;
    }
}


// =============================================================================
// === spqr_happly =============================================================
// =============================================================================

template <typename Entry> void spqr_happly
(
    // input
    int method,     // 0,1,2,3

    Long m,         // X is m-by-n with leading dimension m
    Long n,

    // FUTURE : make H cholmod_sparse:
    Long nh,        // number of Householder vectors
    Long *Hp,       // size nh+1, column pointers for H
    Long *Hi,       // size hnz = Hp [nh], row indices of H
    Entry *Hx,      // size hnz, Householder values.  Note that the first
                    // entry in each column must be equal to 1.0

    Entry *Tau,     // size nh

    // input/output
    Entry *X,       // size m-by-n with leading dimension m

    // workspace
    Long vmax,
    Long hchunk,
    Long *Wi,       // size vmax
    Long *Wmap,     // size MAX(mh,1) where H is mh-by-nh; all EMPTY
    Entry *C,       // size csize
    Entry *V,       // size vsize
    cholmod_common *cc
)
{
    Entry *W ;
    Long h, h1, h2, i, k, hmax, hmin, v, v1, p, done, v2, mh ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    if (m == 0 || n == 0 || nh == 0)
    {
        // nothing to do
        return ;
    }

    // number of rows of H
    mh = (method == 0 || method == 1) ? m : n ;

    W = V + vmax * hchunk ;

    // -------------------------------------------------------------------------
    // apply the Householder vectors
    // -------------------------------------------------------------------------

    if (method == 0 || method == 3)
    {

        // ---------------------------------------------------------------------
        // apply in forward direction
        // ---------------------------------------------------------------------

        PR (("\nHAPPLY Forward, method %d\n", method)) ;

        for (h1 = 0 ; h1 < nh ; h1 = h2)
        {

            // -----------------------------------------------------------------
            // start the panel with Householder vector h1
            // -----------------------------------------------------------------

#ifndef NDEBUG
            for (i = 0 ; i < mh ; i++) ASSERT (Wmap [i] == EMPTY) ;
            PR (("\n ------ h1 %ld\n", h1)) ;
#endif

            v = 0 ;
            for (p = Hp [h1] ; p < Hp [h1+1] ; p++)
            {
                i = Hi [p] ;
                Wmap [i] = v ;
                Wi [v] = i ;
                v++ ;
            }
            Long this_vmax = 2*v + 8 ;               // max # rows in this panel
            this_vmax = MIN (this_vmax, mh) ;
            ASSERT (this_vmax <= vmax) ;

            // -----------------------------------------------------------------
            // acquire pattern of panel of Householder vectors
            // -----------------------------------------------------------------

            hmax = MIN (h1 + hchunk, nh) ;   // at most h1 .. hmax-1 in panel
            done = FALSE ;
            for (h2 = h1+1 ; h2 < hmax ; h2++)
            {
                PR (("try %ld\n", h2)) ;
                p = Hp [h2] ;
                i = Hi [p] ;
                // check to see that this vector fits in the lower triangle
                if (h2-h1 >= v || Wi [h2-h1] != i)
                {
                    // h2 will not be part of this panel
                    PR (("triangle broken\n")) ;
                    break ;
                }
                v1 = v ;      // save this in case h2 is not part of panel
                for ( ; p < Hp [h2+1] ; p++)
                {
                    i = Hi [p] ;
                    if (Wmap [i] == EMPTY)
                    {
                        if (v >= this_vmax)
                        {
                            // h2 is not part of this panel
                            for (v2 = v1 ; v2 < v ; v2++)
                            {
                                // clear the partial column h2 from the panel
                                Wmap [Wi [v2]] = EMPTY ;
                            }
                            v = v1 ;
                            done = TRUE ;
                            PR (("too long\n")) ;
                            break ;
                        }
                        Wmap [i] = v ;
                        Wi [v] = i ;
                        v++ ;
                    }
                }
                if (done)
                {
                    break ;
                }
            }

            // -----------------------------------------------------------------
            // load and apply the panel
            // -----------------------------------------------------------------

            spqr_private_do_panel (method, m, n, v, Wi, h1, h2, Hp, Hi, Hx, Tau,
                Wmap, X, V, C, W, cc) ;
        }

    }
    else
    {

        // ---------------------------------------------------------------------
        // apply in backward direction
        // ---------------------------------------------------------------------

        PR (("\nHAPPLY Backward, method %d\n", method)) ;

        for (h2 = nh ; h2 > 0 ; h2 = h1)
        {

            // -----------------------------------------------------------------
            // start the panel with Householder vector h2-1 as the last one
            // -----------------------------------------------------------------

#ifndef NDEBUG
            for (i = 0 ; i < mh ; i++) ASSERT (Wmap [i] == EMPTY) ;
            PR (("\n ------ h2 %ld\n", h2)) ;
#endif

            // use Wi as a stack, growing upwards starting at Wi [vmax-1]
            h = h2-1 ;
            v = vmax ;
            for (p = Hp [h+1]-1 ; p >= Hp [h] ; p--)
            {
                i = Hi [p] ;
                v-- ;
                Wmap [i] = v ;              // this will be shifted later
                Wi [v] = i ;
            }
            Long this_vmin = v - 32 ;
            this_vmin = MAX (this_vmin, 0) ;

            // -----------------------------------------------------------------
            // acquire pattern of panel of Householder vectors
            // -----------------------------------------------------------------

            hmin = MAX (h2 - hchunk, 0) ;    // at most hmin .. h2-1 in panel
            done = FALSE ;

            for (h1 = h2-2 ; h1 >= hmin ; h1--)
            {
                // try to add h1 to the panel
                PR (("try %ld\n", h1)) ;

                p = Hp [h1] ;

                // check to see that this vector fits in the lower triangle
                Long hlen = Hp [h1+1] - p ;
                if (hlen > 1 && Hi [p+1] != Wi [v])
                {
                    // h1 will not be part of this panel
                    h1++ ;
                    PR (("triangle broken\n")) ;
                    break ;
                }

                // ensure that the first entry of h1 is not in Wi
                i = Hi [p] ;
                if (Wmap [i] != EMPTY)
                {
                    h1++ ;
                    PR (("first entry %ld present; triangle broken\n", i)) ;
                    break ;
                }

                // ensure that all of h1 is already in Wi (except first entry)
                for (p++ ; p < Hp [h1+1] ; p++)
                {
                    i = Hi [p] ;
                    if (Wmap [i] == EMPTY)
                    {
                        // h1 is not in the panel
                        done = TRUE ;
                        PR (("pattern broken\n")) ;
                        h1++ ;
                        break ;
                    }
                }

                if (done)
                {
                    break;
                }

                // h1 is added to the panel
                p = Hp [h1] ;
                i = Hi [p] ;
                v-- ;
                Wi [v] = i ;
                Wmap [i] = v ;              // this will be shifted later

#ifndef NDEBUG
                for (k = v ; k < vmax ; k++) ASSERT (Wmap [Wi [k]] == k) ;
#endif
            }

            h1 = MAX (h1, hmin) ;

            // shift Wi upwards from Wi [v..vmax-1] to Wi [0...], and
            // recompute Wmap

            v2 = 0 ;
            for (k = v ; k < vmax ; k++)
            {
                Wi [v2++] = Wi [k] ;
            }
            v = v2 ;

            for (k = 0 ; k < v ; k++)
            {
                Wmap [Wi [k]] = k ;
            }

            // -----------------------------------------------------------------
            // load and apply the panel
            // -----------------------------------------------------------------

            spqr_private_do_panel (method, m, n, v, Wi, h1, h2, Hp, Hi, Hx, Tau,
                Wmap, X, V, C, W, cc) ;
        }
    }
}


// =============================================================================

template void spqr_happly <double>
(
    // input
    int method,     // 0,1,2,3

    Long m,         // X is m-by-n
    Long n,

    Long nh,        // number of Householder vectors
    Long *Hp,       // size nh+1, column pointers for H
    Long *Hi,       // size hnz = Hp [nh], row indices of H
    double *Hx,     // size hnz, Householder values.  Note that the first
                    // entry in each column must be equal to 1.0

    double *Tau,    // size nh

    // input/output
    double *X,      // size m-by-n with leading dimension m

    // workspace
    Long vmax,
    Long hchunk,
    Long *Wi,       // size vmax
    Long *Wmap,     // size MAX(mh,1) where H is mh-by-nh
    double *C,      // size csize
    double *V,      // size vsize
    cholmod_common *cc
) ;

// =============================================================================

template void spqr_happly <Complex>
(
    // input
    int method,     // 0,1,2,3

    Long m,         // X is m-by-n
    Long n,

    Long nh,        // number of Householder vectors
    Long *Hp,       // size nh+1, column pointers for H
    Long *Hi,       // size hnz = Hp [nh], row indices of H
    Complex *Hx,    // size hnz, Householder values.  Note that the first
                    // entry in each column must be equal to 1.0

    Complex *Tau,   // size nh

    // input/output
    Complex *X,     // size m-by-n with leading dimension m

    // workspace
    Long vmax,
    Long hchunk,
    Long *Wi,       // size vmax
    Long *Wmap,     // size MAX(mh,1) where H is mh-by-nh
    Complex *C,     // size csize
    Complex *V,     // size vsize
    cholmod_common *cc
) ;
