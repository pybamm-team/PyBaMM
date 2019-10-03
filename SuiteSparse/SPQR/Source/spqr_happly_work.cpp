// =============================================================================
// === spqr_happly_work ========================================================
// =============================================================================

// Determines the workspace workspace needed by spqr-happly

#include "spqr.hpp"

int spqr_happly_work
(
    // input
    int method,     // 0,1,2,3 

    Long m,         // X is m-by-n
    Long n,

    // FUTURE : make H cholmod_sparse:
    Long nh,        // number of Householder vectors
    Long *Hp,       // size nh+1, column pointers for H
    Long hchunk,

    // outputs; sizes of workspaces needed
    Long *p_vmax, 
    Long *p_vsize, 
    Long *p_csize
)
{
    Long maxhlen, h, hlen, vmax, mh, vsize, csize, vsize1, vsize2 ;
    int ok = TRUE ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    *p_vmax = 0 ;
    *p_vsize = 0 ;
    *p_csize = 0 ;

    if (m == 0 || n == 0 || nh == 0)
    {
        // nothing to do
        return (TRUE) ;
    }

    // -------------------------------------------------------------------------
    // determine the length of the longest Householder vector
    // -------------------------------------------------------------------------

    maxhlen = 1 ;
    for (h = 0 ; h < nh ; h++)
    {
        hlen = Hp [h+1] - Hp [h] ;
        maxhlen = MAX (maxhlen, hlen) ;
    }

    // number of rows of H
    mh = (method == 0 || method == 1) ? m : n ;

    // -------------------------------------------------------------------------
    // determine workspace sizes
    // -------------------------------------------------------------------------

    // Long overflow cannot occur with vmax since H is already allocated
    if (method == 0 || method == 3)
    {
        // apply H in the forward direction; H(0) first, H(nh-1) last
        vmax = 2 * maxhlen + 8 ;
    }
    else
    {
        // apply H in the backward direction; H(nh-1) first, H(0) last
        vmax = maxhlen + hchunk ;
    }

    vmax = MIN (vmax, mh) ;
    vmax = MAX (vmax, 2) ;

    // csize = vmax * ((method <= 1) ? n : m) ;
    csize = spqr_mult (vmax, (method <= 1) ? n : m, &ok) ;

    // vsize = (hchunk*hchunk + ((method <= 1) ? n : m)*hchunk + vmax*hchunk) ;
    vsize  = spqr_mult (hchunk, hchunk, &ok) ;
    vsize1 = spqr_mult ((method <= 1) ? n : m, hchunk, &ok) ;
    vsize2 = spqr_mult (vmax, hchunk, &ok) ;
    vsize = spqr_add (vsize, vsize1, &ok) ;
    vsize = spqr_add (vsize, vsize2, &ok) ;

    // -------------------------------------------------------------------------
    // return workspace sizes
    // -------------------------------------------------------------------------

    *p_vmax = vmax ;
    *p_vsize = vsize ;
    *p_csize = csize ;
    return (ok) ;
}
