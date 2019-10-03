// =============================================================================
// === SuiteSparseQR_qmult =====================================================
// =============================================================================

// Applies Q in Householder form to a sparse or dense matrix X.  These functions
// use the Householder form in plain sparse column form, as returned by
// the SuiteSparseQR function.
//
// The result Y is sparse if X is sparse, or dense if X is dense.
// The sparse/dense cases are handled via overloaded functions.
//
//  method SPQR_QTX (0): Y = Q'*X      
//  method SPQR_QX  (1): Y = Q*X
//  method SPQR_XQT (2): Y = X*Q'
//  method SPQR_XQ  (3): Y = X*Q
//
//  Q is held in its Householder form, as the mh-by-nh sparse matrix H,
//  a vector HTau of length nh, and a permutation vector HPinv of length mh.
//  mh is m for methods 0 and 1, and n for methods 2 and 3.  If HPinv is
//  NULL, the identity permutation is used.

#include "spqr.hpp"

// =============================================================================
// === SuiteSparseQR_qmult (dense) =============================================
// =============================================================================

#define HCHUNK_DENSE 32        // FUTURE: make this an input parameter

// returns Y of size m-by-n, or NULL on failure
template <typename Entry> cholmod_dense *SuiteSparseQR_qmult
(
    // inputs, not modified
    int method,             // 0,1,2,3
    cholmod_sparse *H,      // either m-by-nh or n-by-nh
    cholmod_dense *HTau,    // size 1-by-nh
    Long *HPinv,            // size mh, identity permutation if NULL
    cholmod_dense *Xdense,  // size m-by-n with leading dimension ldx

    // workspace and parameters
    cholmod_common *cc
)
{
    cholmod_dense *Ydense ;
    Entry *X, *Y, *X1, *Y1, *Z1, *Hx, *C, *V, *Z, *CV, *Tau ;
    Long *Hp, *Hi, *Wi, *Wmap ;
    Long i, k, zsize, nh, mh, vmax, hchunk, vsize, csize, cvsize, wisize, ldx,
        m, n ;
    int ok = TRUE ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (H, NULL) ;
    RETURN_IF_NULL (HTau, NULL) ;
    RETURN_IF_NULL (Xdense, NULL) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (H, NULL) ;
    RETURN_IF_XTYPE_INVALID (HTau, NULL) ;
    RETURN_IF_XTYPE_INVALID (Xdense, NULL) ;
    cc->status = CHOLMOD_OK ;

    Hp = (Long *) H->p ;
    Hi = (Long *) H->i ;
    Hx = (Entry *) H->x ;
    nh = H->ncol ;
    mh = H->nrow ;

    X = (Entry *) Xdense->x ;
    m = Xdense->nrow ;
    n = Xdense->ncol ;
    ldx = Xdense->d ;
    ASSERT (ldx >= m) ;

    if (method == SPQR_QTX || method == SPQR_QX)
    {
        // rows of H and X must be the same
        if (mh != m)
        {
            ERROR (CHOLMOD_INVALID, "mismatched dimensions") ;
            return (NULL) ;
        }
    }
    else if (method == SPQR_XQT || method == SPQR_XQ)
    {
        // rows of H and columns of X must be the same
        if (mh != n)
        {
            ERROR (CHOLMOD_INVALID, "mismatched dimensions") ;
            return (NULL) ;
        }
    }
    else
    {
        ERROR (CHOLMOD_INVALID, "invalid method") ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // allocate result Y
    // -------------------------------------------------------------------------

    Ydense = cholmod_l_allocate_dense (m, n, m, xtype, cc) ;
    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        return (NULL) ;
    }
    Y = (Entry *) Ydense->x ;

    if (m == 0 || n == 0)
    {
        // nothing to do
        return (Ydense) ;    
    }

    // -------------------------------------------------------------------------
    // allocate workspace
    // -------------------------------------------------------------------------

    Z = NULL ;
    zsize = m*n ;
    if (method == SPQR_QX || method ==  SPQR_XQT)
    {
        // Z is needed only for methods SPQR_QX and SPQR_XQT
        Z = (Entry *) cholmod_l_malloc (zsize, sizeof (Entry), cc) ;
    }

    hchunk = MIN (HCHUNK_DENSE, nh) ;
    ok = spqr_happly_work (method, m, n, nh, Hp, hchunk, &vmax, &vsize, &csize);

    ASSERT (vmax <= mh) ;

    wisize = mh + vmax ;
    Wi = (Long *) cholmod_l_malloc (wisize, sizeof (Long), cc) ;
    Wmap = Wi + vmax ;              // Wmap is of size mh, Wi of size vmax

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory; free workspace and result Y
        cholmod_l_free_dense (&Ydense, cc) ;
        cholmod_l_free (zsize,  sizeof (Entry), Z,  cc) ;
        cholmod_l_free (wisize, sizeof (Long), Wi, cc) ;
        return (NULL) ;
    }

    if (method == SPQR_QX || method ==  SPQR_XQT)
    {
        // Z = X
        Z1 = Z ;
        X1 = X ;
        for (k = 0 ; k < n ; k++)
        {
            for (i = 0 ; i < m ; i++)
            {
                Z1 [i] = X1 [i] ;
            }
            X1 += ldx ;
            Z1 += m ;
        }
    }

    for (i = 0 ; i < mh ; i++)
    {
        Wmap [i] = EMPTY ;
    }

    // -------------------------------------------------------------------------
    // allocate O(hchunk) workspace
    // -------------------------------------------------------------------------

    // cvsize = csize + vsize ;
    cvsize = spqr_add (csize, vsize, &ok) ;
    CV = NULL ;

    if (ok)
    {
        CV = (Entry *) cholmod_l_malloc (cvsize, sizeof (Entry), cc) ;
    }

    // -------------------------------------------------------------------------
    // punt if out of memory
    // -------------------------------------------------------------------------

    if (!ok || cc->status < CHOLMOD_OK)
    {
        // PUNT: out of memory; try again with hchunk = 1
        cc->status = CHOLMOD_OK ;
        ok = TRUE ;
        hchunk = 1 ;
        ok = spqr_happly_work (method, m, n, nh, Hp, hchunk,
            &vmax, &vsize, &csize) ;
        // note that vmax has changed, but wisize is left as-is
        // cvsize = csize + vsize ;
        cvsize = spqr_add (csize, vsize, &ok) ;
        if (ok)
        {
            CV = (Entry *) cholmod_l_malloc (cvsize, sizeof (Entry), cc) ;
        }
        if (!ok || cc->status < CHOLMOD_OK)
        {
            // out of memory (or problem too large); free workspace and result Y
            cholmod_l_free_dense (&Ydense, cc) ;
            cholmod_l_free (zsize,  sizeof (Entry), Z,  cc) ;
            cholmod_l_free (wisize, sizeof (Long), Wi, cc) ;
            ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
            return (NULL) ;
        }
    }

    // -------------------------------------------------------------------------
    // split up workspace
    // -------------------------------------------------------------------------

    C = CV ;            // size csize
    V = C + csize ;     // size vsize

    // -------------------------------------------------------------------------
    // Y = Q'*X, Q*X, X*Q, or X*Q'
    // -------------------------------------------------------------------------

    PR (("Method %d m %ld n %ld X %p Y %p P %p\n", method, m, n, X, Y, HPinv)) ;
    PR (("Hp %p Hi %p Hx %p\n", Hp, Hi, Hx)) ;
    ASSERT (IMPLIES ((nh > 0), Hp != NULL && Hi != NULL && Hx != NULL)) ;
    Tau = (Entry *) HTau->x ;
    PR (("Tau %p\n", Tau)) ;
    ASSERT (Tau != NULL) ;

    if (method == SPQR_QTX)
    {

        // ---------------------------------------------------------------------
        // Y = Q'*X
        // ---------------------------------------------------------------------

        // Y (P,:) = X
        X1 = X ;
        Y1 = Y ;
        for (k = 0 ; k < n ; k++)
        {
            for (i = 0 ; i < m ; i++)
            {
                Y1 [HPinv ? HPinv [i] : i] = X1 [i] ;
            }
            X1 += ldx ;
            Y1 += m ;
        }

        // apply H to Y
        spqr_happly (method, m, n, nh, Hp, Hi, Hx, Tau, Y,
            vmax, hchunk, Wi, Wmap, C, V, cc) ;

    }
    else if (method == SPQR_QX)
    {

        // ---------------------------------------------------------------------
        // Y = Q*X
        // ---------------------------------------------------------------------

        // apply H to Z
        spqr_happly (method, m, n, nh, Hp, Hi, Hx, Tau, Z,
            vmax, hchunk, Wi, Wmap, C, V, cc) ;

        // Y = Z (P,:)
        Z1 = Z ;
        Y1 = Y ;
        for (k = 0 ; k < n ; k++)
        {
            for (i = 0 ; i < m ; i++)
            {
                Y1 [i] = Z1 [HPinv ? HPinv [i] : i] ;
            }
            Z1 += m ;
            Y1 += m ;
        }

    }
    else if (method == SPQR_XQT)
    {

        // ---------------------------------------------------------------------
        // Y = X*Q'
        // ---------------------------------------------------------------------

        // apply H to Z
        spqr_happly (method, m, n, nh, Hp, Hi, Hx, Tau, Z,
            vmax, hchunk, Wi, Wmap, C, V, cc) ;

        // Y = Z (:,P)
        Y1 = Y ;
        for (k = 0 ; k < n ; k++)
        {
            ASSERT (IMPLIES (HPinv != NULL, HPinv [k] >= 0 && HPinv [k] < n)) ;
            Z1 = Z + (HPinv ? HPinv [k] : k) * m ;  // m = leading dim. of Z
            for (i = 0 ; i < m ; i++)
            {
                Y1 [i] = Z1 [i] ;
            }
            Y1 += m ;
        }

    }
    else if (method == SPQR_XQ)
    {

        // ---------------------------------------------------------------------
        // Y = X*Q
        // ---------------------------------------------------------------------

        // Y (:,P) = X
        X1 = X ;
        for (k = 0 ; k < n ; k++)
        {
            ASSERT (IMPLIES (HPinv != NULL, HPinv [k] >= 0 && HPinv [k] < n)) ;
            Y1 = Y + (HPinv ? HPinv [k] : k) * m ;  // m = leading dim. of Y
            for (i = 0 ; i < m ; i++)
            {
                Y1 [i] = X1 [i] ;
            }
            X1 += ldx ;
        }

        // apply H to Y
        spqr_happly (method, m, n, nh, Hp, Hi, Hx, Tau, Y,
            vmax, hchunk, Wi, Wmap, C, V, cc) ;
    }

    // -------------------------------------------------------------------------
    // free workspace and return Y
    // -------------------------------------------------------------------------

    cholmod_l_free (cvsize, sizeof (Entry), CV, cc) ;
    cholmod_l_free (zsize,  sizeof (Entry), Z,  cc) ;
    cholmod_l_free (wisize, sizeof (Long), Wi, cc) ;

    if (CHECK_BLAS_INT && !cc->blas_ok)
    {
        ERROR (CHOLMOD_INVALID, "problem too large for the BLAS") ;
        cholmod_l_free_dense (&Ydense, cc) ;
        return (NULL) ;
    }

    return (Ydense) ;
}


// =============================================================================

template cholmod_dense *SuiteSparseQR_qmult <double>
(
    // inputs, not modified
    int method,             // 0,1,2,3
    cholmod_sparse *H,      // either m-by-nh or n-by-nh
    cholmod_dense *HTau,    // size 1-by-nh
    Long *HPinv,            // size mh
    cholmod_dense *Xdense,  // size m-by-n with leading dimension ldx

    // workspace and parameters
    cholmod_common *cc
) ;

// =============================================================================

template cholmod_dense *SuiteSparseQR_qmult <Complex>
(
    // inputs, not modified
    int method,             // 0,1,2,3
    cholmod_sparse *H,      // either m-by-nh or n-by-nh
    cholmod_dense *HTau,    // size 1-by-nh
    Long *HPinv,            // size mh
    cholmod_dense *Xdense,  // size m-by-n with leading dimension ldx

    // workspace and parameters
    cholmod_common *cc
) ;


// =============================================================================
// === SuiteSparseQR_qmult (sparse) ============================================
// =============================================================================

// Applies Q in Householder form to a sparse matrix X>

#define XCHUNK 4            // FUTURE: make a parameter
#define HCHUNK_SPARSE 4     // FUTURE: make a parameter

template <typename Entry> cholmod_sparse *SuiteSparseQR_qmult
(
    // inputs, not modified
    int method,             // 0,1,2,3
    cholmod_sparse *H,      // size m-by-nh or n-by-nh
    cholmod_dense *HTau,    // size 1-by-nh
    Long *HPinv,            // size mh, identity permutation if NULL
    cholmod_sparse *Xsparse,

    // workspace and parameters
    cholmod_common *cc
)
{
    cholmod_sparse *Ysparse ;
    Entry *W, *W1, *Hx, *Xx, *C, *V, *CVW, *Tau ;
    Long *Hp, *Hi, *Xp, *Xi, *Wi, *Wmap ;
    Long i, k, ny, k1, xchunk, p, k2, m, n, nh, vmax, hchunk, vsize,
        csize, cvwsize, wsize, wisize ;
    int ok = TRUE ;

    // -------------------------------------------------------------------------
    // get inputs
    // -------------------------------------------------------------------------

    RETURN_IF_NULL_COMMON (NULL) ;
    RETURN_IF_NULL (H, NULL) ;
    RETURN_IF_NULL (HTau, NULL) ;
    RETURN_IF_NULL (Xsparse, NULL) ;
    Long xtype = spqr_type <Entry> ( ) ;
    RETURN_IF_XTYPE_INVALID (H, NULL) ;
    RETURN_IF_XTYPE_INVALID (HTau, NULL) ;
    RETURN_IF_XTYPE_INVALID (Xsparse, NULL) ;
    cc->status = CHOLMOD_OK ;

    if (method == SPQR_QTX || method == SPQR_QX)
    {
        // rows of H and X must be the same
        if (H->nrow != Xsparse->nrow)
        {
            ERROR (CHOLMOD_INVALID, "mismatched dimensions") ;
            return (NULL) ;
        }
    }
    else if (method == SPQR_XQT || method == SPQR_XQ)
    {
        // rows of H and columns of X must be the same
        if (H->nrow != Xsparse->ncol)
        {
            ERROR (CHOLMOD_INVALID, "mismatched dimensions") ;
            return (NULL) ;
        }
    }
    else
    {
        ERROR (CHOLMOD_INVALID, "invalid method") ;
        return (NULL) ;
    }

    // -------------------------------------------------------------------------
    // methods 2 or 3: Y = X*Q' = (Q*X')' or Y = X*Q = (Q'*X')'
    // -------------------------------------------------------------------------

    if (method == SPQR_XQT || method == SPQR_XQ)
    {
        cholmod_sparse *XT, *YT ;
        XT = cholmod_l_transpose (Xsparse, 2, cc) ;
        YT = SuiteSparseQR_qmult <Entry>
            ((method == SPQR_XQT) ? SPQR_QX : SPQR_QTX, H, HTau, HPinv, XT, cc);
        cholmod_l_free_sparse (&XT, cc) ;
        Ysparse = cholmod_l_transpose (YT, 2, cc) ;
        cholmod_l_free_sparse (&YT, cc) ;
        return (Ysparse) ;
    }

    // -------------------------------------------------------------------------
    // get H and X
    // -------------------------------------------------------------------------

    Hp = (Long *) H->p ;
    Hi = (Long *) H->i ;
    Hx = (Entry *) H->x ;
    m = H->nrow ;
    nh = H->ncol ;

    Xp = (Long *) Xsparse->p ;
    Xi = (Long *) Xsparse->i ;
    Xx = (Entry *) Xsparse->x ;
    n = Xsparse->ncol ;

    Tau = (Entry *) HTau->x ;

    // -------------------------------------------------------------------------
    // allocate Long workspace
    // -------------------------------------------------------------------------

    xchunk = MIN (XCHUNK, n) ;
    hchunk = MIN (HCHUNK_SPARSE, nh) ;

    ok = spqr_happly_work (method, m, xchunk, nh, Hp, hchunk,
        &vmax, &vsize, &csize) ;

    wisize = m + vmax ;
    Wi = (Long *) cholmod_l_malloc (wisize, sizeof (Long), cc) ;
    Wmap = Wi + vmax ;              // Wmap is of size m, Wi is of size vmax

    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        return (NULL) ;
    }

    for (i = 0 ; i < m ; i++)
    {
        Wmap [i] = EMPTY ;
    }

    // -------------------------------------------------------------------------
    // allocate O(xchunk + hchunk) workspace
    // -------------------------------------------------------------------------

    // wsize = xchunk * m ;
    wsize = spqr_mult (xchunk, m, &ok) ;

    // cvwsize = wsize + csize + vsize ;
    cvwsize = spqr_add (wsize, csize, &ok) ;
    cvwsize = spqr_add (cvwsize, vsize, &ok) ;
    CVW = NULL ;

    if (ok)
    {
        CVW = (Entry *) cholmod_l_malloc (cvwsize, sizeof (Entry), cc) ;
    }

    // -------------------------------------------------------------------------
    // punt if out of memory
    // -------------------------------------------------------------------------

    if (!ok || cc->status < CHOLMOD_OK)
    {
        // PUNT: out of memory; try again with xchunk = 1 and hchunk = 1
        cc->status = CHOLMOD_OK ;
        xchunk = 1 ;
        hchunk = 1 ;
        ok = spqr_happly_work (method, m, xchunk, nh, Hp, hchunk,
            &vmax, &vsize, &csize) ;
        wsize = m ;

        // cvwsize = wsize + csize + vsize ;
        cvwsize = spqr_add (wsize, csize, &ok) ;
        cvwsize = spqr_add (cvwsize, vsize, &ok) ;
        if (ok)
        {
            CVW = (Entry *) cholmod_l_malloc (cvwsize, sizeof (Entry), cc) ;
        }
        if (!ok || cc->status < CHOLMOD_OK)
        {
            // still out of memory (or problem too large)
            ERROR (CHOLMOD_OUT_OF_MEMORY, "out of memory") ;
            cholmod_l_free (wisize, sizeof (Long), Wi, cc) ;
            return (NULL) ;
        }
    }

    // -------------------------------------------------------------------------
    // split up workspace
    // -------------------------------------------------------------------------

    C = CVW ;                       // C is of size csize Entry's
    V = C + csize ;                 // V is of size vsize Entry's
    W = V + vsize ;                 // W is of size wsize Entry's

    // -------------------------------------------------------------------------
    // allocate result Y
    // -------------------------------------------------------------------------

    // Y is a sparse matrix of size m-ny-n with space for m+1 entries
    Ysparse = cholmod_l_allocate_sparse (m, n, m+1, TRUE, TRUE, 0, xtype, cc) ;
    if (cc->status < CHOLMOD_OK)
    {
        // out of memory
        cholmod_l_free (cvwsize, sizeof (Entry), CVW, cc) ;
        cholmod_l_free (wisize,  sizeof (Long),  Wi,  cc) ;
        return (NULL) ;
    }
    ny = 0 ;

    // -------------------------------------------------------------------------
    // methods 0 or 1: Y = Q'*X or Q*X
    // -------------------------------------------------------------------------

    if (method == SPQR_QTX)
    {

        // ---------------------------------------------------------------------
        // Y = Q'*X
        // ---------------------------------------------------------------------

        // apply in blocks of xchunk columns each
        for (k1 = 0 ; k1 < n ; k1 += xchunk)
        {
            // scatter W:  W (P,0:3) = X (:, k:k+3)
            W1 = W ;
            k2 = MIN (k1+xchunk, n) ;
            for (k = k1 ; k < k2 ; k++)
            {
                // W1 = 0
                for (i = 0 ; i < m ; i++)
                {
                    W1 [i] = 0 ;
                }
                for (p = Xp [k] ; p < Xp [k+1] ; p++)
                {
                    i = Xi [p] ;
                    W1 [HPinv ? HPinv [i] : i] = Xx [p] ;
                }
                W1 += m ;
            }

            // apply H to W

            spqr_happly (method, m, k2-k1, nh, Hp, Hi, Hx, Tau, W,
                vmax, hchunk, Wi, Wmap, C, V, cc) ;

            // append W onto Y
            W1 = W ;
            for (k = k1 ; k < k2 ; k++)
            {
                spqr_append (W1, NULL, Ysparse, &ny, cc) ;

                if (cc->status < CHOLMOD_OK)
                {
                    // out of memory
                    cholmod_l_free_sparse (&Ysparse, cc) ;
                    cholmod_l_free (cvwsize, sizeof (Entry), CVW, cc) ;
                    cholmod_l_free (wisize,  sizeof (Long),  Wi,  cc) ;
                    return (NULL) ;
                }

                W1 += m ;
            }
        }

    }
    else // if (method == SPQR_QX)
    {

        // ---------------------------------------------------------------------
        // Y = Q*X
        // ---------------------------------------------------------------------

        // apply in blocks of xchunk columns each
        for (k1 = 0 ; k1 < n ; k1 += xchunk)
        {
            // scatter W:  W (:,0:3) = X (:, k:k+3)
            W1 = W ;
            k2 = MIN (k1+xchunk, n) ;
            for (k = k1 ; k < k2 ; k++)
            {
                // W1 = 0
                for (i = 0 ; i < m ; i++)
                {
                    W1 [i] = 0 ;
                }
                for (p = Xp [k] ; p < Xp [k+1] ; p++)
                {
                    i = Xi [p] ;
                    W1 [i] = Xx [p] ;
                }
                W1 += m ;
            }

            // apply H to W
            spqr_happly (method, m, k2-k1, nh, Hp, Hi, Hx, Tau, W,
                vmax, hchunk, Wi, Wmap, C, V, cc) ;

            // append W (P,:) onto Y
            W1 = W ;
            for (k = k1 ; k < k2 ; k++)
            {
                spqr_append (W1, HPinv, Ysparse, &ny, cc) ;
                if (cc->status < CHOLMOD_OK)
                {
                    // out of memory
                    cholmod_l_free_sparse (&Ysparse, cc) ;
                    cholmod_l_free (cvwsize, sizeof (Entry), CVW, cc) ;
                    cholmod_l_free (wisize,  sizeof (Long),  Wi,  cc) ;
                    return (NULL) ;
                }
                W1 += m ;
            }
        }
    }

    // -------------------------------------------------------------------------
    // free workspace and reduce Y in size so that nnz (Y) == nzmax (Y)
    // -------------------------------------------------------------------------

    cholmod_l_free (cvwsize, sizeof (Entry), CVW, cc) ;
    cholmod_l_free (wisize,  sizeof (Long),  Wi,  cc) ;
    cholmod_l_reallocate_sparse (cholmod_l_nnz (Ysparse,cc), Ysparse, cc) ;

    if (CHECK_BLAS_INT && !cc->blas_ok)
    {
        ERROR (CHOLMOD_INVALID, "problem too large for the BLAS") ;
        cholmod_l_free_sparse (&Ysparse, cc) ;
        return (NULL) ;
    }

    return (Ysparse) ;
}


// =============================================================================

template cholmod_sparse *SuiteSparseQR_qmult <double>
(
    // inputs, not modified
    int method,                 // 0,1,2,3
    cholmod_sparse *H,          // size m-by-nh or n-by-nh
    cholmod_dense *HTau,        // size 1-by-nh
    Long *HPinv,                // size mh
    cholmod_sparse *Xsparse,

    // workspace and parameters
    cholmod_common *cc
) ;

// =============================================================================

template cholmod_sparse *SuiteSparseQR_qmult <Complex>
(
    // inputs, not modified
    int method,                 // 0,1,2,3
    cholmod_sparse *H,          // size m-by-nh or n-by-nh
    cholmod_dense *HTau,        // size 1-by-nh
    Long *HPinv,                // size mh
    cholmod_sparse *Xsparse,

    // workspace and parameters
    cholmod_common *cc
) ;
