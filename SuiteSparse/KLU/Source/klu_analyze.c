/* ========================================================================== */
/* === klu_analyze ========================================================== */
/* ========================================================================== */

/* Order the matrix using BTF (or not), and then AMD, COLAMD, the natural
 * ordering, or the user-provided-function on the blocks.  Does not support
 * using a given ordering (use klu_analyze_given for that case). */

#include "klu_internal.h"

/* ========================================================================== */
/* === analyze_worker ======================================================= */
/* ========================================================================== */

static Int analyze_worker       /* returns KLU_OK or < 0 if error */
(
    /* inputs, not modified */
    Int n,              /* A is n-by-n */
    Int Ap [ ],         /* size n+1, column pointers */
    Int Ai [ ],         /* size nz, row indices */
    Int nblocks,        /* # of blocks */
    Int Pbtf [ ],       /* BTF row permutation */
    Int Qbtf [ ],       /* BTF col permutation */
    Int R [ ],          /* size n+1, but only Rbtf [0..nblocks] is used */
    Int ordering,       /* what ordering to use (0, 1, or 3 for this routine) */

    /* output only, not defined on input */
    Int P [ ],          /* size n */
    Int Q [ ],          /* size n */
    double Lnz [ ],     /* size n, but only Lnz [0..nblocks-1] is used */

    /* workspace, not defined on input or output */
    Int Pblk [ ],       /* size maxblock */
    Int Cp [ ],         /* size maxblock+1 */
    Int Ci [ ],         /* size MAX (nz+1, Cilen) */
    Int Cilen,          /* nz+1, or COLAMD_recommend(nz,n,n) for COLAMD */
    Int Pinv [ ],       /* size maxblock */

    /* input/output */
    KLU_symbolic *Symbolic,
    KLU_common *Common
)
{
    double amd_Info [AMD_INFO], lnz, lnz1, flops, flops1 ;
    Int k1, k2, nk, k, block, oldcol, pend, newcol, result, pc, p, newrow,
        maxnz, nzoff, cstats [COLAMD_STATS], ok, err = KLU_INVALID ;

    /* ---------------------------------------------------------------------- */
    /* initializations */
    /* ---------------------------------------------------------------------- */

    /* compute the inverse of Pbtf */
#ifndef NDEBUG
    for (k = 0 ; k < n ; k++)
    {
        P [k] = EMPTY ;
        Q [k] = EMPTY ;
        Pinv [k] = EMPTY ;
    }
#endif
    for (k = 0 ; k < n ; k++)
    {
        ASSERT (Pbtf [k] >= 0 && Pbtf [k] < n) ;
        Pinv [Pbtf [k]] = k ;
    }
#ifndef NDEBUG
    for (k = 0 ; k < n ; k++) ASSERT (Pinv [k] != EMPTY) ;
#endif
    nzoff = 0 ;
    lnz = 0 ;
    maxnz = 0 ;
    flops = 0 ;
    Symbolic->symmetry = EMPTY ;        /* only computed by AMD */

    /* ---------------------------------------------------------------------- */
    /* order each block */
    /* ---------------------------------------------------------------------- */

    for (block = 0 ; block < nblocks ; block++)
    {

        /* ------------------------------------------------------------------ */
        /* the block is from rows/columns k1 to k2-1 */
        /* ------------------------------------------------------------------ */

        k1 = R [block] ;
        k2 = R [block+1] ;
        nk = k2 - k1 ;
        PRINTF (("BLOCK %d, k1 %d k2-1 %d nk %d\n", block, k1, k2-1, nk)) ;

        /* ------------------------------------------------------------------ */
        /* construct the kth block, C */
        /* ------------------------------------------------------------------ */

        Lnz [block] = EMPTY ;
        pc = 0 ;
        for (k = k1 ; k < k2 ; k++)
        {
            newcol = k-k1 ;
            Cp [newcol] = pc ;
            oldcol = Qbtf [k] ;
            pend = Ap [oldcol+1] ;
            for (p = Ap [oldcol] ; p < pend ; p++)
            {
                newrow = Pinv [Ai [p]] ;
                if (newrow < k1)
                {
                    nzoff++ ;
                }
                else
                {
                    /* (newrow,newcol) is an entry in the block */
                    ASSERT (newrow < k2) ;
                    newrow -= k1 ;
                    Ci [pc++] = newrow ;
                }
            }
        }
        Cp [nk] = pc ;
        maxnz = MAX (maxnz, pc) ;
        ASSERT (KLU_valid (nk, Cp, Ci, NULL)) ;

        /* ------------------------------------------------------------------ */
        /* order the block C */
        /* ------------------------------------------------------------------ */

        if (nk <= 3)
        {

            /* -------------------------------------------------------------- */
            /* use natural ordering for tiny blocks (3-by-3 or less) */
            /* -------------------------------------------------------------- */

            for (k = 0 ; k < nk ; k++)
            {
                Pblk [k] = k ;
            }
            lnz1 = nk * (nk + 1) / 2 ;
            flops1 = nk * (nk - 1) / 2 + (nk-1)*nk*(2*nk-1) / 6 ;
            ok = TRUE ;

        }
        else if (ordering == 0)
        {

            /* -------------------------------------------------------------- */
            /* order the block with AMD (C+C') */
            /* -------------------------------------------------------------- */

            result = AMD_order (nk, Cp, Ci, Pblk, NULL, amd_Info) ;
            ok = (result >= AMD_OK) ;
            if (result == AMD_OUT_OF_MEMORY)
            {
                err = KLU_OUT_OF_MEMORY ;
            }

            /* account for memory usage in AMD */
            Common->mempeak = MAX (Common->mempeak,
                Common->memusage + amd_Info [AMD_MEMORY]) ;

            /* get the ordering statistics from AMD */
            lnz1 = (Int) (amd_Info [AMD_LNZ]) + nk ;
            flops1 = 2 * amd_Info [AMD_NMULTSUBS_LU] + amd_Info [AMD_NDIV] ;
            if (pc == maxnz)
            {
                /* get the symmetry of the biggest block */
                Symbolic->symmetry = amd_Info [AMD_SYMMETRY] ;
            }

        }
        else if (ordering == 1)
        {

            /* -------------------------------------------------------------- */
            /* order the block with COLAMD (C) */
            /* -------------------------------------------------------------- */

            /* order (and destroy) Ci, returning column permutation in Cp.
             * COLAMD "cannot" fail since the matrix has already been checked,
             * and Ci allocated. */

            ok = COLAMD (nk, nk, Cilen, Ci, Cp, NULL, cstats) ;
            lnz1 = EMPTY ;
            flops1 = EMPTY ;

            /* copy the permutation from Cp to Pblk */
            for (k = 0 ; k < nk ; k++)
            {
                Pblk [k] = Cp [k] ;
            }

        }
        else
        {

            /* -------------------------------------------------------------- */
            /* pass the block to the user-provided ordering function */
            /* -------------------------------------------------------------- */

            lnz1 = (Common->user_order) (nk, Cp, Ci, Pblk, Common) ;
            flops1 = EMPTY ;
            ok = (lnz1 != 0) ;
        }

        if (!ok)
        {
            return (err) ;  /* ordering method failed */
        }

        /* ------------------------------------------------------------------ */
        /* keep track of nnz(L) and flops statistics */
        /* ------------------------------------------------------------------ */

        Lnz [block] = lnz1 ;
        lnz = (lnz == EMPTY || lnz1 == EMPTY) ? EMPTY : (lnz + lnz1) ;
        flops = (flops == EMPTY || flops1 == EMPTY) ? EMPTY : (flops + flops1) ;

        /* ------------------------------------------------------------------ */
        /* combine the preordering with the BTF ordering */
        /* ------------------------------------------------------------------ */

        PRINTF (("Pblk, 1-based:\n")) ;
        for (k = 0 ; k < nk ; k++)
        {
            ASSERT (k + k1 < n) ;
            ASSERT (Pblk [k] + k1 < n) ;
            Q [k + k1] = Qbtf [Pblk [k] + k1] ;
        }
        for (k = 0 ; k < nk ; k++)
        {
            ASSERT (k + k1 < n) ;
            ASSERT (Pblk [k] + k1 < n) ;
            P [k + k1] = Pbtf [Pblk [k] + k1] ;
        }
    }

    PRINTF (("nzoff %d  Ap[n] %d\n", nzoff, Ap [n])) ;
    ASSERT (nzoff >= 0 && nzoff <= Ap [n]) ;

    /* return estimates of # of nonzeros in L including diagonal */
    Symbolic->lnz = lnz ;           /* EMPTY if COLAMD used */
    Symbolic->unz = lnz ;
    Symbolic->nzoff = nzoff ;
    Symbolic->est_flops = flops ;   /* EMPTY if COLAMD or user-ordering used */
    return (KLU_OK) ;
}


/* ========================================================================== */
/* === order_and_analyze ==================================================== */
/* ========================================================================== */

/* Orders the matrix with or with BTF, then orders each block with AMD, COLAMD,
 * or the user ordering function.  Does not handle the natural or given
 * ordering cases. */

static KLU_symbolic *order_and_analyze  /* returns NULL if error, or a valid
                                           KLU_symbolic object if successful */
(
    /* inputs, not modified */
    Int n,              /* A is n-by-n */
    Int Ap [ ],         /* size n+1, column pointers */
    Int Ai [ ],         /* size nz, row indices */
    /* --------------------- */
    KLU_common *Common
)
{
    double work ;
    KLU_symbolic *Symbolic ;
    double *Lnz ;
    Int *Qbtf, *Cp, *Ci, *Pinv, *Pblk, *Pbtf, *P, *Q, *R ;
    Int nblocks, nz, block, maxblock, k1, k2, nk, do_btf, ordering, k, Cilen,
        *Work ;

    /* ---------------------------------------------------------------------- */
    /* allocate the Symbolic object, and check input matrix */
    /* ---------------------------------------------------------------------- */

    Symbolic = KLU_alloc_symbolic (n, Ap, Ai, Common) ;
    if (Symbolic == NULL)
    {
        return (NULL) ;
    }
    P = Symbolic->P ;
    Q = Symbolic->Q ;
    R = Symbolic->R ;
    Lnz = Symbolic->Lnz ;
    nz = Symbolic->nz ;

    ordering = Common->ordering ;
    if (ordering == 1)
    {
        /* COLAMD */
        Cilen = COLAMD_recommended (nz, n, n) ;
    }
    else if (ordering == 0 || (ordering == 3 && Common->user_order != NULL))
    {
        /* AMD or user ordering function */
        Cilen = nz+1 ;
    }
    else
    {
        /* invalid ordering */
        Common->status = KLU_INVALID ;
        KLU_free_symbolic (&Symbolic, Common) ;
        return (NULL) ;
    }

    /* ---------------------------------------------------------------------- */
    /* allocate workspace for BTF permutation */
    /* ---------------------------------------------------------------------- */

    Pbtf = KLU_malloc (n, sizeof (Int), Common) ;
    Qbtf = KLU_malloc (n, sizeof (Int), Common) ;
    if (Common->status < KLU_OK)
    {
        KLU_free (Pbtf, n, sizeof (Int), Common) ;
        KLU_free (Qbtf, n, sizeof (Int), Common) ;
        KLU_free_symbolic (&Symbolic, Common) ;
        return (NULL) ;
    }

    /* ---------------------------------------------------------------------- */
    /* get the common parameters for BTF and ordering method */
    /* ---------------------------------------------------------------------- */

    do_btf = Common->btf ;
    do_btf = (do_btf) ? TRUE : FALSE ;
    Symbolic->ordering = ordering ;
    Symbolic->do_btf = do_btf ;
    Symbolic->structural_rank = EMPTY ;

    /* ---------------------------------------------------------------------- */
    /* find the block triangular form (if requested) */
    /* ---------------------------------------------------------------------- */

    Common->work = 0 ;

    if (do_btf)
    {
        Work = KLU_malloc (5*n, sizeof (Int), Common) ;
        if (Common->status < KLU_OK)
        {
            /* out of memory */
            KLU_free (Pbtf, n, sizeof (Int), Common) ;
            KLU_free (Qbtf, n, sizeof (Int), Common) ;
            KLU_free_symbolic (&Symbolic, Common) ;
            return (NULL) ;
        }

        nblocks = BTF_order (n, Ap, Ai, Common->maxwork, &work, Pbtf, Qbtf, R,
                &(Symbolic->structural_rank), Work) ;
        Common->structural_rank = Symbolic->structural_rank ;
        Common->work += work ;

        KLU_free (Work, 5*n, sizeof (Int), Common) ;

        /* unflip Qbtf if the matrix does not have full structural rank */
        if (Symbolic->structural_rank < n)
        {
            for (k = 0 ; k < n ; k++)
            {
                Qbtf [k] = BTF_UNFLIP (Qbtf [k]) ;
            }
        }

        /* find the size of the largest block */
        maxblock = 1 ;
        for (block = 0 ; block < nblocks ; block++)
        {
            k1 = R [block] ;
            k2 = R [block+1] ;
            nk = k2 - k1 ;
            PRINTF (("block %d size %d\n", block, nk)) ;
            maxblock = MAX (maxblock, nk) ;
        }
    }
    else
    {
        /* BTF not requested */
        nblocks = 1 ;
        maxblock = n ;
        R [0] = 0 ;
        R [1] = n ;
        for (k = 0 ; k < n ; k++)
        {
            Pbtf [k] = k ;
            Qbtf [k] = k ;
        }
    }

    Symbolic->nblocks = nblocks ;

    PRINTF (("maxblock size %d\n", maxblock)) ;
    Symbolic->maxblock = maxblock ;

    /* ---------------------------------------------------------------------- */
    /* allocate more workspace, for analyze_worker */
    /* ---------------------------------------------------------------------- */

    Pblk = KLU_malloc (maxblock, sizeof (Int), Common) ;
    Cp   = KLU_malloc (maxblock + 1, sizeof (Int), Common) ;
    Ci   = KLU_malloc (MAX (Cilen, nz+1), sizeof (Int), Common) ;
    Pinv = KLU_malloc (n, sizeof (Int), Common) ;

    /* ---------------------------------------------------------------------- */
    /* order each block of the BTF ordering, and a fill-reducing ordering */
    /* ---------------------------------------------------------------------- */

    if (Common->status == KLU_OK)
    {
        PRINTF (("calling analyze_worker\n")) ;
        Common->status = analyze_worker (n, Ap, Ai, nblocks, Pbtf, Qbtf, R,
            ordering, P, Q, Lnz, Pblk, Cp, Ci, Cilen, Pinv, Symbolic, Common) ;
        PRINTF (("analyze_worker done\n")) ;
    }

    /* ---------------------------------------------------------------------- */
    /* free all workspace */
    /* ---------------------------------------------------------------------- */

    KLU_free (Pblk, maxblock, sizeof (Int), Common) ;
    KLU_free (Cp, maxblock+1, sizeof (Int), Common) ;
    KLU_free (Ci, MAX (Cilen, nz+1), sizeof (Int), Common) ;
    KLU_free (Pinv, n, sizeof (Int), Common) ;
    KLU_free (Pbtf, n, sizeof (Int), Common) ;
    KLU_free (Qbtf, n, sizeof (Int), Common) ;

    /* ---------------------------------------------------------------------- */
    /* return the symbolic object */
    /* ---------------------------------------------------------------------- */

    if (Common->status < KLU_OK)
    {
        KLU_free_symbolic (&Symbolic, Common) ;
    }
    return (Symbolic) ;
}


/* ========================================================================== */
/* === KLU_analyze ========================================================== */
/* ========================================================================== */

KLU_symbolic *KLU_analyze       /* returns NULL if error, or a valid
                                   KLU_symbolic object if successful */
(
    /* inputs, not modified */
    Int n,              /* A is n-by-n */
    Int Ap [ ],         /* size n+1, column pointers */
    Int Ai [ ],         /* size nz, row indices */
    /* -------------------- */
    KLU_common *Common
)
{

    /* ---------------------------------------------------------------------- */
    /* get the control parameters for BTF and ordering method */
    /* ---------------------------------------------------------------------- */

    if (Common == NULL)
    {
        return (NULL) ;
    }
    Common->status = KLU_OK ;
    Common->structural_rank = EMPTY ;

    /* ---------------------------------------------------------------------- */
    /* order and analyze */
    /* ---------------------------------------------------------------------- */

    if (Common->ordering == 2)
    {
        /* natural ordering */
        return (KLU_analyze_given (n, Ap, Ai, NULL, NULL, Common)) ;
    }
    else
    {
        /* order with P and Q */
        return (order_and_analyze (n, Ap, Ai, Common)) ;
    }
}
