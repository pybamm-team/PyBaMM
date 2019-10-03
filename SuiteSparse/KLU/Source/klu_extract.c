/* ========================================================================== */
/* === KLU_extract ========================================================== */
/* ========================================================================== */

/* Extract KLU factorization into conventional compressed-column matrices.
 * If any output array is NULL, that part of the LU factorization is not
 * extracted (this is not an error condition).
 *
 * nnz(L) = Numeric->lnz, nnz(U) = Numeric->unz, and nnz(F) = Numeric->Offp [n]
 */

#include "klu_internal.h"

Int KLU_extract     /* returns TRUE if successful, FALSE otherwise */
(
    /* inputs: */
    KLU_numeric *Numeric,
    KLU_symbolic *Symbolic,

    /* outputs, all of which must be allocated on input */

    /* L */
    Int *Lp,        /* size n+1 */
    Int *Li,        /* size nnz(L) */
    double *Lx,     /* size nnz(L) */
#ifdef COMPLEX
    double *Lz,     /* size nnz(L) for the complex case, ignored if real */
#endif

    /* U */
    Int *Up,        /* size n+1 */
    Int *Ui,        /* size nnz(U) */
    double *Ux,     /* size nnz(U) */
#ifdef COMPLEX
    double *Uz,     /* size nnz(U) for the complex case, ignored if real */
#endif

    /* F */
    Int *Fp,        /* size n+1 */
    Int *Fi,        /* size nnz(F) */
    double *Fx,     /* size nnz(F) */
#ifdef COMPLEX
    double *Fz,     /* size nnz(F) for the complex case, ignored if real */
#endif

    /* P, row permutation */
    Int *P,         /* size n */

    /* Q, column permutation */
    Int *Q,         /* size n */

    /* Rs, scale factors */
    double *Rs,     /* size n */

    /* R, block boundaries */
    Int *R,         /* size nblocks+1 */

    KLU_common *Common
)
{
    Int *Lip, *Llen, *Uip, *Ulen, *Li2, *Ui2 ;
    Unit *LU ;
    Entry *Lx2, *Ux2, *Ukk ;
    Int i, k, block, nblocks, n, nz, k1, k2, nk, len, kk, p ;

    if (Common == NULL)
    {
        return (FALSE) ;
    }

    if (Symbolic == NULL || Numeric == NULL)
    {
        Common->status = KLU_INVALID ;
        return (FALSE) ;
    }

    Common->status = KLU_OK ;
    n = Symbolic->n ;
    nblocks = Symbolic->nblocks ;

    /* ---------------------------------------------------------------------- */
    /* extract scale factors */
    /* ---------------------------------------------------------------------- */

    if (Rs != NULL)
    {
        if (Numeric->Rs != NULL)
        {
            for (i = 0 ; i < n ; i++)
            {
                Rs [i] = Numeric->Rs [i] ;
            }
        }
        else
        {
            /* no scaling */
            for (i = 0 ; i < n ; i++)
            {
                Rs [i] = 1 ;
            }
        }
    }

    /* ---------------------------------------------------------------------- */
    /* extract block boundaries */
    /* ---------------------------------------------------------------------- */

    if (R != NULL)
    {
        for (block = 0 ; block <= nblocks ; block++)
        {
            R [block] = Symbolic->R [block] ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* extract final row permutation */
    /* ---------------------------------------------------------------------- */

    if (P != NULL)
    {
        for (k = 0 ; k < n ; k++)
        {
            P [k] = Numeric->Pnum [k] ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* extract column permutation */
    /* ---------------------------------------------------------------------- */

    if (Q != NULL)
    {
        for (k = 0 ; k < n ; k++)
        {
            Q [k] = Symbolic->Q [k] ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* extract each block of L */
    /* ---------------------------------------------------------------------- */

    if (Lp != NULL && Li != NULL && Lx != NULL
#ifdef COMPLEX
        && Lz != NULL
#endif
    )
    {
        nz = 0 ;
        for (block = 0 ; block < nblocks ; block++)
        {
            k1 = Symbolic->R [block] ;
            k2 = Symbolic->R [block+1] ;
            nk = k2 - k1 ;
            if (nk == 1)
            {
                /* singleton block */
                Lp [k1] = nz ;
                Li [nz] = k1 ;
                Lx [nz] = 1 ;
#ifdef COMPLEX
                Lz [nz] = 0 ;
#endif
                nz++ ;
            }
            else
            {
                /* non-singleton block */
                LU = Numeric->LUbx [block] ;
                Lip = Numeric->Lip + k1 ;
                Llen = Numeric->Llen + k1 ;
                for (kk = 0 ; kk < nk ; kk++)
                {
                    Lp [k1+kk] = nz ;
                    /* add the unit diagonal entry */
                    Li [nz] = k1 + kk ;
                    Lx [nz] = 1 ;
#ifdef COMPLEX
                    Lz [nz] = 0 ;
#endif
                    nz++ ;
                    GET_POINTER (LU, Lip, Llen, Li2, Lx2, kk, len) ;
                    for (p = 0 ; p < len ; p++)
                    {
                        Li [nz] = k1 + Li2 [p] ;
                        Lx [nz] = REAL (Lx2 [p]) ;
#ifdef COMPLEX
                        Lz [nz] = IMAG (Lx2 [p]) ;
#endif
                        nz++ ;
                    }
                }
            }
        }
        Lp [n] = nz ;
        ASSERT (nz == Numeric->lnz) ;
    }

    /* ---------------------------------------------------------------------- */
    /* extract each block of U */
    /* ---------------------------------------------------------------------- */

    if (Up != NULL && Ui != NULL && Ux != NULL
#ifdef COMPLEX
        && Uz != NULL
#endif
    )
    {
        nz = 0 ;
        for (block = 0 ; block < nblocks ; block++)
        {
            k1 = Symbolic->R [block] ;
            k2 = Symbolic->R [block+1] ;
            nk = k2 - k1 ;
            Ukk = ((Entry *) Numeric->Udiag) + k1 ;
            if (nk == 1)
            {
                /* singleton block */
                Up [k1] = nz ;
                Ui [nz] = k1 ;
                Ux [nz] = REAL (Ukk [0]) ;
#ifdef COMPLEX
                Uz [nz] = IMAG (Ukk [0]) ;
#endif
                nz++ ;
            }
            else
            {
                /* non-singleton block */
                LU = Numeric->LUbx [block] ;
                Uip = Numeric->Uip + k1 ;
                Ulen = Numeric->Ulen + k1 ;
                for (kk = 0 ; kk < nk ; kk++)
                {
                    Up [k1+kk] = nz ;
                    GET_POINTER (LU, Uip, Ulen, Ui2, Ux2, kk, len) ;
                    for (p = 0 ; p < len ; p++)
                    {
                        Ui [nz] = k1 + Ui2 [p] ;
                        Ux [nz] = REAL (Ux2 [p]) ;
#ifdef COMPLEX
                        Uz [nz] = IMAG (Ux2 [p]) ;
#endif
                        nz++ ;
                    }
                    /* add the diagonal entry */
                    Ui [nz] = k1 + kk ;
                    Ux [nz] = REAL (Ukk [kk]) ;
#ifdef COMPLEX
                    Uz [nz] = IMAG (Ukk [kk]) ;
#endif
                    nz++ ;
                }
            }
        }
        Up [n] = nz ;
        ASSERT (nz == Numeric->unz) ;
    }

    /* ---------------------------------------------------------------------- */
    /* extract the off-diagonal blocks, F */
    /* ---------------------------------------------------------------------- */

    if (Fp != NULL && Fi != NULL && Fx != NULL
#ifdef COMPLEX
        && Fz != NULL
#endif
    )
    {
        for (k = 0 ; k <= n ; k++)
        {
            Fp [k] = Numeric->Offp [k] ;
        }
        nz = Fp [n] ;
        for (k = 0 ; k < nz ; k++)
        {
            Fi [k] = Numeric->Offi [k] ;
        }
        for (k = 0 ; k < nz ; k++)
        {
            Fx [k] = REAL (((Entry *) Numeric->Offx) [k]) ;
#ifdef COMPLEX
            Fz [k] = IMAG (((Entry *) Numeric->Offx) [k]) ;
#endif
        }
    }

    return (TRUE) ;
}
