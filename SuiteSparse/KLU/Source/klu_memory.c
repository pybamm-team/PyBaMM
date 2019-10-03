/* ========================================================================== */
/* === KLU_memory =========================================================== */
/* ========================================================================== */

/* KLU memory management routines:
 *
 * KLU_malloc                   malloc wrapper
 * KLU_free                     free wrapper
 * KLU_realloc                  realloc wrapper
 */

#include "klu_internal.h"

/* ========================================================================== */
/* === KLU_add_size_t ======================================================= */
/* ========================================================================== */

/* Safely compute a+b, and check for size_t overflow */

size_t KLU_add_size_t (size_t a, size_t b, Int *ok)
{
    (*ok) = (*ok) && ((a + b) >= MAX (a,b)) ;
    return ((*ok) ? (a + b) : ((size_t) -1)) ;
}

/* ========================================================================== */
/* === KLU_mult_size_t ====================================================== */
/* ========================================================================== */

/* Safely compute a*k, where k should be small, and check for size_t overflow */

size_t KLU_mult_size_t (size_t a, size_t k, Int *ok)
{
    size_t i, s = 0 ;
    for (i = 0 ; i < k ; i++)
    {
        s = KLU_add_size_t (s, a, ok) ;
    }
    return ((*ok) ? s : ((size_t) -1)) ;
}

/* ========================================================================== */
/* === KLU_malloc =========================================================== */
/* ========================================================================== */

/* Wrapper around malloc routine (mxMalloc for a mexFunction).  Allocates
 * space of size MAX(1,n)*size, where size is normally a sizeof (...).
 *
 * This routine and KLU_realloc do not set Common->status to KLU_OK on success,
 * so that a sequence of KLU_malloc's or KLU_realloc's can be used.  If any of
 * them fails, the Common->status will hold the most recent error status.
 *
 * Usage, for a pointer to Int:
 *
 *      p = KLU_malloc (n, sizeof (Int), Common)
 *
 * Uses a pointer to the malloc routine (or its equivalent) defined in Common.
 */

void *KLU_malloc        /* returns pointer to the newly malloc'd block */
(
    /* ---- input ---- */
    size_t n,           /* number of items */
    size_t size,        /* size of each item */
    /* --------------- */
    KLU_common *Common
)
{
    void *p ;

    if (Common == NULL)
    {
        p = NULL ;
    }
    else if (size == 0)
    {
        /* size must be > 0 */
        Common->status = KLU_INVALID ;
        p = NULL ;
    }
    else if (n >= Int_MAX)
    {
        /* object is too big to allocate; p[i] where i is an Int will not
         * be enough. */
        Common->status = KLU_TOO_LARGE ;
        p = NULL ;
    }
    else
    {
        /* call malloc, or its equivalent */
        p = SuiteSparse_malloc (n, size) ;
        if (p == NULL)
        {
            /* failure: out of memory */
            Common->status = KLU_OUT_OF_MEMORY ;
        }
        else
        {
            Common->memusage += (MAX (1,n) * size) ;
            Common->mempeak = MAX (Common->mempeak, Common->memusage) ;
        }
    }
    return (p) ;
}


/* ========================================================================== */
/* === KLU_free ============================================================= */
/* ========================================================================== */

/* Wrapper around free routine (mxFree for a mexFunction).  Returns NULL,
 * which can be assigned to the pointer being freed, as in:
 *
 *      p = KLU_free (p, n, sizeof (int), Common) ;
 */

void *KLU_free          /* always returns NULL */
(
    /* ---- in/out --- */
    void *p,            /* block of memory to free */
    /* ---- input --- */
    size_t n,           /* size of block to free, in # of items */
    size_t size,        /* size of each item */
    /* --------------- */
    KLU_common *Common
)
{
    if (p != NULL && Common != NULL)
    {
        /* only free the object if the pointer is not NULL */
        /* call free, or its equivalent */
        SuiteSparse_free (p) ;
        Common->memusage -= (MAX (1,n) * size) ;
    }
    /* return NULL, and the caller should assign this to p.  This avoids
     * freeing the same pointer twice. */
    return (NULL) ;
}


/* ========================================================================== */
/* === KLU_realloc ========================================================== */
/* ========================================================================== */

/* Wrapper around realloc routine (mxRealloc for a mexFunction).  Given a
 * pointer p to a block allocated by KLU_malloc, it changes the size of the
 * block pointed to by p to be MAX(1,nnew)*size in size.  It may return a
 * pointer different than p.  This should be used as (for a pointer to Int):
 *
 *      p = KLU_realloc (nnew, nold, sizeof (Int), p, Common) ;
 *
 * If p is NULL, this is the same as p = KLU_malloc (...).
 * A size of nnew=0 is treated as nnew=1.
 *
 * If the realloc fails, p is returned unchanged and Common->status is set
 * to KLU_OUT_OF_MEMORY.  If successful, Common->status is not modified,
 * and p is returned (possibly changed) and pointing to a large block of memory.
 *
 * Uses a pointer to the realloc routine (or its equivalent) defined in Common.
 */

void *KLU_realloc       /* returns pointer to reallocated block */
(
    /* ---- input ---- */
    size_t nnew,        /* requested # of items in reallocated block */
    size_t nold,        /* old # of items */
    size_t size,        /* size of each item */
    /* ---- in/out --- */
    void *p,            /* block of memory to realloc */
    /* --------------- */
    KLU_common *Common
)
{
    void *pnew ;
    int ok = TRUE ;

    if (Common == NULL)
    {
        p = NULL ;
    }
    else if (size == 0)
    {
        /* size must be > 0 */
        Common->status = KLU_INVALID ;
        p = NULL ;
    }
    else if (p == NULL)
    {
        /* A fresh object is being allocated. */
        p = KLU_malloc (nnew, size, Common) ;
    }
    else if (nnew >= Int_MAX)
    {
        /* failure: nnew is too big.  Do not change p */
        Common->status = KLU_TOO_LARGE ;
    }
    else
    {
        /* The object exists, and is changing to some other nonzero size. */
        /* call realloc, or its equivalent */
        pnew = SuiteSparse_realloc (nnew, nold, size, p, &ok) ;
        if (ok)
        {
            /* success: return the new p and change the size of the block */
            Common->memusage += ((nnew-nold) * size) ;
            Common->mempeak = MAX (Common->mempeak, Common->memusage) ;
            p = pnew ;
        }
        else
        {
            /* Do not change p, since it still points to allocated memory */
            Common->status = KLU_OUT_OF_MEMORY ;
        }
    }
    return (p) ;
}
