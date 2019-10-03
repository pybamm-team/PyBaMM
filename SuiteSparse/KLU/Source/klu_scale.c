/* ========================================================================== */
/* === KLU_scale ============================================================ */
/* ========================================================================== */

/* Scale a matrix and check to see if it is valid.  Can be called by the user.
 * This is called by KLU_factor and KLU_refactor.  Returns TRUE if the input
 * matrix is valid, FALSE otherwise.  If the W input argument is non-NULL,
 * then the input matrix is checked for duplicate entries.
 *
 * scaling methods:
 *      <0: no scaling, do not compute Rs, and do not check input matrix.
 *      0: no scaling
 *      1: the scale factor for row i is sum (abs (A (i,:)))
 *      2 or more: the scale factor for row i is max (abs (A (i,:)))
 */

#include "klu_internal.h"

Int KLU_scale           /* return TRUE if successful, FALSE otherwise */
(
    /* inputs, not modified */
    Int scale,          /* 0: none, 1: sum, 2: max */
    Int n,
    Int Ap [ ],         /* size n+1, column pointers */
    Int Ai [ ],         /* size nz, row indices */
    double Ax [ ],
    /* outputs, not defined on input */
    double Rs [ ],      /* size n, can be NULL if scale <= 0 */
    /* workspace, not defined on input or output */
    Int W [ ],          /* size n, can be NULL */
    /* --------------- */
    KLU_common *Common
)
{
    double a ;
    Entry *Az ;
    Int row, col, p, pend, check_duplicates ;

    /* ---------------------------------------------------------------------- */
    /* check inputs */
    /* ---------------------------------------------------------------------- */

    if (Common == NULL)
    {
        return (FALSE) ;
    }
    Common->status = KLU_OK ;

    if (scale < 0)
    {
        /* return without checking anything and without computing the
         * scale factors */
        return (TRUE) ;
    }

    Az = (Entry *) Ax ;

    if (n <= 0 || Ap == NULL || Ai == NULL || Az == NULL ||
        (scale > 0 && Rs == NULL))
    {
        /* Ap, Ai, Ax and Rs must be present, and n must be > 0 */
        Common->status = KLU_INVALID ;
        return (FALSE) ;
    }
    if (Ap [0] != 0 || Ap [n] < 0)
    {
        /* nz = Ap [n] must be >= 0 and Ap [0] must equal zero */
        Common->status = KLU_INVALID ;
        return (FALSE) ;
    }
    for (col = 0 ; col < n ; col++)
    {
        if (Ap [col] > Ap [col+1])
        {
            /* column pointers must be non-decreasing */
            Common->status = KLU_INVALID ;
            return (FALSE) ;
        }
    }

    /* ---------------------------------------------------------------------- */
    /* scale */
    /* ---------------------------------------------------------------------- */

    if (scale > 0)
    {
        /* initialize row sum or row max */
        for (row = 0 ; row < n ; row++)
        {
            Rs [row] = 0 ;
        }
    }

    /* check for duplicates only if W is present */
    check_duplicates = (W != (Int *) NULL) ;
    if (check_duplicates)
    {
        for (row = 0 ; row < n ; row++)
        {
            W [row] = EMPTY ;
        }
    }

    for (col = 0 ; col < n ; col++)
    {
        pend = Ap [col+1] ;
        for (p = Ap [col] ; p < pend ; p++)
        {
            row = Ai [p] ;
            if (row < 0 || row >= n)
            {
                /* row index out of range, or duplicate entry */
                Common->status = KLU_INVALID ;
                return (FALSE) ;
            }
            if (check_duplicates)
            {
                if (W [row] == col)
                {
                    /* duplicate entry */
                    Common->status = KLU_INVALID ;
                    return (FALSE) ;
                }
                /* flag row i as appearing in column col */
                W [row] = col ;
            }
            /* a = ABS (Az [p]) ;*/
            ABS (a, Az [p]) ;
            if (scale == 1)
            {
                /* accumulate the abs. row sum */
                Rs [row] += a ;
            }
            else if (scale > 1)
            {
                /* find the max abs. value in the row */
                Rs [row] = MAX (Rs [row], a) ;
            }
        }
    }

    if (scale > 0)
    {
        /* do not scale empty rows */
        for (row = 0 ; row < n ; row++)
        {
            /* matrix is singular */
            PRINTF (("Rs [%d] = %g\n", row, Rs [row])) ;

            if (Rs [row] == 0.0)
            {
                PRINTF (("Row %d of A is all zero\n", row)) ;
                Rs [row] = 1.0 ;
            }
        }
    }

    return (TRUE) ;
}
