/* ========================================================================== */
/* === KLU_dump ============================================================= */
/* ========================================================================== */

/* Debug routines for klu.  Only used when NDEBUG is not defined at
 * compile-time.
 */

#include "klu_internal.h"

#ifndef NDEBUG

/* ========================================================================== */
/* === KLU_valid ============================================================ */
/* ========================================================================== */

/* Check if a column-form matrix is valid or not.  The matrix A is
 * n-by-n.  The row indices of entries in column j are in
 * Ai [Ap [j] ... Ap [j+1]-1].  Required conditions are:
 *
 *      n >= 0
 *      nz = Ap [n_col] >= 0        number of entries in the matrix
 *      Ap [0] == 0
 *      Ap [j] <= Ap [j+1] for all j in the range 0 to n_col.
 *      row indices in Ai [Ap [j] ... Ap [j+1]-1]
 *          must be in the range 0 to n_row-1,
 *          and no duplicate entries can exist (duplicates not checked here).
 *
 * Not user-callable.  Only used when debugging.
 */

Int KLU_valid (Int n, Int Ap [ ], Int Ai [ ], Entry Ax [ ])
{
    Int nz, j, p1, p2, i, p ;
    PRINTF (("\ncolumn oriented matrix, n = %d\n", n)) ;
    if (n <= 0)
    {
        PRINTF (("n must be >= 0: %d\n", n)) ;
        return (FALSE) ;
    }
    nz = Ap [n] ;
    if (Ap [0] != 0 || nz < 0)
    {
        /* column pointers must start at Ap [0] = 0, and Ap [n] must be >= 0 */
        PRINTF (("column 0 pointer bad or nz < 0\n")) ;
        return (FALSE) ;
    }
    for (j = 0 ; j < n ; j++)
    {
        p1 = Ap [j] ;
        p2 = Ap [j+1] ;
        PRINTF (("\nColumn: %d p1: %d p2: %d\n", j, p1, p2)) ;
        if (p1 > p2)
        {
            /* column pointers must be ascending */
            PRINTF (("column %d pointer bad\n", j)) ;
            return (FALSE) ;
        }
        for (p = p1 ; p < p2 ; p++)
        {
            i = Ai [p] ;
            PRINTF (("row: %d", i)) ;
            if (i < 0 || i >= n)
            {
                /* row index out of range */
                PRINTF (("index out of range, col %d row %d\n", j, i)) ;
                return (FALSE) ;
            }
            if (Ax != (Entry *) NULL)
            {
                PRINT_ENTRY (Ax [p]) ;
            }
            PRINTF (("\n")) ;
        }
    }
    return (TRUE) ;
}


/* ========================================================================== */
/* === KLU_valid_LU ========================================================= */
/* ========================================================================== */

/* This function does the same validity tests as KLU_valid but for the
 * LU factor storage format. The flag flag_test_start_ptr is used to
 * test if Xip [0] = 0. This is not applicable for U. So when calling this
 * function for U, the flag should be set to false.  Only used when debugging.
 */

Int KLU_valid_LU (Int n, Int flag_test_start_ptr, Int Xip [ ],
                   Int Xlen [ ],  Unit LU [ ])
{
    Int *Xi ;
    Entry *Xx ;
    Int j, p1, p2, i, p, len ;

    PRINTF (("\ncolumn oriented matrix, n = %d\n", n)) ;
    if (n <= 0)
    {
        PRINTF (("n must be >= 0: %d\n", n)) ;
        return (FALSE) ;
    }
    if (flag_test_start_ptr && Xip [0] != 0)
    {
        /* column pointers must start at Xip [0] = 0*/
        PRINTF (("column 0 pointer bad\n")) ;
        return (FALSE) ;
    }

    for (j = 0 ; j < n ; j++)
    {
        p1 = Xip [j] ;
        PRINTF (("\nColumn of factor: %d p1: %d ", j, p1)) ;
        if (j < n-1)
        {
            p2 = Xip [j+1] ;
            PRINTF (("p2: %d ", p2)) ;
            if (p1 > p2)
            {
                /* column pointers must be ascending */
                PRINTF (("column %d pointer bad\n", j)) ;
                return (FALSE) ;
            }
        }
        PRINTF (("\n")) ;
        GET_POINTER (LU, Xip, Xlen, Xi, Xx, j, len) ;
        for (p = 0 ; p < len ; p++)
        {
            i = Xi [p] ;
            PRINTF (("row: %d", i)) ;
            if (i < 0 || i >= n)
            {
                /* row index out of range */
                PRINTF (("index out of range, col %d row %d\n", j, i)) ;
                return (FALSE) ;
            }
            if (Xx != (Entry *) NULL)
            {
                PRINT_ENTRY (Xx [p]) ;
            }
            PRINTF (("\n")) ;
        }
    }

    return (TRUE) ;
}
#endif
