/* ========================================================================== */
/* === KLU_free_symbolic ==================================================== */
/* ========================================================================== */

/* Free the KLU Symbolic object. */

#include "klu_internal.h"

Int KLU_free_symbolic
(
    KLU_symbolic **SymbolicHandle,
    KLU_common   *Common
)
{
    KLU_symbolic *Symbolic ;
    Int n ;
    if (Common == NULL)
    {
        return (FALSE) ;
    }
    if (SymbolicHandle == NULL || *SymbolicHandle == NULL)
    {
        return (TRUE) ;
    }
    Symbolic = *SymbolicHandle ;
    n = Symbolic->n ;
    KLU_free (Symbolic->P, n, sizeof (Int), Common) ;
    KLU_free (Symbolic->Q, n, sizeof (Int), Common) ;
    KLU_free (Symbolic->R, n+1, sizeof (Int), Common) ;
    KLU_free (Symbolic->Lnz, n, sizeof (double), Common) ;
    KLU_free (Symbolic, 1, sizeof (KLU_symbolic), Common) ;
    *SymbolicHandle = NULL ;
    return (TRUE) ;
}
