/* ========================================================================== */
/* === MATLAB/sparse2 mexFunction =========================================== */
/* ========================================================================== */

/* Identical to the "sparse" function in MATLAB, just faster.  */

#include "cholmod_matlab.h"

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    sputil_sparse (nargout, pargout, nargin, pargin) ;
}
