#include "mongoose_mex.hpp"

namespace Mongoose
{

EdgeCutProblem *mex_get_graph
(
    const mxArray *Gmatlab, /* The sparse matrix            */
    const mxArray *Amatlab  /* The real-valued vertex weights */
)
{
    // Check for valid sparse matrix
    cs_mex_check (0, -1, -1, 1, 1, 1, Gmatlab) ;

    Int n = mxGetN(Gmatlab);
    Int *Gp = (Int*) mxGetJc(Gmatlab);
    Int *Gi = (Int*) mxGetIr(Gmatlab);
    double *Gx = (double*) mxGetPr(Gmatlab);
    double *Gw = (Amatlab) ? (double*) mxGetPr(Amatlab) : NULL;
    Int nz = Gp[n];
    
    EdgeCutProblem *returner = EdgeCutProblem::create(n, nz, Gp, Gi, Gx, Gw);

    if (!returner)
    {
        return NULL;
    }

    return returner;
}

}
