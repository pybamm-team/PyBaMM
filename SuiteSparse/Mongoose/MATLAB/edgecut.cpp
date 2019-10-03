
#include "mongoose_mex.hpp"

using namespace Mongoose;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    cs Amatrix;
    int i, gtype, values;
    
    const char* usage = "Usage: partition = edgecut(G, (O, A))";
    if(nargout > 1 || nargin < 1 || nargin > 3) mexErrMsgTxt(usage);
    
    const mxArray *matGraph = pargin[0];
    const mxArray *matOptions = (nargin >= 2 ? pargin[1] : NULL);
    const mxArray *matNodeWeights = (nargin >= 3 ? pargin[2] : NULL);
    
    // Get the graph from the MATLAB inputs.
    EdgeCutProblem *G = mex_get_graph(matGraph, matNodeWeights);
    
    if(!G)
        mexErrMsgTxt("Unable to get Graph struct");
    
    // Get the options from the MATLAB inputs.
    EdgeCut_Options *O = mex_get_options(matOptions);

    if(!O)
        mexErrMsgTxt("Unable to get Options struct");

    EdgeCut *result = edge_cut(G, O);
    
    // Copy the partition choices back to MATLAB.
    pargout[0] = gp_mex_put_logical(result->partition, result->n);

    // Cleanup
    O->~EdgeCut_Options();
    G->~EdgeCutProblem();
    result->~EdgeCut();
}
