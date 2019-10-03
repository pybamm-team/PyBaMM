
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
    const char* usage = "Usage: O = edgecut_options()";
    if(nargout > 1 || nargin != 0) mexErrMsgTxt(usage);

    EdgeCut_Options *ret = EdgeCut_Options::create();
    if(ret == NULL) mexErrMsgTxt("Out of memory");

    pargout[0] = mex_put_options(ret);
    ret->~EdgeCut_Options();
}
