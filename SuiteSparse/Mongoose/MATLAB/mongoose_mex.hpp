
#include "mex.h"

#include "Mongoose_Coarsening.hpp"
#include "Mongoose_CSparse.hpp"
#include "Mongoose_EdgeCut.hpp"
#include "Mongoose_EdgeCutOptions.hpp"
#include "Mongoose_EdgeCutProblem.hpp"
#include "Mongoose_Graph.hpp"
#include "Mongoose_Internal.hpp"
#include "Mongoose_Matching.hpp"
#include <algorithm>

namespace Mongoose
{

void shcpDataToMAT
(
    mxArray* matStruct,
    const char* field,
    mxClassID classID,
    void* data,
    size_t size
);

void addFieldWithValue
(
    mxArray* matStruct,     /* The mxArray assumed to be a matlab structure. */
    const char* fieldname,  /* The name of the field to create.              */
    const double value      /* The double value to assign to the new field.  */
);

double readField
(
    const mxArray* matStruct,
    const char* fieldname
);

EdgeCut_Options *mex_get_options(const mxArray *Omatlab = NULL);
mxArray *mex_put_options(const EdgeCut_Options *O);

/* check MATLAB input argument */
void cs_mex_check (csi nel, csi m, csi n, csi square, csi sparse, csi values,
    const mxArray *A);
/* get a MATLAB sparse matrix and convert to cs */
cs *cs_mex_get_sparse (cs *A, csi square, csi values, const mxArray *Amatlab);
/* return a sparse matrix to MATLAB */
mxArray *cs_mex_put_sparse (cs **Ahandle);

EdgeCutProblem *mex_get_graph
(
    const mxArray *Gmatlab,        /* The sparse matrix              */
    const mxArray *Amatlab = NULL  /* The real-valued vertex weights */
);

Int *gp_mex_get_int
(
    Int n,
    const mxArray *Imatlab,
    Int *imax,
    Int lo
);

mxArray *gp_mex_put_int(Int *p, Int n, Int offset, Int do_free);
mxArray *gp_mex_put_logical(bool *p, Int n);
double *gp_mex_get_double (Int n, const mxArray *X) ;
double *gp_mex_put_double (Int n, const double *b, mxArray **X) ;

}