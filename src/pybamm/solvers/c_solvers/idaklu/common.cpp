#include "common.hpp"
#include <iostream>

#include <idas/idas.h>                 /* prototypes for IDAS fcts., consts.    */
#include <idas/idas_bbdpre.h>         /* access to IDABBDPRE preconditioner          */

#include <nvector/nvector_serial.h>  /* access to serial N_Vector            */
#include <nvector/nvector_openmp.h>  /* access to openmp N_Vector            */
#include <sundials/sundials_math.h>  /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_config.h>  /* defs. of SUNRabs, SUNRexp, etc.      */
#include <sundials/sundials_types.h> /* defs. of realtype, sunindextype      */

#if SUNDIALS_VERSION_MAJOR >= 6
  #include <sundials/sundials_context.h>
#endif

#include <sunlinsol/sunlinsol_klu.h> /* access to KLU linear solver          */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense linear solver          */
#include <sunlinsol/sunlinsol_band.h> /* access to dense linear solver          */
#include <sunlinsol/sunlinsol_spbcgs.h> /* access to spbcgs iterative linear solver          */
#include <sunlinsol/sunlinsol_spfgmr.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_sptfqmr.h>

#include <sunmatrix/sunmatrix_sparse.h> /* access to sparse SUNMatrix           */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix           */

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

std::vector<realtype> numpy2realtype(const np_array& input_np) {
  std::vector<realtype> output(input_np.request().size);

  auto const inputData = input_np.unchecked<1>();
  for (int i = 0; i < output.size(); i++) {
    output[i] = inputData[i];
  }

  return output;
}

std::vector<realtype> setDiff(const std::vector<realtype>& A, const std::vector<realtype>& B) {
    std::vector<realtype> result;
    if (!(A.empty())) {
      std::set_difference(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(result));
    }
    return result;
}

std::vector<realtype> makeSortedUnique(const std::vector<realtype>& input) {
    std::unordered_set<realtype> uniqueSet(input.begin(), input.end()); // Remove duplicates
    std::vector<realtype> uniqueVector(uniqueSet.begin(), uniqueSet.end()); // Convert to vector
    std::sort(uniqueVector.begin(), uniqueVector.end()); // Sort the vector
    return uniqueVector;
}

std::vector<realtype> makeSortedUnique(const np_array& input_np) {
    return makeSortedUnique(numpy2realtype(input_np));
}
