#include "common.hpp"

std::vector<sunrealtype> numpy2sunrealtype(const np_array& input_np) {
  std::vector<sunrealtype> output(input_np.request().size);

  auto const inputData = input_np.unchecked<1>();
  for (int i = 0; i < output.size(); i++) {
    output[i] = inputData[i];
  }

  return output;
}



std::vector<sunrealtype> makeSortedUnique(const np_array& input_np) {
    const auto input_vec = numpy2sunrealtype(input_np);
    return makeSortedUnique(input_vec.begin(), input_vec.end());
}

void configure_klu(SUNLinearSolver LS) {
  // AMD (0) instead of the SUNDIALS default COLAMD (1)
  SUNLinSol_KLUSetOrdering(LS, 0);
  // KLU's default tolerance of 0.001 accepts pivots weak enough under AMD to
  // stall the Newton corrector; 0.1 rejects them
  SUNLinSol_KLUGetCommon(LS)->tol = 0.1;
}
