#include "common.hpp"

std::vector<realtype> numpy2realtype(const np_array& input_np) {
  std::vector<realtype> output(input_np.request().size);

  auto const inputData = input_np.unchecked<1>();
  for (int i = 0; i < output.size(); i++) {
    output[i] = inputData[i];
  }

  return output;
}



std::vector<realtype> makeSortedUnique(const np_array& input_np) {
    const auto input_vec = numpy2realtype(input_np);
    return makeSortedUnique(input_vec.begin(), input_vec.end());
}
