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
