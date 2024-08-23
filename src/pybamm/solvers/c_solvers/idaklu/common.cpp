#include "common.hpp"

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
