#include "common.hpp"

std::vector<realtype> numpy2realtype(const np_array& input_np) {
  std::vector<realtype> output(input_np.request().size);

  auto const inputData = input_np.unchecked<1>();
  for (int i = 0; i < output.size(); i++) {
    output[i] = inputData[i];
  }

  return output;
}

std::vector<realtype> setDiff<T>(const T a_begin, const T a_end, const T b_begin, const T b_end) {
    std::vector<realtype> result;
    if (std::distance(a_begin, a_end) > 0) {
      std::set_difference(a_begin, a_end, b_begin, b_end, std::back_inserter(result));
    }
    return result;
}

std::vector<realtype> makeSortedUnique<T>(const T input_begin, const T input_end) {
    std::unordered_set<realtype> uniqueSet(input_begin, input_end); // Remove duplicates
    std::vector<realtype> uniqueVector(uniqueSet.begin(), uniqueSet.end()); // Convert to vector
    std::sort(uniqueVector.begin(), uniqueVector.end()); // Sort the vector
    return uniqueVector;
}

std::vector<realtype> makeSortedUnique(const np_array& input_np) {
    return makeSortedUnique(numpy2realtype(input_np));
}
