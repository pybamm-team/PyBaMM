#ifndef PYBAMM_IDAKLU_IREE_BASE_FUNCTION_HPP
#define PYBAMM_IDAKLU_IREE_BASE_FUNCTION_HPP

#include <string>
#include <vector>

/*
 * @brief Function definition passed from PyBaMM
 */
class IREEBaseFunctionType
{
public:  // methods
  const std::string& get_mlir() const { return mlir; }

public:  // data members
  std::string mlir;  // cppcheck-suppress unusedStructMember
  std::vector<int> kept_var_idx;  // cppcheck-suppress unusedStructMember
  expr_int nnz;  // cppcheck-suppress unusedStructMember
  expr_int numel;  // cppcheck-suppress unusedStructMember
  std::vector<expr_int> col;  // cppcheck-suppress unusedStructMember
  std::vector<expr_int> row;  // cppcheck-suppress unusedStructMember
  std::vector<int> pytree_shape;  // cppcheck-suppress unusedStructMember
  std::vector<int> pytree_sizes;  // cppcheck-suppress unusedStructMember
  expr_int n_args;  // cppcheck-suppress unusedStructMember
};

#endif // PYBAMM_IDAKLU_IREE_BASE_FUNCTION_HPP
