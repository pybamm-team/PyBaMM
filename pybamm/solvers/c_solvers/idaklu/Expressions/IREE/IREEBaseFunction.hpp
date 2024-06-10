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
  std::string mlir;
  std::vector<int> kept_var_idx;
  expr_int nnz;
  std::vector<expr_int> col;
  std::vector<expr_int> row;
  std::vector<int> pytree_shape;
  std::vector<int> pytree_sizes;
  expr_int n_args;
};

#endif // PYBAMM_IDAKLU_IREE_BASE_FUNCTION_HPP
