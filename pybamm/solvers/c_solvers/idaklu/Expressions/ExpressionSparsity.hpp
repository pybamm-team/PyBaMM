#ifndef PYBAMM_EXPRESSION_SPARSITY_HPP
#define PYBAMM_EXPRESSION_SPARSITY_HPP

#include "ExpressionTypes.hpp"
#include <vector>

class ExpressionSparsity
{
public:
  ExpressionSparsity() = default;

  virtual expr_int nnz() = 0;
  virtual std::vector<expr_int> get_row() = 0;
  virtual std::vector<expr_int> get_col() = 0;
};

#endif // PYBAMM_EXPRESSION_SPARSITY_HPP
