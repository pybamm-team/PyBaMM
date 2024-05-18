#ifndef PYBAMM_EXPRESSION_HPP
#define PYBAMM_EXPRESSION_HPP

#include "ExpressionTypes.hpp"
#include "../../common.hpp"
#include "../../Options.hpp"
#include <memory>
#include <vector>

class Expression {
public:
  /**
   * @brief Constructor
   */
  Expression() = default;

  /**
   * @brief Evaluation operator
   */
  virtual void operator()() = 0;

  /**
   * @brief Evaluation operator given data vectors
   */
  virtual void operator()(
    const std::vector<realtype*>& inputs,
    const std::vector<realtype*>& results) = 0;

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  virtual expr_int nnz() = 0;
  virtual expr_int nnz_out() = 0;
  virtual std::vector<expr_int> get_row() = 0;
  virtual std::vector<expr_int> get_col() = 0;

public:
  std::vector<const double *> m_arg;
  std::vector<double *> m_res;

//private:
  std::vector<expr_int> m_iw;
  std::vector<double> m_w;
};

#endif // PYBAMM_EXPRESSION_HPP
