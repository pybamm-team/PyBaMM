#ifndef PYBAMM_EXPRESSION_HPP
#define PYBAMM_EXPRESSION_HPP

#include "ExpressionTypes.hpp"
#include "../../common.hpp"
#include "../../Options.hpp"
#include <memory>
#include <vector>

class Expression {
public:  // method declarations
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
   * @brief Returns the maximum number of elements returned by the k'th output
   *
   * This is used to allocate memory for the output of the function and usual (but
   * not always) corresponds to the number of non-zero elements (NNZ).
   */
  virtual expr_int out_shape(int k) = 0;

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  virtual expr_int nnz() = 0;

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  virtual expr_int nnz_out() = 0;

  /**
   * @brief Returns the row vector of matrix element coordinates in COO format
   */
  virtual std::vector<expr_int> get_row() = 0;

  /**
   * @brief Returns the column vector of matrix element coordinates in COO format
   */
  virtual std::vector<expr_int> get_col() = 0;

public:  // data members
  /**
   * @brief Vector of pointers to the input data
   */
  std::vector<const double *> m_arg;

  /**
   * @brief Vector of pointers to the output data
   */
  std::vector<double *> m_res;
};

#endif // PYBAMM_EXPRESSION_HPP
