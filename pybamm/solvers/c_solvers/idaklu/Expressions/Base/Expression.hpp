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
   * @brief Evaluation operator (for use after setting input and output data references)
   */
  virtual void operator()() = 0;

  /**
   * @brief Evaluation operator (supplying data references)
   */
  virtual void operator()(
    const std::vector<realtype*>& inputs,
    const std::vector<realtype*>& results) = 0;

  /**
   * @brief The maximum number of elements returned by the k'th output
   *
   * This is used to allocate memory for the output of the function and usually (but
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
   * @brief Returns row indices in COO format (where the output data represents sparse matrix elements)
   */
  virtual std::vector<expr_int> get_row() = 0;

  /**
   * @brief Returns column indices in COO format (where the output data represents sparse matrix elements)
   */
  virtual std::vector<expr_int> get_col() = 0;

public:  // data members
  /**
   * @brief Vector of pointers to the input data
   */
  std::vector<const double *> m_arg;  // cppcheck-suppress unusedStructMember

  /**
   * @brief Vector of pointers to the output data
   */
  std::vector<double *> m_res;  // cppcheck-suppress unusedStructMember
};

#endif // PYBAMM_EXPRESSION_HPP
