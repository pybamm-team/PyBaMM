#ifndef PYBAMM_EXPRESSION_HPP
#define PYBAMM_EXPRESSION_HPP

#include "common.hpp"
#include <vector>

/**
 * @brief Base class for handling individual expressions
 */
class Expression
{
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
  virtual void operator()(const std::vector<realtype*>& inputs,
                          const std::vector<realtype*>& results) = 0;
};

#endif // PYBAMM_EXPRESSION_HPP
