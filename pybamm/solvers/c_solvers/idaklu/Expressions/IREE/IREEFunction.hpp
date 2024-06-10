#ifndef PYBAMM_IDAKLU_IREE_FUNCTION_HPP
#define PYBAMM_IDAKLU_IREE_FUNCTION_HPP

#include "../../Options.hpp"
#include "../Expressions.hpp"
#include <memory>
#include "iree_jit.hpp"
#include "IREEBaseFunction.hpp"

/**
 * @brief Class for handling individual iree functions
 */
class IREEFunction : public Expression
{
public:
  typedef IREEBaseFunctionType BaseFunctionType;

  /*
   * @brief Constructor
   */
  explicit IREEFunction(const BaseFunctionType &f);

  // Method overrides
  void operator()() override;
  void operator()(const std::vector<realtype*>& inputs,
                  const std::vector<realtype*>& results) override;
  expr_int out_shape(int k) override;
  expr_int nnz() override;
  expr_int nnz_out() override;
  std::vector<expr_int> get_col() override;
  std::vector<expr_int> get_row() override;

  /*
   * @brief Evaluate the MLIR function
   */
  void evaluate();

  /*
   * @brief Evaluate the MLIR function
   * @param n_outputs The number of outputs to return
   */
  void evaluate(int n_outputs);

public:
  std::unique_ptr<IREESession> session;
  std::vector<std::vector<float>> result;
  std::vector<std::vector<int>> input_shape;
  std::vector<std::vector<int>> output_shape;
  std::vector<std::vector<float>> input_data;
  
  BaseFunctionType m_func;
  std::string module_name;
  std::string function_name;
  std::vector<int> m_arg_argno;
  std::vector<int> m_arg_argix;
  std::vector<int> numel;
};

#endif // PYBAMM_IDAKLU_IREE_FUNCTION_HPP
