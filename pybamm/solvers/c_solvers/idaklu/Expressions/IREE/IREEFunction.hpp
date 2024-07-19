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
  std::vector<std::vector<float>> result;  // cppcheck-suppress unusedStructMember
  std::vector<std::vector<int>> input_shape;  // cppcheck-suppress unusedStructMember
  std::vector<std::vector<int>> output_shape;  // cppcheck-suppress unusedStructMember
  std::vector<std::vector<float>> input_data;  // cppcheck-suppress unusedStructMember

  BaseFunctionType m_func;  // cppcheck-suppress unusedStructMember
  std::string module_name;  // cppcheck-suppress unusedStructMember
  std::string function_name;  // cppcheck-suppress unusedStructMember
  std::vector<int> m_arg_argno;  // cppcheck-suppress unusedStructMember
  std::vector<int> m_arg_argix;  // cppcheck-suppress unusedStructMember
  std::vector<int> numel;  // cppcheck-suppress unusedStructMember
};

#endif // PYBAMM_IDAKLU_IREE_FUNCTION_HPP
