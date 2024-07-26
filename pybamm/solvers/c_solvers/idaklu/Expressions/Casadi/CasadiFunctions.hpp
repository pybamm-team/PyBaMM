#ifndef PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
#define PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP

#include "../../Options.hpp"
#include "../Expressions.hpp"
#include <casadi/casadi.hpp>
#include <casadi/core/function.hpp>
#include <casadi/core/sparsity.hpp>
#include <memory>

/**
 * @brief Class for handling individual casadi functions
 */
class CasadiFunction : public Expression
{
public:

  typedef casadi::Function BaseFunctionType;

  /**
   * @brief Constructor
   */
  explicit CasadiFunction(const BaseFunctionType &f);

  // Method overrides
  void operator()() override;
  void operator()(const std::vector<realtype*>& inputs,
                  const std::vector<realtype*>& results) override;
  expr_int out_shape(int k) override;
  expr_int nnz() override;
  expr_int nnz_out() override;
  std::vector<expr_int> get_row() override;
  std::vector<expr_int> get_row(expr_int ind);
  std::vector<expr_int> get_col() override;
  std::vector<expr_int> get_col(expr_int ind);

public:
  /*
   * @brief Casadi function
   */
  BaseFunctionType m_func;

private:
  std::vector<expr_int> m_iw;  // cppcheck-suppress unusedStructMember
  std::vector<double> m_w;  // cppcheck-suppress unusedStructMember
};

/**
 * @brief Class for handling casadi functions
 */
class CasadiFunctions : public ExpressionSet<CasadiFunction>
{
public:

  typedef CasadiFunction::BaseFunctionType BaseFunctionType;  // expose typedef in class

  /**
   * @brief Create a new CasadiFunctions object
   */
  CasadiFunctions(
    const BaseFunctionType &rhs_alg,
    const BaseFunctionType &jac_times_cjmass,
    const int jac_times_cjmass_nnz,
    const int jac_bandwidth_lower,
    const int jac_bandwidth_upper,
    const np_array_int &jac_times_cjmass_rowvals_arg,
    const np_array_int &jac_times_cjmass_colptrs_arg,
    const int inputs_length,
    const BaseFunctionType &jac_action,
    const BaseFunctionType &mass_action,
    const BaseFunctionType &sens,
    const BaseFunctionType &events,
    const int n_s,
    const int n_e,
    const int n_p,
    const std::vector<BaseFunctionType*>& var_fcns,
    const std::vector<BaseFunctionType*>& dvar_dy_fcns,
    const std::vector<BaseFunctionType*>& dvar_dp_fcns,
    const SetupOptions& setup_opts
  ) :
    rhs_alg_casadi(rhs_alg),
    jac_times_cjmass_casadi(jac_times_cjmass),
    jac_action_casadi(jac_action),
    mass_action_casadi(mass_action),
    sens_casadi(sens),
    events_casadi(events),
    ExpressionSet<CasadiFunction>(
      static_cast<Expression*>(&rhs_alg_casadi),
      static_cast<Expression*>(&jac_times_cjmass_casadi),
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      jac_times_cjmass_rowvals_arg,
      jac_times_cjmass_colptrs_arg,
      inputs_length,
      static_cast<Expression*>(&jac_action_casadi),
      static_cast<Expression*>(&mass_action_casadi),
      static_cast<Expression*>(&sens_casadi),
      static_cast<Expression*>(&events_casadi),
      n_s, n_e, n_p,
      setup_opts)
  {
    // convert BaseFunctionType list to CasadiFunction list
    // NOTE: You must allocate ALL std::vector elements before taking references
    for (auto& var : var_fcns)
      var_fcns_casadi.push_back(CasadiFunction(*var));
    for (int k = 0; k < var_fcns_casadi.size(); k++)
      ExpressionSet::var_fcns.push_back(&this->var_fcns_casadi[k]);

    for (auto& var : dvar_dy_fcns)
      dvar_dy_fcns_casadi.push_back(CasadiFunction(*var));
    for (int k = 0; k < dvar_dy_fcns_casadi.size(); k++)
      this->dvar_dy_fcns.push_back(&this->dvar_dy_fcns_casadi[k]);

    for (auto& var : dvar_dp_fcns)
      dvar_dp_fcns_casadi.push_back(CasadiFunction(*var));
    for (int k = 0; k < dvar_dp_fcns_casadi.size(); k++)
      this->dvar_dp_fcns.push_back(&this->dvar_dp_fcns_casadi[k]);

    // copy across numpy array values
    const int n_row_vals = jac_times_cjmass_rowvals_arg.request().size;
    auto p_jac_times_cjmass_rowvals = jac_times_cjmass_rowvals_arg.unchecked<1>();
    jac_times_cjmass_rowvals.resize(n_row_vals);
    for (int i = 0; i < n_row_vals; i++) {
      jac_times_cjmass_rowvals[i] = p_jac_times_cjmass_rowvals[i];
    }

    const int n_col_ptrs = jac_times_cjmass_colptrs_arg.request().size;
    auto p_jac_times_cjmass_colptrs = jac_times_cjmass_colptrs_arg.unchecked<1>();
    jac_times_cjmass_colptrs.resize(n_col_ptrs);
    for (int i = 0; i < n_col_ptrs; i++) {
      jac_times_cjmass_colptrs[i] = p_jac_times_cjmass_colptrs[i];
    }

    inputs.resize(inputs_length);
  }

  CasadiFunction rhs_alg_casadi;
  CasadiFunction jac_times_cjmass_casadi;
  CasadiFunction jac_action_casadi;
  CasadiFunction mass_action_casadi;
  CasadiFunction sens_casadi;
  CasadiFunction events_casadi;

  std::vector<CasadiFunction> var_fcns_casadi;
  std::vector<CasadiFunction> dvar_dy_fcns_casadi;
  std::vector<CasadiFunction> dvar_dp_fcns_casadi;

  realtype* get_tmp_state_vector() override {
    return tmp_state_vector.data();
  }
  realtype* get_tmp_sparse_jacobian_data() override {
    return tmp_sparse_jacobian_data.data();
  }
};

#endif // PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
