#ifndef PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
#define PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP

#include "../../Options.hpp"
#include "../Expressions.hpp"
#include <casadi/casadi.hpp>
#include <casadi/core/function.hpp>
#include <casadi/core/sparsity.hpp>
#include <memory>

class CasadiSparsity : public ExpressionSparsity
{
public:
  CasadiSparsity() = default;

  expr_int nnz() override { return _nnz; }
  std::vector<expr_int> get_row() override { return _get_row; }
  std::vector<expr_int> get_col() override { return _get_col; }
  
  expr_int _nnz = 0;
  std::vector<expr_int> _get_row;
  std::vector<expr_int> _get_col;
};

/**
 * @brief Class for handling individual casadi functions
 */
class CasadiFunction : public Expression
{
public:
  /**
   * @brief Constructor
   */
  explicit CasadiFunction(const casadi::Function &f);
  
  void operator()() override;

  void operator()(const std::vector<realtype*>& inputs,
                  const std::vector<realtype*>& results) override;

  casadi::Function m_func;

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  expr_int nnz_out() override;

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  ExpressionSparsity *sparsity_out(casadi_int ind) override;
};

/**
 * @brief Class for handling casadi functions
 */
class CasadiFunctions : public ExpressionSet<CasadiFunction>
{
public:
  /**
   * @brief Create a new CasadiFunctions object
   */
  CasadiFunctions(
    const casadi::Function &rhs_alg,
    const casadi::Function &jac_times_cjmass,
    const int jac_times_cjmass_nnz,
    const int jac_bandwidth_lower,
    const int jac_bandwidth_upper,
    const np_array_int &jac_times_cjmass_rowvals_arg,
    const np_array_int &jac_times_cjmass_colptrs_arg,
    const int inputs_length,
    const casadi::Function &jac_action,
    const casadi::Function &mass_action,
    const casadi::Function &sens,
    const casadi::Function &events,
    const int n_s,
    const int n_e,
    const int n_p,
    const std::vector<casadi::Function*>& var_fcns,
    const std::vector<casadi::Function*>& dvar_dy_fcns,
    const std::vector<casadi::Function*>& dvar_dp_fcns,
    const Options& options
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
      options)
  {
    // convert casadi::Function list to CasadiFunction list
    for (auto& var : var_fcns) {
      this->var_fcns.push_back(CasadiFunction(*var));
    }
    for (auto& var : dvar_dy_fcns) {
      this->dvar_dy_fcns.push_back(CasadiFunction(*var));
    }
    for (auto& var : dvar_dp_fcns) {
      this->dvar_dp_fcns.push_back(CasadiFunction(*var));
    }

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

  realtype* get_tmp_state_vector() override {
    return tmp_state_vector.data();
  }
  realtype* get_tmp_sparse_jacobian_data() override {
    return tmp_sparse_jacobian_data.data();
  }
};

#endif // PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
