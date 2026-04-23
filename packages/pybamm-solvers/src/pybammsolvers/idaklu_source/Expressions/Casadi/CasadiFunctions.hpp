#ifndef PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
#define PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP

#include "../../Options.hpp"
#include "../Expressions.hpp"
#include <casadi/casadi.hpp>
#include <casadi/core/function.hpp>
#include <stdexcept>
#include <string>

struct SerializedCasadiFunctions {
  std::string rhs_alg;
  std::string jac_times_cjmass;
  std::string jac_action;
  std::string mass_action;
  std::string sens;
  std::string events;
  std::vector<std::string> var_fcns;
  std::vector<std::string> dvar_dy_fcns;
  std::vector<std::string> dvar_dp_fcns;
};

SerializedCasadiFunctions serialize_casadi_functions(
  const casadi::Function &rhs_alg,
  const casadi::Function &jac_times_cjmass,
  const casadi::Function &jac_action,
  const casadi::Function &mass_action,
  const casadi::Function &sens,
  const casadi::Function &events,
  const std::vector<casadi::Function*>& var_fcns,
  const std::vector<casadi::Function*>& dvar_dy_fcns,
  const std::vector<casadi::Function*>& dvar_dp_fcns
);

/**
 * @brief Class for handling individual casadi functions
 */
class CasadiFunction : public Expression
{
public:

  typedef casadi::Function BaseFunctionType;

  /**
   * @brief Constructor
   * @param f The CasADi function to wrap
   * @param deep_copy If true, creates an independent copy via serialize/deserialize
   *                  to avoid thread-unsafe shared state in parallel execution
   */
  explicit CasadiFunction(
    const BaseFunctionType &f,
    bool deep_copy = false,
    const std::string *serialized = nullptr
  );

  // Method overrides
  void operator()() override;
  void operator()(const std::vector<sunrealtype*>& inputs,
                  const std::vector<sunrealtype*>& results) override;
  expr_int out_shape(int k) override;
  expr_int nnz() override;
  expr_int nnz_out() override;
  const std::vector<expr_int>& get_row() override;
  const std::vector<expr_int>& get_col() override;

public:
  /*
   * @brief Casadi function
   */
  BaseFunctionType m_func;

private:
  std::vector<expr_int> m_iw;  // cppcheck-suppress unusedStructMember
  std::vector<double> m_w;  // cppcheck-suppress unusedStructMember
  std::vector<expr_int> m_rows;  // cppcheck-suppress unusedStructMember
  std::vector<expr_int> m_cols;  // cppcheck-suppress unusedStructMember
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
    const SetupOptions& setup_opts,
    const SerializedCasadiFunctions *serialized_fcns = nullptr
  ) :
    rhs_alg_casadi(
      rhs_alg,
      setup_opts.num_solvers > 1,
      serialized_fcns != nullptr ? &serialized_fcns->rhs_alg : nullptr),
    jac_times_cjmass_casadi(
      jac_times_cjmass,
      setup_opts.num_solvers > 1,
      serialized_fcns != nullptr ? &serialized_fcns->jac_times_cjmass : nullptr),
    jac_action_casadi(
      jac_action,
      setup_opts.num_solvers > 1,
      serialized_fcns != nullptr ? &serialized_fcns->jac_action : nullptr),
    mass_action_casadi(
      mass_action,
      setup_opts.num_solvers > 1,
      serialized_fcns != nullptr ? &serialized_fcns->mass_action : nullptr),
    sens_casadi(
      sens,
      setup_opts.num_solvers > 1,
      serialized_fcns != nullptr ? &serialized_fcns->sens : nullptr),
    events_casadi(
      events,
      setup_opts.num_solvers > 1,
      serialized_fcns != nullptr ? &serialized_fcns->events : nullptr),
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
    const bool deep_copy = setup_opts.num_solvers > 1;
    if (serialized_fcns != nullptr) {
      if (serialized_fcns->var_fcns.size() != var_fcns.size()
          || serialized_fcns->dvar_dy_fcns.size() != dvar_dy_fcns.size()
          || serialized_fcns->dvar_dp_fcns.size() != dvar_dp_fcns.size()) {
        throw std::invalid_argument("Serialized CasADi function cache has mismatched sizes");
      }
    }

    var_fcns_casadi.reserve(var_fcns.size());
    dvar_dy_fcns_casadi.reserve(dvar_dy_fcns.size());
    dvar_dp_fcns_casadi.reserve(dvar_dp_fcns.size());

    for (size_t k = 0; k < var_fcns.size(); k++) {
      const std::string *serialized =
        serialized_fcns != nullptr ? &serialized_fcns->var_fcns[k] : nullptr;
      var_fcns_casadi.push_back(CasadiFunction(*var_fcns[k], deep_copy, serialized));
    }
    for (int k = 0; k < var_fcns_casadi.size(); k++)
      ExpressionSet::var_fcns.push_back(&this->var_fcns_casadi[k]);

    for (size_t k = 0; k < dvar_dy_fcns.size(); k++) {
      const std::string *serialized =
        serialized_fcns != nullptr ? &serialized_fcns->dvar_dy_fcns[k] : nullptr;
      dvar_dy_fcns_casadi.push_back(CasadiFunction(*dvar_dy_fcns[k], deep_copy, serialized));
    }
    for (int k = 0; k < dvar_dy_fcns_casadi.size(); k++)
      this->dvar_dy_fcns.push_back(&this->dvar_dy_fcns_casadi[k]);

    for (size_t k = 0; k < dvar_dp_fcns.size(); k++) {
      const std::string *serialized =
        serialized_fcns != nullptr ? &serialized_fcns->dvar_dp_fcns[k] : nullptr;
      dvar_dp_fcns_casadi.push_back(CasadiFunction(*dvar_dp_fcns[k], deep_copy, serialized));
    }
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

  sunrealtype* get_tmp_state_vector() override {
    return tmp_state_vector.data();
  }
  sunrealtype* get_tmp_sparse_jacobian_data() override {
    return tmp_sparse_jacobian_data.data();
  }
};

#endif // PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
