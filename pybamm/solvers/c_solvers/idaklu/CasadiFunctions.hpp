#ifndef PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
#define PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP

#include "common.hpp"
#include "Options.hpp"
#include <casadi/casadi.hpp>
#include <casadi/core/sparsity.hpp>
#include <memory>

class Expression
{
public:
  /**
   * @brief Constructor
   */
  explicit Expression(const casadi::Function &f) : m_func(f) {}

  /**
   * @brief Evaluation operator
   */
  virtual void operator()() = 0;

  /**
   * @brief Evaluation operator given data vectors
   */
  virtual void operator()(const std::vector<realtype*>& inputs,
                  const std::vector<realtype*>& results) = 0;

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  virtual casadi_int nnz_out() = 0;

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  virtual casadi::Sparsity sparsity_out(casadi_int ind) = 0;

public:
  std::vector<const double *> m_arg;
  std::vector<double *> m_res;

//private:
  const casadi::Function &m_func;
  std::vector<casadi_int> m_iw;
  std::vector<double> m_w;
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

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  casadi_int nnz_out() override;

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  casadi::Sparsity sparsity_out(casadi_int ind) override;
};

template <class T>
class ExpressionSet
{
public:

  /**
   * @brief Constructor
   */
  ExpressionSet(
    const casadi::Function &rhs_alg, const casadi::Function &jac_times_cjmass,
    const int jac_times_cjmass_nnz,
    const int jac_bandwidth_lower, const int jac_bandwidth_upper,
    const np_array_int &jac_times_cjmass_rowvals_arg,
    const np_array_int &jac_times_cjmass_colptrs_arg,
    const int inputs_length, const casadi::Function &jac_action,
    const casadi::Function &mass_action, const casadi::Function &sens, const casadi::Function &events,
    const int n_s, int n_e, const int n_p,
    const std::vector<casadi::Function*>& var_casadi_fcns,
    const std::vector<casadi::Function*>& dvar_dy_fcns,
    const std::vector<casadi::Function*>& dvar_dp_fcns,
    const Options& options)
    : number_of_states(n_s), number_of_events(n_e), number_of_parameters(n_p),
      number_of_nnz(jac_times_cjmass_nnz),
      jac_bandwidth_lower(jac_bandwidth_lower), jac_bandwidth_upper(jac_bandwidth_upper),
      rhs_alg(rhs_alg),
      jac_times_cjmass(jac_times_cjmass), jac_action(jac_action),
      mass_action(mass_action), sens(sens), events(events),
      tmp_state_vector(number_of_states),
      tmp_sparse_jacobian_data(jac_times_cjmass_nnz),
      options(options) {};

  int number_of_states;
  int number_of_parameters;
  int number_of_events;
  int number_of_nnz;
  int jac_bandwidth_lower;
  int jac_bandwidth_upper;

  T rhs_alg;
  T sens;
  T jac_times_cjmass;
  T jac_action;
  T mass_action;
  T events;

  // NB: cppcheck-suppress unusedStructMember is used because codacy reports
  //     these members as unused even though they are important
  std::vector<T> var_casadi_fcns;  // cppcheck-suppress unusedStructMember
  std::vector<T> dvar_dy_fcns;  // cppcheck-suppress unusedStructMember
  std::vector<T> dvar_dp_fcns;  // cppcheck-suppress unusedStructMember

  std::vector<int64_t> jac_times_cjmass_rowvals;
  std::vector<int64_t> jac_times_cjmass_colptrs;
  std::vector<realtype> inputs;

  Options options;

  virtual realtype *get_tmp_state_vector() = 0;
  virtual realtype *get_tmp_sparse_jacobian_data() = 0;

//private:
  std::vector<realtype> tmp_state_vector;
  std::vector<realtype> tmp_sparse_jacobian_data;
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
    const std::vector<casadi::Function*>& var_casadi_fcns,
    const std::vector<casadi::Function*>& dvar_dy_fcns,
    const std::vector<casadi::Function*>& dvar_dp_fcns,
    const Options& options
  ) : ExpressionSet<CasadiFunction>(
    rhs_alg, jac_times_cjmass,
    jac_times_cjmass_nnz,
    jac_bandwidth_lower, jac_bandwidth_upper,
    jac_times_cjmass_rowvals_arg,
    jac_times_cjmass_colptrs_arg,
    inputs_length, jac_action,
    mass_action, sens, events,
    n_s, n_e, n_p,
    var_casadi_fcns,
    dvar_dy_fcns,
    dvar_dp_fcns,
    options)
  {
    // convert casadi::Function list to CasadiFunction list
    for (auto& var : var_casadi_fcns) {
      this->var_casadi_fcns.push_back(CasadiFunction(*var));
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

  realtype* get_tmp_state_vector() override {
    return tmp_state_vector.data();
  }
  realtype* get_tmp_sparse_jacobian_data() override {
    return tmp_sparse_jacobian_data.data();
  }
};

#endif // PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
