#ifndef PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
#define PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP

#include "common.hpp"
#include "Options.hpp"
#include <casadi/casadi.hpp>
#include <casadi/core/sparsity.hpp>
#include <memory>

using Function = casadi::Function;

/**
 * @brief Class for handling individual casadi functions
 */
class CasadiFunction
{
public:
  /**
   * @brief Constructor
   */
  explicit CasadiFunction(const Function &f);

  /**
   * @brief Evaluation operator
   */
  void operator()();

  /**
   * @brief Evaluation operator given data vectors
   */
  void operator()(const std::vector<realtype*>& inputs,
                  const std::vector<realtype*>& results);

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  casadi_int nnz_out();

  /**
   * @brief Return the number of non-zero elements for the function output
   */
  casadi::Sparsity sparsity_out(casadi_int ind);

public:
  std::vector<const double *> m_arg;
  std::vector<double *> m_res;

private:
  const Function &m_func;
  std::vector<casadi_int> m_iw;
  std::vector<double> m_w;
};

/**
 * @brief Class for handling casadi functions
 */
class CasadiFunctions
{
public:
  /**
   * @brief Create a new CasadiFunctions object
   */
  CasadiFunctions(
    const Function &rhs_alg,
    const Function &jac_times_cjmass,
    const int jac_times_cjmass_nnz,
    const int jac_bandwidth_lower,
    const int jac_bandwidth_upper,
    const np_array_int &jac_times_cjmass_rowvals,
    const np_array_int &jac_times_cjmass_colptrs,
    const int inputs_length,
    const Function &jac_action,
    const Function &mass_action,
    const Function &sens,
    const Function &events,
    const int n_s,
    const int n_e,
    const int n_p,
    const std::vector<Function*>& var_casadi_fcns,
    const std::vector<Function*>& dvar_dy_fcns,
    const std::vector<Function*>& dvar_dp_fcns,
    const Options& options
  );

public:
  int number_of_states;
  int number_of_parameters;
  int number_of_events;
  int number_of_nnz;
  int jac_bandwidth_lower;
  int jac_bandwidth_upper;

  CasadiFunction rhs_alg;
  CasadiFunction sens;
  CasadiFunction jac_times_cjmass;
  CasadiFunction jac_action;
  CasadiFunction mass_action;
  CasadiFunction events;

  // NB: cppcheck-suppress unusedStructMember is used because codacy reports
  //     these members as unused even though they are important
  std::vector<CasadiFunction> var_casadi_fcns;  // cppcheck-suppress unusedStructMember
  std::vector<CasadiFunction> dvar_dy_fcns;  // cppcheck-suppress unusedStructMember
  std::vector<CasadiFunction> dvar_dp_fcns;  // cppcheck-suppress unusedStructMember

  std::vector<int64_t> jac_times_cjmass_rowvals;
  std::vector<int64_t> jac_times_cjmass_colptrs;
  std::vector<realtype> inputs;

  Options options;

  realtype *get_tmp_state_vector();
  realtype *get_tmp_sparse_jacobian_data();

private:
  std::vector<realtype> tmp_state_vector;
  std::vector<realtype> tmp_sparse_jacobian_data;
};

#endif // PYBAMM_IDAKLU_CASADI_FUNCTIONS_HPP
