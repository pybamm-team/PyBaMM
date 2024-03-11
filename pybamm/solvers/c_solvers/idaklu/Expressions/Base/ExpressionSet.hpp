#ifndef PYBAMM_IDAKLU_EXPRESSION_SET_HPP
#define PYBAMM_IDAKLU_EXPRESSION_SET_HPP

#include "ExpressionTypes.hpp"
#include "ExpressionSparsity.hpp"
#include "Expression.hpp"
#include "../../common.hpp"
#include "../../Options.hpp"
#include <memory>

template <class T, class TBase>
class ExpressionSet
{
public:

  /**
   * @brief Constructor
   */
  ExpressionSet(
    const TBase &rhs_alg,
    const TBase &jac_times_cjmass,
    const int jac_times_cjmass_nnz,
    const int jac_bandwidth_lower,
    const int jac_bandwidth_upper,
    const np_array_int &jac_times_cjmass_rowvals_arg,
    const np_array_int &jac_times_cjmass_colptrs_arg,
    const int inputs_length,
    const TBase &jac_action,
    const TBase &mass_action,
    const TBase &sens,
    const TBase &events,
    const int n_s,
    const int n_e,
    const int n_p,
    const Options& options)
      : number_of_states(n_s),
        number_of_events(n_e),
        number_of_parameters(n_p),
        number_of_nnz(jac_times_cjmass_nnz),
        jac_bandwidth_lower(jac_bandwidth_lower),
        jac_bandwidth_upper(jac_bandwidth_upper),
        rhs_alg(rhs_alg),
        jac_times_cjmass(jac_times_cjmass),
        jac_action(jac_action),
        mass_action(mass_action),
        sens(sens),
        events(events),
        tmp_state_vector(number_of_states),
        tmp_sparse_jacobian_data(jac_times_cjmass_nnz),
        options(options)
      {};

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
  std::vector<T> var_fcns;  // cppcheck-suppress unusedStructMember
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

#endif // PYBAMM_IDAKLU_EXPRESSION_SET_HPP
