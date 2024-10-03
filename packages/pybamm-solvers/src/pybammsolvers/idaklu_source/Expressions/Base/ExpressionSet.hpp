#ifndef PYBAMM_IDAKLU_EXPRESSION_SET_HPP
#define PYBAMM_IDAKLU_EXPRESSION_SET_HPP

#include "ExpressionTypes.hpp"
#include "Expression.hpp"
#include "../../common.hpp"
#include "../../Options.hpp"
#include <memory>

template <class T>
class ExpressionSet
{
public:

  /**
   * @brief Constructor
   */
  ExpressionSet(
    Expression* rhs_alg,
    Expression* jac_times_cjmass,
    const int jac_times_cjmass_nnz,
    const int jac_bandwidth_lower,
    const int jac_bandwidth_upper,
    const np_array_int &jac_times_cjmass_rowvals_arg,  // cppcheck-suppress unusedStructMember
    const np_array_int &jac_times_cjmass_colptrs_arg,  // cppcheck-suppress unusedStructMember
    const int inputs_length,
    Expression* jac_action,
    Expression* mass_action,
    Expression* sens,
    Expression* events,
    const int n_s,
    const int n_e,
    const int n_p,
    const SetupOptions& options)
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
        setup_opts(options)
      {};

  int number_of_states;
  int number_of_parameters;
  int number_of_events;
  int number_of_nnz;
  int jac_bandwidth_lower;
  int jac_bandwidth_upper;

  Expression *rhs_alg = nullptr;
  Expression *jac_times_cjmass = nullptr;
  Expression *jac_action = nullptr;
  Expression *mass_action = nullptr;
  Expression *sens = nullptr;
  Expression *events = nullptr;

  // `cppcheck-suppress unusedStructMember` is used because codacy reports
  // these members as unused, but they are inherited through variadics
  std::vector<Expression*> var_fcns;  // cppcheck-suppress unusedStructMember
  std::vector<Expression*> dvar_dy_fcns;  // cppcheck-suppress unusedStructMember
  std::vector<Expression*> dvar_dp_fcns;  // cppcheck-suppress unusedStructMember

  std::vector<int64_t> jac_times_cjmass_rowvals;  // cppcheck-suppress unusedStructMember
  std::vector<int64_t> jac_times_cjmass_colptrs;  // cppcheck-suppress unusedStructMember
  std::vector<realtype> inputs;  // cppcheck-suppress unusedStructMember

  SetupOptions setup_opts;

  virtual realtype *get_tmp_state_vector() = 0;
  virtual realtype *get_tmp_sparse_jacobian_data() = 0;

protected:
  std::vector<realtype> tmp_state_vector;
  std::vector<realtype> tmp_sparse_jacobian_data;
};

#endif // PYBAMM_IDAKLU_EXPRESSION_SET_HPP
