#ifndef PYBAMM_IDAKLU_EXPRESSION_SET_HPP
#define PYBAMM_IDAKLU_EXPRESSION_SET_HPP

#include "Expression.hpp"
#include "../../common.hpp"
#include "../../Options.hpp"

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
    const np_array_int &jac_times_cjmass_rowvals_arg,
    const np_array_int &jac_times_cjmass_colptrs_arg,
    const int inputs_length,
    Expression* jac_action,
    Expression* mass_action,
    Expression* sens,
    Expression* events,
    const int n_s,
    const int n_e,
    const int n_p,
    const SetupOptions& options,
    Expression* alg_res = nullptr,
    Expression* alg_jac = nullptr)
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
        alg_res(alg_res),
        alg_jac(alg_jac),
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
  Expression *alg_res = nullptr;
  Expression *alg_jac = nullptr;

  std::vector<Expression*> var_fcns;
  std::vector<Expression*> dvar_dy_fcns;
  std::vector<Expression*> dvar_dp_fcns;

  std::vector<int64_t> jac_times_cjmass_rowvals;
  std::vector<int64_t> jac_times_cjmass_colptrs;
  // The inputs, followed by the values the precompute function derives from them
  // once per solve
  std::vector<sunrealtype> inputs;
  int number_of_inputs = 0;
  Expression *precompute = nullptr;

  SetupOptions setup_opts;

  // Per-solver preconditioner
  std::unique_ptr<Preconditioner> preconditioner;

  // While set, the Jacobian-vector product and the block preconditioner use
  // this state and cj rather than IDA's (stale outside of IDASolve)
  const sunrealtype *linearisation_y = nullptr;
  sunrealtype linearisation_t = 0.0;
  sunrealtype linearisation_cj = 0.0;
  // If set, differential rows (id > 0.5) of the linearisation are identity rows
  const sunrealtype *linearisation_id = nullptr;

  virtual sunrealtype *get_tmp_state_vector() = 0;
  virtual sunrealtype *get_tmp_sparse_jacobian_data() = 0;

protected:
  std::vector<sunrealtype> tmp_state_vector;
  std::vector<sunrealtype> tmp_sparse_jacobian_data;
};

#endif // PYBAMM_IDAKLU_EXPRESSION_SET_HPP
