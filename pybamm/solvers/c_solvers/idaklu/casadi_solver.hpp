#ifndef PYBAMM_IDAKLU_CREATE_CASADI_SOLVER_HPP
#define PYBAMM_IDAKLU_CREATE_CASADI_SOLVER_HPP

#include "CasadiSolver.hpp"

/**
 * Creates a concrete casadi solver given a linear solver, as specified in
 * options_cpp.linear_solver.
 * @brief Create a concrete casadi solver given a linear solver
 */
CasadiSolver *create_casadi_solver(
  int number_of_states,
  int number_of_parameters,
  const Function &rhs_alg,
  const Function &jac_times_cjmass,
  const np_array_int &jac_times_cjmass_colptrs,
  const np_array_int &jac_times_cjmass_rowvals,
  const int jac_times_cjmass_nnz,
  const int jac_bandwidth_lower,
  const int jac_bandwidth_upper,
  const Function &jac_action,
  const Function &mass_action,
  const Function &sens,
  const Function &event,
  const int number_of_events,
  np_array rhs_alg_id,
  np_array atol_np,
  double rel_tol,
  int inputs_length,
  const std::vector<Function*>& var_casadi_fcns,
  const std::vector<Function*>& dvar_dy_fcns,
  const std::vector<Function*>& dvar_dp_fcns,
  py::dict options
);

#endif // PYBAMM_IDAKLU_CREATE_CASADI_SOLVER_HPP
