#ifndef PYBAMM_IDAKLU_CREATE_CASADI_SOLVER_HPP
#define PYBAMM_IDAKLU_CREATE_CASADI_SOLVER_HPP

#include "Expressions/Casadi/CasadiFunctions.hpp"
#include "IDAKLUSolverOpenMP_solvers.hpp"
#include <idas/idas.h>
#include <memory>

/**
 * Creates a concrete casadi solver given a linear solver, as specified in
 * options_cpp.linear_solver.
 * @brief Create a concrete casadi solver given a linear solver
 */
template<class CExprSet, class CExpr, class CExprBase>
IDAKLUSolver *create_idaklu_solver(
  int number_of_states,
  int number_of_parameters,
  const CExprBase &rhs_alg,
  const CExprBase &jac_times_cjmass,
  const np_array_int &jac_times_cjmass_colptrs,
  const np_array_int &jac_times_cjmass_rowvals,
  const int jac_times_cjmass_nnz,
  const int jac_bandwidth_lower,
  const int jac_bandwidth_upper,
  const CExprBase &jac_action,
  const CExprBase &mass_action,
  const CExprBase &sens,
  const CExprBase &events,
  const int number_of_events,
  np_array rhs_alg_id,
  np_array atol_np,
  double rel_tol,
  int inputs_length,
  const std::vector<CExprBase*>& var_fcns,
  const std::vector<CExprBase*>& dvar_dy_fcns,
  const std::vector<CExprBase*>& dvar_dp_fcns,
  py::dict options
) {
  auto options_cpp = Options(options);
  auto functions = std::make_unique<CExprSet>(
    rhs_alg,
    jac_times_cjmass,
    jac_times_cjmass_nnz,
    jac_bandwidth_lower,
    jac_bandwidth_upper,
    jac_times_cjmass_rowvals,
    jac_times_cjmass_colptrs,
    inputs_length,
    jac_action,
    mass_action,
    sens,
    events,
    number_of_states,
    number_of_events,
    number_of_parameters,
    var_fcns,
    dvar_dy_fcns,
    dvar_dp_fcns,
    options_cpp
  );

  IDAKLUSolver *idakluSolver = nullptr;

  // Instantiate solver class
  if (options_cpp.linear_solver == "SUNLinSol_Dense")
  {
    DEBUG("\tsetting SUNLinSol_Dense linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_Dense<CExprSet, CExpr>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      options_cpp
     );
  }
  else if (options_cpp.linear_solver == "SUNLinSol_KLU")
  {
    DEBUG("\tsetting SUNLinSol_KLU linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_KLU<CExprSet, CExpr>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      options_cpp
     );
  }
  else if (options_cpp.linear_solver == "SUNLinSol_Band")
  {
    DEBUG("\tsetting SUNLinSol_Band linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_Band<CExprSet, CExpr>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      options_cpp
     );
  }
  else if (options_cpp.linear_solver == "SUNLinSol_SPBCGS")
  {
    DEBUG("\tsetting SUNLinSol_SPBCGS_linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPBCGS<CExprSet, CExpr>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      options_cpp
     );
  }
  else if (options_cpp.linear_solver == "SUNLinSol_SPFGMR")
  {
    DEBUG("\tsetting SUNLinSol_SPFGMR_linear solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPFGMR<CExprSet, CExpr>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      options_cpp
     );
  }
  else if (options_cpp.linear_solver == "SUNLinSol_SPGMR")
  {
    DEBUG("\tsetting SUNLinSol_SPGMR solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPGMR<CExprSet, CExpr>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      options_cpp
     );
  }
  else if (options_cpp.linear_solver == "SUNLinSol_SPTFQMR")
  {
    DEBUG("\tsetting SUNLinSol_SPGMR solver");
    idakluSolver = new IDAKLUSolverOpenMP_SPTFQMR<CExprSet, CExpr>(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      options_cpp
     );
  }

  if (idakluSolver == nullptr) {
    throw std::invalid_argument("Unsupported solver requested");
  }

  return idakluSolver;
}

#endif // PYBAMM_IDAKLU_CREATE_CASADI_SOLVER_HPP
