#include "casadi_solver.hpp"
#include "CasadiSolver.hpp"
#include "CasadiSolverOpenMP_solvers.hpp"
#include "casadi_sundials_functions.hpp"
#include "common.hpp"
#include <idas/idas.h>
#include <memory>
#include <omp.h>

CasadiSolverGroup *create_casadi_solver_group(
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
  const Function &events,
  const int number_of_events,
  np_array rhs_alg_id,
  np_array atol_np,
  double rel_tol,
  int inputs_length,
  const std::vector<Function*>& var_casadi_fcns,
  const std::vector<Function*>& dvar_dy_fcns,
  const std::vector<Function*>& dvar_dp_fcns,
  py::dict options,
  const int nsolvers
) {
  const int nthreads = options["num_threads"].cast<int>();
  const int nsolvers_limited = std::min(nsolvers, nthreads);
  auto options_cpp = Options(options, nsolvers_limited);
  auto functions = std::make_unique<CasadiFunctions>(
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
    var_casadi_fcns,
    dvar_dy_fcns,
    dvar_dp_fcns,
    options_cpp
  );

  std::vector<std::unique_ptr<CasadiSolver>> solvers;
  for (int i = 0; i < nsolvers_limited; i++) {
    solvers.emplace_back(create_casadi_solver(
      std::make_unique<CasadiFunctions>(*functions),
      number_of_parameters,
      jac_times_cjmass_colptrs,
      jac_times_cjmass_rowvals,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      number_of_events,
      rhs_alg_id,
      atol_np,
      rel_tol,
      inputs_length,
      options_cpp
    ));
  }

  // calculate length of return vector as needed for allocating ouput
  int length_of_return_vector = 0;
  if (functions->var_casadi_fcns.size() > 0) {
    // return only the requested variables list after computation
    for (auto& var_fcn : functions->var_casadi_fcns) {
      length_of_return_vector += var_fcn.nnz_out();
    }
  } else {
    // Return full y state-vector
    length_of_return_vector = number_of_states;
  }

  const bool is_output_variables = functions->var_casadi_fcns.size() > 0;
  return new CasadiSolverGroup(std::move(solvers), number_of_states, number_of_parameters, length_of_return_vector, is_output_variables);
}

std::unique_ptr<CasadiSolver> create_casadi_solver(
  std::unique_ptr<CasadiFunctions> functions,
  int number_of_parameters,
  const np_array_int &jac_times_cjmass_colptrs,
  const np_array_int &jac_times_cjmass_rowvals,
  const int jac_times_cjmass_nnz,
  const int jac_bandwidth_lower,
  const int jac_bandwidth_upper,
  const int number_of_events,
  np_array rhs_alg_id,
  np_array atol_np,
  double rel_tol,
  int inputs_length,
  Options options_cpp
) {


  // Instantiate solver class
  if (options_cpp.linear_solver == "SUNLinSol_Dense")
  {
    DEBUG("\tsetting SUNLinSol_Dense linear solver");
    return std::unique_ptr<CasadiSolver>(new CasadiSolverOpenMP_Dense(
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
     ));
  }
  else if (options_cpp.linear_solver == "SUNLinSol_KLU")
  {
    DEBUG("\tsetting SUNLinSol_KLU linear solver");
    return std::unique_ptr<CasadiSolver>(new CasadiSolverOpenMP_KLU(
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
     ));
  }
  else if (options_cpp.linear_solver == "SUNLinSol_Band")
  {
    DEBUG("\tsetting SUNLinSol_Band linear solver");
    return std::unique_ptr<CasadiSolver>(new CasadiSolverOpenMP_Band(
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
     ));
  }
  else if (options_cpp.linear_solver == "SUNLinSol_SPBCGS")
  {
    DEBUG("\tsetting SUNLinSol_SPBCGS_linear solver");
    return std::unique_ptr<CasadiSolver>(new CasadiSolverOpenMP_SPBCGS(
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
     ));
  }
  else if (options_cpp.linear_solver == "SUNLinSol_SPFGMR")
  {
    DEBUG("\tsetting SUNLinSol_SPFGMR_linear solver");
    return std::unique_ptr<CasadiSolver>(new CasadiSolverOpenMP_SPFGMR(
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
     ));
  }
  else if (options_cpp.linear_solver == "SUNLinSol_SPGMR")
  {
    DEBUG("\tsetting SUNLinSol_SPGMR solver");
    return std::unique_ptr<CasadiSolver>(new CasadiSolverOpenMP_SPGMR(
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
     ));
  }
  else if (options_cpp.linear_solver == "SUNLinSol_SPTFQMR")
  {
    DEBUG("\tsetting SUNLinSol_SPGMR solver");
    return std::unique_ptr<CasadiSolver>(new CasadiSolverOpenMP_SPTFQMR(
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
     ));
  }
  throw std::invalid_argument("Unsupported solver requested");
}
