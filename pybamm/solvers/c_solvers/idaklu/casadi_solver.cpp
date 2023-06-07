#include "casadi_solver.hpp"
#include "CasadiSolver.hpp"
#include "CasadiSolverOpenMP_solvers.hpp"
#include "casadi_sundials_functions.hpp"
#include "common.hpp"
#include <idas/idas.h>
#include <memory>

#define CUDA 1
#ifdef CUDA
  #include "CasadiSolverCuda_solvers.hpp"
#endif

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
    const Function &events,
    const int number_of_events,
    np_array rhs_alg_id,
    np_array atol_np,
    double rel_tol,
    int inputs_length,
    py::dict options
) {
  auto options_cpp = Options(options);
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
    options_cpp
  );

  CasadiSolver *casadiSolver = nullptr;
  
  // Instantiate solver class
  if (options_cpp.linear_solver == "SUNLinSol_Dense")
  {
    DEBUG("\tsetting SUNLinSol_Dense linear solver");
    casadiSolver = new CasadiSolverOpenMP_Dense(
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
    casadiSolver = new CasadiSolverOpenMP_KLU(
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
    casadiSolver = new CasadiSolverOpenMP_Band(
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
    casadiSolver = new CasadiSolverOpenMP_SPBCGS(
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
    casadiSolver = new CasadiSolverOpenMP_SPFGMR(
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
    casadiSolver = new CasadiSolverOpenMP_SPGMR(
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
    casadiSolver = new CasadiSolverOpenMP_SPTFQMR(
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
#ifdef CUDA
  else if (options_cpp.linear_solver == "SUNLinSol_cuSolverSp_batchQR")
  {
    DEBUG("\tsetting SUNLinSol_cuSolverSp_batchQR solver");
    casadiSolver = new CasadiSolverCuda_cuSolverSp_batchQR(
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
#endif

  if (casadiSolver == nullptr) {
    throw std::invalid_argument("Unsupported solver requested");
  }

  return casadiSolver;
}
