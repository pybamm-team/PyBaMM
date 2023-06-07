#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_CUDA_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_CUDA_HPP

#include "CasadiSolver.hpp"
#include "casadi_solver.hpp"

class CasadiSolverCuda_cuSolverSp_batchQR : public CasadiSolver {
public:
  cusparseHandle_t cusp;
  cusolverSpHandle_t cusol;
public:
  CasadiSolverCuda_cuSolverSp_batchQR(
    np_array atol_np,
    double rel_tol,
    np_array rhs_alg_id,
    int number_of_parameters,
    int number_of_events,
    int jac_times_cjmass_nnz,
    int jac_bandwidth_lower,
    int jac_bandwidth_upper,
    std::unique_ptr<CasadiFunctions> functions,
    const Options& options
  ) {
    Initialize();
  }
  void Initialize() override;
  Solution solve(
    np_array t_np,
    np_array y0_np,
    np_array yp0_np,
    np_array_dense inputs) override;
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_CUDA_HPP
