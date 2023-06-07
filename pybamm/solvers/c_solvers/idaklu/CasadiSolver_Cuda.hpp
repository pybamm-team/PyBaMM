#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_CUDA_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_CUDA_HPP

#include "casadi_solver.hpp"

class CasadiSolver_cuSolverSp_batchQR : public CasadiSolver {
public:
  cusparseHandle_t cusp;
  cusolverSpHandle_t cusol;
public:
  CasadiSolver_cuSolverSp_batchQR(
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
  ) :
    CasadiSolver(
      atol_np,
      rel_tol,
      rhs_alg_id,
      number_of_parameters,
      number_of_events,
      jac_times_cjmass_nnz,
      jac_bandwidth_lower,
      jac_bandwidth_upper,
      std::move(functions),
      options
    )
  {
    Initialize();
  }
  void SetLinearSolver() override;
  void AllocateVectors();
  void SetMatrix();
  void ChildDestructors();
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_CUDA_HPP
