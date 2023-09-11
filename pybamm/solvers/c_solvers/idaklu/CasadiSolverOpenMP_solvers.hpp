#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP

#include "CasadiSolverOpenMP.hpp"
#include "casadi_solver.hpp"

/**
 * @brief CasadiSolver Dense implementation with OpenMP class
 */
class CasadiSolverOpenMP_Dense : public CasadiSolverOpenMP {
public:
  CasadiSolverOpenMP_Dense(
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
    CasadiSolverOpenMP(
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
};

/**
 * @brief CasadiSolver KLU implementation with OpenMP class
 */
class CasadiSolverOpenMP_KLU : public CasadiSolverOpenMP {
public:
  CasadiSolverOpenMP_KLU(
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
    CasadiSolverOpenMP(
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
};

/**
 * @brief CasadiSolver Banded implementation with OpenMP class
 */
class CasadiSolverOpenMP_Band : public CasadiSolverOpenMP {
public:
  CasadiSolverOpenMP_Band(
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
    CasadiSolverOpenMP(
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
};

/**
 * @brief CasadiSolver SPBCGS implementation with OpenMP class
 */
class CasadiSolverOpenMP_SPBCGS : public CasadiSolverOpenMP {
public:
  CasadiSolverOpenMP_SPBCGS(
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
    CasadiSolverOpenMP(
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
};

/**
 * @brief CasadiSolver SPFGMR implementation with OpenMP class
 */
class CasadiSolverOpenMP_SPFGMR : public CasadiSolverOpenMP {
public:
  CasadiSolverOpenMP_SPFGMR(
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
    CasadiSolverOpenMP(
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
};

/**
 * @brief CasadiSolver SPGMR implementation with OpenMP class
 */
class CasadiSolverOpenMP_SPGMR : public CasadiSolverOpenMP {
public:
  CasadiSolverOpenMP_SPGMR(
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
    CasadiSolverOpenMP(
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
};

/**
 * @brief CasadiSolver SPTFQMR implementation with OpenMP class
 */
class CasadiSolverOpenMP_SPTFQMR : public CasadiSolverOpenMP {
public:
  CasadiSolverOpenMP_SPTFQMR(
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
    CasadiSolverOpenMP(
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
};

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
