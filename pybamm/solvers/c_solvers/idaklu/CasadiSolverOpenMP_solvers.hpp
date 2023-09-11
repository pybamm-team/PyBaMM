#ifndef PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
#define PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP

#include "CasadiSolverOpenMP.hpp"
#include "casadi_solver.hpp"

/**
 * Macro to generate CasadiSolver OpenMP implementations with specified linear
 * solvers
 */
#define CASADISOLVER_NEWCLASS(CLASSNAME, FCN_CALL) \
class CasadiSolverOpenMP_##CLASSNAME : public CasadiSolverOpenMP { \
public: \
  CasadiSolverOpenMP_##CLASSNAME( \
    np_array atol_np, \
    double rel_tol, \
    np_array rhs_alg_id, \
    int number_of_parameters, \
    int number_of_events, \
    int jac_times_cjmass_nnz, \
    int jac_bandwidth_lower, \
    int jac_bandwidth_upper, \
    std::unique_ptr<CasadiFunctions> functions, \
    const Options& options \
  ) : \
    CasadiSolverOpenMP( \
      atol_np, \
      rel_tol, \
      rhs_alg_id, \
      number_of_parameters, \
      number_of_events, \
      jac_times_cjmass_nnz, \
      jac_bandwidth_lower, \
      jac_bandwidth_upper, \
      std::move(functions), \
      options \
    ) \
  { \
    Initialize(); \
  } \
  void SetLinearSolver() override { LS = FCN_CALL; }; \
};

/**
 * @brief CasadiSolver Dense implementation with OpenMP class
 */
CASADISOLVER_NEWCLASS(
  Dense,
  SUNLinSol_Dense(yy, J, sunctx)
)

/**
 * @brief CasadiSolver KLU implementation with OpenMP class
 */
CASADISOLVER_NEWCLASS(
  KLU,
  SUNLinSol_KLU(yy, J, sunctx)
)

/**
 * @brief CasadiSolver Banded implementation with OpenMP class
 */
CASADISOLVER_NEWCLASS(
  Band,
  SUNLinSol_Band(yy, J, sunctx)
)

/**
 * @brief CasadiSolver SPBCGS implementation with OpenMP class
 */
CASADISOLVER_NEWCLASS(
  SPBCGS,
  LS = SUNLinSol_SPBCGS(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
)

/**
 * @brief CasadiSolver SPFGMR implementation with OpenMP class
 */
CASADISOLVER_NEWCLASS(
  SPFGMR,
  LS = SUNLinSol_SPFGMR(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
)

/**
 * @brief CasadiSolver SPGMR implementation with OpenMP class
 */
CASADISOLVER_NEWCLASS(
  SPGMR,
  LS = SUNLinSol_SPGMR(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
)

/**
 * @brief CasadiSolver SPTFQMR implementation with OpenMP class
 */
CASADISOLVER_NEWCLASS(
  SPTFQMR,
  LS = SUNLinSol_SPTFQMR(
    yy,
    precon_type,
    options.linsol_max_iterations,
    sunctx
  );
)

#endif // PYBAMM_IDAKLU_CASADI_SOLVER_OPENMP_HPP
