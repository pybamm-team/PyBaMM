#ifndef PYBAMM_IDAKLU_CASADISOLVEROPENMP_HPP
#define PYBAMM_IDAKLU_CASADISOLVEROPENMP_HPP

#include "CasadiSolver.hpp"
#include <casadi/casadi.hpp>
using Function = casadi::Function;

#include "casadi_functions.hpp"
#include "common.hpp"
#include "options.hpp"
#include "solution.hpp"
#include "sundials_legacy_wrapper.hpp"

/**
 * @brief Abstract solver class based on OpenMP vectors
 *
 * An abstract class that implements a solution based on OpenMP
 * vectors but needs to be provided with a suitable linear solver.
 *
 * This class broadly implements the following skeleton workflow:
 * (https://sundials.readthedocs.io/en/latest/ida/Usage/index.html)
 *    1. (N/A) Initialize parallel or multi-threaded environment
 *    2. Create the SUNDIALS context object
 *    3. Create the vector of initial values
 *    4. Create matrix object (if appropriate)
 *    5. Create linear solver object
 *    6. (N/A) Create nonlinear solver object
 *    7. Create IDA object
 *    8. Initialize IDA solver
 *    9. Specify integration tolerances
 *   10. Attach the linear solver
 *   11. Set linear solver optional inputs
 *   12. (N/A) Attach nonlinear solver module
 *   13. (N/A) Set nonlinear solver optional inputs
 *   14. Specify rootfinding problem (optional)
 *   15. Set optional inputs
 *   16. Correct initial values (optional)
 *   17. Advance solution in time
 *   18. Get optional outputs
 *   19. Destroy objects
 *   20. (N/A) Finalize MPI
 */
class CasadiSolverOpenMP : public CasadiSolver
{
public:
  void *ida_mem = nullptr;
  np_array atol_np;
  double rel_tol;
  np_array rhs_alg_id;
  int number_of_states;
  int number_of_parameters;
  int number_of_events;
  int precon_type;
  N_Vector yy, yp, avtol;   // y, y', and absolute tolerance
  N_Vector *yyS, *ypS;      // y, y' for sensitivities
  N_Vector id;              // rhs_alg_id
  realtype rtol;
  const int jac_times_cjmass_nnz;
  int jac_bandwidth_lower;
  int jac_bandwidth_upper;
  SUNMatrix J;
  SUNLinearSolver LS;
  std::unique_ptr<CasadiFunctions> functions;
  realtype *res;
  realtype *res_dvar_dy;
  realtype *res_dvar_dp;
  Options options;

#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext sunctx;
#endif

public:
  /**
   * @brief Constructor
   */
  CasadiSolverOpenMP(
    np_array atol_np,
    double rel_tol,
    np_array rhs_alg_id,
    int number_of_parameters,
    int number_of_events,
    int jac_times_cjmass_nnz,
    int jac_bandwidth_lower,
    int jac_bandwidth_upper,
    std::unique_ptr<CasadiFunctions> functions,
    const Options& options);

  /**
   * @brief Destructor
   */
  ~CasadiSolverOpenMP();

  /**
   * Evaluate casadi functions (including sensitivies) for each requested
   * variable and store
   * @brief Evaluate casadi functions
   */
  void CalcVars(
    realtype *y_return,
    size_t length_of_return_vector,
    size_t t_i,
    realtype *tret,
    realtype *yval,
    const std::vector<realtype*>& ySval,
    realtype *yS_return,
    size_t *ySk);

  /**
   * @brief Evaluate casadi functions for sensitivities
   */
  void CalcVarsSensitivities(
    realtype *tret,
    realtype *yval,
    const std::vector<realtype*>& ySval,
    realtype *yS_return,
    size_t *ySk);

  /**
   * @brief The main solve method that solves for each variable and time step
   */
  Solution solve(
    np_array t_np,
    np_array y0_np,
    np_array yp0_np,
    np_array_dense inputs) override;

  /**
   * @brief Concrete implementation of initialization method
   */
  void Initialize() override;

  /**
   * @brief Allocate memory for OpenMP vectors
   */
  void AllocateVectors();

  /**
   * @brief Allocate memory for matrices (noting appropriate matrix format/types)
   */
  void SetMatrix();

  /**
   * @Brief Abstract method to set the linear solver
   */
  virtual void SetLinearSolver() = 0;
};

#endif // PYBAMM_IDAKLU_CASADISOLVEROPENMP_HPP
