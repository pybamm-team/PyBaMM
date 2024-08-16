#ifndef PYBAMM_IDAKLU_SOLVEROPENMP_HPP
#define PYBAMM_IDAKLU_SOLVEROPENMP_HPP

#include "IDAKLUSolver.hpp"
#include "common.hpp"
#include "Options.hpp"
#include "Solution.hpp"
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
template <class ExprSet>
class IDAKLUSolverOpenMP : public IDAKLUSolver
{
  // NB: cppcheck-suppress unusedStructMember is used because codacy reports
  //     these members as unused even though they are important in child
  //     classes, but are passed by variadic arguments (and are therefore unnamed)
public:
  void *ida_mem = nullptr;
  np_array atol_np;
  np_array rhs_alg_id;
  int number_of_states;  // cppcheck-suppress unusedStructMember
  int number_of_parameters;  // cppcheck-suppress unusedStructMember
  int number_of_events;  // cppcheck-suppress unusedStructMember
  int precon_type;  // cppcheck-suppress unusedStructMember
  N_Vector yy, yp, avtol;  // y, y', and absolute tolerance
  N_Vector *yyS;  // cppcheck-suppress unusedStructMember
  N_Vector *ypS;  // cppcheck-suppress unusedStructMember
  N_Vector id;              // rhs_alg_id
  realtype rtol;
  const int jac_times_cjmass_nnz;  // cppcheck-suppress unusedStructMember
  int jac_bandwidth_lower;  // cppcheck-suppress unusedStructMember
  int jac_bandwidth_upper;  // cppcheck-suppress unusedStructMember
  SUNMatrix J;
  SUNLinearSolver LS = nullptr;
  std::unique_ptr<ExprSet> functions;
  std::vector<realtype> res;
  std::vector<realtype> res_dvar_dy;
  std::vector<realtype> res_dvar_dp;
  SetupOptions setup_opts;
  SolverOptions solver_opts;

#if SUNDIALS_VERSION_MAJOR >= 6
  SUNContext sunctx;
#endif

public:
  /**
   * @brief Constructor
   */
  IDAKLUSolverOpenMP(
    np_array atol_np,
    double rel_tol,
    np_array rhs_alg_id,
    int number_of_parameters,
    int number_of_events,
    int jac_times_cjmass_nnz,
    int jac_bandwidth_lower,
    int jac_bandwidth_upper,
    std::unique_ptr<ExprSet> functions,
    const SetupOptions &setup_opts,
    const SolverOptions &solver_opts
  );

  /**
   * @brief Destructor
   */
  ~IDAKLUSolverOpenMP();

  /**
   * Evaluate functions (including sensitivies) for each requested
   * variable and store
   * @brief Evaluate functions
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
   * @brief Evaluate functions for sensitivities
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
   * @brief Apply user-configurable IDA options
   */
  void SetSolverOptions();

  /**
   * @brief Check the return flag for errors
   */
  void CheckErrors(int const & flag);
};

#include "IDAKLUSolverOpenMP.inl"

#endif // PYBAMM_IDAKLU_SOLVEROPENMP_HPP
